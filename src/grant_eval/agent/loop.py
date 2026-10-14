"""The agent loop: read pages and call tools until the agent submits or runs out of steps.

The loop is the same for every model; only the model string changes between arms. It
does no compaction, no summarising and no hidden sub-agents, and it adds nothing to the
conversation except a short notice when few steps are left. If a small model overflows
its context, that is recorded as a failure of that model.

`step_budget` counts exploration steps. After a rejected submit the agent gets one
repair turn, which does not use up a step. Any other completion, including one that
only reasons and calls no tool, uses one step.

The router returns normalised `Reply`/`ToolCall` objects, so the loop rebuilds each
assistant turn in OpenAI/LiteLLM shape (a `tool_calls` list with JSON-string
`arguments`) so tool results can be matched by `tool_call_id` on the next turn. Each
step can be written to an append-only JSONL trace.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, replace

from grant_eval.agent.prompts import render
from grant_eval.agent.router import Usage
from grant_eval.schema import ExtractedOpportunity

# Must match the name in tools.py; a test checks this against the real Toolset.
SUBMIT_TOOL_NAME = "submit_opportunities"


@dataclass
class LoopResult:
    """The outcome of one loop run.

    Outcome flags:

    * ``schema_valid``: a valid submit happened (an empty one counts).
    * ``abstained``: the valid submit was empty.
    * ``step_budget_exhausted``: steps ran out with no valid submit.
    * ``submit_failed``: a second invalid submit after the repair turn.

    ``steps_used`` counts exploration steps and never exceeds ``step_budget``.
    ``repair_turns_used`` (0 or 1) is kept separate so repair rates can be compared
    across models. ``usage`` covers both, since the repair turn still costs money.
    """

    opportunities: list[ExtractedOpportunity]
    abstained: bool
    step_budget_exhausted: bool
    submit_failed: bool
    schema_valid: bool
    steps_used: int
    usage: Usage
    trace_path: str | None
    repair_turns_used: int = 0
    # >1 means an earlier attempt was cut short by a 429 and its spend is only in the
    # trace. Set by `_finish`.
    attempt: int = 1


def run_loop(
    *,
    backend,
    model,
    temperature,
    toolset,
    strategy,
    entry_url,
    funder,
    step_budget,
    snapshot_date,
    trace_path=None,
    carry_reasoning: bool = True,
    budget_notice_at: int = 3,
    _messages_sink=None,
) -> LoopResult:
    """Run the read/act loop to a terminal submit or the step-budget cap.

    Args:
        backend:      a ``RouterBackend`` (``.name``, ``.complete(...)``).
        model:        the model string, the only thing that varies between arms.
        temperature:  sampling temperature, held constant within an arm.
        toolset:      a ``Toolset`` (``.tool_schemas()``, ``.dispatch(call)``).
        strategy:     prompt strategy name for ``prompts.render``.
        entry_url:    the funder entry URL the agent starts from.
        funder:       the funder display name.
        step_budget:  maximum exploration steps. The one repair turn is on top, so
                      a run makes at most ``step_budget + 1`` completions.
        snapshot_date: capture date of the mirror. The inclusion rule is defined
                      relative to it.
        budget_notice_at: when this many steps or fewer remain, add one short user
                      turn saying how many are left and asking for a submit. Without
                      it a model has no way to know it is about to run out. 0
                      disables it. Never sent on the repair turn.
        trace_path:   append-only JSONL trace path. ``None`` writes nothing.
        carry_reasoning: when a reasoning model puts its turn in ``reasoning``
                      rather than ``content``, use it as the content so the turn is
                      not empty.
        _messages_sink: for tests only. A list passed here becomes the live
                      ``messages`` list so the transcript can be inspected.

    Returns:
        LoopResult with the outcome flags and total usage.
    """
    system, user = render(
        strategy, entry_url=entry_url, funder=funder, snapshot_date=snapshot_date
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if _messages_sink is not None:
        # share list identity so the caller sees appends as they happen
        _messages_sink.clear()
        _messages_sink.extend(messages)
        messages = _messages_sink

    tools = toolset.tool_schemas()
    total = Usage(0, 0, 0.0, backend.name, model)
    invalid_submits = 0

    # The repair turn is counted separately so a model that formats badly is not also
    # docked an exploration step.
    steps_used = 0  # exploration completions used; capped by step_budget
    repair_turns_used = 0  # 0 or 1: the free repair turn, once it has actually run
    repair_pending = False  # the NEXT completion is the free repair turn

    tracer = _Tracer(trace_path)
    tracer.emit(
        {
            "event": "prompt",
            "system": system,
            "user": user,
            "config": {
                "backend": backend.name,
                "model": model,
                "temperature": temperature,
                "strategy": strategy,
                "entry_url": entry_url,
                "funder": funder,
                "step_budget": step_budget,
                # Older traces charged the repair turn to the budget.
                "repair_turn_outside_budget": True,
                "carry_reasoning": carry_reasoning,
                "budget_notice_at": budget_notice_at,
                "snapshot_date": str(snapshot_date),
            },
        }
    )

    while repair_pending or steps_used < step_budget:
        is_repair = repair_pending
        repair_pending = False
        # The repair turn reuses the index of the step whose submit was rejected.
        step = steps_used - 1 if is_repair else steps_used

        # Without this, runs kept planning more page reads until the budget ended with
        # nothing submitted. Not sent on the repair turn, which is outside the budget.
        remaining = step_budget - steps_used
        if not is_repair and budget_notice_at and remaining <= budget_notice_at:
            notice = (
                f"You have {remaining} step{'' if remaining == 1 else 's'} left. "
                f"Call submit_opportunities now with everything you have gathered so "
                f"far. Submitting an empty list is valid if nothing is in scope."
            )
            messages.append({"role": "user", "content": notice})
            tracer.emit({"event": "budget_notice", "step": step, "remaining": remaining})

        reply, usage = backend.complete(messages, model, temperature, tools)
        if is_repair:
            repair_turns_used += 1
        else:
            steps_used += 1
        total += usage
        tracer.emit(
            {
                "event": "completion",
                "step": step,
                "repair": is_repair,
                "text": reply.text,
                # Tells a thinking-only step apart from an empty one.
                "reasoning": reply.reasoning,
                "empty": reply.is_empty,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments}
                    for c in reply.tool_calls
                ],
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                    "backend": usage.backend,
                    "model": usage.model,
                    # Per step, so a provider change mid-run shows up in the trace.
                    "provider": usage.provider,
                },
            }
        )

        # Rebuild the assistant turn in OpenAI/LiteLLM shape so the next completion can
        # match tool results by id.
        #
        # A reasoning model may return its prose in `reasoning` with `content: None`.
        # Appending an empty turn like that sent one DeepSeek run into 20 empty steps
        # out of 24, so the reasoning is used as the content instead.
        content = reply.text
        if not (content or "").strip() and carry_reasoning and reply.reasoning:
            content = reply.reasoning
        assistant = {"role": "assistant", "content": content}
        if reply.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in reply.tool_calls
            ]
        messages.append(assistant)

        if reply.has_tool_calls:
            for call in reply.tool_calls:
                result = toolset.dispatch(call)
                tracer.emit(
                    {
                        "event": "tool_result",
                        "step": step,
                        "repair": is_repair,
                        "tool_call_id": call.id,
                        "name": call.name,
                        "terminate": result.terminate,
                        "content": result.content,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                    }
                )

                if call.name == SUBMIT_TOOL_NAME and result.terminate:
                    # Valid submit (incl. valid empty = abstention): terminate.
                    return _finish(
                        tracer,
                        LoopResult(
                            opportunities=result.submission or [],
                            abstained=result.abstained,
                            step_budget_exhausted=False,
                            submit_failed=False,
                            schema_valid=True,
                            steps_used=steps_used,
                            usage=total,
                            trace_path=trace_path,
                            repair_turns_used=repair_turns_used,
                        ),
                    )

                if call.name == SUBMIT_TOOL_NAME and not result.terminate:
                    # Invalid submit. The validation error is already in the transcript and
                    # the agent gets one repair turn; a second invalid submit fails.
                    invalid_submits += 1
                    grant = invalid_submits < 2
                    tracer.emit(
                        {
                            "event": "schema_rejection",
                            "step": step,
                            "repair": is_repair,
                            "invalid_submits": invalid_submits,
                            "repair_granted": grant,
                            "remaining": step_budget - steps_used,
                        }
                    )
                    if not grant:
                        return _finish(
                            tracer,
                            LoopResult(
                                opportunities=[],
                                abstained=False,
                                step_budget_exhausted=False,
                                submit_failed=True,
                                schema_valid=False,
                                steps_used=steps_used,
                                usage=total,
                                trace_path=trace_path,
                                repair_turns_used=repair_turns_used,
                            ),
                        )
                    # The repair turn runs next and does not use a step, so a rejection
                    # on the last step can still be repaired.
                    repair_pending = True
            continue

        # Plain-text reply with no tool calls: it used a step. No guidance is added.
        continue

    # Out of steps with no valid submit.
    return _finish(
        tracer,
        LoopResult(
            opportunities=[],
            abstained=False,
            step_budget_exhausted=True,
            submit_failed=False,
            schema_valid=False,
            steps_used=steps_used,
            usage=total,
            trace_path=trace_path,
            repair_turns_used=repair_turns_used,
        ),
    )


def _finish(tracer: "_Tracer", result: LoopResult) -> LoopResult:
    """Stamp the tracer's attempt index onto the result, emit the final summary event,
    and return it."""
    result = replace(result, attempt=tracer.attempt)
    tracer.emit(
        {
            "event": "result",
            "opportunities": len(result.opportunities),
            "abstained": result.abstained,
            "step_budget_exhausted": result.step_budget_exhausted,
            "submit_failed": result.submit_failed,
            "schema_valid": result.schema_valid,
            "steps_used": result.steps_used,
            "repair_turns_used": result.repair_turns_used,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_usd": result.usage.cost_usd,
                "backend": result.usage.backend,
                "model": result.usage.model,
                "provider": result.usage.provider,
            },
        }
    )
    return result


class _Tracer:
    """JSONL tracer, one JSON object per line. A ``None`` path writes nothing.

    Appends rather than overwrites, so a cell restarted after a 429 keeps the earlier
    attempt's steps and cost. Each attempt starts with an ``attempt`` event and every
    line carries the attempt number. If only ``<path>.gz`` exists (the cell finished and
    is being re-run), it is decompressed first so the new attempt is added to it.
    """

    def __init__(self, trace_path: str | None) -> None:
        self._path = trace_path
        self._started = False
        self.attempt = 1
        if self._path is not None:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._restore_gzipped()
            self.attempt = self._previous_attempts() + 1

    def _restore_gzipped(self) -> None:
        """Decompress ``<path>.gz`` back to ``<path>`` and delete the archive.

        Does nothing if a plain ``.jsonl`` already exists. Fsynced before the delete so
        a crash cannot lose both copies."""
        archive = self._path + ".gz"
        if not os.path.exists(archive) or os.path.exists(self._path):
            return
        with gzip.open(archive, "rb") as fin, open(self._path, "wb") as fout:
            fout.writelines(fin)
            fout.flush()
            os.fsync(fout.fileno())
        os.remove(archive)

    def _previous_attempts(self) -> int:
        """How many attempts this trace already holds. Unparseable lines (such as a
        half-written last line from a killed run) are skipped."""
        count = 0
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        if json.loads(line).get("event") == "attempt":
                            count += 1
                    except ValueError:
                        continue
        except OSError:
            return 0
        return count

    def emit(self, event: dict) -> None:
        if self._path is None:
            return
        with open(self._path, "a", encoding="utf-8") as fh:
            if not self._started:
                self._started = True
                fh.write(json.dumps({"event": "attempt", "attempt": self.attempt}) + "\n")
            fh.write(json.dumps({"attempt": self.attempt, **event}) + "\n")
