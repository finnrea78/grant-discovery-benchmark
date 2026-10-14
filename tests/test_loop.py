"""Tests for the agent loop, run offline with MockBackend and a fake toolset.

Covers the four outcome flags, the single repair turn, the rebuilt message history
and the JSONL trace.
"""
from __future__ import annotations

import json

from grant_eval.agent.loop import SUBMIT_TOOL_NAME, LoopResult, run_loop
from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage
from grant_eval.agent.tools import ToolResult
from grant_eval.schema import ExtractedOpportunity


class FakeToolset:
    """Scripted stand-in for grant_eval.agent.tools.Toolset.

    `plan` maps a tool name to one ToolResult (returned every call) or a list of
    ToolResults (used in order). Every dispatched ToolCall is recorded.
    """

    def __init__(self, plan: dict):
        self._plan = {k: list(v) if isinstance(v, list) else v for k, v in plan.items()}
        self.dispatched: list[ToolCall] = []

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": SUBMIT_TOOL_NAME,
                    "description": "Submit the final opportunities and end the run.",
                    "parameters": {
                        "type": "object",
                        "properties": {"opportunities": {"type": "array"}},
                        "required": ["opportunities"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "Fetch a page.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        self.dispatched.append(call)
        entry = self._plan[call.name]
        if isinstance(entry, list):
            return entry.pop(0)
        return entry


def _opp(oid: str) -> ExtractedOpportunity:
    return ExtractedOpportunity(
        funder="Leverhulme Trust",
        opportunity_id=oid,
        title=f"Title {oid}",
        url=f"https://example.org/{oid}",
    )


_COMMON = dict(
    model="fake/model",
    temperature=0.0,
    strategy="minimal",
    entry_url="https://example.org/",
    funder="Leverhulme Trust",
    snapshot_date="2026-08-09",
)


def test_fetch_then_submit_two_and_message_reconstruction():
    u1 = Usage(100, 20, 0.0, "mock", "fake/model")
    u2 = Usage(200, 40, 0.0, "mock", "fake/model")
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": "https://example.org/"})]),
            Reply(
                tool_calls=[
                    ToolCall(
                        "c2",
                        SUBMIT_TOOL_NAME,
                        {"opportunities": [{"a": 1}, {"b": 2}]},
                    )
                ]
            ),
        ],
        usages=[u1, u2],
    )
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE TEXT", terminate=False),
            SUBMIT_TOOL_NAME: ToolResult(
                content="Submitted 2.",
                terminate=True,
                submission=[_opp("o1"), _opp("o2")],
                abstained=False,
            ),
        }
    )
    messages: list[dict] = []
    result = run_loop(
        backend=backend, toolset=toolset, step_budget=8, _messages_sink=messages, **_COMMON
    )

    assert isinstance(result, LoopResult)
    assert len(result.opportunities) == 2
    assert result.abstained is False
    assert result.schema_valid is True
    assert result.step_budget_exhausted is False
    assert result.submit_failed is False
    assert result.steps_used == 2
    # usage is the sum of the two scripted usages
    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 60
    assert result.usage.cost_usd == 0.0

    # the rebuilt history should be in the OpenAI/LiteLLM message shape
    assistant_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if m["tool_calls"][0]["id"] == "c1":
                assistant_idx = i
                break
    assert assistant_idx is not None, "no reconstructed assistant turn for c1"
    tc = messages[assistant_idx]["tool_calls"][0]
    assert tc["id"] == "c1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "fetch_page"
    # arguments are a JSON string, not a dict
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"url": "https://example.org/"}
    nxt = messages[assistant_idx + 1]
    assert nxt["role"] == "tool"
    assert nxt["tool_call_id"] == "c1"
    assert nxt["content"] == "PAGE TEXT"


def test_abstention_empty_submit():
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": []})]),
        ]
    )
    toolset = FakeToolset(
        {
            SUBMIT_TOOL_NAME: ToolResult(
                content="Submitted 0 (abstention).",
                terminate=True,
                submission=[],
                abstained=True,
            ),
        }
    )
    result = run_loop(backend=backend, toolset=toolset, step_budget=8, **_COMMON)

    assert result.abstained is True
    assert result.schema_valid is True
    assert result.opportunities == []
    assert result.step_budget_exhausted is False
    assert result.submit_failed is False
    assert result.steps_used == 1


def test_n1_repair_exhausted_two_invalid_submits():
    invalid = ToolResult(content="validation error: fix and resubmit", terminate=False)
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 2}]})]),
        ]
    )
    toolset = FakeToolset({SUBMIT_TOOL_NAME: [invalid, invalid]})
    result = run_loop(backend=backend, toolset=toolset, step_budget=8, **_COMMON)

    assert result.submit_failed is True
    assert result.schema_valid is False
    assert result.step_budget_exhausted is False
    assert result.abstained is False
    assert result.opportunities == []
    # the second submit used the free repair turn, so only one step was spent
    assert result.steps_used == 1
    assert result.repair_turns_used == 1
    assert backend.calls == 2


def test_n1_repair_succeeds_invalid_then_valid():
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"ok": 1}]})]),
        ]
    )
    toolset = FakeToolset(
        {
            SUBMIT_TOOL_NAME: [
                ToolResult(content="validation error", terminate=False),
                ToolResult(
                    content="Submitted 1.",
                    terminate=True,
                    submission=[_opp("o1")],
                    abstained=False,
                ),
            ]
        }
    )
    result = run_loop(backend=backend, toolset=toolset, step_budget=8, **_COMMON)

    assert result.schema_valid is True
    assert result.submit_failed is False
    assert result.abstained is False
    assert len(result.opportunities) == 1
    assert result.opportunities[0].opportunity_id == "o1"
    # the resubmit used the free repair turn
    assert result.steps_used == 1
    assert result.repair_turns_used == 1
    assert backend.calls == 2


# The repair turn sits outside `step_budget`, which only counts exploration steps.
# Otherwise a rejected submit on the last step could never be repaired.
def _invalid() -> ToolResult:
    return ToolResult(content="validation error: fix and resubmit", terminate=False)


def _valid(n: int = 1) -> ToolResult:
    return ToolResult(
        content=f"Submitted {n}.",
        terminate=True,
        submission=[_opp(f"o{i}") for i in range(1, n + 1)],
        abstained=False,
    )


def test_repair_turn_runs_after_step_budget_is_exhausted():
    """A submit rejected on the last step can still be repaired."""
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": "https://example.org/"})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c3", SUBMIT_TOOL_NAME, {"opportunities": [{"ok": 1}]})]),
        ]
    )
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: [_invalid(), _valid(2)],
        }
    )
    result = run_loop(backend=backend, toolset=toolset, step_budget=2, **_COMMON)

    assert result.schema_valid is True
    assert result.submit_failed is False
    assert result.step_budget_exhausted is False
    assert len(result.opportunities) == 2
    # the exploration budget was fully spent, and the repair ran anyway
    assert result.steps_used == 2
    assert result.repair_turns_used == 1
    assert backend.calls == 3  # steps_used + repair_turns_used


def test_second_invalid_submit_still_fails_hard_on_the_repair_turn():
    """There is only one repair turn, so a second invalid submit fails the run."""
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 2}]})]),
        ]
    )
    toolset = FakeToolset({SUBMIT_TOOL_NAME: [_invalid(), _invalid()]})
    result = run_loop(backend=backend, toolset=toolset, step_budget=1, **_COMMON)

    assert result.submit_failed is True
    assert result.schema_valid is False
    assert result.step_budget_exhausted is False
    assert result.steps_used == 1
    assert result.repair_turns_used == 1
    assert backend.calls == 2
    # the loop stopped at the second rejection, not at the budget
    assert len(toolset.dispatched) == 2


def test_repair_turn_does_not_consume_exploration_budget():
    """A rejection mid-run does not use up any exploration steps."""
    nav = Reply(tool_calls=[ToolCall("cn", "fetch_page", {"url": "https://example.org/"})])
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            nav,  # the repair turn, spent on navigation instead
            nav,
            nav,
        ]
    )
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: [_invalid()],
        }
    )
    result = run_loop(backend=backend, toolset=toolset, step_budget=3, **_COMMON)

    assert result.step_budget_exhausted is True
    assert result.steps_used == 3  # never more than the budget
    assert result.repair_turns_used == 1
    assert backend.calls == 4


def test_no_budget_notice_on_the_repair_turn():
    """The budget notice counts exploration steps and is not sent on the repair turn."""
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": "u"})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c3", SUBMIT_TOOL_NAME, {"opportunities": [{"ok": 1}]})]),
        ]
    )
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: [_invalid(), _valid(1)],
        }
    )
    messages: list[dict] = []
    result = run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=2,
        budget_notice_at=2,
        _messages_sink=messages,
        **_COMMON,
    )

    assert result.schema_valid is True
    notices = [m for m in messages
               if m["role"] == "user" and "left" in (m.get("content") or "")]
    assert len(notices) == 2, [m["content"] for m in notices]
    assert "2 steps left" in notices[0]["content"]
    assert "1 step left" in notices[1]["content"]
    # the repair turn (c3) should see the validation error last, not a notice
    repair_idx = next(
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
        and m["tool_calls"][0]["id"] == "c3"
    )
    before = messages[repair_idx - 1]
    assert before["role"] == "tool"
    assert "validation error" in before["content"]


def test_trace_marks_the_repair_turn_and_the_rejection(tmp_path):
    """The trace records the rejection and marks the repair turn."""
    trace_path = tmp_path / "repair.jsonl"
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": [{"bad": 1}]})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": [{"ok": 1}]})]),
        ]
    )
    toolset = FakeToolset({SUBMIT_TOOL_NAME: [_invalid(), _valid(1)]})
    run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=4,
        trace_path=str(trace_path),
        **_COMMON,
    )
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]

    completions = [e for e in events if e["event"] == "completion"]
    assert len(completions) == 2
    assert [e["repair"] for e in completions] == [False, True]

    rejections = [e for e in events if e["event"] == "schema_rejection"]
    assert len(rejections) == 1
    assert rejections[0]["invalid_submits"] == 1
    assert rejections[0]["repair_granted"] is True
    # the rejection did not use up a step
    assert rejections[0]["remaining"] == 3

    (final,) = [e for e in events if e["event"] == "result"]
    assert final["steps_used"] == 1
    assert final["repair_turns_used"] == 1

    (prompt,) = [e for e in events if e["event"] == "prompt"]
    assert prompt["config"]["repair_turn_outside_budget"] is True


def test_reasoning_only_turn_counts_against_step_budget():
    """A turn with no tool call still counts as a step, so reasoning runs stay bounded."""
    budget = 3
    backend = MockBackend(
        replies=[Reply(text=None, reasoning=f"thinking {i}") for i in range(budget)]
    )
    toolset = FakeToolset({})
    messages: list[dict] = []
    result = run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=budget,
        _messages_sink=messages,
        **_COMMON,
    )

    assert result.step_budget_exhausted is True
    assert result.steps_used == budget
    assert result.repair_turns_used == 0
    assert backend.calls == budget
    assert toolset.dispatched == []
    # reasoning is carried over as assistant content by default
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        f"thinking {i}" for i in range(budget)
    ]


def test_budget_exhaustion_distinct_from_abstention():
    budget = 4
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall(f"c{i}", "fetch_page", {"url": "https://example.org/"})])
            for i in range(budget)
        ]
    )
    toolset = FakeToolset({"fetch_page": ToolResult(content="PAGE", terminate=False)})
    result = run_loop(backend=backend, toolset=toolset, step_budget=budget, **_COMMON)

    assert result.step_budget_exhausted is True
    assert result.abstained is False
    assert result.schema_valid is False
    assert result.submit_failed is False
    assert result.opportunities == []
    assert result.steps_used == budget


def test_plaintext_reply_burns_a_step():
    budget = 2
    backend = MockBackend(
        replies=[
            Reply(text="thinking...", tool_calls=[]),
            Reply(text="still thinking...", tool_calls=[]),
        ]
    )
    toolset = FakeToolset({})
    messages: list[dict] = []
    result = run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=budget,
        _messages_sink=messages,
        **_COMMON,
    )

    assert result.step_budget_exhausted is True
    assert result.steps_used == budget
    assert toolset.dispatched == []
    # plain-text turns are kept as they are, with no nudge added
    texts = [m["content"] for m in messages if m.get("role") == "assistant"]
    assert "thinking..." in texts
    assert "still thinking..." in texts


def test_trace_file(tmp_path):
    trace_path = tmp_path / "traces" / "run1.jsonl"
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": []})]),
        ]
    )
    toolset = FakeToolset(
        {SUBMIT_TOOL_NAME: ToolResult(content="abstain", terminate=True, submission=[], abstained=True)}
    )
    result = run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=8,
        trace_path=str(trace_path),
        **_COMMON,
    )
    assert result.trace_path == str(trace_path)
    assert trace_path.exists()

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    kinds = [e.get("event") for e in events]

    # the prompt event holds the exact system and user prompt
    prompt_events = [e for e in events if e.get("event") == "prompt"]
    assert len(prompt_events) == 1
    from grant_eval.agent.prompts import render

    system, user = render(
        "minimal",
        entry_url=_COMMON["entry_url"],
        funder=_COMMON["funder"],
        snapshot_date=_COMMON["snapshot_date"],
    )
    assert prompt_events[0]["system"] == system
    assert prompt_events[0]["user"] == user

    assert "result" in kinds


def test_trace_carries_the_served_provider_per_step_and_in_the_final_summary(tmp_path):
    trace_path = tmp_path / "traces" / "run1.jsonl"
    backend = MockBackend(
        replies=[
            Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": "https://example.org/x"})]),
            Reply(tool_calls=[ToolCall("c2", SUBMIT_TOOL_NAME, {"opportunities": []})]),
        ],
        usages=[
            Usage(1, 1, 0.0, "mock", "fake/model", provider="DeepInfra"),
            Usage(1, 1, 0.0, "mock", "fake/model", provider="Together"),
        ],
    )
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="page", terminate=False),
            SUBMIT_TOOL_NAME: ToolResult(content="abstain", terminate=True, submission=[], abstained=True),
        }
    )
    run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=8,
        trace_path=str(trace_path),
        **_COMMON,
    )

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    completion_events = [e for e in events if e.get("event") == "completion"]
    assert [e["usage"]["provider"] for e in completion_events] == ["DeepInfra", "Together"]

    # the result event takes the provider of the last completion, like Usage.__add__
    result_events = [e for e in events if e.get("event") == "result"]
    assert result_events[0]["usage"]["provider"] == "Together"


def test_trace_provider_is_none_when_the_backend_never_reports_one(tmp_path):
    """With no provider reported, the trace stores null rather than dropping the key."""
    trace_path = tmp_path / "traces" / "run1.jsonl"
    backend = MockBackend(
        replies=[Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": []})])]
    )
    toolset = FakeToolset(
        {SUBMIT_TOOL_NAME: ToolResult(content="abstain", terminate=True, submission=[], abstained=True)}
    )
    run_loop(
        backend=backend,
        toolset=toolset,
        step_budget=8,
        trace_path=str(trace_path),
        **_COMMON,
    )

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    completion_events = [e for e in events if e.get("event") == "completion"]
    assert completion_events[0]["usage"]["provider"] is None
    result_events = [e for e in events if e.get("event") == "result"]
    assert result_events[0]["usage"]["provider"] is None


# Budget notice: the only message the loop adds itself.
def test_budget_notice_fires_only_near_the_end():
    """With budget 4 and threshold 2, the notice lands before steps 2 and 3 only."""
    backend = MockBackend(
        replies=[Reply(tool_calls=[ToolCall(f"c{i}", "fetch_page", {"url": "u"})])
                 for i in range(4)],
        usages=[Usage(1, 1, 0.0, "fake", "m") for _ in range(4)],
    )
    toolset = FakeToolset({"fetch_page": ToolResult(content="PAGE", terminate=False)})
    messages: list[dict] = []
    common = {**_COMMON}
    result = run_loop(backend=backend, toolset=toolset, step_budget=4,
                      budget_notice_at=2, _messages_sink=messages, **common)

    assert result.step_budget_exhausted is True
    notices = [m for m in messages
               if m["role"] == "user" and "step" in (m.get("content") or "")
               and "left" in (m.get("content") or "")]
    assert len(notices) == 2, [m["content"] for m in notices]
    assert "2 steps left" in notices[0]["content"]
    assert "1 step left" in notices[1]["content"]
    assert "submit_opportunities" in notices[0]["content"]


def test_budget_notice_can_be_disabled():
    backend = MockBackend(
        replies=[Reply(tool_calls=[ToolCall(f"c{i}", "fetch_page", {"url": "u"})])
                 for i in range(3)],
        usages=[Usage(1, 1, 0.0, "fake", "m") for _ in range(3)],
    )
    toolset = FakeToolset({"fetch_page": ToolResult(content="PAGE", terminate=False)})
    messages: list[dict] = []
    run_loop(backend=backend, toolset=toolset, step_budget=3,
             budget_notice_at=0, _messages_sink=messages, **_COMMON)

    assert not [m for m in messages
                if m["role"] == "user" and "steps left" in (m.get("content") or "")]


def _one_traced_run(trace, steps: int = 1):
    """One run to `trace`: `steps - 1` fetches, then a valid empty submit."""
    replies = [Reply(tool_calls=[ToolCall(f"f{i}", "fetch_page", {"url": "u"})])
               for i in range(steps - 1)]
    replies.append(Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME,
                                              {"opportunities": []})]))
    backend = MockBackend(replies=replies)
    toolset = FakeToolset({
        "fetch_page": ToolResult(content="PAGE", terminate=False),
        SUBMIT_TOOL_NAME: ToolResult(
            content="Submitted 0.", terminate=True, submission=[], abstained=True),
    })
    return run_loop(backend=backend, toolset=toolset, step_budget=16,
                    trace_path=str(trace), **_COMMON)


def _events(trace):
    return [json.loads(line) for line in
            trace.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_restarted_cell_appends_and_stays_distinguishable_from_one_clean_run(tmp_path):
    """A restarted run appends to the trace, and each event is tagged with its attempt."""
    trace = tmp_path / "iter1.jsonl"

    _one_traced_run(trace, steps=2)
    first = _events(trace)
    result = _one_traced_run(trace, steps=4)
    second = _events(trace)

    assert len(second) > len(first), "the abandoned attempt was erased"
    assert [e["attempt"] for e in second if e.get("event") == "attempt"] == [1, 2]
    assert all("attempt" in e for e in second), "an event nothing can attribute"

    # each attempt can be read back as a full run on its own
    attempt_1 = [e for e in second if e["attempt"] == 1]
    attempt_2 = [e for e in second if e["attempt"] == 2]
    assert attempt_1 == [dict(e) for e in first]
    for events in (attempt_1, attempt_2):
        assert events[-1]["event"] == "result"
    steps = lambda evs: [e["step"] for e in evs if e.get("event") == "completion"]
    assert steps(attempt_1) == [0, 1]
    assert steps(attempt_2) == [0, 1, 2, 3]

    # the result says which attempt it was, since its cost only covers that attempt
    assert result.attempt == 2


def test_a_re_run_of_a_gzipped_trace_restores_it_and_appends(tmp_path):
    """A re-run unpacks a gzipped trace first, so one file holds every attempt."""
    import gzip

    trace = tmp_path / "iter1.jsonl"
    _one_traced_run(trace, steps=2)
    archive = trace.with_suffix(".jsonl.gz")
    with trace.open("rb") as fin, gzip.open(archive, "wb") as fout:
        fout.writelines(fin)
    trace.unlink()

    _one_traced_run(trace, steps=3)

    assert not archive.exists(), "the .gz must not survive beside the restored .jsonl"
    events = _events(trace)
    assert [e["attempt"] for e in events if e.get("event") == "attempt"] == [1, 2]
    assert sum(1 for e in events if e.get("event") == "result") == 2


# SUBMIT_TOOL_NAME has to match a real tool in tool_schemas().
def test_submit_tool_name_matches_real_toolset():
    from grant_eval.agent.tools import Toolset

    ts = Toolset(base_url="http://localhost:0", agent_config={})
    names = {t["function"]["name"] for t in ts.tool_schemas()}
    assert SUBMIT_TOOL_NAME in names
