"""Run one extraction end to end and package it as an `ExtractionResult`.

Token and cost figures are summed over every Ralph iteration, since each one was paid
for. The opportunities and outcome flags come from the scored (final) iteration. Cost
from an earlier attempt that was cut short by a 429 is not included; it is only in the
trace, and `attempts` says which attempt the cost belongs to.

`config_hash` covers the experimental settings only (model, prompt strategy and a hash
of its template, temperature, agent config, Ralph and loop settings). Funder, replicate
and snapshot date are left out because they are stored as separate fields.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from grant_eval.agent.ralph import run_ralph
from grant_eval.agent.router import Usage
from grant_eval.agent.tools import Toolset
from grant_eval.config import AGENT_CONFIGS, LOOP, RALPH
from grant_eval.schema import ExtractionResult, Split

# Fixed values to render the prompt template with, so the fingerprint does not vary by
# funder. Read-only because `prompt_fingerprint` is cached on `strategy` alone: changing
# this mapping would give one config two different hashes.
SENTINEL_PROMPT_CONTEXT = MappingProxyType(
    {
        "entry_url": "https://example.invalid/",
        "funder": "SENTINEL FUNDER",
        "snapshot_date": date(2000, 1, 1),
    }
)


@lru_cache(maxsize=None)
def prompt_fingerprint(strategy: str) -> str:
    """SHA-256 of the prompt template for ``strategy``.

    Editing a prompt's text changes the hash even if its name stays the same. An
    unimplemented strategy returns ``unimplemented:<strategy>`` so the grid can still be
    planned; preflight refuses to run such a cell.
    """
    from grant_eval.agent import prompts

    try:
        system, user = prompts.render(strategy, **SENTINEL_PROMPT_CONTEXT)
    except NotImplementedError:
        return f"unimplemented:{strategy}"
    encoded = json.dumps([system, user], sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    """Settings and identity of one grid run for one funder."""

    funder: str
    split: Split                 # e.g. Leverhulme = Split.TUNE
    snapshot_date: date
    model: str                   # full model string, e.g. "openrouter/amazon/nova-lite-v1"
    prompt_strategy: str         # e.g. "minimal"
    temperature: float
    agent_config: str            # label from `config.AGENT_CONFIGS`
    replicate: int


def config_hash(
    *, model: str, prompt_strategy: str, temperature: float,
    agent_config: str, agent_cfg: dict, ralph_config: dict, loop_config: dict,
) -> str:
    """SHA-256 over the experimental settings of a run.

    Funder, replicate, snapshot date and ``config.GRID_SEED`` are left out.
    ``loop_config`` is included because its settings change results: moving
    ``budget_notice_at`` from 2 to 3 changed one funder's count from 5 to 7. Callers use
    the hash before the run to name its output and trace paths.
    """
    payload = {
        "model": model,
        "prompt_strategy": prompt_strategy,
        "prompt_fingerprint": prompt_fingerprint(prompt_strategy),
        "temperature": temperature,
        "agent_config": agent_config,
        "agent_cfg": agent_cfg,
        "ralph_config": ralph_config,
        "loop_config": loop_config,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_extraction(
    *,
    run_config: RunConfig,
    base_url: str,
    backend_factory,
    entry_url: str,
    agent_configs=AGENT_CONFIGS,
    ralph_config=RALPH,
    loop_config=LOOP,
    trace_dir=None,
    out_path=None,
) -> ExtractionResult:
    """Drive one full Ralph run over a mirror and assemble an ``ExtractionResult``.

    Args:
        run_config: the frozen run identity + experimental knobs (see ``RunConfig``).
        base_url: base URL of a running ``grant_eval.dataset.serve`` for this
            funder snapshot. The Toolset can only reach this server.
        backend_factory: zero-arg callable returning a new ``RouterBackend`` per
            Ralph iteration, so nothing carries between iterations.
        entry_url: the funder entry URL the agent starts from.
        agent_configs: label -> agent settings (defaults to the grid's).
        ralph_config: Ralph settings (defaults to ``config.RALPH``).
        loop_config: loop settings (defaults to ``config.LOOP``). Part of ``config_hash``.
        trace_dir: optional dir; iteration traces go to ``<trace_dir>/iterN.jsonl``.
        out_path: optional path; if given, the result JSON is written there (parent dirs
            created) before returning.

    Returns:
        The ``ExtractionResult`` for this cell.
    """
    agent_cfg = agent_configs[run_config.agent_config]
    toolset = Toolset(base_url, agent_cfg)

    start = time.perf_counter()
    ralph = run_ralph(
        backend_factory=backend_factory,
        model=run_config.model,
        temperature=run_config.temperature,
        toolset=toolset,
        strategy=run_config.prompt_strategy,
        entry_url=entry_url,
        funder=run_config.funder,
        step_budget=agent_cfg["step_budget"],
        ralph_config=ralph_config,
        loop_config=loop_config,
        snapshot_date=run_config.snapshot_date,
        trace_dir=trace_dir,
    )
    latency_s = time.perf_counter() - start

    # The total ends up tagged with the last iteration's backend and model.
    total = Usage(0, 0, 0.0, "", run_config.model)
    for it in ralph.iterations:
        total += it.usage

    final = ralph.final

    cfg_hash = config_hash(
        model=run_config.model,
        prompt_strategy=run_config.prompt_strategy,
        temperature=run_config.temperature,
        agent_config=run_config.agent_config,
        agent_cfg=agent_cfg,
        ralph_config=ralph_config,
        loop_config=loop_config,
    )

    result = ExtractionResult(
        snapshot_date=run_config.snapshot_date,
        funder=run_config.funder,
        split=run_config.split,
        model=run_config.model,
        prompt_strategy=run_config.prompt_strategy,
        temperature=run_config.temperature,
        agent_config=run_config.agent_config,
        replicate=run_config.replicate,
        config_hash=cfg_hash,
        # from the scored (final) iteration
        opportunities=final.opportunities,
        # summed over the whole run
        input_tokens=total.input_tokens,
        output_tokens=total.output_tokens,
        cost_usd=total.cost_usd,
        latency_s=latency_s,
        ralph_iterations=ralph.ralph_iterations,
        scored_iteration_policy=ralph.scored_iteration_policy,
        backend=total.backend,
        provider=total.provider,
        attempts=final.attempt,
        # Summed over all iterations: a repair cost tokens whichever iteration it was in.
        repair_turns_used=sum(it.repair_turns_used for it in ralph.iterations),
        # outcome flags from the scored iteration
        step_budget_exhausted=final.step_budget_exhausted,
        abstained=final.abstained,
        schema_valid=final.schema_valid,
        submit_failed=final.submit_failed,
    )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    return result
