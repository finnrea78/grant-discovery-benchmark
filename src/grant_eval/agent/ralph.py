"""The Ralph outer driver: re-runs the agent loop under fixed settings from `config.RALPH`.

Each iteration is a fresh run. Nothing carries over: `backend_factory` makes a new
backend every time and `run_loop` rebuilds its messages. The scored result is always
the last iteration (`final`); choosing the best iteration would need the ground truth.

The driver stops when an iteration is schema-valid and finds nothing that the previous
iteration did not. On iteration 1 a non-empty result always triggers a second run,
while a valid empty result stops at once. Opportunities are compared with the same
match key the scorer uses (`grant_eval.scoring.identity`), so the two cannot disagree
about what counts as the same opportunity.
"""
from __future__ import annotations

from collections.abc import Container, Iterable
from dataclasses import dataclass

from grant_eval.agent.loop import LoopResult, run_loop
from grant_eval.schema import ExtractedOpportunity
from grant_eval.scoring.identity import (
    MatchKey,
    normalise_url,
    opportunity_match_key,
    shared_urls,
)

# Old name kept for the tests that still use it.
canonical_url = normalise_url


def identity_key(
    opportunity: ExtractedOpportunity,
    funder: str,
    shared: Container[str] = frozenset(),
) -> MatchKey:
    """Convergence key for one extracted opportunity.

    `(funder, normalised_url, normalised_title)`, where the title is only used when the
    URL is in `shared`. The scorer treats a URL as shared when more than one
    ground-truth row has it. Ralph cannot see the ground truth, so it uses the agent's
    own records instead: a URL is shared if more than one extracted record has it.
    Keying on the URL alone would merge several schemes listed on one page (all 8
    `society-antiquaries` schemes, for example) into one key and make convergence
    meaningless. Always using the title would treat a reworded title as a new find.
    """
    return opportunity_match_key(opportunity, funder, shared)


def shared_urls_in(opportunities: Iterable[ExtractedOpportunity]) -> frozenset[str]:
    """Normalised URLs that appear on more than one of these records (see `identity_key`)."""
    return shared_urls((o.url or "") for o in opportunities)


@dataclass
class RalphResult:
    """Outcome of one Ralph run for one funder cell.

    * ``final``: the scored iteration, which is the last loop run.
    * ``ralph_iterations``: how many loop runs happened (1..max_iterations).
    * ``scored_iteration_policy``: copied from config (always ``"final"``).
    * ``iterations``: every iteration's ``LoopResult`` in order.
    """

    final: LoopResult
    ralph_iterations: int
    scored_iteration_policy: str
    iterations: list[LoopResult]


def run_ralph(
    *,
    backend_factory,
    model,
    temperature,
    toolset,
    strategy,
    entry_url,
    funder,
    step_budget,
    ralph_config,
    snapshot_date,
    loop_config=None,
    trace_dir=None,
) -> RalphResult:
    """Re-run the agent loop under fixed parameters until convergence or budget.

    Args:
        backend_factory: zero-arg callable returning a new ``RouterBackend``. Called
            once per iteration so nothing carries over.
        model, temperature, strategy, entry_url, funder, step_budget, snapshot_date:
            passed straight through to ``run_loop`` unchanged every iteration.
        toolset: shared across iterations; it is a stateless HTTP client.
        ralph_config: the ``grant_eval.config.RALPH`` dict.
        loop_config: the ``grant_eval.config.LOOP`` dict, passed to ``run_loop``.
            ``None`` uses run_loop's defaults.
        trace_dir: optional directory; if given, iteration ``i`` traces to
            ``<trace_dir>/iter{i}.jsonl``.

    Returns:
        RalphResult with the scored iteration and every iteration's result.
    """
    max_iter = ralph_config["max_iterations"]
    policy = ralph_config["scored_iteration_policy"]

    prev: LoopResult | None = None  # None => no previous iteration yet
    results: list[LoopResult] = []
    # URLs that need the title to tell records apart. Only ever grows, so a record's
    # key cannot change meaning during the run.
    shared: frozenset[str] = frozenset()

    for i in range(1, max_iter + 1):
        backend = backend_factory()  # new each iteration
        trace_path = f"{trace_dir}/iter{i}.jsonl" if trace_dir else None
        lr = run_loop(
            backend=backend,
            model=model,
            temperature=temperature,
            toolset=toolset,
            strategy=strategy,
            entry_url=entry_url,
            funder=funder,
            step_budget=step_budget,
            snapshot_date=snapshot_date,
            trace_path=trace_path,
            **(loop_config or {}),
        )
        results.append(lr)

        # Key both iterations with the same shared set so the comparison is fair.
        shared |= shared_urls_in(lr.opportunities)
        cur_keys = {identity_key(o, funder, shared) for o in lr.opportunities}
        seen = (
            set()
            if prev is None
            else {identity_key(o, funder, shared) for o in prev.opportunities}
        )
        no_new = cur_keys.issubset(seen)

        if lr.schema_valid and no_new:
            return RalphResult(
                final=lr,
                ralph_iterations=i,
                scored_iteration_policy=policy,
                iterations=results,
            )
        prev = lr

    # No convergence within the budget: score the last run.
    return RalphResult(
        final=results[-1],
        ralph_iterations=max_iter,
        scored_iteration_policy=policy,
        iterations=results,
    )
