"""Grid run CLI: `uv run python -m grant_eval.run`.

A run is scoped if any of `--funder`, `--model`, `--prompt`, `--replicate` or `--limit`
is given. A full-grid run refuses to start on any failed preflight check; a scoped run
only enforces the checks its scope needs and warns about the rest.

`--split validation` refuses to run unless `LOCKED_CONFIG_PATH` holds a valid lock file,
and then runs only the cells it names, since the held-out funders can only be used once.

The cap flags are stored in `grid_run.json` and a resume that changes any of them is
refused without `--force-resume`. `--max-spend-usd` applies to the whole run across
sessions, not to one process, so `SpendGuard` starts from the spend already recorded.
Each session's outcome is also written to `grid_run.json`.

INFO goes to stdout and DEBUG to `<results-root>/grid_run.log` (see `grid/logs.py`).
`--dry-run` prints the seeded plan and exits without touching anything.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from grant_eval import config
from grant_eval.dataset.registry import DEFAULT_REGISTRY_PATH, load_registry
from grant_eval.grid import freeze, logs, preflight, status, store
from grant_eval.grid.cells import Cell, enumerate_cells, ordered_plan
from grant_eval.grid.executor import (
    MAX_RATE_LIMIT_REQUEUES,
    GridOutcome,
    run_grid,
)
from grant_eval.grid.policy import CircuitBreaker, SpendGuard
from grant_eval.schema import Split

_LOG = logging.getLogger(__name__)

# Lock file for `--split validation`. A module attribute so tests can monkeypatch it.
LOCKED_CONFIG_PATH: Path = Path("eval/locked-config.json")

# A lock must say which cells, when, and on what evidence.
_LOCK_REQUIRED_KEYS = frozenset({"cells", "locked_at", "tune_evidence"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grant_eval.run",
        description="Run the evaluation grid: 1,920 tune cells or, with "
        "--split validation and a committed lock file, the validation "
        "cells it names.",
    )
    parser.add_argument("--split", choices=["tune", "validation"], default="tune")
    parser.add_argument("--funder", action="append", default=None,
                        help="restrict to this funder slug; repeatable")
    parser.add_argument("--model", action="append", default=None,
                        help="restrict to this grid model label; repeatable")
    parser.add_argument("--prompt", action="append", default=None,
                        help="restrict to this prompt strategy; repeatable")
    parser.add_argument("--replicate", type=int, action="append", default=None,
                        help=f"restrict to this replicate number "
                             f"(1..{config.REPLICATES}); repeatable")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N cells of the seeded plan")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-arm-limit", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the seeded plan and exit; no side effects")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force-resume", action="store_true",
                        help="continue past a result_sha mismatch")
    parser.add_argument("--max-spend-usd", type=float, default=150.0,
                        help="cumulative across sessions, not per process")
    # Per model request, not per Ralph iteration, which can make up to
    # `step_budget + 1` requests.
    parser.add_argument("--iteration-timeout-s", type=float, default=180.0,
                        help="per MODEL REQUEST timeout in seconds; one Ralph "
                             "iteration issues up to step_budget + 1 requests")
    parser.add_argument("--cell-timeout-s", type=float, default=5400.0)
    parser.add_argument("--max-rate-limit-requeues", type=int,
                        default=MAX_RATE_LIMIT_REQUEUES,
                        help="429 re-queues one cell may take before it is FAILED, "
                             "so a saturated arm reaches the circuit breaker")
    parser.add_argument("--results-root", type=Path,
                        default=Path("eval/final_grid"))
    # Not `.../grid`: `Cell.trace_dir` adds `grid/` itself.
    parser.add_argument("--traces-root", type=Path,
                        default=Path("eval/final_grid_traces"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    # Console verbosity only; the run log always gets DEBUG. It does not affect results,
    # so it is not recorded as a cap.
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help=f"console log level (default INFO, or "
                             f"${logs.LEVEL_ENV_VAR})")
    return parser


def is_scoped(args: argparse.Namespace) -> bool:
    """Whether any of `--funder`, `--model`, `--prompt`, `--replicate` or `--limit` is
    given. `--limit 0` still counts as scoped."""
    return bool(
        args.funder or args.model or args.prompt or args.replicate
    ) or args.limit is not None


def _checked_replicates(values: list[int] | None) -> list[int] | None:
    """Return `values`, or exit if any replicate is outside 1..`config.REPLICATES`,
    since it would match no cell."""
    if values is None:
        return None
    bad = sorted({v for v in values if not 1 <= v <= config.REPLICATES})
    if bad:
        raise SystemExit(
            f"--replicate: {', '.join(str(v) for v in bad)} out of range. Valid "
            f"replicates are 1..{config.REPLICATES} (config.REPLICATES). An "
            "out-of-range replicate names no cell, so the run would plan zero cells."
        )
    return values


def _checked_requeues(value: int) -> int:
    """Return `value`, or exit if it is below 1: then the first 429 would fail the cell
    and count toward the circuit breaker."""
    if value < 1:
        raise SystemExit(
            f"--max-rate-limit-requeues: {value} must be at least 1. At zero or below "
            "the first 429 on an arm fails the cell outright, and a failed cell counts "
            "toward the circuit breaker, so a transient rate limit would halt the run."
        )
    return value


def _print_plan(plan: list[Cell], *, split: Split) -> None:
    """Print every cell id in run order, then a summary line. No timestamps, so the
    same arguments always print the same text."""
    for cell in plan:
        print(cell.cell_id)
    print(f"dry run: {len(plan)} cells (split={split.value}, seed={config.GRID_SEED})")


def _load_validation_lock(path: Path) -> list[str]:
    """The cell ids in the lock file at `path`, or exit saying what is wrong with it
    (missing, unreadable, or missing a required key)."""
    if not path.is_file():
        raise SystemExit(
            f"--split validation refuses to run: no committed lock file at {path}. "
            "Validation funders are held out and can only be run once, so they "
            "need a committed lock file first. Emit one with "
            "`uv run lock_config` (see `grid/lock_config.py`), then commit it."
        )

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"--split validation refuses to run: {path} could not be read as "
            f"JSON ({exc}). A malformed lock file is refused exactly like a "
            "missing one."
        ) from exc

    if not isinstance(data, dict):
        raise SystemExit(
            f"--split validation refuses to run: {path} must contain a JSON "
            f"object, got {type(data).__name__}."
        )

    missing = _LOCK_REQUIRED_KEYS - data.keys()
    if missing:
        raise SystemExit(
            f"--split validation refuses to run: {path} is missing required "
            f"key(s): {', '.join(sorted(missing))}. Expected "
            f"{{'cells': [...], 'locked_at': <iso8601>, 'tune_evidence': <str>}}."
        )

    cells = data["cells"]
    if not isinstance(cells, list) or not cells or not all(
        isinstance(cell_id, str) for cell_id in cells
    ):
        raise SystemExit(
            f"--split validation refuses to run: {path}'s 'cells' must be a "
            "non-empty list of cell id strings."
        )
    if not isinstance(data["locked_at"], str) or not isinstance(
        data["tune_evidence"], str
    ):
        raise SystemExit(
            f"--split validation refuses to run: {path}'s 'locked_at' and "
            "'tune_evidence' must both be strings."
        )
    return cells


def _apply_validation_lock(cells: list[Cell], *, path: Path) -> list[Cell]:
    """Keep only the cells whose ids the lock file at `path` names.

    Matched by `Cell.cell_id` so the right cells run, not just the right number. Exits
    if the lock names an id the current grid no longer produces.
    """
    locked_ids = _load_validation_lock(path)
    by_id = {cell.cell_id: cell for cell in cells}
    unknown = [cell_id for cell_id in locked_ids if cell_id not in by_id]
    if unknown:
        raise SystemExit(
            "--split validation refuses to run: the lock file names cell "
            f"id(s) not produced by the current grid definition: "
            f"{', '.join(unknown)}. This means the grid changed since the "
            "config was locked; re-derive the lock file before running."
        )
    return [by_id[cell_id] for cell_id in locked_ids]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Checked before `--dry-run` returns so a dry run catches a bad value too.
    max_rate_limit_requeues = _checked_requeues(args.max_rate_limit_requeues)
    scoped = is_scoped(args)
    split = Split(args.split)

    registry = load_registry(args.registry)
    cells = enumerate_cells(
        registry, split,
        funders=args.funder, models=args.model, prompts=args.prompt,
        replicates=_checked_replicates(args.replicate),
    )

    if split is Split.VALIDATION:
        # Before the dry-run return, so a dry run is refused the same way.
        cells = _apply_validation_lock(cells, path=LOCKED_CONFIG_PATH)

    plan = ordered_plan(cells, config.GRID_SEED)

    if args.limit is not None:
        plan = plan[: args.limit]

    if args.dry_run:
        # Returns before anything is written.
        _print_plan(plan, split=split)
        return 0

    results_root = Path(args.results_root)
    traces_root = Path(args.traces_root)

    # The run log sits beside `grid_run.json` and is appended to across sessions.
    console_level = (
        logs.level_from_env() if args.log_level is None
        else logging.getLevelName(args.log_level)
    )
    run_log_path = logs.configure(results_root, console_level=console_level)
    _LOG.info("run log: %s", run_log_path)

    # Read only; the session entry in the header needs the count.
    todo = store.pending(plan, results_root, retry_failed=args.retry_failed)

    # The spend cap covers the whole run, so start from what is already spent.
    prior_spend = status.summarise(
        results_root, plan, workers=args.workers
    ).total_cost_usd

    caps = {
        "cell_timeout_s": args.cell_timeout_s,
        "iteration_timeout_s": args.iteration_timeout_s,
        "max_rate_limit_requeues": max_rate_limit_requeues,
        "max_spend_usd": args.max_spend_usd,
        "per_arm_limit": args.per_arm_limit,
    }

    header_path = results_root / "grid_run.json"
    try:
        freeze.verify_or_raise(
            header_path,
            force_resume=args.force_resume,
            seed=config.GRID_SEED,
            split=split.value,
            planned_cells=len(plan),
            workers=args.workers,
            cells_pending=len(todo),
            caps=caps,
        )
    except (freeze.FreezeViolation, freeze.CorruptRunHeader) as exc:
        raise SystemExit(str(exc)) from exc

    # Imported here because `litellm` is slow to import and a dry run does not need it.
    from grant_eval.agent.router import OpenRouterBackend

    backend_factory = lambda: OpenRouterBackend()  # noqa: E731

    checks = preflight.preflight(
        plan, registry=registry, scoped=scoped,
        results_root=results_root, backend_factory=backend_factory,
    )
    try:
        preflight.enforce(checks, scoped=scoped)
    except preflight.PreflightError as exc:
        raise SystemExit(str(exc)) from exc

    # For the log only: `run_grid` works out its own pending cells from the full plan.
    _LOG.info(
        "%d of %d cells pending ($%.2f already spent on this plan)",
        len(todo), len(plan), prior_spend,
    )

    def on_cell(cell: Cell, cell_status: str, result) -> None:  # noqa: ANN001
        # One line per finished cell, failures included, so losses are visible.
        _LOG.info("%s: %s", cell_status, cell.cell_id)

    outcome = GridOutcome(skipped=len(plan), total_cost_usd=prior_spend)
    try:
        outcome = run_grid(
            plan,
            results_root=results_root,
            traces_root=traces_root,
            workers=args.workers,
            per_arm_limit=args.per_arm_limit,
            backend_factory=backend_factory,
            breaker=CircuitBreaker(),
            spend_guard=SpendGuard(args.max_spend_usd, total=prior_spend),
            iteration_timeout_s=args.iteration_timeout_s,
            cell_timeout_s=args.cell_timeout_s,
            max_rate_limit_requeues=max_rate_limit_requeues,
            on_cell=on_cell,
            retry_failed=args.retry_failed,
        )
    except KeyboardInterrupt:
        # Cells in flight are dropped and nothing is marked failed, so the run can be
        # resumed. Recorded so the header does not show a clean finish.
        outcome = replace(
            outcome, halted_kind="interrupted", halted_reason="interrupted by operator"
        )
        _LOG.warning("interrupted: cells in flight abandoned, nothing marked failed")
    finally:
        # A halt or Ctrl-C can leave traces that were never compressed, so sweep them.
        swept = store.sweep_traces(traces_root)
        if swept:
            _LOG.info("swept %d orphaned trace file(s) under %s", swept, traces_root)

        # An interrupted session still records what it did.
        freeze.record_outcome(
            header_path,
            completed=outcome.completed,
            failed=outcome.failed,
            skipped=outcome.skipped,
            halted_kind=outcome.halted_kind,
            halted_reason=outcome.halted_reason,
            halted_cell=outcome.halted_cell,
            session_cost_usd=outcome.total_cost_usd - prior_spend,
            total_cost_usd=outcome.total_cost_usd,
        )

    _LOG.info(
        "completed=%d failed=%d skipped=%d deferred=%d total_cost_usd=%.2f "
        "halted_kind=%s halted_reason=%s",
        outcome.completed, outcome.failed, outcome.skipped, outcome.deferred,
        outcome.total_cost_usd, outcome.halted_kind, outcome.halted_reason,
    )
    if outcome.halted_cell is not None:

        _LOG.error("halted on cell: %s", json.dumps(outcome.halted_cell))
    if outcome.deferred:
        # Deferred cells are unfinished, so do not report success.
        return 1
    return 0 if outcome.halted_kind is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
