"""`grid_status`: progress, failures, spend and ETA for a grid run.

Everything is read from the results directory, so it works whether or not a run is
still going. Totals come from the enumerated cells, so ones not yet started show as pending.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grant_eval.dataset.registry import DEFAULT_REGISTRY_PATH, load_registry
from grant_eval.grid import store
from grant_eval.grid.cells import enumerate_cells
from grant_eval.schema import ExtractionResult, Split

if TYPE_CHECKING:
    from grant_eval.grid.cells import Cell

# With fewer usable latencies than this, an ETA would be noise, so report none.
_MIN_RESULTS_FOR_ETA: int = 5


@dataclass(frozen=True)
class StatusReport:
    """A grid run's progress. `complete + failed + pending` always equals `total`."""

    total: int
    complete: int
    failed: int
    pending: int
    total_cost_usd: float
    mean_latency_s: float | None
    eta_hours: float | None


def _read_result(cell: "Cell", results_root: Path) -> ExtractionResult | None:
    """The parsed result for `cell`, or `None` if incomplete or unreadable."""
    try:
        raw = cell.result_path(results_root).read_text(encoding="utf-8")
        return ExtractionResult.model_validate_json(raw)
    except (OSError, ValueError):
        return None


def summarise(
    results_root: Path, plan: list["Cell"], *, workers: int = 8
) -> StatusReport:
    """Summarise `plan`'s progress under `results_root`.

    `workers` is only used for the ETA; pass the run's actual `--workers` for an
    accurate one.
    """
    results_root = Path(results_root)
    complete = 0
    failed = 0
    total_cost_usd = 0.0
    latencies: list[float] = []

    for cell in plan:
        if store.is_complete(cell, results_root):
            complete += 1
            result = _read_result(cell, results_root)
            if result is not None:
                if result.cost_usd is not None:
                    total_cost_usd += result.cost_usd
                if result.latency_s is not None:
                    latencies.append(result.latency_s)
        elif store.is_failed(cell, results_root):
            failed += 1

    pending = len(plan) - complete - failed
    mean_latency_s = sum(latencies) / len(latencies) if latencies else None

    eta_hours = None
    if mean_latency_s is not None and len(latencies) >= _MIN_RESULTS_FOR_ETA:
        eta_hours = pending * mean_latency_s / workers / 3600

    return StatusReport(
        total=len(plan),
        complete=complete,
        failed=failed,
        pending=pending,
        total_cost_usd=total_cost_usd,
        mean_latency_s=mean_latency_s,
        eta_hours=eta_hours,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid_status",
        description="Report progress, failures, spend and ETA for a grid run, "
        "derived entirely from the results directory.",
    )
    parser.add_argument("--split", choices=["tune", "validation"], default="tune")
    parser.add_argument("--funder", action="append", default=None,
                        help="restrict to this funder slug; repeatable")
    parser.add_argument("--model", action="append", default=None,
                        help="restrict to this grid model label; repeatable")
    parser.add_argument("--prompt", action="append", default=None,
                        help="restrict to this prompt strategy; repeatable")
    parser.add_argument("--workers", type=int, default=8,
                        help="workers assumed for eta_hours (default: 8)")
    parser.add_argument("--results-root", type=Path,
                        default=Path("eval/final_grid"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    plan = enumerate_cells(
        registry, Split(args.split),
        funders=args.funder, models=args.model, prompts=args.prompt,
    )
    report = summarise(args.results_root, plan, workers=args.workers)

    eta = f"{report.eta_hours:.1f}h" if report.eta_hours is not None else "n/a"
    mean_latency = (
        f"{report.mean_latency_s:.1f}s" if report.mean_latency_s is not None else "n/a"
    )
    print(
        f"total={report.total} complete={report.complete} failed={report.failed} "
        f"pending={report.pending} total_cost_usd={report.total_cost_usd:.2f} "
        f"mean_latency={mean_latency} eta={eta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
