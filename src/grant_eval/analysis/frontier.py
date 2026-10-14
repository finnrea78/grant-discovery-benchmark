"""Cost-effectiveness Pareto frontier over configs.

Cost is the realised total of each config's billed runs (runs that never billed are
counted, not treated as free). Macro F1 is resampled by funder to estimate how often
each config stays on the frontier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dataset as ds
from . import stats as st

__all__ = ["FrontierPoint", "pareto_frontier", "cost_effectiveness"]


@dataclass(frozen=True)
class FrontierPoint:
    """One config's position in cost/F1 space, and whether it is Pareto-optimal.

    `dominated_by` lists every config that dominates this one, not just the first.
    `n_cost_excluded` counts runs that failed before billing. `n_resample_funder_gaps`
    counts resampled funders with no F1 value for this config (only possible under a
    `runs=` filter).
    """

    config_label: str
    cost_usd: float
    macro_f1: float
    on_frontier: bool
    dominated_by: tuple[str, ...]
    frontier_probability: float | None
    n_cost_excluded: int = 0
    n_resample_funder_gaps: int = 0


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if `a` is no worse than `b` on cost and F1 and strictly better on one."""
    cost_a, f1_a = a
    cost_b, f1_b = b
    not_worse = cost_a <= cost_b and f1_a >= f1_b
    strictly_better = cost_a < cost_b or f1_a > f1_b
    return not_worse and strictly_better


def pareto_frontier(
    points: dict[str, tuple[float, float]],
) -> tuple[FrontierPoint, ...]:
    """The Pareto frontier over `{label: (cost, f1)}`.

    O(n^2), which is fine for 24 configs. `frontier_probability` is left as `None`;
    `cost_effectiveness` fills it in.
    """
    labels = list(points)
    results = []
    for label in labels:
        dominated_by = tuple(
            other for other in labels if other != label and _dominates(points[other], points[label])
        )
        results.append(
            FrontierPoint(
                config_label=label,
                cost_usd=points[label][0],
                macro_f1=points[label][1],
                on_frontier=len(dominated_by) == 0,
                dominated_by=dominated_by,
                frontier_probability=None,
            )
        )
    return tuple(results)


def _config_costs(
    data: ds.AnalysisDataset, runs: tuple[ds.RunRecord, ...] | None
) -> dict[str, tuple[float, int]]:
    """`{config_label: (total_cost, n_excluded)}`, counting unbilled runs separately."""
    pool = data.runs if runs is None else runs
    totals: dict[str, float] = {}
    excluded: dict[str, int] = {}
    for r in pool:
        totals.setdefault(r.config_label, 0.0)
        excluded.setdefault(r.config_label, 0)
        if r.cost_usd is None:
            excluded[r.config_label] += 1
        else:
            totals[r.config_label] += r.cost_usd
    return {label: (totals[label], excluded[label]) for label in totals}


def _macro_f1_from_funder_means(funder_means: dict[str, float]) -> float | None:
    """Unweighted mean over funder means, or `None` if there are none."""
    if not funder_means:
        return None
    return sum(funder_means.values()) / len(funder_means)


def cost_effectiveness(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    n_resamples: int = 2000,
    seed: int = st.DEFAULT_SEED,
) -> tuple[FrontierPoint, ...]:
    """The cost-effectiveness frontier, with the share of funder resamples each
    config is on it.

    Cost is a realised total so it is held fixed; only macro F1 is resampled.
    """
    pool = data.runs if runs is None else runs
    funder_means = data.funder_means(value="f1", runs=runs)
    costs = _config_costs(data, runs)

    labels = sorted(set(funder_means) | set(costs))
    point_estimate: dict[str, tuple[float, float]] = {}
    for label in labels:
        total_cost, _ = costs.get(label, (0.0, 0))
        macro_f1 = _macro_f1_from_funder_means(funder_means.get(label, {}))
        if macro_f1 is None:
            continue
        point_estimate[label] = (total_cost, macro_f1)

    base_frontier = pareto_frontier(point_estimate)

    funders = tuple(sorted({r.funder for r in pool}))
    n_frontier_hits = {label: 0 for label in point_estimate}
    n_funder_gaps = {label: 0 for label in point_estimate}
    if n_resamples > 0 and funders:
        rng = np.random.default_rng(seed)
        n_funders = len(funders)
        for _ in range(n_resamples):
            draw = rng.integers(0, n_funders, size=n_funders)
            resampled_points: dict[str, tuple[float, float]] = {}
            for label in point_estimate:
                values = [
                    funder_means[label][funders[i]]
                    for i in draw
                    if funders[i] in funder_means[label]
                ]
                # Drop funders with no value for this config, but count them so the
                # smaller resample is visible.
                n_funder_gaps[label] += len(draw) - len(values)
                if not values:
                    continue
                cost = point_estimate[label][0]
                resampled_points[label] = (cost, sum(values) / len(values))
            resample_frontier = pareto_frontier(resampled_points)
            for point in resample_frontier:
                if point.on_frontier:
                    n_frontier_hits[point.config_label] += 1
        probabilities = {
            label: n_frontier_hits[label] / n_resamples for label in point_estimate
        }
    else:
        probabilities = {label: None for label in point_estimate}

    results = []
    for point in base_frontier:
        _, n_excluded = costs.get(point.config_label, (0.0, 0))
        results.append(
            FrontierPoint(
                config_label=point.config_label,
                cost_usd=point.cost_usd,
                macro_f1=point.macro_f1,
                on_frontier=point.on_frontier,
                dominated_by=point.dominated_by,
                frontier_probability=probabilities[point.config_label],
                n_cost_excluded=n_excluded,
                n_resample_funder_gaps=n_funder_gaps.get(point.config_label, 0),
            )
        )
    return tuple(sorted(results, key=lambda p: p.config_label))
