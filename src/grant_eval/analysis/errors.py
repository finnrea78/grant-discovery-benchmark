"""False-positive breakdown by reason and fabrication counts, grouped by model or config.

A missed opportunity and an invented URL both lower F1, but they are very different
failures for users, so this module separates them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import dataset as ds

__all__ = [
    "FalsePositiveProfile",
    "fp_reason_totals",
    "FabricationProfile",
    "fabrication_totals",
]

#: How runs can be grouped.
_KEYS = {"model": lambda r: r.model, "config": lambda r: r.config_label}


def _key_fn(key: str):
    """Return the grouping function for `key`, raising `ValueError` for an unknown key."""
    if key not in _KEYS:
        raise ValueError(f"key must be one of {sorted(_KEYS)}, got {key!r}")
    return _KEYS[key]


@dataclass(frozen=True)
class FalsePositiveProfile:
    """One group's false positives, split by the scorer's reason classification.

    `share` is each reason's fraction of `n_false_positives`, and is empty when the
    group has no false positives.
    """

    label: str
    n_runs: int
    n_false_positives: int
    by_reason: dict[str, int] = field(default_factory=dict)
    share: dict[str, float] = field(default_factory=dict)


def fp_reason_totals(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
) -> tuple[FalsePositiveProfile, ...]:
    """False positives per group, split by the scorer's reason, commonest first.

    Reasons a group never hit are left out of `by_reason` rather than set to 0.
    """
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_runs: dict[str, int] = defaultdict(int)
    for r in pool:
        label = extract(r)
        n_runs[label] += 1
        for reason, n in (r.fp_by_reason or {}).items():
            counts[label][reason] += n

    profiles = []
    for label in sorted(n_runs):
        by_reason = dict(sorted(counts[label].items(), key=lambda kv: (-kv[1], kv[0])))
        total = sum(by_reason.values())
        share = {k: v / total for k, v in by_reason.items()} if total else {}
        profiles.append(
            FalsePositiveProfile(
                label=label,
                n_runs=n_runs[label],
                n_false_positives=total,
                by_reason=by_reason,
                share=share,
            )
        )
    return tuple(profiles)


@dataclass(frozen=True)
class FabricationProfile:
    """One group's fabricated output.

    `mean_hallucination_rate` averages only runs with a defined rate (`n_rate_defined`
    of them) and is `None` if there are none. Runs that extracted nothing have no rate.
    `n_runs_fabricating` counts runs with at least one fabricated item, to tell one bad
    run apart from a regular habit.
    """

    label: str
    n_runs: int
    n_rate_defined: int
    mean_hallucination_rate: float | None
    total_fabricated: int
    n_runs_fabricating: int


def fabrication_totals(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
) -> tuple[FabricationProfile, ...]:
    """Fabricated-item counts and mean hallucination rate per group.

    Runs with `n_fabricated` of `None` (mirror unavailable) count towards `n_runs` only.
    """
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)

    n_runs: dict[str, int] = defaultdict(int)
    rates: dict[str, list[float]] = defaultdict(list)
    total: dict[str, int] = defaultdict(int)
    fabricating: dict[str, int] = defaultdict(int)

    for r in pool:
        label = extract(r)
        n_runs[label] += 1
        if r.hallucination_rate is not None:
            rates[label].append(r.hallucination_rate)
        if r.n_fabricated:
            total[label] += r.n_fabricated
            fabricating[label] += 1

    profiles = []
    for label in sorted(n_runs):
        vals = rates[label]
        profiles.append(
            FabricationProfile(
                label=label,
                n_runs=n_runs[label],
                n_rate_defined=len(vals),
                mean_hallucination_rate=(sum(vals) / len(vals)) if vals else None,
                total_fabricated=total[label],
                n_runs_fabricating=fabricating[label],
            )
        )
    return tuple(profiles)
