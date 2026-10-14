"""Run telemetry: latency, abstention, cost per true positive and runtime health.

Latency is `None` for context overflows, and timeouts record the time cap rather than
a real duration. Percentiles use numpy's default linear interpolation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dataset as ds

__all__ = [
    "RuntimeHealth",
    "runtime_health",
    "LatencyProfile",
    "latency_profiles",
    "AbstentionProfile",
    "abstention_profiles",
    "CostPerTruePositive",
    "cost_per_true_positive",
]

#: How runs are grouped. Same keys as `errors._KEYS`.
_KEYS = {"model": lambda r: r.model, "config": lambda r: r.config_label}


def _key_fn(key: str):
    """The grouping accessor for `key`; raises `ValueError` for an unknown key."""
    if key not in _KEYS:
        raise ValueError(f"key must be one of {sorted(_KEYS)}, got {key!r}")
    return _KEYS[key]



@dataclass(frozen=True)
class RuntimeHealth:
    """Whether the harness worked, separate from whether the answers were right.

    Token totals are included because they carry over to other prices, unlike dollars.
    `n_fp_unattributable` counts false positives matching no ground-truth row.
    """

    n_runs: int
    n_schema_invalid: int
    n_retried: int
    max_attempts: int | None
    n_step_budget_exhausted: int
    n_submit_failed: int
    total_input_tokens: int
    total_output_tokens: int
    n_fp_unattributable: int
    by_model_schema_invalid: dict[str, int]


def runtime_health(
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> RuntimeHealth:
    """Schema validity, retries and token totals, read from `runs.jsonl`.

    These fields are not on `RunRecord`, so the raw file is read directly.
    """
    n_runs = 0
    n_schema_invalid = 0
    n_retried = 0
    attempts: list[int] = []
    n_step_budget = 0
    n_submit_failed = 0
    tokens_in = 0
    tokens_out = 0
    fp_unattributable = 0
    by_model: dict[str, int] = defaultdict(int)

    with (Path(scored_dir) / "runs.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            telemetry = payload.get("telemetry", {})
            n_runs += 1
            if not telemetry.get("schema_valid", True):
                n_schema_invalid += 1
                by_model[payload["model"]] += 1
            attempt = telemetry.get("attempts")
            if attempt is not None:
                attempts.append(attempt)
                if attempt > 1:
                    n_retried += 1
            if telemetry.get("step_budget_exhausted"):
                n_step_budget += 1
            if telemetry.get("submit_failed"):
                n_submit_failed += 1
            tokens_in += telemetry.get("input_tokens") or 0
            tokens_out += telemetry.get("output_tokens") or 0
            fp_unattributable += payload.get("fp_unattributable", 0)

    return RuntimeHealth(
        n_runs=n_runs,
        n_schema_invalid=n_schema_invalid,
        n_retried=n_retried,
        max_attempts=max(attempts) if attempts else None,
        n_step_budget_exhausted=n_step_budget,
        n_submit_failed=n_submit_failed,
        total_input_tokens=tokens_in,
        total_output_tokens=tokens_out,
        n_fp_unattributable=fp_unattributable,
        by_model_schema_invalid=dict(by_model),
    )


@dataclass(frozen=True)
class LatencyProfile:
    """One group's wall-clock latency in seconds.

    Statistics are `None` if no run has a latency. `n_defined` is the count behind them.
    """

    label: str
    n_runs: int
    n_defined: int
    mean_s: float | None
    p50_s: float | None
    p95_s: float | None
    max_s: float | None


def latency_profiles(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
) -> tuple[LatencyProfile, ...]:
    """Wall-clock latency per group, with p50 and p95 because the times are skewed."""
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)

    n_runs: dict[str, int] = defaultdict(int)
    values: dict[str, list[float]] = defaultdict(list)
    for r in pool:
        label = extract(r)
        n_runs[label] += 1
        if r.latency_s is not None:
            values[label].append(r.latency_s)

    profiles = []
    for label in sorted(n_runs):
        vals = values[label]
        if vals:
            arr = np.asarray(vals, dtype=float)
            mean_s = float(arr.mean())
            p50_s = float(np.percentile(arr, 50))
            p95_s = float(np.percentile(arr, 95))
            max_s = float(arr.max())
        else:
            mean_s = p50_s = p95_s = max_s = None
        profiles.append(
            LatencyProfile(
                label=label,
                n_runs=n_runs[label],
                n_defined=len(vals),
                mean_s=mean_s,
                p50_s=p50_s,
                p95_s=p95_s,
                max_s=max_s,
            )
        )
    return tuple(profiles)


@dataclass(frozen=True)
class AbstentionProfile:
    """How often a group declined to answer. `rate` is `None` for an empty group."""

    label: str
    n_runs: int
    n_abstained: int
    rate: float | None


def abstention_profiles(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
) -> tuple[AbstentionProfile, ...]:
    """Abstention count and rate per group."""
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)

    n_runs: dict[str, int] = defaultdict(int)
    n_abstained: dict[str, int] = defaultdict(int)
    for r in pool:
        label = extract(r)
        n_runs[label] += 1
        if r.abstained:
            n_abstained[label] += 1

    return tuple(
        AbstentionProfile(
            label=label,
            n_runs=n_runs[label],
            n_abstained=n_abstained[label],
            rate=(n_abstained[label] / n_runs[label]) if n_runs[label] else None,
        )
        for label in sorted(n_runs)
    )


@dataclass(frozen=True)
class CostPerTruePositive:
    """Dollar cost per correctly found opportunity.

    `cost_per_tp` is `None` when the group found nothing. `n_cost_excluded` counts runs
    that failed before billing and so add nothing to `cost_usd`.
    """

    label: str
    cost_usd: float
    n_true_positives: int
    cost_per_tp: float | None
    n_cost_excluded: int


def cost_per_true_positive(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "config",
) -> tuple[CostPerTruePositive, ...]:
    """Realised spend per true positive, per group (by config by default)."""
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)

    cost: dict[str, float] = defaultdict(float)
    tp: dict[str, int] = defaultdict(int)
    excluded: dict[str, int] = defaultdict(int)
    for r in pool:
        label = extract(r)
        tp[label] += r.tp
        if r.cost_usd is None:
            excluded[label] += 1
        else:
            cost[label] += r.cost_usd

    return tuple(
        CostPerTruePositive(
            label=label,
            cost_usd=cost[label],
            n_true_positives=tp[label],
            cost_per_tp=(cost[label] / tp[label]) if tp[label] else None,
            n_cost_excluded=excluded[label],
        )
        for label in sorted(tp)
    )
