"""Run failures, temperature stability, and how results change without failed runs.

Failed runs (timeouts and context overflows) score F1 = 0 in the main analysis.
Abstentions are a model choice, not a failure, so they always count. Every context
overflow is nova-lite and every timeout is deepseek, so failure type cannot be
separated from model here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dataset as ds

__all__ = [
    "FailureBreakdown",
    "failure_breakdown",
    "surviving_runs",
    "SensitivityComparison",
    "compare_with_and_without_failures",
    "TemperatureStability",
    "temperature_stability",
]


@dataclass(frozen=True)
class FailureBreakdown:
    """How many runs failed, of what kind, and in which config and model.

    `n_failed` is not forced to equal the sum of the two kinds, so a run that failed
    for some other reason would still show up. `n_abstained` is shown for comparison.
    """

    n_runs: int
    n_failed: int
    n_timed_out: int
    n_context_overflow: int
    n_abstained: int
    by_config: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)


def _raw_failure_flags(scored_dir: Path) -> dict[tuple[str, str, int], tuple[bool, bool]]:
    """`{(config_hash, funder, replicate): (timed_out, context_overflow)}` from `runs.jsonl`.

    `RunRecord` only keeps `run_failed`, so the failure kind is read from the raw file.
    """
    scored_dir = Path(scored_dir)
    runs_path = scored_dir / "runs.jsonl"
    flags: dict[tuple[str, str, int], tuple[bool, bool]] = {}
    with runs_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            telemetry = payload["telemetry"]
            key = (payload["config_hash"], payload["funder"], payload["replicate"])
            flags[key] = (
                bool(telemetry.get("timed_out")),
                bool(telemetry.get("context_overflow")),
            )
    return flags


def failure_breakdown(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> FailureBreakdown:
    """Count failed runs by kind, config and model.

    Pass the `scored_dir` that `data` was loaded from. Raises `KeyError` if a failed run
    is missing from its telemetry.
    """
    pool = data.runs if runs is None else runs
    flags = _raw_failure_flags(scored_dir)

    n_failed = 0
    n_timed_out = 0
    n_context_overflow = 0
    n_abstained = 0
    by_config: dict[str, int] = {}
    by_model: dict[str, int] = {}

    for r in pool:
        if r.abstained:
            n_abstained += 1
        if not r.run_failed:
            continue
        n_failed += 1
        by_config[r.config_label] = by_config.get(r.config_label, 0) + 1
        by_model[r.model] = by_model.get(r.model, 0) + 1
        key = (r.config_hash, r.funder, r.replicate)
        if key not in flags:
            raise KeyError(
                f"failure_breakdown: no raw telemetry for {key} at {scored_dir}; "
                "pass the scored_dir `data` was actually loaded from"
            )
        timed_out, context_overflow = flags[key]
        if timed_out:
            n_timed_out += 1
        if context_overflow:
            n_context_overflow += 1

    return FailureBreakdown(
        n_runs=len(pool),
        n_failed=n_failed,
        n_timed_out=n_timed_out,
        n_context_overflow=n_context_overflow,
        n_abstained=n_abstained,
        by_config=by_config,
        by_model=by_model,
    )


def surviving_runs(data: ds.AnalysisDataset) -> tuple[ds.RunRecord, ...]:
    """Every run that did not fail, including abstentions."""
    return tuple(r for r in data.runs if not r.run_failed)


@dataclass(frozen=True)
class SensitivityComparison:
    """One config's macro `value`, with and without its failed runs.

    `primary` counts every run (the pre-registered figure). `sensitivity` drops failed
    runs and is `None` if none survive. `delta` is `sensitivity - primary`.
    """

    label: str
    primary: float
    sensitivity: float | None
    delta: float | None
    n_excluded: int


def _macro_mean(funder_values: dict[str, float]) -> float | None:
    """Unweighted mean over funder means, or `None` if there are none."""
    if not funder_values:
        return None
    return sum(funder_values.values()) / len(funder_values)


def compare_with_and_without_failures(
    data: ds.AnalysisDataset, value: str = "f1"
) -> tuple[SensitivityComparison, ...]:
    """Per config, macro `value` over all runs against macro `value` over surviving runs."""
    survivors = surviving_runs(data)
    primary_means = data.funder_means(value=value, runs=data.runs)
    survivor_means = data.funder_means(value=value, runs=survivors)

    n_excluded_by_config: dict[str, int] = {}
    for r in data.runs:
        if r.run_failed:
            n_excluded_by_config[r.config_label] = n_excluded_by_config.get(r.config_label, 0) + 1

    comparisons = []
    for label in data.config_labels():
        primary = _macro_mean(primary_means.get(label, {}))
        sensitivity = _macro_mean(survivor_means.get(label, {}))
        delta = None if (primary is None or sensitivity is None) else sensitivity - primary
        comparisons.append(
            SensitivityComparison(
                label=label,
                primary=primary,
                sensitivity=sensitivity,
                delta=delta,
                n_excluded=n_excluded_by_config.get(label, 0),
            )
        )
    return tuple(comparisons)


@dataclass(frozen=True)
class TemperatureStability:
    """Replicate spread at t0.0 against t0.7 for the same model, prompt and funder.

    `delta` is `replicate_sd_t07 - replicate_sd_t0`, so positive means t0.7 is noisier.
    """

    model: str
    prompt: str
    funder: str
    replicate_sd_t0: float | None
    replicate_sd_t07: float | None
    delta: float | None


def temperature_stability(
    data: ds.AnalysisDataset, value: str = "f1"
) -> tuple[TemperatureStability, ...]:
    """Population SD of the replicates in each cell, paired t0.0 against t0.7.

    Pairing within the same model, prompt and funder keeps other effects out of the
    comparison. A side with no defined values gets `None`.
    """
    models = sorted({r.model for r in data.runs})
    prompts = sorted({r.prompt for r in data.runs})
    funders = data.funders()

    rows = []
    for model in models:
        for prompt in prompts:
            for funder in funders:
                label_t0 = ds.config_label(model, prompt, 0.0)
                label_t07 = ds.config_label(model, prompt, 0.7)
                vals_t0 = [
                    getattr(r, value)
                    for r in data.cell(label_t0, funder)
                    if getattr(r, value) is not None
                ]
                vals_t07 = [
                    getattr(r, value)
                    for r in data.cell(label_t07, funder)
                    if getattr(r, value) is not None
                ]
                sd_t0 = float(np.std(vals_t0, ddof=0)) if vals_t0 else None
                sd_t07 = float(np.std(vals_t07, ddof=0)) if vals_t07 else None
                delta = None if (sd_t0 is None or sd_t07 is None) else sd_t07 - sd_t0
                rows.append(
                    TemperatureStability(
                        model=model,
                        prompt=prompt,
                        funder=funder,
                        replicate_sd_t0=sd_t0,
                        replicate_sd_t07=sd_t07,
                        delta=delta,
                    )
                )
    return tuple(rows)
