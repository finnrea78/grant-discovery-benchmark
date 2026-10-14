"""Averages per-run scores from `score.py` across replicates and funders, per config.

A config is `(model, prompt_strategy, temperature, agent_config)`. `macro_f1`, the
unweighted mean of each funder's F1, is the main aggregate because funders are the
unit we resample; `micro_f1` pools all opportunities and is reported as well.
Funders with no in-scope rows are left out of the detection metrics and scored on
abstention instead, but still count towards cost. As in `score.py`, a ratio with an
empty denominator has a matching `*_undefined` flag next to its placeholder value.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, field

from statsmodels.stats.inter_rater import fleiss_kappa as _statsmodels_fleiss_kappa

from grant_eval.scoring import score
from grant_eval.scoring.groundtruth import DEFAULT_GROUND_TRUTH_DIR, load_ground_truth
from grant_eval.scoring.score import RunScore

#: RQ3 reliability bar: an opportunity is "stably surfaced" when at least this many of
#: a config's five replicates find it.
STABLE_THRESHOLD = 4


@dataclass(frozen=True)
class ConfigKey:
    """`(model, prompt_strategy, temperature, agent_config)`, the cell replicates share."""

    model: str
    prompt_strategy: str
    temperature: float
    agent_config: str

    @property
    def label(self) -> str:
        return f"{self.model} | {self.prompt_strategy} | t{self.temperature} | {self.agent_config}"


@dataclass
class FunderAggregate:
    """One config's scores on one funder, averaged over that funder's replicates.

    `precision` is averaged only over replicates where it is defined, since counting
    an abstaining run's placeholder `0.0` would invent a precision. Those replicates
    appear as `None` in `replicate_precision`.
    """

    funder: str
    n_replicates: int
    n_in_scope: int
    precision: float
    recall: float
    f1: float
    tp: float
    fp: float
    fn: float
    precision_undefined: bool = False
    n_precision_undefined: int = 0
    replicate_precision: list[float | None] = field(default_factory=list)
    replicate_recall: list[float] = field(default_factory=list)
    replicate_f1: list[float] = field(default_factory=list)

    @property
    def zero_positive(self) -> bool:
        """True when the funder has no in-scope rows (left out of macro/micro)."""
        return self.n_in_scope == 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _funder_aggregate(funder: str, runs: list[RunScore]) -> FunderAggregate:
    n_in_scope_values = sorted({r.detection.n_in_scope for r in runs})
    if len(n_in_scope_values) > 1:
        raise ValueError(
            f"funder {funder!r}: replicates disagree on n_in_scope "
            f"({n_in_scope_values}) — two different ground-truth vintages were mixed "
            f"into one aggregation call"
        )

    replicate_precision: list[float | None] = [
        None if r.detection.precision_undefined else r.detection.precision for r in runs
    ]
    defined_precision = [p for p in replicate_precision if p is not None]
    precision_undefined = not defined_precision

    return FunderAggregate(
        funder=funder,
        n_replicates=len(runs),
        n_in_scope=n_in_scope_values[0],
        precision=0.0 if precision_undefined else _mean(defined_precision),
        recall=_mean([r.detection.recall for r in runs]),
        f1=_mean([r.detection.f1 for r in runs]),
        tp=_mean([r.detection.tp for r in runs]),
        fp=_mean([r.detection.fp for r in runs]),
        fn=_mean([r.detection.fn for r in runs]),
        precision_undefined=precision_undefined,
        n_precision_undefined=len(replicate_precision) - len(defined_precision),
        replicate_precision=replicate_precision,
        replicate_recall=[r.detection.recall for r in runs],
        replicate_f1=[r.detection.f1 for r in runs],
    )


@dataclass
class ConsistencyAggregate:
    """How consistently a config finds the same opportunities across replicates (RQ3).

    Mean recall cannot tell "the same 60% every run" from "a different 60% each time",
    so this is tracked separately. `stable_rate` is out of the opportunities found by at
    least one replicate, not all in-scope rows. It is undefined when there are fewer
    replicates than the threshold. `fleiss_kappa` treats in-scope rows as items and
    replicates as raters; a `nan` result becomes `None`.
    """

    funder: str
    n_replicates: int
    n_surfaced_any: int
    n_stable: int
    stable_rate: float
    stable_rate_undefined: bool
    fleiss_kappa: float | None
    fleiss_kappa_undefined: bool
    threshold: int = STABLE_THRESHOLD


def _fleiss_kappa(n_replicates: int, item_surfaced_counts: dict[str, int]) -> tuple[float | None, bool]:
    """`fleiss_kappa` over the given items, or `(None, True)` when it is undefined."""
    if n_replicates < 2 or not item_surfaced_counts:
        return None, True

    table = [
        [n_replicates - count, count] for count in item_surfaced_counts.values()
    ]
    with warnings.catch_warnings():
        # statsmodels warns on complete agreement, which becomes (None, True) below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kappa = _statsmodels_fleiss_kappa(table)
    if math.isnan(kappa):
        return None, True
    return float(kappa), False


def _consistency_aggregate(
    funder: str, runs: list[RunScore], *, threshold: int = STABLE_THRESHOLD
) -> ConsistencyAggregate:
    n_replicates = len(runs)

    surfaced_counts: dict[str, int] = {}
    for r in runs:
        for tp in r.detection.true_positives:
            surfaced_counts[tp.opportunity_id] = surfaced_counts.get(tp.opportunity_id, 0) + 1
        for fn in r.detection.false_negatives:
            surfaced_counts.setdefault(fn.opportunity_id, 0)

    surfaced_any = {oid: c for oid, c in surfaced_counts.items() if c >= 1}
    n_surfaced_any = len(surfaced_any)
    n_stable = sum(1 for c in surfaced_any.values() if c >= threshold)

    stable_rate_undefined = n_surfaced_any == 0 or n_replicates < threshold
    stable_rate = 0.0 if stable_rate_undefined else n_stable / n_surfaced_any

    fleiss_kappa_value, fleiss_kappa_undefined = _fleiss_kappa(n_replicates, surfaced_counts)

    return ConsistencyAggregate(
        funder=funder,
        n_replicates=n_replicates,
        n_surfaced_any=n_surfaced_any,
        n_stable=n_stable,
        stable_rate=stable_rate,
        stable_rate_undefined=stable_rate_undefined,
        fleiss_kappa=fleiss_kappa_value,
        fleiss_kappa_undefined=fleiss_kappa_undefined,
        threshold=threshold,
    )


@dataclass
class AbstentionAggregate:
    """Abstention for one funder with no in-scope rows: the share of runs that return nothing."""

    funder: str
    n_runs: int
    mean_extracted: float
    abstention_rate: float


def _abstention_aggregate(funder: str, runs: list[RunScore]) -> AbstentionAggregate:
    n = len(runs)
    n_zero = sum(1 for r in runs if r.detection.n_extracted == 0)
    return AbstentionAggregate(
        funder=funder,
        n_runs=n,
        mean_extracted=_mean([r.detection.n_extracted for r in runs]),
        abstention_rate=n_zero / n,
    )


@dataclass
class ConformanceAggregate:
    """Schema conformance across a config's runs.

    A valid empty submission counts as conformant. Older runs without the
    `schema_valid` flag are counted in `n_unknown` and left out of `rate`.
    """

    n_runs: int
    n_valid: int
    n_invalid: int
    n_unknown: int
    rate: float
    rate_undefined: bool
    n_submit_failed: int
    n_abstained: int
    n_budget_exhausted: int


def _conformance_aggregate(runs: list[RunScore]) -> ConformanceAggregate:
    n_valid = sum(1 for r in runs if r.result.schema_valid is True)
    n_invalid = sum(1 for r in runs if r.result.schema_valid is False)
    n_unknown = sum(1 for r in runs if r.result.schema_valid is None)
    denom = n_valid + n_invalid
    rate_undefined = denom == 0
    rate = 0.0 if rate_undefined else n_valid / denom
    return ConformanceAggregate(
        n_runs=len(runs),
        n_valid=n_valid,
        n_invalid=n_invalid,
        n_unknown=n_unknown,
        rate=rate,
        rate_undefined=rate_undefined,
        n_submit_failed=sum(1 for r in runs if r.result.submit_failed is True),
        n_abstained=sum(1 for r in runs if r.result.abstained),
        n_budget_exhausted=sum(1 for r in runs if r.result.step_budget_exhausted),
    )


@dataclass
class CostAggregate:
    """Cost and tokens summed over all of a config's runs, including zero-positive funders.

    `total_cost_undefined` is set when no run reported a cost, and `cost_per_tp` is
    `None` when there are no true positives.
    """

    n_runs: int
    total_cost_usd: float
    total_cost_undefined: bool
    total_input_tokens: int
    total_output_tokens: int
    mean_latency_s: float | None
    cost_per_tp: float | None
    cost_per_tp_undefined: bool
    n_missing_cost: int
    n_missing_latency: int


def _cost_aggregate(runs: list[RunScore]) -> CostAggregate:
    costs = [r.result.cost_usd for r in runs if r.result.cost_usd is not None]
    inputs = [r.result.input_tokens for r in runs if r.result.input_tokens is not None]
    outputs = [r.result.output_tokens for r in runs if r.result.output_tokens is not None]
    latencies = [r.result.latency_s for r in runs if r.result.latency_s is not None]

    total_tp = sum(r.detection.tp for r in runs)
    total_cost = sum(costs)
    total_cost_undefined = len(costs) == 0
    cost_per_tp_undefined = total_tp == 0 or total_cost_undefined

    return CostAggregate(
        n_runs=len(runs),
        total_cost_usd=total_cost,
        total_cost_undefined=total_cost_undefined,
        total_input_tokens=sum(inputs),
        total_output_tokens=sum(outputs),
        mean_latency_s=_mean(latencies) if latencies else None,
        cost_per_tp=None if cost_per_tp_undefined else total_cost / total_tp,
        cost_per_tp_undefined=cost_per_tp_undefined,
        n_missing_cost=len(runs) - len(costs),
        n_missing_latency=len(runs) - len(latencies),
    )


@dataclass
class ConfigAggregate:
    """One config averaged over its replicates and funders."""

    key: ConfigKey
    funders: list[FunderAggregate]
    excluded_funders: list[str]
    n_funders: int
    n_runs: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_f1_undefined: bool
    macro_precision_undefined: bool
    micro_precision: float
    micro_recall: float
    micro_f1: float
    micro_precision_undefined: bool
    micro_recall_undefined: bool
    micro_f1_undefined: bool
    abstentions: list[AbstentionAggregate]
    conformance: ConformanceAggregate
    cost: CostAggregate
    consistency: dict[str, ConsistencyAggregate]
    mean_stable_rate: float
    mean_stable_rate_undefined: bool
    mean_fleiss_kappa: float | None
    mean_fleiss_kappa_undefined: bool

    @property
    def label(self) -> str:
        return self.key.label


def _config_aggregate(
    key: ConfigKey, runs: list[RunScore], *, threshold: int = STABLE_THRESHOLD
) -> ConfigAggregate:
    by_funder: dict[str, list[RunScore]] = {}
    for r in runs:
        by_funder.setdefault(r.funder, []).append(r)

    funder_aggs = [_funder_aggregate(f, rs) for f, rs in sorted(by_funder.items())]
    included = [fa for fa in funder_aggs if not fa.zero_positive]
    excluded_funders = sorted(fa.funder for fa in funder_aggs if fa.zero_positive)

    macro_f1_undefined = len(included) == 0
    if macro_f1_undefined:
        macro_recall = macro_f1 = 0.0
    else:
        macro_recall = _mean([fa.recall for fa in included])
        macro_f1 = _mean([fa.f1 for fa in included])

    # Separate flag: a funder whose replicates all abstained drops out of this mean only.
    precision_included = [fa for fa in included if not fa.precision_undefined]
    macro_precision_undefined = len(precision_included) == 0
    macro_precision = (
        0.0 if macro_precision_undefined else _mean([fa.precision for fa in precision_included])
    )

    included_funder_names = {fa.funder for fa in included}
    included_runs = [r for r in runs if r.funder in included_funder_names]
    total_tp = sum(r.detection.tp for r in included_runs)
    total_fp = sum(r.detection.fp for r in included_runs)
    total_fn = sum(r.detection.fn for r in included_runs)

    micro_precision_undefined = (total_tp + total_fp) == 0
    micro_precision = 0.0 if micro_precision_undefined else total_tp / (total_tp + total_fp)
    micro_recall_undefined = (total_tp + total_fn) == 0
    micro_recall = 0.0 if micro_recall_undefined else total_tp / (total_tp + total_fn)
    micro_f1_undefined = micro_precision_undefined and micro_recall_undefined
    micro_f1 = (
        0.0
        if (micro_precision + micro_recall) == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    abstentions = [
        _abstention_aggregate(fa.funder, by_funder[fa.funder])
        for fa in funder_aggs
        if fa.zero_positive
    ]

    # Computed for every funder; each mean only uses funders where it is defined.
    consistency = {
        fa.funder: _consistency_aggregate(fa.funder, by_funder[fa.funder], threshold=threshold)
        for fa in funder_aggs
    }
    stable_rate_defined = [c for c in consistency.values() if not c.stable_rate_undefined]
    mean_stable_rate_undefined = len(stable_rate_defined) == 0
    mean_stable_rate = (
        0.0 if mean_stable_rate_undefined else _mean([c.stable_rate for c in stable_rate_defined])
    )

    kappa_defined = [c for c in consistency.values() if not c.fleiss_kappa_undefined]
    mean_fleiss_kappa_undefined = len(kappa_defined) == 0
    mean_fleiss_kappa = (
        None if mean_fleiss_kappa_undefined else _mean([c.fleiss_kappa for c in kappa_defined])
    )

    return ConfigAggregate(
        key=key,
        funders=funder_aggs,
        excluded_funders=excluded_funders,
        n_funders=len(included),
        n_runs=len(runs),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        macro_f1_undefined=macro_f1_undefined,
        macro_precision_undefined=macro_precision_undefined,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        micro_precision_undefined=micro_precision_undefined,
        micro_recall_undefined=micro_recall_undefined,
        micro_f1_undefined=micro_f1_undefined,
        abstentions=abstentions,
        conformance=_conformance_aggregate(runs),
        cost=_cost_aggregate(runs),
        consistency=consistency,
        mean_stable_rate=mean_stable_rate,
        mean_stable_rate_undefined=mean_stable_rate_undefined,
        mean_fleiss_kappa=mean_fleiss_kappa,
        mean_fleiss_kappa_undefined=mean_fleiss_kappa_undefined,
    )


def aggregate_runs(
    scores: list[RunScore], *, threshold: int = STABLE_THRESHOLD
) -> list[ConfigAggregate]:
    """Group `scores` by config and aggregate each one, sorted by label.

    Raises `ValueError` if replicates of the same config and funder disagree on
    `n_in_scope`.
    """
    by_config: dict[ConfigKey, list[RunScore]] = {}
    for r in scores:
        key = ConfigKey(
            model=r.result.model,
            prompt_strategy=r.result.prompt_strategy,
            temperature=r.result.temperature,
            agent_config=r.result.agent_config,
        )
        by_config.setdefault(key, []).append(r)

    aggregates = [
        _config_aggregate(key, runs, threshold=threshold) for key, runs in by_config.items()
    ]
    return sorted(aggregates, key=lambda ca: ca.key.label)


def aggregate_to_dict(config: ConfigAggregate) -> dict:
    """A JSON-serialisable view of one `ConfigAggregate`.

    Undefined values are written as `null` here (unlike the in-memory `0.0`), because
    pandas reads them back as `NaN` and leaves them out of `.mean()`. The `_undefined`
    flags are still included.
    """
    return {
        "config": {
            "model": config.key.model,
            "prompt_strategy": config.key.prompt_strategy,
            "temperature": config.key.temperature,
            "agent_config": config.key.agent_config,
            "label": config.key.label,
        },
        "n_funders": config.n_funders,
        "n_runs": config.n_runs,
        "excluded_funders": config.excluded_funders,
        "macro_precision": None if config.macro_precision_undefined else config.macro_precision,
        "macro_recall": None if config.macro_f1_undefined else config.macro_recall,
        "macro_f1": None if config.macro_f1_undefined else config.macro_f1,
        "macro_f1_undefined": config.macro_f1_undefined,
        "macro_precision_undefined": config.macro_precision_undefined,
        "micro_precision": None if config.micro_precision_undefined else config.micro_precision,
        "micro_recall": None if config.micro_recall_undefined else config.micro_recall,
        "micro_f1": None if config.micro_f1_undefined else config.micro_f1,
        "micro_precision_undefined": config.micro_precision_undefined,
        "micro_recall_undefined": config.micro_recall_undefined,
        "micro_f1_undefined": config.micro_f1_undefined,
        "funders": [
            {
                "funder": fa.funder,
                "n_replicates": fa.n_replicates,
                "n_in_scope": fa.n_in_scope,
                "zero_positive": fa.zero_positive,
                "precision": None if fa.precision_undefined else fa.precision,
                "precision_undefined": fa.precision_undefined,
                "n_precision_undefined": fa.n_precision_undefined,
                "recall": fa.recall,
                "f1": fa.f1,
                "tp": fa.tp,
                "fp": fa.fp,
                "fn": fa.fn,
                "replicate_precision": fa.replicate_precision,
                "replicate_recall": fa.replicate_recall,
                "replicate_f1": fa.replicate_f1,
            }
            for fa in config.funders
        ],
        "abstentions": [
            {
                "funder": ab.funder,
                "n_runs": ab.n_runs,
                "mean_extracted": ab.mean_extracted,
                "abstention_rate": ab.abstention_rate,
            }
            for ab in config.abstentions
        ],
        "conformance": {
            "n_runs": config.conformance.n_runs,
            "n_valid": config.conformance.n_valid,
            "n_invalid": config.conformance.n_invalid,
            "n_unknown": config.conformance.n_unknown,
            "rate": None if config.conformance.rate_undefined else config.conformance.rate,
            "rate_undefined": config.conformance.rate_undefined,
            "n_submit_failed": config.conformance.n_submit_failed,
            "n_abstained": config.conformance.n_abstained,
            "n_budget_exhausted": config.conformance.n_budget_exhausted,
        },
        "cost": {
            "n_runs": config.cost.n_runs,
            "total_cost_usd": (
                None if config.cost.total_cost_undefined else config.cost.total_cost_usd
            ),
            "total_cost_undefined": config.cost.total_cost_undefined,
            "total_input_tokens": config.cost.total_input_tokens,
            "total_output_tokens": config.cost.total_output_tokens,
            "mean_latency_s": config.cost.mean_latency_s,
            "cost_per_tp": config.cost.cost_per_tp,
            "cost_per_tp_undefined": config.cost.cost_per_tp_undefined,
            "n_missing_cost": config.cost.n_missing_cost,
            "n_missing_latency": config.cost.n_missing_latency,
        },
        "consistency": {
            funder: {
                "n_replicates": c.n_replicates,
                "n_surfaced_any": c.n_surfaced_any,
                "n_stable": c.n_stable,
                "threshold": c.threshold,
                "stable_rate": None if c.stable_rate_undefined else c.stable_rate,
                "stable_rate_undefined": c.stable_rate_undefined,
                "fleiss_kappa": c.fleiss_kappa,
                "fleiss_kappa_undefined": c.fleiss_kappa_undefined,
            }
            for funder, c in config.consistency.items()
        },
        "mean_stable_rate": None if config.mean_stable_rate_undefined else config.mean_stable_rate,
        "mean_stable_rate_undefined": config.mean_stable_rate_undefined,
        "mean_fleiss_kappa": config.mean_fleiss_kappa,
        "mean_fleiss_kappa_undefined": config.mean_fleiss_kappa_undefined,
    }


def _fmt(value: float | None, decimals: int = 3) -> str:
    """Format a value, showing a dash when it is `None`."""
    return "—" if value is None else f"{value:.{decimals}f}"


def _fmtu(value: float, undefined: bool, decimals: int = 3) -> str:
    """Format a value, showing a dash when its `_undefined` flag is set."""
    return "—" if undefined else f"{value:.{decimals}f}"


def format_aggregate_report(config: ConfigAggregate, *, by_funder: bool = False) -> str:
    """A readable report for one config, like `score.format_report`.

    `by_funder=True` adds a table of the funders in the macro mean. Zero-positive
    funders are listed at the end with their abstention rate.
    """
    out: list[str] = []
    out.append("=" * 78)
    out.append(config.label)
    out.append("=" * 78)
    out.append(
        f"  {config.n_funders} funder(s) in the macro mean, "
        f"{len(config.excluded_funders)} excluded (zero-positive), "
        f"{config.n_runs} run(s) across all funders"
    )

    macro_notes = []
    if config.macro_f1_undefined:
        macro_notes.append("undefined — every funder is zero-positive")
    elif config.macro_precision_undefined:
        macro_notes.append("precision undefined — no included funder extracted anything")
    macro_note = f"  ({'; '.join(macro_notes)})" if macro_notes else ""
    out.append(
        f"detection (macro)  precision "
        f"{_fmtu(config.macro_precision, config.macro_precision_undefined)}"
        f"   recall {_fmtu(config.macro_recall, config.macro_f1_undefined)}"
        f"   F1 {_fmtu(config.macro_f1, config.macro_f1_undefined)}{macro_note}"
    )
    micro_notes = []
    if config.micro_precision_undefined:
        micro_notes.append("precision undefined — nothing extracted")
    if config.micro_recall_undefined:
        micro_notes.append("recall/F1 undefined — no included funder had anything to find")
    micro_note = f"  ({'; '.join(micro_notes)})" if micro_notes else ""
    out.append(
        f"detection (micro)  precision {_fmtu(config.micro_precision, config.micro_precision_undefined)}"
        f"   recall {_fmtu(config.micro_recall, config.micro_recall_undefined)}"
        f"   F1 {_fmtu(config.micro_f1, config.micro_f1_undefined)}{micro_note}"
    )

    conf = config.conformance
    conf_note = (
        "  (undefined — only unknown-schema runs)" if conf.rate_undefined else ""
    )
    out.append(
        f"conformance        rate {_fmtu(conf.rate, conf.rate_undefined)}"
        f"   valid {conf.n_valid}   invalid {conf.n_invalid}   unknown {conf.n_unknown}{conf_note}"
    )

    cost = config.cost
    total_cost = (
        "—" if cost.total_cost_undefined else f"${cost.total_cost_usd:.4f}"
    )
    out.append(
        f"cost               total {total_cost}   cost/TP {_fmt(cost.cost_per_tp)}"
        f"   mean latency {_fmt(cost.mean_latency_s)}s   ({cost.n_runs} run(s),"
        f" missing cost {cost.n_missing_cost}/{cost.n_runs},"
        f" missing latency {cost.n_missing_latency}/{cost.n_runs})"
    )

    out.append(
        f"consistency (RQ3)  mean stable-rate "
        f"{_fmtu(config.mean_stable_rate, config.mean_stable_rate_undefined)}"
        f"   mean Fleiss' kappa {_fmt(config.mean_fleiss_kappa)}"
    )

    if by_funder:
        included = [fa for fa in config.funders if not fa.zero_positive]
        if included:
            out.append("")
            out.append(
                f"  {'funder':<40}{'n':>4}{'reps':>5}{'prec':>7}{'rec':>7}{'F1':>7}"
                f"{'stable':>8}{'kappa':>8}"
            )
            for fa in included:
                c = config.consistency[fa.funder]
                out.append(
                    f"  {fa.funder:<40}{fa.n_in_scope:>4}{fa.n_replicates:>5}"
                    f"{_fmtu(fa.precision, fa.precision_undefined):>7}"
                    f"{fa.recall:>7.3f}{fa.f1:>7.3f}"
                    f"{_fmtu(c.stable_rate, c.stable_rate_undefined):>8}"
                    f"{_fmt(c.fleiss_kappa):>8}"
                )

    out.append("")
    if config.excluded_funders:
        out.append(
            "excluded (zero-positive) funders, scored on abstention only: "
            + ", ".join(config.excluded_funders)
        )
        for ab in config.abstentions:
            out.append(
                f"  {ab.funder}: abstention rate {ab.abstention_rate:.3f}"
                f"  (mean extracted {ab.mean_extracted:.3f}, n={ab.n_runs})"
            )
    else:
        out.append("no zero-positive funders excluded")

    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aggregate",
        description=(
            "Aggregate score.py's per-run RunScores across replicates and funders, one "
            "config -- (model, prompt_strategy, temperature, agent_config) -- at a "
            "time. Macro F1 (the unweighted mean over funders) is the primary "
            "aggregate; see the module docstring for why."
        ),
    )
    p.add_argument(
        "results", nargs="+", help="ExtractionResult JSON file(s), or directories of them"
    )
    p.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH_DIR),
        help=f"ground-truth CSV directory (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of the report")
    p.add_argument(
        "--show-rejected",
        action="store_true",
        help="list the CSVs in the ground-truth directory that were NOT loaded, and why",
    )
    p.add_argument(
        "--by-funder",
        action="store_true",
        help="also print the per-funder detection/consistency table within each config",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    gt = load_ground_truth(args.ground_truth)
    paths = score._expand(args.results)
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    if args.show_rejected and not args.json:
        print(
            f"ground truth: {len(gt.files)} file(s), {len(gt.rows)} rows, "
            f"{len(gt.in_scope)} in scope, {len(gt.funders)} funders"
        )
        for rejected_path, reason in gt.rejected:
            print(f"  rejected {rejected_path.name}: {reason}")
        print()

    scores: list[RunScore] = []
    for path in paths:
        try:
            scores.append(score.score_run(score.load_result(path), gt, source_path=path))
        except KeyError as exc:  # unlabelled funder: report it and carry on
            print(f"skipped {path}: {exc}", file=sys.stderr)

    if not scores:
        return 1

    aggregates = aggregate_runs(scores)

    if args.json:
        print(json.dumps([aggregate_to_dict(a) for a in aggregates], indent=2, default=str))
        return 0

    for a in aggregates:
        print(format_aggregate_report(a, by_funder=args.by_funder))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
