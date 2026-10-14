"""Accuracy of the extracted structured fields, and an end-to-end rate that combines it with detection.

Field outcomes are read from the `fields` block in `runs.jsonl`, which `RunRecord`
does not keep. Field accuracy only covers opportunities a run found, so weak models are
scored on an easier subset. The end-to-end rate avoids this by counting every in-scope
opportunity: a missed one counts as a failure.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from . import dataset as ds
from . import frontier as fr
from . import stats as st
from . import strata as strat
from grant_eval.scoring.groundtruth import GroundTruth

__all__ = [
    "FIELDS",
    "FieldAccuracy",
    "field_accuracy",
    "FieldFactorAccuracy",
    "field_accuracy_by_factor",
    "FieldDirection",
    "field_directions",
    "TagFieldAccuracy",
    "field_accuracy_by_tag",
    "EndToEnd",
    "end_to_end",
    "end_to_end_frontier",
]

#: The four structured fields scored by exact match, weakest first.
FIELDS = ("deadlines", "amount_max_gbp", "amount_min_gbp", "status")

#: How runs can be grouped, same as `errors._KEYS` and `telemetry._KEYS`.
_KEYS = {"model": lambda r: r.model, "config": lambda r: r.config_label}


def _key_fn(key: str):
    if key not in _KEYS:
        raise ValueError(f"key must be one of {sorted(_KEYS)}, got {key!r}")
    return _KEYS[key]


def _uncertain_index(gt: GroundTruth) -> dict[str, frozenset[str]]:
    """`{opportunity_id: fields the labeller marked uncertain}` for every row."""
    index: dict[str, frozenset[str]] = {}
    for funder_gt in gt.funders.values():
        for row in funder_gt.rows:
            index[row.opportunity_id] = frozenset(row.uncertain_fields or [])
    return index


@dataclass(frozen=True)
class _RawFields:
    """One run's raw `fields` block, reduced to what this module uses."""

    mismatches: dict[str, frozenset[str]]
    gt_blank_agent_asserted: dict[str, int]
    gt_value_agent_blank: dict[str, int]
    excluded_uncertain: dict[str, int]
    recorded_correct: dict[str, int]
    recorded_incorrect: dict[str, int]


def _raw_fields(scored_dir: Path) -> dict[tuple[str, str, int], _RawFields]:
    """`{(config_hash, funder, replicate): _RawFields}` from the frozen artefact.

    Reads `runs.jsonl` directly because `RunRecord` drops the `fields` block.
    """
    out: dict[tuple[str, str, int], _RawFields] = {}
    with (Path(scored_dir) / "runs.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            blocks = payload.get("fields") or {}
            key = (payload["config_hash"], payload["funder"], payload["replicate"])
            out[key] = _RawFields(
                mismatches={
                    f: frozenset(m[0] for m in (blocks.get(f) or {}).get("mismatches", []))
                    for f in FIELDS
                },
                gt_blank_agent_asserted={
                    f: (blocks.get(f) or {}).get("gt_blank_agent_asserted", 0) for f in FIELDS
                },
                gt_value_agent_blank={
                    f: (blocks.get(f) or {}).get("gt_value_agent_blank", 0) for f in FIELDS
                },
                excluded_uncertain={
                    f: (blocks.get(f) or {}).get("excluded_uncertain", 0) for f in FIELDS
                },
                recorded_correct={f: (blocks.get(f) or {}).get("correct", 0) for f in FIELDS},
                recorded_incorrect={f: (blocks.get(f) or {}).get("incorrect", 0) for f in FIELDS},
            )
    return out


def _outcomes(run: ds.RunRecord, raw: _RawFields, uncertain: dict[str, frozenset[str]], field: str):
    """`(evaluated_ids, incorrect_ids)` for one run and field.

    `evaluated` is the run's true positives minus those whose field is marked
    uncertain; `incorrect` is the recorded mismatch ids. A test checks this matches
    the recorded counts.
    """
    incorrect = raw.mismatches[field]
    evaluated = frozenset(
        oid for oid in run.true_positive_ids if field not in uncertain.get(oid, frozenset())
    )
    return evaluated, incorrect


@dataclass(frozen=True)
class FieldAccuracy:
    """One group's exact-match accuracy on one structured field.

    `accuracy` is `None` when nothing was evaluated. Groups can have very different
    `n_evaluated`, since accuracy depends on what was detected.
    """

    label: str
    field: str
    n_evaluated: int
    n_correct: int
    n_incorrect: int
    n_excluded_uncertain: int
    accuracy: float | None


def field_accuracy(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> tuple[FieldAccuracy, ...]:
    """Exact-match accuracy per group and field, pooled over opportunities.

    Pooling stops a run that found two opportunities counting as much as one that
    found twenty.
    """
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)
    raw_by_key = _raw_fields(scored_dir)
    uncertain = _uncertain_index(gt)

    correct: dict[tuple[str, str], int] = defaultdict(int)
    incorrect: dict[tuple[str, str], int] = defaultdict(int)
    skipped: dict[tuple[str, str], int] = defaultdict(int)
    labels: set[str] = set()

    for r in pool:
        raw = raw_by_key.get((r.config_hash, r.funder, r.replicate))
        if raw is None:
            raise KeyError(
                f"field_accuracy: no raw fields for {(r.config_hash, r.funder, r.replicate)} "
                f"at {scored_dir}; pass the scored_dir `data` was actually loaded from"
            )
        label = extract(r)
        labels.add(label)
        for f in FIELDS:
            evaluated, wrong = _outcomes(r, raw, uncertain, f)
            incorrect[(label, f)] += len(wrong)
            correct[(label, f)] += len(evaluated) - len(wrong)
            skipped[(label, f)] += raw.excluded_uncertain[f]

    out = []
    for label in sorted(labels):
        for f in FIELDS:
            c, i = correct[(label, f)], incorrect[(label, f)]
            out.append(
                FieldAccuracy(
                    label=label,
                    field=f,
                    n_evaluated=c + i,
                    n_correct=c,
                    n_incorrect=i,
                    n_excluded_uncertain=skipped[(label, f)],
                    accuracy=(c / (c + i)) if (c + i) else None,
                )
            )
    return tuple(out)


#: Factors to break field accuracy down by, within each model.
_FACTORS = {"prompt": lambda r: r.prompt, "temperature": lambda r: f"t{r.temperature}"}


@dataclass(frozen=True)
class FieldFactorAccuracy:
    """One `(model, factor level, field)` cell's exact-match accuracy.

    Split within model because model has by far the largest effect, so pooling over
    models would hide prompt effects such as the scaffold helping DeepSeek's deadlines.
    """

    model: str
    factor: str
    level: str
    field: str
    n_evaluated: int
    n_correct: int
    accuracy: float | None


def field_accuracy_by_factor(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    factor: str = "prompt",
    runs: tuple[ds.RunRecord, ...] | None = None,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> tuple[FieldFactorAccuracy, ...]:
    """Field accuracy per `(model, level of factor, field)`, pooled over opportunities.

    Pooled as in `field_accuracy`; compare `n_evaluated` before comparing cells.
    """
    if factor not in _FACTORS:
        raise ValueError(f"factor must be one of {sorted(_FACTORS)}, got {factor!r}")
    level_of = _FACTORS[factor]

    pool = data.runs if runs is None else runs
    raw_by_key = _raw_fields(scored_dir)
    uncertain = _uncertain_index(gt)

    correct: dict[tuple[str, str, str], int] = defaultdict(int)
    total: dict[tuple[str, str, str], int] = defaultdict(int)
    cells: set[tuple[str, str]] = set()

    for r in pool:
        raw = raw_by_key.get((r.config_hash, r.funder, r.replicate))
        if raw is None:
            raise KeyError(
                f"field_accuracy_by_factor: no raw fields for "
                f"{(r.config_hash, r.funder, r.replicate)} at {scored_dir}"
            )
        level = level_of(r)
        cells.add((r.model, level))
        for f in FIELDS:
            evaluated, wrong = _outcomes(r, raw, uncertain, f)
            total[(r.model, level, f)] += len(evaluated)
            correct[(r.model, level, f)] += len(evaluated) - len(wrong)

    return tuple(
        FieldFactorAccuracy(
            model=model,
            factor=factor,
            level=level,
            field=f,
            n_evaluated=total[(model, level, f)],
            n_correct=correct[(model, level, f)],
            accuracy=(
                correct[(model, level, f)] / total[(model, level, f)]
                if total[(model, level, f)]
                else None
            ),
        )
        for model, level in sorted(cells)
        for f in FIELDS
    )


@dataclass(frozen=True)
class FieldDirection:
    """Which way a field's errors run.

    `asserted_where_blank` counts values the agent gave where the ground truth has
    none; `blank_where_value` counts values it left out. The first is worse for users,
    who may act on an invented value.
    """

    label: str
    field: str
    asserted_where_blank: int
    blank_where_value: int


def field_directions(
    data: ds.AnalysisDataset,
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "model",
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> tuple[FieldDirection, ...]:
    """Directional error counts per group and field."""
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)
    raw_by_key = _raw_fields(scored_dir)

    asserted: dict[tuple[str, str], int] = defaultdict(int)
    blank: dict[tuple[str, str], int] = defaultdict(int)
    labels: set[str] = set()
    for r in pool:
        raw = raw_by_key[(r.config_hash, r.funder, r.replicate)]
        label = extract(r)
        labels.add(label)
        for f in FIELDS:
            asserted[(label, f)] += raw.gt_blank_agent_asserted[f]
            blank[(label, f)] += raw.gt_value_agent_blank[f]

    return tuple(
        FieldDirection(
            label=label, field=f,
            asserted_where_blank=asserted[(label, f)],
            blank_where_value=blank[(label, f)],
        )
        for label in sorted(labels)
        for f in FIELDS
    )


@dataclass(frozen=True)
class TagFieldAccuracy:
    """Field accuracy on tagged rows against untagged rows, paired within funder.

    Same design as `edgecases.tag_contrast`, but asks whether a tagged opportunity is
    harder to read correctly once found (e.g. deadlines on `multi_deadline` rows).
    """

    tag: str
    field: str
    n_rows: int
    n_paired_clusters: int
    accuracy_tagged: float | None
    accuracy_untagged: float | None
    difference: st.BootstrapCI | None
    reason: str


def _accuracy_by_funder(pool, raw_by_key, uncertain, field, predicate, tags):
    """`{funder: accuracy}` over opportunities matching `predicate`, pooled over runs."""
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for r in pool:
        raw = raw_by_key[(r.config_hash, r.funder, r.replicate)]
        evaluated, wrong = _outcomes(r, raw, uncertain, field)
        for oid in evaluated:
            if oid not in tags or not predicate(tags[oid]):
                continue
            total[r.funder] += 1
            if oid not in wrong:
                correct[r.funder] += 1
    return {f: correct[f] / total[f] for f in total if total[f]}


def field_accuracy_by_tag(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    field: str,
    tag: str,
    runs: tuple[ds.RunRecord, ...] | None = None,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
    n_resamples: int = 10_000,
    seed: int = st.DEFAULT_SEED,
) -> TagFieldAccuracy:
    """Accuracy on `field` for rows carrying `tag`, against untagged rows, paired
    within funder."""
    pool = data.runs if runs is None else runs
    raw_by_key = _raw_fields(scored_dir)
    uncertain = _uncertain_index(gt)
    tags = strat.opportunity_tags(gt, data.funders())

    tagged = _accuracy_by_funder(
        pool, raw_by_key, uncertain, field, lambda ts: tag in ts, tags
    )
    untagged = _accuracy_by_funder(
        pool, raw_by_key, uncertain, field, lambda ts: not ts, tags
    )
    paired = sorted(set(tagged) & set(untagged))
    n_rows = sum(1 for ts in tags.values() if tag in ts)

    if not paired:
        return TagFieldAccuracy(
            tag=tag, field=field, n_rows=n_rows, n_paired_clusters=0,
            accuracy_tagged=None, accuracy_untagged=None, difference=None,
            reason=(
                "no funder has both a scorable tagged opportunity and a scorable "
                "untagged one for this field"
            ),
        )

    a = [tagged[f] for f in paired]
    b = [untagged[f] for f in paired]
    difference = st.bootstrap_ci(
        [x - y for x, y in zip(a, b)], n_resamples=n_resamples, seed=seed
    )
    reason = (
        "" if len(paired) >= strat.MIN_CLUSTERS
        else f"{len(paired)} paired funder-cluster(s) (< {strat.MIN_CLUSTERS})"
    )
    return TagFieldAccuracy(
        tag=tag, field=field, n_rows=n_rows, n_paired_clusters=len(paired),
        accuracy_tagged=sum(a) / len(a), accuracy_untagged=sum(b) / len(b),
        difference=difference, reason=reason,
    )


@dataclass(frozen=True)
class EndToEnd:
    """One config's share of in-scope opportunities both found and correctly read.

    `detection_rate` uses the same denominator, so the gap between the two rates is
    what extraction errors cost. `field` is the field that must be correct, or `"all"`.
    Rates are macro means over funders, or `None` if no funder could be evaluated.
    """

    label: str
    field: str
    n_opportunities: int
    n_excluded_uncertain: int
    detection_rate: float | None
    end_to_end_rate: float | None
    n_funders: int
    cost_usd: float = 0.0
    cost_per_usable: float | None = None
    on_frontier: bool = False
    dominated_by: tuple[str, ...] = dc_field(default_factory=tuple)


def end_to_end(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    field: str = "deadlines",
    runs: tuple[ds.RunRecord, ...] | None = None,
    key: str = "config",
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> tuple[EndToEnd, ...]:
    """Share of in-scope opportunities that were found AND had `field` correct.

    `field="all"` requires every field to be correct. The denominator is every
    in-scope opportunity not marked uncertain for the required fields, including ones
    the run missed.
    """
    pool = data.runs if runs is None else runs
    extract = _key_fn(key)
    raw_by_key = _raw_fields(scored_dir)
    uncertain = _uncertain_index(gt)
    required = FIELDS if field == "all" else (field,)

    denominator: dict[str, set[str]] = {}
    n_skipped = 0
    for funder in data.funders():
        keep = set()
        for row in gt.for_funder(funder).in_scope:
            unc = frozenset(row.uncertain_fields or [])
            if any(f in unc for f in required):
                n_skipped += 1
                continue
            keep.add(row.opportunity_id)
        denominator[funder] = keep

    found: dict[tuple[str, str], list[float]] = defaultdict(list)
    usable: dict[tuple[str, str], list[float]] = defaultdict(list)
    labels: set[str] = set()

    for r in pool:
        raw = raw_by_key[(r.config_hash, r.funder, r.replicate)]
        label = extract(r)
        labels.add(label)
        pool_ids = denominator[r.funder]
        if not pool_ids:
            continue
        tp = set(r.true_positive_ids) & pool_ids
        wrong: set[str] = set()
        for f in required:
            wrong |= set(raw.mismatches[f])
        good = tp - wrong
        found[(label, r.funder)].append(len(tp) / len(pool_ids))
        usable[(label, r.funder)].append(len(good) / len(pool_ids))

    costs: dict[str, float] = defaultdict(float)
    for r in pool:
        if r.cost_usd is not None:
            costs[extract(r)] += r.cost_usd

    n_opportunities = sum(len(v) for v in denominator.values())
    out = []
    for label in sorted(labels):
        f_vals = [
            sum(v) / len(v) for (l, _), v in found.items() if l == label and v
        ]
        u_vals = [
            sum(v) / len(v) for (l, _), v in usable.items() if l == label and v
        ]
        rate = (sum(u_vals) / len(u_vals)) if u_vals else None
        n_usable = rate * n_opportunities if rate is not None else 0.0
        out.append(
            EndToEnd(
                label=label,
                field=field,
                n_opportunities=n_opportunities,
                n_excluded_uncertain=n_skipped,
                detection_rate=(sum(f_vals) / len(f_vals)) if f_vals else None,
                end_to_end_rate=rate,
                n_funders=len(u_vals),
                cost_usd=costs[label],
                cost_per_usable=(costs[label] / n_usable) if n_usable else None,
            )
        )
    return tuple(out)


def end_to_end_frontier(
    rows: tuple[EndToEnd, ...],
) -> tuple[EndToEnd, ...]:
    """The cost/quality Pareto frontier over `end_to_end` rows.

    Uses the end-to-end rate in place of macro F1, so the frontier can differ from the
    F1 one.
    """
    points = {
        r.label: (r.cost_usd, r.end_to_end_rate)
        for r in rows
        if r.end_to_end_rate is not None
    }
    if not points:
        return rows
    verdicts = {p.config_label: p for p in fr.pareto_frontier(points)}
    return tuple(
        EndToEnd(
            label=r.label, field=r.field, n_opportunities=r.n_opportunities,
            n_excluded_uncertain=r.n_excluded_uncertain,
            detection_rate=r.detection_rate, end_to_end_rate=r.end_to_end_rate,
            n_funders=r.n_funders, cost_usd=r.cost_usd,
            cost_per_usable=r.cost_per_usable,
            on_frontier=bool(verdicts[r.label].on_frontier) if r.label in verdicts else False,
            dominated_by=verdicts[r.label].dominated_by if r.label in verdicts else (),
        )
        for r in rows
    )
