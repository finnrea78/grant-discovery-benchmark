"""Scores one `ExtractionResult` against the frozen ground truth.

Computes opportunity-detection precision, recall and F1 for one (funder, run) cell,
plus per-stratum and field-level scores. The TP/FP/FN records are kept so errors can
be inspected, and each false positive has a reason:

* `distractor`: matched a row with `in_scope = no` (a real page, wrongly included).
* `unlabelled_page` / `fabricated_url`: the URL matches no row. The mirror tells these
  apart; only `fabricated_url` (not in the snapshot) is a hallucination. Without a
  mirror the reason is `unmatched` and the hallucination rate is undefined.
* `ambiguous_title`: a shared ground-truth URL whose title matched no row on it.
* `duplicate`: a second record for a row already claimed in the same run. A staged
  call is one row, so one record per stage gives one TP and one duplicate.

All of these count against precision. A run that extracts nothing has precision 0.0
with `precision_undefined = True`.

Per-stratum precision only uses false positives that matched a row, since unmatched
ones have no stratum, so per-stratum recall is the more reliable figure.

Field accuracy is scored over true positives only. A blank ground-truth value means
"the page doesn't say" and is a real target. A field listed in `uncertain_fields` has
no reliable target and is excluded. A blank `status` is a labelling gap and is also
excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from grant_eval.schema import (
    DatePrecision,
    ExtractedOpportunity,
    ExtractionResult,
    GroundTruthOpportunity,
    PartialDate,
    Status,
)
from grant_eval.scoring.groundtruth import (
    DEFAULT_GROUND_TRUTH_DIR,
    FunderGroundTruth,
    GroundTruth,
    load_ground_truth,
)
from grant_eval.scoring.identity import MatchKey, match_key
from grant_eval.scoring.mirror import (
    DEFAULT_REGISTRY,
    DEFAULT_SNAPSHOT_DIR,
    MirrorIndex,
    load_mirror,
)

#: The tag used for the restricted-vs-open analysis. Other tags are descriptive only.
STRATUM_TAG = "restricted"

#: Every value `FalsePositive.reason` can take. `unmatched` only appears when no mirror
#: was supplied, in place of `unlabelled_page` / `fabricated_url`.
FP_REASONS = (
    "distractor",
    "duplicate",
    "ambiguous_title",
    "unlabelled_page",
    "fabricated_url",
    "unmatched",
)

#: The one reason that counts as a hallucination: a URL absent from the frozen mirror.
FABRICATED = "fabricated_url"

#: Fields scored for accuracy; listing one in `uncertain_fields` excludes it.
SCORED_FIELDS = ("status", "deadlines", "amount_min_gbp", "amount_max_gbp")

#: Run-level telemetry copied into `to_dict`, so frozen scores include cost and failures.
TELEMETRY_FIELDS = (
    "split", "cost_usd", "latency_s", "input_tokens", "output_tokens",
    "timed_out", "context_overflow", "abstained", "schema_valid",
    "submit_failed", "step_budget_exhausted", "attempts", "provider",
)


def _telemetry(result: ExtractionResult) -> dict:
    """One run's telemetry, plus `run_failed`.

    `run_failed` is `timed_out or context_overflow`, defined once here so every analysis
    uses the same rule. Abstaining is model behaviour, not a failure. Older results may
    lack some fields, so these come back as `None` (a missing cost is not a zero cost).
    """
    telemetry = {name: getattr(result, name, None) for name in TELEMETRY_FIELDS}
    telemetry["split"] = getattr(result.split, "value", result.split)
    telemetry["run_failed"] = bool(telemetry["timed_out"] or telemetry["context_overflow"])
    return telemetry


def deadline_key(pd: PartialDate) -> str:
    """A comparable string for one `PartialDate`, at its own precision.

    Ground truth and agent output can store the same month differently (one with
    `resolved = None`, one resolved to the 1st), so we compare at the stated precision:

        DAY   -> `YYYY-MM-DD`
        MONTH -> `YYYY-MM`
        YEAR  -> `YYYY`
        NONE  -> the casefolded, whitespace-collapsed `raw` text

    A coarse date is never treated as matching a precise one.
    """
    raw = " ".join((pd.raw or "").split()).casefold()
    if pd.precision is DatePrecision.DAY:
        return pd.resolved.isoformat() if pd.resolved else raw
    if pd.precision is DatePrecision.MONTH:
        if pd.resolved:
            return f"{pd.resolved.year:04d}-{pd.resolved.month:02d}"
        return raw
    if pd.precision is DatePrecision.YEAR:
        if pd.resolved:
            return f"{pd.resolved.year:04d}"
        return raw
    return raw


def deadline_set(deadlines) -> frozenset[str]:
    """The full deadline set as comparable strings; the whole set must match."""
    return frozenset(deadline_key(d) for d in (deadlines or []))


def _amount_bound(opportunity, bound: str) -> float | None:
    """`amount.min_gbp` or `amount.max_gbp`, or None if there is no `amount`."""
    amount = getattr(opportunity, "amount", None)
    if amount is None:
        return None
    value = getattr(amount, bound)
    return None if value is None else round(float(value), 2)


def _field_values(opportunity, name: str):
    if name == "status":
        return opportunity.status
    if name == "deadlines":
        return deadline_set(opportunity.deadlines)
    if name == "amount_min_gbp":
        return _amount_bound(opportunity, "min_gbp")
    if name == "amount_max_gbp":
        return _amount_bound(opportunity, "max_gbp")
    raise KeyError(name)  # pragma: no cover


def _is_blank(name: str, value) -> bool:
    """True if the value is the "page doesn't say" blank for its field."""
    if name == "deadlines":
        return not value
    return value is None


@dataclass(frozen=True)
class TruePositive:
    """One extracted record matched to one in-scope ground-truth row."""

    key: MatchKey
    extracted: ExtractedOpportunity
    gold: GroundTruthOpportunity

    @property
    def opportunity_id(self) -> str:
        return self.gold.opportunity_id


@dataclass(frozen=True)
class FalsePositive:
    """One extracted record that claims no unclaimed in-scope row.

    `reason` is one of `FP_REASONS`; `gold` is the matched row, if any.
    """

    key: MatchKey
    extracted: ExtractedOpportunity
    reason: str
    gold: GroundTruthOpportunity | None = None


@dataclass(frozen=True)
class FalseNegative:
    """One in-scope ground-truth row no extracted record claimed."""

    key: MatchKey
    gold: GroundTruthOpportunity

    @property
    def opportunity_id(self) -> str:
        return self.gold.opportunity_id


@dataclass
class DetectionScore:
    """Detection precision / recall / F1 with the records behind them."""

    true_positives: list[TruePositive] = field(default_factory=list)
    false_positives: list[FalsePositive] = field(default_factory=list)
    false_negatives: list[FalseNegative] = field(default_factory=list)
    n_in_scope: int = 0
    n_extracted: int = 0
    #: True when a mirror was supplied, so the hallucination rate is defined.
    mirror_resolved: bool = False

    @property
    def tp(self) -> int:
        return len(self.true_positives)

    @property
    def fp(self) -> int:
        return len(self.false_positives)

    @property
    def fn(self) -> int:
        return len(self.false_negatives)

    @property
    def precision_undefined(self) -> bool:
        """True when the run extracted nothing, so precision has no denominator."""
        return self.n_extracted == 0

    @property
    def precision(self) -> float:
        return 0.0 if self.n_extracted == 0 else self.tp / self.n_extracted

    @property
    def recall(self) -> float:
        return 0.0 if self.n_in_scope == 0 else self.tp / self.n_in_scope

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    @property
    def n_fabricated(self) -> int:
        """Extracted records whose URL is not in the snapshot.

        Always 0 without a mirror, so check `hallucination_rate_undefined` too.
        """
        return sum(1 for f in self.false_positives if f.reason == FABRICATED)

    @property
    def hallucination_rate_undefined(self) -> bool:
        """True when there is no mirror or nothing was extracted."""
        return not self.mirror_resolved or self.n_extracted == 0

    @property
    def hallucination_rate(self) -> float:
        """Share of extracted records that do not exist in the frozen mirror.

        Returns 0.0 when undefined, like `precision`.
        """
        if self.hallucination_rate_undefined:
            return 0.0
        return self.n_fabricated / self.n_extracted

    def fp_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.false_positives:
            counts[f.reason] = counts.get(f.reason, 0) + 1
        return counts


@dataclass
class StratumScore:
    """Detection within one side of the `restricted` split.

    `precision` and `f1` only use false positives that matched a row in this stratum.
    """

    name: str
    n_in_scope: int = 0
    tp: int = 0
    fn: int = 0
    fp_attributable: int = 0

    @property
    def recall(self) -> float:
        return 0.0 if self.n_in_scope == 0 else self.tp / self.n_in_scope

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp_attributable
        return 0.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


@dataclass
class FieldAccuracy:
    """Accuracy of one field over the true-positive pairs of one run."""

    name: str
    correct: int = 0
    incorrect: int = 0
    excluded_uncertain: int = 0
    excluded_no_target: int = 0
    gt_blank_agent_asserted: int = 0
    gt_value_agent_blank: int = 0
    #: `(opportunity_id, gold_value, extracted_value)` for every mismatch.
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def scored(self) -> int:
        return self.correct + self.incorrect

    @property
    def accuracy(self) -> float | None:
        return None if self.scored == 0 else self.correct / self.scored


@dataclass
class RunScore:
    """Everything computed for one (funder, run) cell."""

    funder: str
    result: ExtractionResult
    detection: DetectionScore
    strata: dict[str, StratumScore]
    fields: dict[str, FieldAccuracy]
    #: In-scope row counts per edge-case tag (descriptive).
    tag_counts: dict[str, int] = field(default_factory=dict)
    #: False positives that could not be attributed to a stratum (no ground-truth row).
    fp_unattributable: int = 0
    source_path: Path | None = None

    @property
    def run_label(self) -> str:
        r = self.result
        return (
            f"{r.funder} | {r.model} | {r.prompt_strategy} | t{r.temperature} | "
            f"{r.agent_config} | rep{r.replicate}"
        )


def _unmatched_reason(key: MatchKey, gt_urls, opp, mirror: MirrorIndex | None) -> str:
    """Why this record matched no ground-truth row.

    If the URL is a ground-truth URL, only the title failed (`ambiguous_title`).
    Otherwise the mirror decides whether the page exists; with no mirror the reason
    is `unmatched`.
    """
    if key[1] in gt_urls:
        return "ambiguous_title"
    if mirror is None:
        return "unmatched"
    return "unlabelled_page" if mirror.contains(opp.url or "") else FABRICATED


def score_detection(
    opportunities,
    funder_gt: FunderGroundTruth,
    funder: str,
    mirror: MirrorIndex | None = None,
) -> DetectionScore:
    """Match one run's extracted records against one funder's ground-truth rows.

    Records are taken in emitted order, so the first claim on a row is the TP and later
    ones are duplicates. `mirror` only relabels unmatched false positives; it never
    changes TP, FN, precision, recall or F1.
    """
    score = DetectionScore(
        n_in_scope=len(funder_gt.in_scope),
        n_extracted=len(opportunities),
        mirror_resolved=mirror is not None,
    )
    claimed: dict[MatchKey, ExtractedOpportunity] = {}
    urls = {k[1] for k in funder_gt.by_key}

    for opp in opportunities:
        key = match_key(
            funder=funder,
            url=opp.url or "",
            title=opp.title or "",
            shared=funder_gt.shared_urls,
        )
        gold = funder_gt.by_key.get(key)

        if gold is None:
            reason = _unmatched_reason(key, urls, opp, mirror)
            score.false_positives.append(FalsePositive(key=key, extracted=opp, reason=reason))
            continue
        if key in claimed:
            score.false_positives.append(
                FalsePositive(key=key, extracted=opp, reason="duplicate", gold=gold)
            )
            continue
        claimed[key] = opp
        if not gold.in_scope:
            score.false_positives.append(
                FalsePositive(key=key, extracted=opp, reason="distractor", gold=gold)
            )
            continue
        score.true_positives.append(TruePositive(key=key, extracted=opp, gold=gold))

    matched = {tp.key for tp in score.true_positives}
    for key, gold in funder_gt.by_key.items():
        if gold.in_scope and key not in matched:
            score.false_negatives.append(FalseNegative(key=key, gold=gold))
    return score


def score_strata(detection: DetectionScore, funder_gt: FunderGroundTruth, tag: str = STRATUM_TAG):
    """Split detection by the `restricted` tag. Returns `(strata, n_unattributable_fp)`."""

    def stratum_of(row: GroundTruthOpportunity) -> str:
        return tag if any(t.value == tag for t in row.edge_case_tags) else f"not_{tag}"

    strata = {
        tag: StratumScore(name=tag),
        f"not_{tag}": StratumScore(name=f"not_{tag}"),
    }
    for row in funder_gt.in_scope:
        strata[stratum_of(row)].n_in_scope += 1
    for tp in detection.true_positives:
        strata[stratum_of(tp.gold)].tp += 1
    for fn in detection.false_negatives:
        strata[stratum_of(fn.gold)].fn += 1

    unattributable = 0
    for fp in detection.false_positives:
        if fp.gold is None:
            unattributable += 1
        else:
            strata[stratum_of(fp.gold)].fp_attributable += 1
    return strata, unattributable


def score_fields(detection: DetectionScore) -> dict[str, FieldAccuracy]:
    """Field accuracy over the true-positive pairs."""
    fields = {name: FieldAccuracy(name=name) for name in SCORED_FIELDS}

    for tp in detection.true_positives:
        disclaimed = {f.strip() for f in tp.gold.uncertain_fields}
        for name in SCORED_FIELDS:
            acc = fields[name]
            if name in disclaimed:
                acc.excluded_uncertain += 1
                continue
            gold_value = _field_values(tp.gold, name)
            if name == "status" and gold_value is Status.UNKNOWN:
                # Blank mandatory column: a labelling gap, not "the page doesn't say".
                acc.excluded_no_target += 1
                continue
            ext_value = _field_values(tp.extracted, name)
            gold_blank = _is_blank(name, gold_value)
            ext_blank = _is_blank(name, ext_value)
            if gold_value == ext_value:
                acc.correct += 1
                continue
            acc.incorrect += 1
            if gold_blank and not ext_blank:
                acc.gt_blank_agent_asserted += 1
            elif ext_blank and not gold_blank:
                acc.gt_value_agent_blank += 1
            acc.mismatches.append(
                (tp.opportunity_id, _render(gold_value), _render(ext_value))
            )
    return fields


def _render(value) -> str:
    if isinstance(value, Status):
        return value.value
    if isinstance(value, frozenset):
        return "{" + ", ".join(sorted(value)) + "}" if value else "{}"
    return "" if value is None else str(value)


def score_run(
    result: ExtractionResult,
    ground_truth: GroundTruth,
    *,
    mirror: MirrorIndex | None = None,
    source_path: Path | None = None,
) -> RunScore:
    """Score one `ExtractionResult` (one funder, one run).

    `mirror` is this funder's snapshot index, or `None` to skip the hallucination split.
    """
    funder_gt = ground_truth.for_funder(result.funder)
    detection = score_detection(result.opportunities, funder_gt, funder_gt.funder, mirror)
    strata, unattributable = score_strata(detection, funder_gt)

    tag_counts: dict[str, int] = {}
    for row in funder_gt.in_scope:
        for t in row.edge_case_tags:
            tag_counts[t.value] = tag_counts.get(t.value, 0) + 1

    return RunScore(
        funder=funder_gt.funder,
        result=result,
        detection=detection,
        strata=strata,
        fields=score_fields(detection),
        tag_counts=tag_counts,
        fp_unattributable=unattributable,
        source_path=source_path,
    )


def load_result(path: str | Path) -> ExtractionResult:
    """Read one `ExtractionResult` JSON off disk."""
    return ExtractionResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def score_paths(paths, ground_truth: GroundTruth, mirror=None) -> list[RunScore]:
    """Score every result JSON in `paths` against one loaded ground truth.

    `mirror` is the whole `Mirror`; each result uses its own funder's index.
    """
    scores = []
    for p in paths:
        result = load_result(p)
        index = mirror.for_funder(result.funder) if mirror is not None else None
        scores.append(score_run(result, ground_truth, mirror=index, source_path=Path(p)))
    return scores


def format_report(score: RunScore, *, show_records: bool = True) -> str:
    """A readable per-run report. This is what the CLI prints."""
    d = score.detection
    out: list[str] = []
    out.append("=" * 78)
    out.append(score.run_label)
    if score.source_path:
        out.append(f"result: {score.source_path}")
    out.append("=" * 78)
    undef = "  (undefined — no records extracted)" if d.precision_undefined else ""
    out.append(
        f"detection   precision {d.precision:.3f}{undef}   recall {d.recall:.3f}   "
        f"F1 {d.f1:.3f}"
    )
    out.append(
        f"            TP {d.tp}   FP {d.fp}   FN {d.fn}   "
        f"(extracted {d.n_extracted}, in-scope rows {d.n_in_scope})"
    )
    by_reason = d.fp_by_reason()
    if by_reason:
        out.append(
            "            FP by reason: "
            + ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items()))
        )
    if d.hallucination_rate_undefined:
        why = (
            "no mirror supplied — the split was not resolved"
            if not d.mirror_resolved
            else "nothing extracted"
        )
        out.append(f"            hallucination rate: —  ({why})")
    else:
        out.append(
            f"            hallucination rate: {d.hallucination_rate:.3f}  "
            f"({d.n_fabricated} of {d.n_extracted} extracted URL(s) are in no snapshot)"
        )

    if show_records:
        out.append("")
        out.append(f"TRUE POSITIVES ({d.tp})")
        for tp in d.true_positives:
            out.append(f"  + {tp.opportunity_id}")
            out.append(f"      gold: {tp.gold.title}")
            out.append(f"      got : {tp.extracted.title}")
        out.append(f"FALSE POSITIVES ({d.fp})")
        for fp in d.false_positives:
            gold = f"  [row {fp.gold.opportunity_id}]" if fp.gold else ""
            out.append(f"  ! {fp.reason}{gold}")
            out.append(f"      got : {fp.extracted.title}")
            out.append(f"      url : {fp.extracted.url}")
            if fp.gold is not None and not fp.gold.in_scope and fp.gold.exclusion_reason:
                out.append(f"      why out of scope: {fp.gold.exclusion_reason}")
        out.append(f"FALSE NEGATIVES ({d.fn})")
        for fn in d.false_negatives:
            out.append(f"  - {fn.opportunity_id}")
            out.append(f"      gold: {fn.gold.title}  [{fn.gold.url}]")

    out.append("")
    out.append(f"STRATUM ({STRATUM_TAG}, D4)")
    out.append(f"  {'stratum':<16}{'n':>4}{'TP':>5}{'FN':>5}{'FP*':>5}{'recall':>9}{'prec*':>8}{'F1*':>8}")
    for name in (STRATUM_TAG, f"not_{STRATUM_TAG}"):
        s = score.strata[name]
        out.append(
            f"  {name:<16}{s.n_in_scope:>4}{s.tp:>5}{s.fn:>5}{s.fp_attributable:>5}"
            f"{s.recall:>9.3f}{s.precision:>8.3f}{s.f1:>8.3f}"
        )
    out.append(
        f"  * attributable FPs only; {score.fp_unattributable} FP(s) matched no row and "
        f"have no stratum."
    )
    if score.tag_counts:
        out.append(
            "  in-scope tag counts (descriptive; pdf_only/js_portal are NOT strata, D5): "
            + ", ".join(f"{k} {v}" for k, v in sorted(score.tag_counts.items()))
        )

    out.append("")
    out.append("FIELD ACCURACY (over true-positive pairs)")
    out.append(
        f"  {'field':<18}{'acc':>8}{'ok':>5}{'bad':>5}{'unc':>5}{'n/t':>5}"
        f"{'blank>said':>12}{'value>blank':>13}"
    )
    for name in SCORED_FIELDS:
        a = score.fields[name]
        acc = "  —  " if a.accuracy is None else f"{a.accuracy:.3f}"
        out.append(
            f"  {name:<18}{acc:>8}{a.correct:>5}{a.incorrect:>5}{a.excluded_uncertain:>5}"
            f"{a.excluded_no_target:>5}{a.gt_blank_agent_asserted:>12}"
            f"{a.gt_value_agent_blank:>13}"
        )
    out.append("  unc = excluded, labeller flagged the field uncertain (protocol: no target)")
    out.append("  n/t = excluded, no target in the ground truth (blank mandatory column)")
    if show_records:
        for name in SCORED_FIELDS:
            for oid, gold, got in score.fields[name].mismatches:
                out.append(f"  ~ {name}  {oid}")
                out.append(f"      gold: {gold!r}")
                out.append(f"      got : {got!r}")
    return "\n".join(out)


def to_dict(score: RunScore) -> dict:
    """A JSON-serialisable view of a `RunScore`, records included."""
    d = score.detection
    return {
        "funder": score.funder,
        "model": score.result.model,
        "prompt_strategy": score.result.prompt_strategy,
        "temperature": score.result.temperature,
        "agent_config": score.result.agent_config,
        "replicate": score.result.replicate,
        "config_hash": score.result.config_hash,
        "source_path": str(score.source_path) if score.source_path else None,
        "telemetry": _telemetry(score.result),
        "detection": {
            "precision": d.precision,
            "precision_undefined": d.precision_undefined,
            "recall": d.recall,
            "f1": d.f1,
            "tp": d.tp,
            "fp": d.fp,
            "fn": d.fn,
            "n_extracted": d.n_extracted,
            "n_in_scope": d.n_in_scope,
            # null when undefined (as in aggregate_to_dict) so pandas reads NaN, not 0.
            "mirror_resolved": d.mirror_resolved,
            "n_fabricated": d.n_fabricated if d.mirror_resolved else None,
            "hallucination_rate": (
                None if d.hallucination_rate_undefined else d.hallucination_rate
            ),
            "hallucination_rate_undefined": d.hallucination_rate_undefined,
            "fp_by_reason": d.fp_by_reason(),
            "true_positives": [tp.opportunity_id for tp in d.true_positives],
            "false_negatives": [fn.opportunity_id for fn in d.false_negatives],
            "false_positives": [
                {
                    "reason": fp.reason,
                    "title": fp.extracted.title,
                    "url": fp.extracted.url,
                    "matched_row": fp.gold.opportunity_id if fp.gold else None,
                }
                for fp in d.false_positives
            ],
        },
        "strata": {
            name: {
                "n_in_scope": s.n_in_scope,
                "tp": s.tp,
                "fn": s.fn,
                "fp_attributable": s.fp_attributable,
                "recall": s.recall,
                "precision": s.precision,
                "f1": s.f1,
            }
            for name, s in score.strata.items()
        },
        "fp_unattributable": score.fp_unattributable,
        "tag_counts": score.tag_counts,
        "fields": {
            name: {
                "accuracy": a.accuracy,
                "correct": a.correct,
                "incorrect": a.incorrect,
                "excluded_uncertain": a.excluded_uncertain,
                "excluded_no_target": a.excluded_no_target,
                "gt_blank_agent_asserted": a.gt_blank_agent_asserted,
                "gt_value_agent_blank": a.gt_value_agent_blank,
                "mismatches": a.mismatches,
            }
            for name, a in score.fields.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="score",
        description=(
            "Score ExtractionResult JSON file(s) against the frozen ground truth "
            "(match key: funder + normalised URL, title only where the URL is shared)."
        ),
    )
    p.add_argument("results", nargs="+", help="ExtractionResult JSON file(s), or directories of them")
    p.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH_DIR),
        help=f"ground-truth CSV directory (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of the report")
    p.add_argument("--quiet-records", action="store_true", help="counts only, no record lists")
    p.add_argument(
        "--show-rejected",
        action="store_true",
        help="list the CSVs in the ground-truth directory that were NOT loaded, and why",
    )
    p.add_argument(
        "--snapshots",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help=(
            "frozen snapshot root, used to tell a fabricated URL from a real page the "
            f"labeller never enumerated (default: {DEFAULT_SNAPSHOT_DIR})"
        ),
    )
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help=f"funder registry locating each snapshot (default: {DEFAULT_REGISTRY})",
    )
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help=(
            "skip mirror resolution: false positives stay in the undivided `unmatched` "
            "bucket and the hallucination rate is reported as undefined"
        ),
    )
    return p


def _expand(paths) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        out.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    return out


def _resolve_mirror(args):
    """Load the mirror, or return `None` and print why to stderr.

    A missing mirror is not fatal: detection scores still work, but the hallucination
    rate is undefined.
    """
    if args.no_mirror:
        return None
    try:
        mirror = load_mirror(registry=args.registry, snapshots_dir=args.snapshots)
    except (OSError, ValueError) as exc:
        print(
            f"mirror unavailable ({exc}); false positives stay `unmatched` and the "
            f"hallucination rate is undefined",
            file=sys.stderr,
        )
        return None
    for slug, reason in mirror.missing:
        print(f"no mirror for {slug}: {reason}", file=sys.stderr)
    return mirror


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    gt = load_ground_truth(args.ground_truth)
    mirror = _resolve_mirror(args)
    paths = _expand(args.results)
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    if args.show_rejected and not args.json:
        print(
            f"ground truth: {len(gt.files)} file(s), {len(gt.rows)} rows, "
            f"{len(gt.in_scope)} in scope, {len(gt.funders)} funders"
        )
        for path, reason in gt.rejected:
            print(f"  rejected {path.name}: {reason}")
        print()

    scores = []
    for path in paths:
        try:
            result = load_result(path)
            index = mirror.for_funder(result.funder) if mirror is not None else None
            scores.append(score_run(result, gt, mirror=index, source_path=path))
        except KeyError as exc:  # unlabelled funder: report it and carry on
            print(f"skipped {path}: {exc}", file=sys.stderr)

    if not scores:
        return 1

    if args.json:
        print(json.dumps([to_dict(s) for s in scores], indent=2, default=str))
        return 0

    for s in scores:
        print(format_report(s, show_records=not args.quiet_records))
        print()

    if len(scores) > 1:
        print("=" * 78)
        print("SUMMARY")
        print(f"  {'precision':>9}{'recall':>8}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}  run")
        for s in scores:
            d = s.detection
            print(
                f"  {d.precision:>9.3f}{d.recall:>8.3f}{d.f1:>7.3f}{d.tp:>5}{d.fp:>5}"
                f"{d.fn:>5}  {s.source_path.name if s.source_path else s.run_label}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
