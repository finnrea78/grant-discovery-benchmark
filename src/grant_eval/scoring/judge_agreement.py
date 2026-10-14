"""Human spot-check of the LLM judge and judge-human agreement.

Exports a stratified, blinded sample of judged claims for hand-coding, joins the coded
file back to the stored verdicts, and reports raw agreement and Krippendorff's alpha.
It reports statistics only and does not apply a pass/fail threshold. The CLI flags in
`judge.py` (`--export-spotcheck`/`--out`, `--agreement`, `--test-retest`/
`--spotcheck-csv`) call into this module.

The sampling unit is the claim, because the judge's main metric is claim-level.
`human_code` has three values: `supported`, `unsupported` and `not_a_claim`. The third
catches a judge that invents or merges claims. Agreement and alpha use the binary
subset; `not_a_claim` is reported as its own rate.

Sampling uses 8 cells, `field` x `dimension` x judge verdict, 25 claims each by
default. Equal allocation already oversamples the rarer `unsupported` verdicts. A cell
that is short passes its remainder to the same field and verdict in the other
dimension. Funder is not a stratum (there would be 128 cells); the funder spread is
reported at export time instead.

Each exported claim stores its cell's sampling probability. Agreement is reported per
stratum and as one overall figure reweighted by 1 / probability (Horvitz-Thompson).
The reweighted raw agreement is the figure to compare with a threshold; the reweighted
alpha is illustrative only. Pooled figures that are not reweighted are printed with
`UNWEIGHTED_SAMPLE_CAVEAT`.

`krippendorff_alpha_binary` implements the unweighted binary case and is checked
against a published example. The weighted version is our own extension and reduces to
the standard formula when all weights are 1.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from grant_eval.scoring import score as scoremod
from grant_eval.scoring.judge import (
    DIMENSIONS,
    JUDGED_FIELDS,
    JudgeBackend,
    JudgeConfig,
    JudgeOutcome,
    _extract_mirror_page_text,
    build_completeness_prompt,
    build_groundedness_prompt,
    build_mirror_lookup,
    funder_slug,
    read_verdict_file,
    resolve_mirror_path,
    run_judge,
    verdict_path,
)

#: field x dimension x judge verdict: the 8 sampling cells.
ALL_CELLS: tuple[tuple[str, str, bool], ...] = tuple(
    itertools.product(JUDGED_FIELDS, DIMENSIONS, (True, False))
)
SPOTCHECK_CELL_TARGET = 25  # 200 claims / 8 cells
TEST_RETEST_TARGET = 200

#: Fixed seeds so re-running an export gives the same sample and finished coding stays
#: valid. The two seeds differ so the two draws are independent.
SPOTCHECK_SEED = 20260908
TEST_RETEST_SEED = 20260909

_VALID_HUMAN_CODES = ("supported", "unsupported", "not_a_claim")


def compute_item_id(
    run_config_hash: str, funder_slug: str, replicate: int,
    opportunity_id: str, field: str, dimension: str, claim_index: int,
) -> str:
    """Stable row key for the export: SHA-256 over all the identifying fields.

    `funder_slug` and `replicate` are needed because `run_config_hash` is shared by
    every funder and replicate of one grid arm, and `opportunity_id` (the raw id, not
    the match key) is not unique across funders. `--agreement` recomputes this from
    the coded row as a corruption check.
    """
    payload = f"{run_config_hash}|{funder_slug}|{replicate}|{opportunity_id}|{field}|{dimension}|{claim_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




@dataclass(frozen=True)
class JudgedRecord:
    """One (run, opportunity, field, dimension) cell whose stored outcome was `"ok"`.

    `claims` and `record_supported` come from the saved verdict. The text fields are
    rebuilt from the original result, ground truth and mirror. `comparison_text` is
    what the judge sees (gold text or mirror page text); `comparison_display` is what
    the human sees (gold text or mirror file path).
    """

    run_config_hash: str
    judge_config_hash: str
    funder: str
    funder_slug: str
    replicate: int
    opportunity_id: str
    field: str
    dimension: str
    extracted_text: str
    comparison_text: str
    comparison_display: str
    snapshot_date: object
    claims: list[dict]
    record_supported: bool


@dataclass(frozen=True)
class LoadedRecords:
    """Result of `load_judged_records`, with every exclusion named and counted."""

    records: list[JudgedRecord]
    #: No persisted verdict file at all for this (run, funder, replicate, judge config).
    skipped_unjudged: list[Path]
    #: No ground truth for this funder, or no registry slug to build a verdict path.
    skipped_unlabelled_funder: list[Path]
    #: The stored verdict's `funder`/`replicate` disagree with the result. Should
    #: always be empty; kept as a check against stale files or slug collisions.
    mismatched_funder: list[dict]
    #: Records whose outcome is not `"ok"`, so there are no claims to sample.
    excluded_records: list[dict]


def load_judged_records(
    paths, *, ground_truth, registry, snapshot_root: Path, results_root: Path, judge_config_hash: str,
) -> LoadedRecords:
    """Rebuild the judged records from the original results and the saved verdicts.

    Follows the same populations as `score_run_judge` (completeness over true
    positives, groundedness over records found in the mirror). Read-only: makes no
    judge calls.
    """
    records: list[JudgedRecord] = []
    skipped_unjudged: list[Path] = []
    skipped_unlabelled_funder: list[Path] = []
    mismatched_funder: list[dict] = []
    excluded_records: list[dict] = []

    for path in paths:
        result = scoremod.load_result(path)
        try:
            run_score = scoremod.score_run(result, ground_truth, source_path=Path(path))
        except KeyError:
            skipped_unlabelled_funder.append(Path(path))
            continue

        slug = funder_slug(result.funder, registry)
        if slug is None:
            skipped_unlabelled_funder.append(Path(path))
            continue

        run_config_hash = result.config_hash
        vpath = verdict_path(
            run_config_hash, judge_config_hash,
            funder_slug=slug, replicate=result.replicate, root=results_root,
        )
        if not vpath.is_file():
            skipped_unjudged.append(Path(path))
            continue
        stored = read_verdict_file(vpath)
        if stored.get("funder") != result.funder or stored.get("replicate") != result.replicate:
            mismatched_funder.append({
                "path": str(path), "expected_funder": result.funder, "expected_replicate": result.replicate,
                "stored_funder": stored.get("funder"), "stored_replicate": stored.get("replicate"),
            })
            continue

        def _records_by_opportunity(f: str, d: str) -> dict:
            return {r["opportunity_id"]: r for r in stored["fields"][f"{f}:{d}"]["records"]}

        completeness = {f: _records_by_opportunity(f, "completeness") for f in JUDGED_FIELDS}
        groundedness = {f: _records_by_opportunity(f, "groundedness") for f in JUDGED_FIELDS}

        for tp in run_score.detection.true_positives:
            for f in JUDGED_FIELDS:
                extracted_text = getattr(tp.extracted, f)
                if not (extracted_text or "").strip():
                    continue
                rec = completeness[f].get(tp.extracted.opportunity_id)
                if rec is None or rec["outcome"] != "ok":
                    excluded_records.append({
                        "path": str(path), "opportunity_id": tp.extracted.opportunity_id,
                        "field": f, "dimension": "completeness",
                        "reason": "no_stored_record" if rec is None else rec["outcome"],
                        # Stored failure text, so this can be diagnosed without a new call.
                        "error": None if rec is None else rec.get("error"),
                    })
                    continue
                gold_text = getattr(tp.gold, f) or ""
                records.append(JudgedRecord(
                    run_config_hash=run_config_hash, judge_config_hash=judge_config_hash,
                    funder=result.funder, funder_slug=slug, replicate=result.replicate,
                    opportunity_id=tp.extracted.opportunity_id,
                    field=f, dimension="completeness",
                    extracted_text=extracted_text, comparison_text=gold_text, comparison_display=gold_text,
                    snapshot_date=result.snapshot_date, claims=rec["claims"] or [],
                    record_supported=bool(rec["record_supported"]),
                ))

        lookup = build_mirror_lookup(result.funder, result.snapshot_date, registry=registry, snapshot_root=snapshot_root)
        for opp in result.opportunities:
            mirror_path = resolve_mirror_path(lookup, opp.url)
            if mirror_path is None:
                continue
            page_text: str | None = None  # extracted once per opportunity, not per field
            for f in JUDGED_FIELDS:
                extracted_text = getattr(opp, f)
                if not (extracted_text or "").strip():
                    continue
                rec = groundedness[f].get(opp.opportunity_id)
                if rec is None or rec["outcome"] != "ok":
                    excluded_records.append({
                        "path": str(path), "opportunity_id": opp.opportunity_id,
                        "field": f, "dimension": "groundedness",
                        "reason": "no_stored_record" if rec is None else rec["outcome"],
                        "error": None if rec is None else rec.get("error"),
                    })
                    continue
                if page_text is None:
                    page_text = _extract_mirror_page_text(mirror_path).text
                records.append(JudgedRecord(
                    run_config_hash=run_config_hash, judge_config_hash=judge_config_hash,
                    funder=result.funder, funder_slug=slug, replicate=result.replicate,
                    opportunity_id=opp.opportunity_id,
                    field=f, dimension="groundedness",
                    extracted_text=extracted_text, comparison_text=page_text,
                    comparison_display=str(mirror_path),
                    snapshot_date=result.snapshot_date, claims=rec["claims"] or [],
                    record_supported=bool(rec["record_supported"]),
                ))

    return LoadedRecords(
        records=records, skipped_unjudged=skipped_unjudged,
        skipped_unlabelled_funder=skipped_unlabelled_funder,
        mismatched_funder=mismatched_funder, excluded_records=excluded_records,
    )


def record_key(rec: JudgedRecord) -> tuple[str, str, int, str, str, str]:
    """Record-level key used to keep the test-retest sample apart from the spot-check."""
    return (rec.run_config_hash, rec.funder_slug, rec.replicate, rec.opportunity_id, rec.field, rec.dimension)


@dataclass(frozen=True)
class JudgedClaim:
    """One claim from a `JudgedRecord`.

    `judge_verdict` is used for stratifying but is never written to the export CSV.
    """

    item_id: str
    run_config_hash: str
    judge_config_hash: str
    funder: str
    funder_slug: str
    replicate: int
    opportunity_id: str
    field: str
    dimension: str
    claim_index: int
    claim_text: str
    claim_type: str
    judge_verdict: bool
    extracted_text: str
    comparison_display: str


def explode_claims(records: list[JudgedRecord]) -> list[JudgedClaim]:
    claims: list[JudgedClaim] = []
    for rec in records:
        for i, c in enumerate(rec.claims):
            claims.append(JudgedClaim(
                item_id=compute_item_id(
                    rec.run_config_hash, rec.funder_slug, rec.replicate,
                    rec.opportunity_id, rec.field, rec.dimension, i,
                ),
                run_config_hash=rec.run_config_hash, judge_config_hash=rec.judge_config_hash,
                funder=rec.funder, funder_slug=rec.funder_slug, replicate=rec.replicate,
                opportunity_id=rec.opportunity_id,
                field=rec.field, dimension=rec.dimension, claim_index=i,
                claim_text=c["text"], claim_type=c["claim_type"], judge_verdict=bool(c["supported"]),
                extracted_text=rec.extracted_text, comparison_display=rec.comparison_display,
            ))
    return claims


def allocate_targets(population: dict[tuple[str, str, bool], int], *, cell_target: int) -> dict[tuple[str, str, bool], int]:
    """Target sample size for each of the 8 cells.

    Each cell gets `min(cell_target, population)`. A short cell passes its remainder to
    the same field and verdict in the other dimension, once only. If both are short the
    pair simply comes in under target. Missing cells count as zero.
    """
    targets = {cell: min(cell_target, population.get(cell, 0)) for cell in ALL_CELLS}
    for cell_field, dimension, verdict in ALL_CELLS:
        shortfall = cell_target - targets[(cell_field, dimension, verdict)]
        if shortfall <= 0:
            continue
        other_dimension = next(d for d in DIMENSIONS if d != dimension)
        other_cell = (cell_field, other_dimension, verdict)
        headroom = population.get(other_cell, 0) - targets[other_cell]
        targets[other_cell] += min(shortfall, max(0, headroom))
    return targets


@dataclass(frozen=True)
class SpotcheckSelection:
    claims: list[JudgedClaim]                       # selected, row order randomised
    probabilities: dict[str, float]                  # item_id -> sampling probability
    population: dict[tuple[str, str, bool], int]      # cell -> corpus-wide population
    achieved: dict[tuple[str, str, bool], int]        # cell -> n actually selected


def select_spotcheck_sample(
    claims: list[JudgedClaim], *, cell_target: int = SPOTCHECK_CELL_TARGET, rng: random.Random,
) -> SpotcheckSelection:
    """Draw the stratified sample and record each claim's sampling probability.

    Each cell is sampled without replacement, so the probability is
    `achieved / population`. The final list is shuffled so row order reveals nothing.
    """
    by_cell: dict[tuple[str, str, bool], list[JudgedClaim]] = {cell: [] for cell in ALL_CELLS}
    for c in claims:
        by_cell[(c.field, c.dimension, c.judge_verdict)].append(c)
    population = {cell: len(items) for cell, items in by_cell.items()}
    targets = allocate_targets(population, cell_target=cell_target)

    selected: list[JudgedClaim] = []
    probabilities: dict[str, float] = {}
    for cell, target in targets.items():
        items = by_cell[cell]
        chosen = items if target >= len(items) else rng.sample(items, target)
        p = 0.0 if not items else target / len(items)
        for c in chosen:
            probabilities[c.item_id] = p
        selected.extend(chosen)
    rng.shuffle(selected)
    return SpotcheckSelection(claims=selected, probabilities=probabilities, population=population, achieved=targets)


#: `funder_slug` and `replicate` let `--agreement` rebuild verdict paths and item ids
#: without the registry; `funder` records the funder spread in the export itself.
SPOTCHECK_FIELDNAMES: tuple[str, ...] = (
    "item_id", "run_config_hash", "judge_config_hash", "funder", "funder_slug", "replicate",
    "opportunity_id", "field", "dimension",
    "claim_index", "claim_text", "claim_type", "extracted_text", "comparison_reference",
    "sampling_probability", "human_code", "human_notes",
)


def write_spotcheck_csv(selection: SpotcheckSelection, out_path: Path) -> None:
    """Write the coding sheet with empty `human_code`/`human_notes` columns.

    There is no judge verdict or justification column, so the coder stays blind.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPOTCHECK_FIELDNAMES)
        writer.writeheader()
        for c in selection.claims:
            writer.writerow({
                "item_id": c.item_id, "run_config_hash": c.run_config_hash,
                "judge_config_hash": c.judge_config_hash,
                "funder": c.funder, "funder_slug": c.funder_slug, "replicate": c.replicate,
                "opportunity_id": c.opportunity_id,
                "field": c.field, "dimension": c.dimension, "claim_index": c.claim_index,
                "claim_text": c.claim_text, "claim_type": c.claim_type,
                "extracted_text": c.extracted_text, "comparison_reference": c.comparison_display,
                "sampling_probability": selection.probabilities[c.item_id],
                "human_code": "", "human_notes": "",
            })


def krippendorff_alpha_binary(
    pairs, weights: list[float] | None = None,
) -> tuple[float | None, bool]:
    """Krippendorff's alpha for two raters and binary values, with no missing data.

    `pairs` is `[(rater_a, rater_b), ...]`, one per item. Uses the coincidence-matrix
    form alpha = 1 - (n-1) * o_mismatch / (n_a * n_b) from Krippendorff (2011),
    "Computing Krippendorff's Alpha-Reliability", Section A; the tests check it against
    that paper's worked example.

    Optional `weights` scale each item's contribution (our own Horvitz-Thompson style
    extension). They are rescaled to sum to `len(pairs)` so that multiplying all
    weights by a constant changes nothing. Returns `(None, True)` when undefined:
    fewer than 2 items, no positive weight, or only one value seen.
    """
    if weights is None:
        weights = [1.0] * len(pairs)
    if len(pairs) < 2:
        return None, True
    total_weight = sum(weights)
    if total_weight <= 0:
        return None, True
    scale = len(pairs) / total_weight
    weights = [w * scale for w in weights]

    o_mismatch = 0.0
    n_by_value: dict = {}
    n = 0.0
    for (a, b), w in zip(pairs, weights):
        if a != b:
            o_mismatch += w
        n_by_value[a] = n_by_value.get(a, 0.0) + w
        n_by_value[b] = n_by_value.get(b, 0.0) + w
        n += 2 * w

    if len(n_by_value) < 2:
        return None, True
    if len(n_by_value) > 2:
        raise ValueError(
            f"krippendorff_alpha_binary expects exactly 2 distinct values, got {sorted(map(str, n_by_value))!r}"
        )
    n_a, n_b = n_by_value.values()
    denominator = n_a * n_b
    if denominator == 0:
        return None, True
    return 1.0 - (n - 1.0) * o_mismatch / denominator, False


def _weighted_rate(flags: list[bool], weights: list[float] | None = None) -> tuple[float | None, bool]:
    """Weighted share of `True` flags (unweighted if `weights` is None).

    Returns `(None, True)` for empty input or zero total weight.
    """
    if not flags:
        return None, True
    if weights is None:
        weights = [1.0] * len(flags)
    total = sum(weights)
    if total == 0:
        return None, True
    return sum(w for flag, w in zip(flags, weights) if flag) / total, False


def _raw_agreement(pairs, weights: list[float] | None = None) -> tuple[float | None, bool]:
    """Weighted share of pairs that agree (unweighted if `weights` is None)."""
    return _weighted_rate([a == b for a, b in pairs], weights)


def load_coded_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass(frozen=True)
class JoinedRow:
    item_id: str
    field: str
    dimension: str
    human_code: str  # "supported" | "unsupported" | "not_a_claim"
    judge_verdict: bool
    sampling_probability: float


def join_coded_rows(rows: list[dict], *, results_root: Path) -> tuple[list[JoinedRow], list[dict], int]:
    """Join each coded row to its stored claim verdict by `item_id`.

    Returns `(joined, unjoinable, n_uncoded)`. Each unjoinable row comes with a reason.
    Rows with a blank `human_code` are counted in `n_uncoded`.
    """
    joined: list[JoinedRow] = []
    unjoinable: list[dict] = []
    n_uncoded = 0
    verdict_cache: dict[tuple[str, str, str, int], dict | None] = {}

    for row in rows:
        item_id = row.get("item_id", "")
        human_code = (row.get("human_code") or "").strip()
        if not human_code:
            n_uncoded += 1
            continue
        if human_code not in _VALID_HUMAN_CODES:
            unjoinable.append({"item_id": item_id, "reason": f"unrecognised human_code {human_code!r}"})
            continue

        try:
            run_config_hash = row["run_config_hash"]
            judge_config_hash = row["judge_config_hash"]
            row_funder_slug = row["funder_slug"]
            replicate = int(row["replicate"])
            opportunity_id = row["opportunity_id"]
            row_field = row["field"]
            dimension = row["dimension"]
            claim_index = int(row["claim_index"])
            sampling_probability = float(row["sampling_probability"])
        except (KeyError, ValueError) as exc:
            unjoinable.append({"item_id": item_id, "reason": f"malformed row: {exc}"})
            continue

        expected = compute_item_id(
            run_config_hash, row_funder_slug, replicate, opportunity_id, row_field, dimension, claim_index,
        )
        if expected != item_id:
            unjoinable.append({"item_id": item_id, "reason": "item_id does not match its own row fields"})
            continue
        if sampling_probability <= 0:
            unjoinable.append({"item_id": item_id, "reason": f"non-positive sampling_probability {sampling_probability!r}"})
            continue

        # The config hashes alone do not identify a run, so include funder and replicate.
        cache_key = (run_config_hash, row_funder_slug, replicate, judge_config_hash)
        if cache_key not in verdict_cache:
            vpath = verdict_path(
                run_config_hash, judge_config_hash,
                funder_slug=row_funder_slug, replicate=replicate, root=results_root,
            )
            verdict_cache[cache_key] = read_verdict_file(vpath) if vpath.is_file() else None
        stored = verdict_cache[cache_key]
        if stored is None:
            unjoinable.append({"item_id": item_id, "reason": f"no stored verdict file for run {run_config_hash!r}"})
            continue

        cell = stored["fields"].get(f"{row_field}:{dimension}")
        record = None if cell is None else next(
            (r for r in cell["records"] if r["opportunity_id"] == opportunity_id), None
        )
        if record is None or record["outcome"] != "ok" or claim_index >= len(record["claims"] or []):
            unjoinable.append({"item_id": item_id, "reason": "no matching judged claim in the stored verdict"})
            continue

        joined.append(JoinedRow(
            item_id=item_id, field=row_field, dimension=dimension, human_code=human_code,
            judge_verdict=bool(record["claims"][claim_index]["supported"]),
            sampling_probability=sampling_probability,
        ))

    return joined, unjoinable, n_uncoded


@dataclass(frozen=True)
class AgreementReport:
    """Agreement figures from the joined spot-check rows.

    Equal-sized cells oversample `unsupported`, so any figure pooled across strata is
    given both reweighted (1 / sampling_probability, comparable with the corpus) and
    unweighted (the sample only). `by_stratum` needs no reweighting, since each stratum
    has a single sampling probability.
    """

    n_joined: int
    not_a_claim_rate: float | None                 # sample only (unweighted)
    not_a_claim_rate_undefined: bool
    not_a_claim_rate_reweighted: float | None      # corpus-comparable
    not_a_claim_rate_reweighted_undefined: bool
    by_stratum: dict[str, dict]           # "<field>:<dimension>:<supported|unsupported>" -> {n, raw_agreement, raw_agreement_undefined, alpha, alpha_undefined}
    raw_agreement_overall_reweighted: float | None
    raw_agreement_overall_reweighted_undefined: bool
    alpha_overall_reweighted: float | None
    alpha_overall_reweighted_undefined: bool
    by_field: dict[str, dict]             # field -> {n, raw_agreement(+_undefined), raw_agreement_reweighted(+_undefined)}
    by_dimension: dict[str, dict]         # dimension -> same shape as by_field
    confusion: dict[str, dict[str, int]]  # human_code -> {"supported": n, "unsupported": n}


def _human_bool(code: str) -> bool:
    return code == "supported"


def compute_agreement(rows: list[JoinedRow]) -> AgreementReport:
    """Compute everything `--agreement` reports from the joined rows.

    Alpha and raw agreement use only `supported`/`unsupported` rows; `not_a_claim` is
    reported as its own rate.
    """
    n_joined = len(rows)
    not_a_claim_flags = [r.human_code == "not_a_claim" for r in rows]
    not_a_claim_rate, not_a_claim_undefined = _weighted_rate(not_a_claim_flags)
    not_a_claim_rate_rw, not_a_claim_rw_undefined = _weighted_rate(
        not_a_claim_flags, [1.0 / r.sampling_probability for r in rows]
    )

    confusion: dict[str, dict[str, int]] = {
        code: {"supported": 0, "unsupported": 0} for code in _VALID_HUMAN_CODES
    }
    for r in rows:
        confusion[r.human_code]["supported" if r.judge_verdict else "unsupported"] += 1

    binary = [r for r in rows if r.human_code != "not_a_claim"]

    by_stratum: dict[str, dict] = {}
    for f in JUDGED_FIELDS:
        for d in DIMENSIONS:
            for verdict in (True, False):
                key = f"{f}:{d}:{'supported' if verdict else 'unsupported'}"
                cell_rows = [r for r in binary if r.field == f and r.dimension == d and r.judge_verdict is verdict]
                pairs = [(_human_bool(r.human_code), r.judge_verdict) for r in cell_rows]
                raw, raw_undef = _raw_agreement(pairs)
                alpha, alpha_undef = krippendorff_alpha_binary(pairs)
                by_stratum[key] = {
                    "n": len(cell_rows),
                    "raw_agreement": raw, "raw_agreement_undefined": raw_undef,
                    "alpha": alpha, "alpha_undefined": alpha_undef,
                }

    pairs_all = [(_human_bool(r.human_code), r.judge_verdict) for r in binary]
    weights = [1.0 / r.sampling_probability for r in binary]
    raw_overall, raw_overall_undef = _raw_agreement(pairs_all, weights=weights)
    alpha_overall, alpha_overall_undef = krippendorff_alpha_binary(pairs_all, weights=weights)

    def _pooled_cell(subset: list[JoinedRow]) -> dict:
        """One `by_field`/`by_dimension` cell, reweighted and unweighted."""
        pairs = [(_human_bool(r.human_code), r.judge_verdict) for r in subset]
        raw, undef = _raw_agreement(pairs)
        raw_rw, rw_undef = _raw_agreement(pairs, [1.0 / r.sampling_probability for r in subset])
        return {
            "n": len(pairs),
            "raw_agreement": raw, "raw_agreement_undefined": undef,
            "raw_agreement_reweighted": raw_rw, "raw_agreement_reweighted_undefined": rw_undef,
        }

    by_field = {f: _pooled_cell([r for r in binary if r.field == f]) for f in JUDGED_FIELDS}
    by_dimension = {d: _pooled_cell([r for r in binary if r.dimension == d]) for d in DIMENSIONS}

    return AgreementReport(
        n_joined=n_joined,
        not_a_claim_rate=not_a_claim_rate, not_a_claim_rate_undefined=not_a_claim_undefined,
        not_a_claim_rate_reweighted=not_a_claim_rate_rw,
        not_a_claim_rate_reweighted_undefined=not_a_claim_rw_undefined,
        by_stratum=by_stratum,
        raw_agreement_overall_reweighted=raw_overall, raw_agreement_overall_reweighted_undefined=raw_overall_undef,
        alpha_overall_reweighted=alpha_overall, alpha_overall_reweighted_undefined=alpha_overall_undef,
        by_field=by_field, by_dimension=by_dimension, confusion=confusion,
    )


#: Gap between raw agreement and alpha above which a note is printed.
DIVERGENCE_NOTE_THRESHOLD = 0.10

#: The reweighted alpha is our own extension, not a published estimator, so it is
#: always shown with this caveat.
ALPHA_REWEIGHTED_CAVEAT = (
    "NOT a published estimator -- this module's own probability-reweighted "
    "extension of Krippendorff's formula (see krippendorff_alpha_binary), not "
    "independently validated. The per-stratum alpha above is the published, "
    "pinned figure; treat this one as illustrative only."
)

#: Strata include the judge verdict, so the judge is constant within each one and
#: alpha there can only be undefined or <= 0. That reflects the design, not the judge.
PER_STRATUM_ALPHA_CAVEAT = (
    "per-stratum alpha: the judge verdict is CONSTANT within each stratum by "
    "construction (strata are field x dimension x verdict), so alpha there can "
    "only be undefined or <= 0 -- a negative value here means human disagreement "
    "exists, not that the judge is worse than chance."
)


#: Printed next to every unweighted pooled figure so it is not read as a corpus figure.
UNWEIGHTED_SAMPLE_CAVEAT = (
    "each bracketed unweighted figure is a proportion over the SPOT-CHECK SAMPLE only. "
    "The design draws 25 claims per field x dimension x verdict cell, which massively "
    "oversamples `unsupported` relative to the corpus, so an unweighted pooled figure "
    "is NOT corpus-representative -- the reweighted figure beside it is. They are "
    "printed together so the sample itself stays visible, not as two rival estimates."
)


def agreement_report_to_dict(report: AgreementReport, *, n_uncoded: int, unjoinable: list[dict]) -> dict:
    """JSON view of an `AgreementReport`. Undefined rates are `null`, `unjoinable` is kept in
    full, and the reweighted alpha has its caveat in a field next to it."""
    return {
        "n_joined": report.n_joined, "n_uncoded": n_uncoded,
        "n_unjoinable": len(unjoinable), "unjoinable": unjoinable,
        "not_a_claim_rate": report.not_a_claim_rate,
        "not_a_claim_rate_undefined": report.not_a_claim_rate_undefined,
        "not_a_claim_rate_reweighted": report.not_a_claim_rate_reweighted,
        "not_a_claim_rate_reweighted_undefined": report.not_a_claim_rate_reweighted_undefined,
        "unweighted_sample_caveat": UNWEIGHTED_SAMPLE_CAVEAT,
        "by_stratum": report.by_stratum,
        "by_stratum_alpha_caveat": PER_STRATUM_ALPHA_CAVEAT,
        "raw_agreement_overall_reweighted": report.raw_agreement_overall_reweighted,
        "raw_agreement_overall_reweighted_undefined": report.raw_agreement_overall_reweighted_undefined,
        "alpha_overall_reweighted": report.alpha_overall_reweighted,
        "alpha_overall_reweighted_undefined": report.alpha_overall_reweighted_undefined,
        "alpha_overall_reweighted_is_published_estimator": False,
        "alpha_overall_reweighted_caveat": ALPHA_REWEIGHTED_CAVEAT,
        "by_field": report.by_field, "by_dimension": report.by_dimension, "confusion": report.confusion,
    }


def _fmtu(value: float | None, undefined: bool, decimals: int = 3) -> str:
    return "undefined" if undefined or value is None else f"{value:.{decimals}f}"


def format_agreement_report(report: AgreementReport, *, n_uncoded: int, unjoinable: list[dict]) -> str:
    """Readable agreement report: per-stratum figures first, then the reweighted overall.

    Raw agreement and alpha are always shown together, with a note when they diverge
    (the kappa paradox under skewed prevalence, Feinstein and Cicchetti 1990).
    """
    lines = ["Judge-human agreement"]
    lines.append(f"  n joined: {report.n_joined}  n uncoded: {n_uncoded}  n unjoinable: {len(unjoinable)}")
    lines.append(
        f"  not_a_claim rate: "
        f"reweighted={_fmtu(report.not_a_claim_rate_reweighted, report.not_a_claim_rate_reweighted_undefined)} "
        f"(sample-only unweighted={_fmtu(report.not_a_claim_rate, report.not_a_claim_rate_undefined)})"
    )

    lines.append("")
    lines.append("  PRIMARY -- by stratum (field:dimension:verdict), published/pinned alpha:")
    for key, cell in sorted(report.by_stratum.items()):
        lines.append(
            f"    {key}: n={cell['n']} raw={_fmtu(cell['raw_agreement'], cell['raw_agreement_undefined'])} "
            f"alpha={_fmtu(cell['alpha'], cell['alpha_undefined'])}"
        )
    lines.append(f"  NOTE: {PER_STRATUM_ALPHA_CAVEAT}")

    lines.append("")
    lines.append("  SECONDARY -- reweighted overall (probability-weighted, corrects for oversampling):")
    lines.append(
        f"    raw agreement: "
        f"{_fmtu(report.raw_agreement_overall_reweighted, report.raw_agreement_overall_reweighted_undefined)}"
    )
    lines.append(
        f"    Krippendorff's alpha: "
        f"{_fmtu(report.alpha_overall_reweighted, report.alpha_overall_reweighted_undefined)}"
    )
    lines.append(f"    NOTE: {ALPHA_REWEIGHTED_CAVEAT}")
    raw_v, raw_u = report.raw_agreement_overall_reweighted, report.raw_agreement_overall_reweighted_undefined
    alpha_v, alpha_u = report.alpha_overall_reweighted, report.alpha_overall_reweighted_undefined
    if not raw_u and not alpha_u and abs(raw_v - alpha_v) > DIVERGENCE_NOTE_THRESHOLD:
        lines.append(
            f"    NOTE: raw agreement and alpha diverge by more than {DIVERGENCE_NOTE_THRESHOLD:.2f} -- "
            "skewed prevalence deflates chance-corrected agreement while leaving raw agreement "
            "high (the kappa paradox, Feinstein & Cicchetti 1990); report both, never one alone."
        )

    def _pooled_line(label: str, cell: dict) -> str:
        return (
            f"    {label}: n={cell['n']} "
            f"reweighted={_fmtu(cell['raw_agreement_reweighted'], cell['raw_agreement_reweighted_undefined'])} "
            f"(sample-only unweighted={_fmtu(cell['raw_agreement'], cell['raw_agreement_undefined'])})"
        )

    lines.append("")
    lines.append(f"  NOTE: {UNWEIGHTED_SAMPLE_CAVEAT}")
    lines.append("  by field (raw agreement, pooled across strata):")
    for f, cell in sorted(report.by_field.items()):
        lines.append(_pooled_line(f, cell))
    lines.append("  by dimension (raw agreement, pooled across strata):")
    for d, cell in sorted(report.by_dimension.items()):
        lines.append(_pooled_line(d, cell))
    if unjoinable:
        lines.append("  unjoinable item_ids (reported, not dropped):")
        for u in unjoinable:
            lines.append(f"    {u['item_id']}: {u['reason']}")
    return "\n".join(lines)


def claim_record_key(c: JudgedClaim) -> tuple[str, str, int, str, str, str]:
    """The `record_key` of the record a claim came from."""
    return (c.run_config_hash, c.funder_slug, c.replicate, c.opportunity_id, c.field, c.dimension)


def spotcheck_record_keys(
    records: list[JudgedRecord], *, cell_target: int = SPOTCHECK_CELL_TARGET, rng: random.Random | None = None,
) -> set[tuple[str, str, int, str, str, str]]:
    """Record keys the default spot-check export would select, regenerated from the seed.

    Only correct if the export used the default `--export-spotcheck 200` over the same
    results. Prefer `spotcheck_record_keys_from_csv` when the exported file is available.
    """
    claims = explode_claims(records)
    selection = select_spotcheck_sample(claims, cell_target=cell_target, rng=rng or random.Random(SPOTCHECK_SEED))
    return {claim_record_key(c) for c in selection.claims}


def spotcheck_record_keys_from_csv(path: Path) -> tuple[set[tuple[str, str, int, str, str, str]], list[dict]]:
    """Record keys read from the exported spot-check CSV.

    Safer than `spotcheck_record_keys`, since the exclusion set is exactly what was
    exported. Returns `(keys, malformed)`; rows that give no key are listed in
    `malformed` so they are not silently let back into the draw.
    """
    keys: set[tuple[str, str, int, str, str, str]] = set()
    malformed: list[dict] = []
    with open(Path(path), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                keys.add((
                    row["run_config_hash"], row["funder_slug"], int(row["replicate"]),
                    row["opportunity_id"], row["field"], row["dimension"],
                ))
            except (KeyError, TypeError, ValueError) as exc:
                malformed.append({"item_id": row.get("item_id", ""), "reason": f"malformed row: {exc}"})
    return keys, malformed


def select_test_retest_sample(
    records: list[JudgedRecord], *, exclude_keys: set, n: int = TEST_RETEST_TARGET,
    rng: random.Random | None = None,
) -> list[JudgedRecord]:
    """Draw up to `n` records not in `exclude_keys` (the spot-check records).

    Works per record, not per claim, because two separate claim decompositions cannot
    be lined up without a human. Uses its own fixed seed, `TEST_RETEST_SEED`.
    """
    rng = rng or random.Random(TEST_RETEST_SEED)
    pool = [r for r in records if record_key(r) not in exclude_keys]
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


@dataclass(frozen=True)
class TestRetestResult:
    n_compared: int
    n_agree: int
    n_fresh_failed: int
    n_skipped_spend_cap: int
    total_spend_usd: float
    cap_hit: bool
    self_agreement: float | None
    self_agreement_undefined: bool


def run_test_retest(
    records: list[JudgedRecord], *, backend: JudgeBackend, judge_config: JudgeConfig,
    max_spend_usd: float = 25.0,
) -> TestRetestResult:
    """Re-judge each record and compare the new `record_supported` with the saved one.

    Calls `run_judge` directly, which skips the cache. Failed fresh calls are counted
    in `n_fresh_failed` rather than treated as disagreement. Stops once the backend's
    total spend reaches `max_spend_usd` and counts the rest in `n_skipped_spend_cap`.
    """
    if not hasattr(backend, "total_cost_usd"):
        raise AttributeError(
            f"{backend!r} has no `total_cost_usd` attribute, so the spend cap cannot "
            "be enforced against it (see JudgeBackend.total_cost_usd). Refusing to "
            "run any test-retest calls rather than running uncapped."
        )
    n_agree = 0
    n_compared = 0
    n_fresh_failed = 0
    n_skipped_spend_cap = 0
    cap_hit = False
    for rec in records:
        if backend.total_cost_usd >= max_spend_usd:
            cap_hit = True
            n_skipped_spend_cap += 1
            continue

        if rec.dimension == "completeness":
            prompt = build_completeness_prompt(
                field=rec.field, extracted_text=rec.extracted_text,
                gold_text=rec.comparison_text, snapshot_date=rec.snapshot_date,
            )
        else:
            prompt = build_groundedness_prompt(
                field=rec.field, extracted_text=rec.extracted_text,
                page_text=rec.comparison_text, snapshot_date=rec.snapshot_date,
            )
        result = run_judge(
            backend=backend, judge_config=judge_config, prompt=prompt,
            field=rec.field, dimension=rec.dimension,
        )
        if result.outcome is not JudgeOutcome.OK:
            n_fresh_failed += 1
            continue
        n_compared += 1
        if result.verdict.record_supported == rec.record_supported:
            n_agree += 1

    undefined = n_compared == 0
    return TestRetestResult(
        n_compared=n_compared, n_agree=n_agree, n_fresh_failed=n_fresh_failed,
        n_skipped_spend_cap=n_skipped_spend_cap, total_spend_usd=backend.total_cost_usd, cap_hit=cap_hit,
        self_agreement=None if undefined else n_agree / n_compared,
        self_agreement_undefined=undefined,
    )


def cli_export_spotcheck(
    paths, *, ground_truth, registry, snapshot_root: Path, results_root: Path,
    judge_config_hash: str, n_total: int, out_path: Path | None, json_output: bool,
    rng: random.Random | None = None,
) -> int:
    if out_path is None:
        print("--export-spotcheck requires --out FILE", file=sys.stderr)
        return 1

    loaded = load_judged_records(
        paths, ground_truth=ground_truth, registry=registry, snapshot_root=snapshot_root,
        results_root=results_root, judge_config_hash=judge_config_hash,
    )
    claims = explode_claims(loaded.records)
    cell_target = max(0, n_total // len(ALL_CELLS))
    selection = select_spotcheck_sample(claims, cell_target=cell_target, rng=rng or random.Random(SPOTCHECK_SEED))
    write_spotcheck_csv(selection, out_path)

    achieved = {f"{f}:{d}:{'supported' if v else 'unsupported'}": n for (f, d, v), n in selection.achieved.items()}
    population = {f"{f}:{d}:{'supported' if v else 'unsupported'}": n for (f, d, v), n in selection.population.items()}
    funders = sorted({c.funder for c in selection.claims})
    report = {
        "n_selected": len(selection.claims), "n_target": n_total, "cell_target": cell_target,
        "achieved_allocation": achieved, "population": population,
        "n_distinct_funders": len(funders), "funders": funders,
        "n_skipped_unjudged_runs": len(loaded.skipped_unjudged),
        "skipped_unjudged_runs": [str(p) for p in loaded.skipped_unjudged],
        "n_skipped_unlabelled_funder_runs": len(loaded.skipped_unlabelled_funder),
        "skipped_unlabelled_funder_runs": [str(p) for p in loaded.skipped_unlabelled_funder],
        "n_mismatched_funder": len(loaded.mismatched_funder),
        "mismatched_funder": loaded.mismatched_funder,
        "n_excluded_records": len(loaded.excluded_records),
        #: Listed in full: `no_stored_record` entries appear nowhere else.
        "excluded_records": loaded.excluded_records,
        "out_path": str(out_path),
    }
    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"wrote {len(selection.claims)} claim(s) to {out_path}")
        print(f"achieved allocation: {achieved}")
        print(f"funders in sample: {len(funders)}")
        print(
            f"excluded/skipped: {len(loaded.excluded_records)} record(s) not stored ok, "
            f"{len(loaded.skipped_unjudged)} run(s) not yet judged, "
            f"{len(loaded.skipped_unlabelled_funder)} run(s) unlabelled/unregistered funder"
        )
        if loaded.mismatched_funder:
            print(
                f"WARNING: {len(loaded.mismatched_funder)} stored verdict(s) disagreed with the "
                "result they were loaded for on funder/replicate -- see mismatched_funder in --json",
                file=sys.stderr,
            )
        if loaded.skipped_unjudged:
            print(
                f"{len(loaded.skipped_unjudged)} run(s) skipped: no stored verdict for this judge "
                "config -- run `judge` on them first", file=sys.stderr,
            )
    return 0


def cli_agreement(coded_csv_path: Path, *, results_root: Path, json_output: bool) -> int:
    rows = load_coded_csv(coded_csv_path)
    joined, unjoinable, n_uncoded = join_coded_rows(rows, results_root=results_root)
    report = compute_agreement(joined)
    if json_output:
        print(json.dumps(agreement_report_to_dict(report, n_uncoded=n_uncoded, unjoinable=unjoinable), indent=2))
    else:
        print(format_agreement_report(report, n_uncoded=n_uncoded, unjoinable=unjoinable))
    return 0


def cli_test_retest(
    paths, *, ground_truth, registry, snapshot_root: Path, results_root: Path,
    judge_config_hash: str, backend: JudgeBackend, judge_config: JudgeConfig,
    n: int, max_spend_usd: float, json_output: bool, spotcheck_csv: Path | None = None,
) -> int:
    """Run the test-retest check on records outside the spot-check sample.

    Makes a live, paid call per record, capped by `max_spend_usd`. Pass `spotcheck_csv`
    to exclude exactly what was exported; otherwise the default selection is
    regenerated, with a warning. The output records which source was used.
    """
    loaded = load_judged_records(
        paths, ground_truth=ground_truth, registry=registry, snapshot_root=snapshot_root,
        results_root=results_root, judge_config_hash=judge_config_hash,
    )
    malformed_spotcheck_rows: list[dict] = []
    if spotcheck_csv is None:
        exclusion_source = "regenerated_default_selection"
        exclude_keys = spotcheck_record_keys(loaded.records)
        print(
            "WARNING: no --spotcheck-csv given, so the excluded set is the DEFAULT "
            f"spot-check selection regenerated at cell_target={SPOTCHECK_CELL_TARGET} "
            "over these paths. That is the right set ONLY if the export ran at "
            "--export-spotcheck 200 over this same result set with the corpus "
            "unchanged since; otherwise the two samples may overlap. Pass "
            "--spotcheck-csv <the exported file> to exclude what was really exported.",
            file=sys.stderr,
        )
    else:
        exclusion_source = "exported_csv"
        exclude_keys, malformed_spotcheck_rows = spotcheck_record_keys_from_csv(spotcheck_csv)
        if malformed_spotcheck_rows:
            print(
                f"WARNING: {len(malformed_spotcheck_rows)} row(s) in {spotcheck_csv} yielded no "
                "record key and are NOT excluded -- see malformed_spotcheck_rows in --json",
                file=sys.stderr,
            )
    sample = select_test_retest_sample(loaded.records, exclude_keys=exclude_keys, n=n)
    result = run_test_retest(sample, backend=backend, judge_config=judge_config, max_spend_usd=max_spend_usd)

    report = {
        "n_candidate_records": len(loaded.records), "n_excluded_as_spotcheck": len(exclude_keys),
        "exclusion_source": exclusion_source,
        "spotcheck_csv": None if spotcheck_csv is None else str(spotcheck_csv),
        #: Only matters for `regenerated_default_selection`, but always recorded.
        "spotcheck_cell_target": SPOTCHECK_CELL_TARGET,
        "n_malformed_spotcheck_rows": len(malformed_spotcheck_rows),
        "malformed_spotcheck_rows": malformed_spotcheck_rows,
        "n_drawn": len(sample), "n_compared": result.n_compared, "n_agree": result.n_agree,
        "n_fresh_failed": result.n_fresh_failed, "n_skipped_spend_cap": result.n_skipped_spend_cap,
        "cap_hit": result.cap_hit, "total_spend_usd": result.total_spend_usd,
        "self_agreement": result.self_agreement, "self_agreement_undefined": result.self_agreement_undefined,
    }
    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"test-retest: {result.n_compared} record(s) compared, {result.n_fresh_failed} fresh call(s) failed")
        print(
            f"excluded as spot-check: {len(exclude_keys)} record(s) "
            f"(source: {exclusion_source}"
            + (f", {spotcheck_csv}" if spotcheck_csv is not None else f", cell_target={SPOTCHECK_CELL_TARGET}")
            + ")"
        )
        print(
            f"self-agreement: {'undefined' if result.self_agreement_undefined else f'{result.self_agreement:.3f}'} "
            f"(bounds how much of any judge-human disagreement is judge noise, not genuine disagreement)"
        )
        print(f"total spend: ${result.total_spend_usd:.4f}")
        if result.cap_hit:
            print(f"spend cap ${max_spend_usd:.2f} reached: {result.n_skipped_spend_cap} record(s) not re-judged", file=sys.stderr)
    return 0
