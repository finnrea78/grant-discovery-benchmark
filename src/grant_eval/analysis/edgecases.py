"""Whether edge-case tags cost recall, as a paired within-funder contrast.

For each tag, a funder's mean recall on tagged rows is compared with its recall on rows
with no tag at all. Rows with other tags are not used as the baseline, since tags often
co-occur. Pairing within funder matters because funders differ a lot in difficulty.

A contrast is only inferential if the tag's stratum policy is inferential and enough
funders have both tagged and untagged rows (`strata.MIN_CLUSTERS`). Tags that cannot be
assessed are still reported, with `None` statistics and a reason.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import dataset as ds
from . import stats as st
from . import strata as strat
from grant_eval.scoring.groundtruth import GroundTruth

__all__ = [
    "TagContrast",
    "tag_contrast",
    "tag_contrast_family",
    "TagCountRecall",
    "recall_by_tag_count",
    "tag_cooccurrence",
]


def _recall(run: ds.RunRecord, tags: dict[str, frozenset[str]], predicate) -> tuple[int, int]:
    """`(tp, n)` over the in-scope opportunities of `run` whose tag set matches `predicate`.

    Every in-scope row is either a TP or an FN, so `tp + fn` is the denominator.
    """
    tp = sum(1 for oid in run.true_positive_ids if predicate(tags[oid]))
    fn = sum(1 for oid in run.false_negative_ids if predicate(tags[oid]))
    return tp, tp + fn


def _mean_by_funder(runs, tags, predicate) -> dict[str, float]:
    """`{funder: mean recall}`, skipping runs with no matching rows."""
    per_run: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        tp, n = _recall(r, tags, predicate)
        if n:
            per_run[r.funder].append(tp / n)
    return {f: sum(v) / len(v) for f, v in per_run.items()}


@dataclass(frozen=True)
class TagContrast:
    """One edge-case tag's recall against the untagged baseline, paired within funder.

    `difference` is a bootstrap CI on `mean_tagged - mean_untagged`, or `None` if no
    funder could be paired. A negative value means the tag costs recall.
    `n_tag_clusters` counts funders with any tagged row; `n_paired_clusters` counts
    funders with both tagged and untagged rows.
    """

    tag: str
    n_rows: int
    n_rows_alone: int
    n_tag_clusters: int
    n_paired_clusters: int
    mean_tagged: float | None
    mean_untagged: float | None
    difference: st.BootstrapCI | None
    inferential: bool
    reason: str
    excluded_funders: tuple[tuple[str, str], ...] = ()


def tag_contrast(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    tag: str,
    runs: tuple[ds.RunRecord, ...] | None = None,
    alone: bool = False,
    n_resamples: int = 10_000,
    seed: int = st.DEFAULT_SEED,
) -> TagContrast:
    """Recall on `tag`'s rows versus recall on untagged rows, paired within funder.

    `alone=True` only counts rows carrying this tag and no other, which removes the
    overlap between tags but leaves far fewer rows.
    """
    pool = data.runs if runs is None else runs
    funders = data.funders()
    tags = strat.opportunity_tags(gt, funders)
    policies = strat.stratum_policies(gt, funders)
    policy = policies.get(tag)

    target = frozenset({tag})
    tagged_pred = (lambda ts: ts == target) if alone else (lambda ts: tag in ts)
    untagged_pred = lambda ts: not ts  # noqa: E731

    n_rows = sum(1 for ts in tags.values() if tag in ts)
    n_rows_alone = sum(1 for ts in tags.values() if ts == target)

    tagged = _mean_by_funder(pool, tags, tagged_pred)
    untagged = _mean_by_funder(pool, tags, untagged_pred)
    paired = sorted(set(tagged) & set(untagged))

    excluded = []
    for f in sorted(set(tagged) | set(untagged)):
        if f not in tagged:
            excluded.append((f, "no in-scope row carries this tag"))
        elif f not in untagged:
            excluded.append((f, "every in-scope row carries some tag; no baseline"))

    n_tag_clusters = policy.n_clusters if policy else 0
    inferential = bool(policy and policy.inferential) and len(paired) >= strat.MIN_CLUSTERS

    if not paired:
        reason = (
            f"{n_rows} in-scope row(s) carry this tag, but no funder has both tagged "
            "and untagged rows: the paired contrast cannot be formed"
        )
        return TagContrast(
            tag=tag, n_rows=n_rows, n_rows_alone=n_rows_alone,
            n_tag_clusters=n_tag_clusters, n_paired_clusters=0,
            mean_tagged=None, mean_untagged=None, difference=None,
            inferential=False, reason=reason, excluded_funders=tuple(excluded),
        )

    a = [tagged[f] for f in paired]
    b = [untagged[f] for f in paired]
    difference = st.bootstrap_ci(
        [x - y for x, y in zip(a, b)], n_resamples=n_resamples, seed=seed
    )

    if inferential:
        reason = ""
    else:
        failures = []
        if not (policy and policy.inferential):
            failures.append(f"D-C policy: {policy.reason if policy else 'no tune rows'}")
        if len(paired) < strat.MIN_CLUSTERS:
            failures.append(
                f"{len(paired)} paired funder-cluster(s) (< {strat.MIN_CLUSTERS})"
            )
        reason = "; ".join(failures)

    return TagContrast(
        tag=tag,
        n_rows=n_rows,
        n_rows_alone=n_rows_alone,
        n_tag_clusters=n_tag_clusters,
        n_paired_clusters=len(paired),
        mean_tagged=sum(a) / len(a),
        mean_untagged=sum(b) / len(b),
        difference=difference,
        inferential=inferential,
        reason=reason,
        excluded_funders=tuple(excluded),
    )


def tag_contrast_family(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    runs: tuple[ds.RunRecord, ...] | None = None,
    alone: bool = False,
    n_resamples: int = 10_000,
    seed: int = st.DEFAULT_SEED,
) -> tuple[TagContrast, ...]:
    """One `TagContrast` per tag with at least one in-scope row, commonest first."""
    policies = strat.stratum_policies(gt, data.funders())
    ordered = sorted(policies, key=lambda t: (-policies[t].n_rows, t))
    return tuple(
        tag_contrast(data, gt, tag, runs=runs, alone=alone,
                     n_resamples=n_resamples, seed=seed)
        for tag in ordered
    )


@dataclass(frozen=True)
class TagCountRecall:
    """Mean recall over rows carrying exactly `n_tags` edge-case tags.

    Checks whether edge cases compound. `ci` is a bootstrap CI over funders, or `None`
    if no funder has a row with that many tags.
    """

    n_tags: int
    n_rows: int
    n_clusters: int
    mean_recall: float | None
    ci: st.BootstrapCI | None


def recall_by_tag_count(
    data: ds.AnalysisDataset,
    gt: GroundTruth,
    runs: tuple[ds.RunRecord, ...] | None = None,
    n_resamples: int = 10_000,
    seed: int = st.DEFAULT_SEED,
) -> tuple[TagCountRecall, ...]:
    """Mean recall grouped by how many edge-case tags a row carries.

    Descriptive only: the grouping was chosen after looking at the data.
    """
    pool = data.runs if runs is None else runs
    tags = strat.opportunity_tags(gt, data.funders())
    bands = sorted({len(ts) for ts in tags.values()})

    out = []
    for k in bands:
        pred = (lambda ts, k=k: len(ts) == k)
        by_funder = _mean_by_funder(pool, tags, pred)
        n_rows = sum(1 for ts in tags.values() if len(ts) == k)
        values = list(by_funder.values())
        ci = (
            st.bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
            if values
            else None
        )
        out.append(
            TagCountRecall(
                n_tags=k,
                n_rows=n_rows,
                n_clusters=len(values),
                mean_recall=(sum(values) / len(values)) if values else None,
                ci=ci,
            )
        )
    return tuple(out)


def tag_cooccurrence(gt: GroundTruth, funders: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """`{tag_a: {tag_b: rows carrying both}}`; the diagonal is each tag's row count.

    Shows how much the per-tag contrasts overlap.
    """
    tags = strat.opportunity_tags(gt, funders)
    all_tags = sorted({t for ts in tags.values() for t in ts})
    matrix: dict[str, dict[str, int]] = {a: {b: 0 for b in all_tags} for a in all_tags}
    for ts in tags.values():
        for a in ts:
            for b in ts:
                matrix[a][b] += 1
    return matrix
