"""Recall per edge-case stratum, recomputed from opportunity IDs.

`score.py` only splits by one tag, but each run stores its true-positive and
false-negative IDs, so joining them to the ground-truth tags gives recall for any tag.
Precision is not computed because unattributable false positives have no stratum.
"""
from __future__ import annotations

from dataclasses import dataclass

from grant_eval.analysis.dataset import AnalysisDataset, RunRecord
from grant_eval.scoring.groundtruth import GroundTruth

__all__ = [
    "MIN_ROWS",
    "MIN_CLUSTERS",
    "StratumPolicy",
    "stratum_policies",
    "opportunity_tags",
    "stratum_recall",
    "stratum_recall_by_funder",
]

#: A stratum needs this many rows and funders to be tested; fewer is descriptive only.
#: Fixed before the counts were looked at.
MIN_ROWS = 20
MIN_CLUSTERS = 8


@dataclass(frozen=True)
class StratumPolicy:
    """Whether one edge-case tag has enough tune data for a statistical test.

    `n_clusters` is the number of distinct funders with a row carrying the tag.
    `reason` says which threshold failed, or is empty.
    """

    tag: str
    n_rows: int
    n_clusters: int
    inferential: bool
    reason: str


def opportunity_tags(gt: GroundTruth, funders: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """`{opportunity_id: frozenset(tag values)}` over the in-scope rows of `funders`.

    Distractors are skipped because they never appear as true positives or false
    negatives.
    """
    tags: dict[str, frozenset[str]] = {}
    for funder in funders:
        funder_gt = gt.for_funder(funder)
        for row in funder_gt.in_scope:
            tags[row.opportunity_id] = frozenset(t.value for t in row.edge_case_tags)
    return tags


def stratum_policies(
    gt: GroundTruth, funders: tuple[str, ...]
) -> dict[str, StratumPolicy]:
    """One `StratumPolicy` per edge-case tag with at least one in-scope row."""
    rows_by_tag: dict[str, int] = {}
    funders_by_tag: dict[str, set[str]] = {}
    for funder in funders:
        funder_gt = gt.for_funder(funder)
        for row in funder_gt.in_scope:
            for t in row.edge_case_tags:
                tag = t.value
                rows_by_tag[tag] = rows_by_tag.get(tag, 0) + 1
                funders_by_tag.setdefault(tag, set()).add(funder)

    policies: dict[str, StratumPolicy] = {}
    for tag, n_rows in rows_by_tag.items():
        n_clusters = len(funders_by_tag[tag])
        rows_ok = n_rows >= MIN_ROWS
        clusters_ok = n_clusters >= MIN_CLUSTERS
        inferential = rows_ok and clusters_ok
        if inferential:
            reason = ""
        else:
            failures = []
            if not rows_ok:
                failures.append(f"{n_rows} rows (< {MIN_ROWS})")
            if not clusters_ok:
                failures.append(f"{n_clusters} funder-clusters (< {MIN_CLUSTERS})")
            reason = " and ".join(failures)
        policies[tag] = StratumPolicy(
            tag=tag,
            n_rows=n_rows,
            n_clusters=n_clusters,
            inferential=inferential,
            reason=reason,
        )
    return policies


def stratum_recall(
    run: RunRecord, tags: dict[str, frozenset[str]], tag: str
) -> tuple[int, int]:
    """`(tp_in_stratum, n_in_stratum)` for one run, joined on opportunity ID.

    Every in-scope opportunity is a TP or an FN, so `tp + fn` is the full denominator.
    """
    tp = sum(1 for oid in run.true_positive_ids if tag in tags[oid])
    fn = sum(1 for oid in run.false_negative_ids if tag in tags[oid])
    return tp, tp + fn


def stratum_recall_by_funder(
    data: AnalysisDataset,
    gt: GroundTruth,
    tag: str,
    runs: tuple[RunRecord, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """`{config_label: {funder: mean recall over replicates}}` for `tag`.

    Funders with no rows in the stratum are left out rather than scored 0.
    """
    if runs is None:
        runs = data.runs
    tags = opportunity_tags(gt, data.funders())

    grouped: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        tp, n = stratum_recall(run, tags, tag)
        if n == 0:
            continue
        grouped.setdefault(run.config_label, {}).setdefault(run.funder, []).append(tp / n)

    return {
        config: {funder: sum(vals) / len(vals) for funder, vals in by_funder.items()}
        for config, by_funder in grouped.items()
    }
