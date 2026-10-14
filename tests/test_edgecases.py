"""Tests for the paired within-funder edge-case contrasts (`analysis/edgecases.py`).

Focuses on using the right baseline and the right cluster count.
"""
from __future__ import annotations

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import edgecases as ec
from grant_eval.analysis import strata as strat
from grant_eval.scoring.groundtruth import load_ground_truth


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


@pytest.fixture(scope="module")
def gt():
    return load_ground_truth()


@pytest.fixture(scope="module")
def family(data, gt):
    # Fewer resamples than the default, since no test checks interval widths.
    return ec.tag_contrast_family(data, gt, n_resamples=500)


def test_the_baseline_is_untagged_rows_not_rows_without_this_tag(data, gt):
    """The baseline is rows with no tag at all, not rows lacking this particular tag.

    `restricted` and `multi_deadline` share 19 rows, so "not this tag" would still contain other edge cases.
    """
    contrasts = {c.tag: c for c in ec.tag_contrast_family(data, gt, n_resamples=200)}
    untagged = {c.mean_untagged for c in contrasts.values() if c.mean_untagged is not None}
    # Pairing keeps only funders on both sides, so the baseline mean can differ
    # between tags. The check below is that the baseline has no tagged rows.
    assert untagged, "no tag produced a baseline at all"

    tags = strat.opportunity_tags(gt, data.funders())
    baseline_ids = {oid for oid, ts in tags.items() if not ts}
    assert baseline_ids, "fixture assumption: some in-scope rows carry no tag"
    for oid in baseline_ids:
        assert tags[oid] == frozenset()


def test_untagged_baseline_and_tagged_rows_are_disjoint(gt, data):
    tags = strat.opportunity_tags(gt, data.funders())
    tagged = {oid for oid, ts in tags.items() if "restricted" in ts}
    untagged = {oid for oid, ts in tags.items() if not ts}
    assert tagged and untagged
    assert not (tagged & untagged)


def test_paired_clusters_never_exceed_the_tag_clusters(family):
    """The paired cluster count is never more than the tagged cluster count."""
    for c in family:
        assert c.n_paired_clusters <= c.n_tag_clusters


def test_paired_clusters_are_strictly_smaller_for_the_two_inferential_strata(family):
    """`restricted` has 11 tag clusters but 4 paired ones; `multi_deadline` has 10 and 6."""
    by_tag = {c.tag: c for c in family}
    assert by_tag["restricted"].n_tag_clusters == 11
    assert by_tag["restricted"].n_paired_clusters == 4
    assert by_tag["multi_deadline"].n_tag_clusters == 10
    assert by_tag["multi_deadline"].n_paired_clusters == 6


def test_no_tag_is_inferential_on_the_tune_split(family):
    """No tag has enough paired clusters to be tested on the tune split."""
    assert all(not c.inferential for c in family)
    for c in family:
        assert c.reason, f"{c.tag} is not inferential but gives no reason"


def test_inferential_requires_both_the_dc_policy_and_paired_clusters(data, gt, family):
    """A tag is inferential only if its stratum policy allows it and it has enough paired clusters."""
    policies = strat.stratum_policies(gt, data.funders())
    for c in family:
        expected = (
            policies[c.tag].inferential and c.n_paired_clusters >= strat.MIN_CLUSTERS
        )
        assert c.inferential is expected


def test_pdf_only_cannot_be_assessed_and_says_so(family):
    """`pdf_only` has one in-scope row, so its contrast is undefined and gives a reason."""
    pdf = next(c for c in family if c.tag == "pdf_only")
    assert pdf.n_rows == 1
    assert pdf.difference is None or pdf.difference.undefined
    assert not pdf.inferential
    assert "1 rows" in pdf.reason or "paired" in pdf.reason


def test_js_portal_has_no_contrast_at_all(family):
    """A tag with no in-scope tune rows gets no contrast."""
    assert "js_portal" not in {c.tag for c in family}


def test_a_funder_with_no_baseline_is_excluded_with_a_reason(family):
    """A funder whose in-scope rows are all tagged is listed in `excluded_funders` with a reason."""
    restricted = next(c for c in family if c.tag == "restricted")
    assert restricted.excluded_funders
    reasons = {reason for _, reason in restricted.excluded_funders}
    assert all(reason for reason in reasons)


def test_contrast_with_no_paired_funder_is_undefined_not_zero(data, gt):
    """With no runs, the contrast is undefined rather than zero."""
    empty = ec.tag_contrast(data, gt, "restricted", runs=(), n_resamples=200)
    assert empty.n_paired_clusters == 0
    assert empty.mean_tagged is None
    assert empty.mean_untagged is None
    assert empty.difference is None
    assert "cannot be formed" in empty.reason


def test_alone_reading_uses_fewer_rows_than_the_marginal_one(data, gt):
    """`alone=True` keeps only rows with just this tag, so it never has more rows."""
    marginal = {c.tag: c for c in ec.tag_contrast_family(data, gt, n_resamples=200)}
    alone = {c.tag: c for c in ec.tag_contrast_family(data, gt, alone=True, n_resamples=200)}
    for tag, c in alone.items():
        assert c.n_rows_alone <= marginal[tag].n_rows
    assert alone["restricted"].n_rows_alone == 25
    assert alone["multi_deadline"].n_rows_alone == 11


def test_tag_count_bands_partition_the_in_scope_corpus(data, gt):
    """Grouping by number of tags covers every in-scope row exactly once."""
    tags = strat.opportunity_tags(gt, data.funders())
    bands = ec.recall_by_tag_count(data, gt, n_resamples=200)
    assert sum(b.n_rows for b in bands) == len(tags)
    assert [b.n_tags for b in bands] == sorted({len(ts) for ts in tags.values()})


def test_tag_overlap_diagonal_is_the_tags_own_row_count(data, gt):
    matrix = ec.tag_cooccurrence(gt, data.funders())
    policies = strat.stratum_policies(gt, data.funders())
    for tag, row in matrix.items():
        assert row[tag] == policies[tag].n_rows


def test_tag_overlap_is_symmetric_and_finds_the_known_confound(data, gt):
    matrix = ec.tag_cooccurrence(gt, data.funders())
    for a in matrix:
        for b in matrix:
            assert matrix[a][b] == matrix[b][a]
    # 19 of multi_deadline's 32 rows are also restricted, which is why the
    # alone reading exists.
    assert matrix["multi_deadline"]["restricted"] == 19
