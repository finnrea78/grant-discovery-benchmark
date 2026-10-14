"""Tests for per-stratum recall, joining opportunity IDs to ground-truth tags."""
from __future__ import annotations

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import strata as st
from grant_eval.scoring.groundtruth import load_ground_truth

BEST = "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"


class _FakeTag:
    """Duck-types `EdgeCaseTag`: only `.value` is read by `stratum_policies`."""

    def __init__(self, value):
        self.value = value


class _FakeRow:
    """Duck-types a `GroundTruthOpportunity`: only `.edge_case_tags` is read."""

    def __init__(self, tags):
        self.edge_case_tags = [_FakeTag(t) for t in tags]


class _FakeFunderGT:
    """Duck-types `FunderGroundTruth`: only `.in_scope` is read."""

    def __init__(self, rows):
        self.in_scope = rows


class _FakeGroundTruth:
    """Duck-types `GroundTruth`: only `.for_funder` is read."""

    def __init__(self, by_funder):
        self._by_funder = by_funder

    def for_funder(self, funder):
        return self._by_funder[funder]


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


@pytest.fixture(scope="module")
def gt():
    return load_ground_truth()


def test_tune_stratum_sizes_match_the_recorded_counts(data, gt):
    """Stratum sizes on the tune split (about half the pre-registration's all-funder counts)."""
    policies = st.stratum_policies(gt, data.funders())
    assert {t: p.n_rows for t, p in policies.items()} == {
        "restricted": 50, "multi_deadline": 32, "rolling": 10,
        "ambiguous_eligibility": 5, "pdf_only": 1,
    }
    assert "js_portal" not in policies


def test_the_policy_rule_is_applied_mechanically(data, gt):
    policies = st.stratum_policies(gt, data.funders())
    assert [t for t, p in policies.items() if p.inferential] == [
        "restricted", "multi_deadline"
    ]
    assert policies["rolling"].n_clusters == 6
    assert "clusters" in policies["rolling"].reason


def test_a_stratum_meeting_only_one_threshold_stays_descriptive(data, gt):
    """A stratum is inferential only if it meets both the row and cluster thresholds."""
    policies = st.stratum_policies(gt, data.funders())
    assert st.MIN_ROWS == 20 and st.MIN_CLUSTERS == 8
    for policy in policies.values():
        assert policy.inferential == (
            policy.n_rows >= st.MIN_ROWS and policy.n_clusters >= st.MIN_CLUSTERS
        )


def test_the_and_rule_rejects_a_stratum_that_meets_only_one_threshold():
    """Strata that meet only the row threshold or only the cluster threshold stay descriptive.

    No real tune tag meets just one threshold, so both cases are built with fake data.
    """
    funders = tuple(f"funder{i}" for i in range(8))
    by_funder = {f: _FakeFunderGT([_FakeRow(["wide_thin"])]) for f in funders}
    # deep_narrow: 25 rows, all under a single funder -> rows clear, clusters do not.
    by_funder["funder0"] = _FakeFunderGT(
        by_funder["funder0"].in_scope + [_FakeRow(["deep_narrow"]) for _ in range(25)]
    )
    gt_fake = _FakeGroundTruth(by_funder)

    policies = st.stratum_policies(gt_fake, funders)

    # wide_thin: 8 rows (one per funder) over 8 funders -> clusters clear, rows do not.
    wide_thin = policies["wide_thin"]
    assert (wide_thin.n_rows, wide_thin.n_clusters) == (8, 8)
    assert not wide_thin.inferential

    deep_narrow = policies["deep_narrow"]
    assert (deep_narrow.n_rows, deep_narrow.n_clusters) == (25, 1)
    assert not deep_narrow.inferential


def test_stratum_recall_counts_only_that_stratums_rows(data, gt):
    tags = st.opportunity_tags(gt, data.funders())
    run = data.cell(BEST, "Wolfson Foundation")[0]
    tp, n = st.stratum_recall(run, tags, "multi_deadline")
    assert n == 12
    assert 0 <= tp <= n
    assert tp == sum(1 for oid in run.true_positive_ids if "multi_deadline" in tags[oid])

    # Wolfson's only FN is tagged multi_deadline, so it can't show that FNs are filtered
    # by tag. RSC has 6 FNs with only 1 tagged, which can.
    rsc_run = data.cell(BEST, "Royal Society of Chemistry")[0]
    assert len(rsc_run.false_negative_ids) == 6
    rsc_tp, rsc_n = st.stratum_recall(rsc_run, tags, "multi_deadline")
    rsc_filtered_fn = sum(
        1 for oid in rsc_run.false_negative_ids if "multi_deadline" in tags[oid]
    )
    assert rsc_filtered_fn == 1
    assert rsc_n == rsc_tp + rsc_filtered_fn
    assert rsc_n != rsc_tp + len(rsc_run.false_negative_ids)


def test_recall_over_all_strata_reconciles_with_the_run(data, gt):
    """TP + FN equals n_in_scope and every ID has an entry in the tag map."""
    tags = st.opportunity_tags(gt, data.funders())
    run = data.cell(BEST, "The Leverhulme Trust")[0]
    assert len(run.true_positive_ids) + len(run.false_negative_ids) == run.n_in_scope
    assert all(oid in tags for oid in run.true_positive_ids + run.false_negative_ids)


def test_a_funder_with_no_rows_in_the_stratum_is_absent_not_zero(data, gt):
    """Nuffield has no `restricted` rows, so it is left out rather than given 0.0."""
    by_funder = st.stratum_recall_by_funder(data, gt, "restricted")
    assert "Nuffield Foundation" not in by_funder[BEST]
    assert "Wolfson Foundation" in by_funder[BEST]
    assert len(by_funder[BEST]) == 11


def test_multi_deadline_recall_is_available_although_score_py_only_strata_restricted(
    data, gt
):
    by_funder = st.stratum_recall_by_funder(data, gt, "multi_deadline")
    assert len(by_funder[BEST]) == 10
    assert all(0.0 <= v <= 1.0 for v in by_funder[BEST].values())


def test_restricted_recall_agrees_with_the_scorers_own_stratum(data, gt):
    """Recomputing `restricted` recall from opportunity IDs matches the value `score.py` stored.

    Five funders have no `restricted` rows, so those runs are always (0, 0); the
    `non_trivial` count tracks the comparisons that could actually fail.
    """
    import json, pathlib
    frozen = {
        (r["funder"], r["config_hash"], r["replicate"]): r["strata"]["restricted"]
        for r in (
            json.loads(l)
            for l in pathlib.Path("eval/scored/grid/runs.jsonl").read_text().splitlines()
        )
    }
    tags = st.opportunity_tags(gt, data.funders())
    checked = 0
    non_trivial = 0
    for run in data.runs:
        theirs = frozen[(run.funder, run.config_hash, run.replicate)]
        tp, n = st.stratum_recall(run, tags, "restricted")
        checked += 1
        if theirs["n_in_scope"] > 0:
            non_trivial += 1
            assert (tp, n) == (theirs["tp"], theirs["n_in_scope"])
        else:
            # only checks that the join doesn't invent rows
            assert (tp, n) == (0, 0)
    assert checked == 1920
    assert non_trivial > 1000
