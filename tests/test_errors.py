"""Tests for false-positive breakdowns and fabrication counts (`analysis/errors.py`).

The breakdowns have to add up to the same false positives that precision uses.
"""
from __future__ import annotations

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import errors as err


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


def test_fp_reason_totals_reconcile_with_the_fp_column(data):
    """Grouped false-positive reason totals equal the grouped `fp` sums."""
    profiles = err.fp_reason_totals(data, key="model")
    charged = {}
    for r in data.runs:
        charged[r.model] = charged.get(r.model, 0) + r.fp
    assert {p.label: p.n_false_positives for p in profiles} == charged
    # Check the grid total too, in case errors cancel out across models.
    assert sum(p.n_false_positives for p in profiles) == sum(r.fp for r in data.runs)


def test_fp_reason_shares_sum_to_one_and_match_their_counts(data):
    for p in err.fp_reason_totals(data, key="model"):
        assert p.n_false_positives == sum(p.by_reason.values())
        assert sum(p.share.values()) == pytest.approx(1.0)
        for reason, n in p.by_reason.items():
            assert p.share[reason] == pytest.approx(n / p.n_false_positives)


def test_fp_reason_composition_differs_by_model(data):
    """nova-lite's most common false positive is `unlabelled_page`; DeepSeek's is `ambiguous_title`."""
    by_label = {p.label: p for p in err.fp_reason_totals(data, key="model")}
    nova = by_label["openrouter/amazon/nova-lite-v1"]
    deepseek = by_label["openrouter/deepseek/deepseek-v4-flash-0731"]
    assert max(nova.by_reason, key=nova.by_reason.get) == "unlabelled_page"
    assert max(deepseek.by_reason, key=deepseek.by_reason.get) == "ambiguous_title"


def test_fp_reason_share_is_empty_not_zeros_when_nothing_was_charged(data):
    """A group with no false positives gets empty dicts rather than dicts of zeros."""
    clean = tuple(r for r in data.runs if r.fp == 0)
    assert clean, "fixture assumption: some runs charge no false positives"
    for p in err.fp_reason_totals(data, runs=clean, key="model"):
        assert p.n_false_positives == 0
        assert p.share == {}
        assert p.by_reason == {}


def test_fp_reason_totals_rejects_an_unknown_key(data):
    with pytest.raises(ValueError, match="key must be one of"):
        err.fp_reason_totals(data, key="funder")


def test_fp_reason_totals_by_config_partitions_the_same_total(data):
    """Grouping by config gives the same total as grouping by model."""
    by_model = err.fp_reason_totals(data, key="model")
    by_config = err.fp_reason_totals(data, key="config")
    assert sum(p.n_false_positives for p in by_config) == sum(
        p.n_false_positives for p in by_model
    )
    assert len(by_config) == 24


def test_fabrication_totals_match_a_hand_count(data):
    profiles = {p.label: p for p in err.fabrication_totals(data, key="model")}
    expected = {}
    for r in data.runs:
        if r.n_fabricated:
            expected[r.model] = expected.get(r.model, 0) + r.n_fabricated
    for label, p in profiles.items():
        assert p.total_fabricated == expected.get(label, 0)


def test_deepseek_fabricated_nothing_and_that_is_a_real_zero(data):
    """DeepSeek fabricated nothing, and its mean rate is a defined 0.0 rather than None."""
    profiles = {p.label: p for p in err.fabrication_totals(data, key="model")}
    deepseek = profiles["openrouter/deepseek/deepseek-v4-flash-0731"]
    assert deepseek.total_fabricated == 0
    assert deepseek.n_runs_fabricating == 0
    assert deepseek.mean_hallucination_rate == pytest.approx(0.0)
    assert deepseek.n_rate_defined > 0


def test_mean_hallucination_rate_excludes_runs_with_no_defined_rate(data):
    """Runs with no defined rate are left out of the mean rather than counted as 0."""
    for p in err.fabrication_totals(data, key="model"):
        defined = [
            r.hallucination_rate
            for r in data.runs
            if r.model == p.label and r.hallucination_rate is not None
        ]
        assert p.n_rate_defined == len(defined)
        assert p.n_rate_defined < p.n_runs or not any(
            r.hallucination_rate is None for r in data.runs if r.model == p.label
        )
        assert p.mean_hallucination_rate == pytest.approx(sum(defined) / len(defined))


def test_mean_hallucination_rate_is_none_not_zero_when_nothing_is_defined(data):
    """A group with no defined rates has a mean of None."""
    none_defined = tuple(r for r in data.runs if r.hallucination_rate is None)
    assert none_defined, "fixture assumption: some runs have no defined rate"
    for p in err.fabrication_totals(data, runs=none_defined, key="model"):
        assert p.n_rate_defined == 0
        assert p.mean_hallucination_rate is None


def test_runs_fabricating_separates_one_bad_run_from_a_habit(data):
    """`n_runs_fabricating` is consistent with `total_fabricated` and `n_runs`."""
    for p in err.fabrication_totals(data, key="model"):
        assert p.n_runs_fabricating <= p.total_fabricated or p.total_fabricated == 0
        assert p.n_runs_fabricating <= p.n_runs
