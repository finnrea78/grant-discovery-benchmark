"""Tests for latency, abstention and cost per true positive (`analysis/telemetry.py`).

Mostly about which runs go into each denominator and what happens when it is empty.
"""
from __future__ import annotations

import numpy as np
import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import telemetry as tel


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


def test_latency_excludes_runs_with_no_defined_latency(data):
    """Runs with no latency count in `n_runs` but are left out of the statistics, not treated as 0."""
    for p in tel.latency_profiles(data, key="model"):
        defined = [
            r.latency_s for r in data.runs if r.model == p.label and r.latency_s is not None
        ]
        assert p.n_runs == sum(1 for r in data.runs if r.model == p.label)
        assert p.n_defined == len(defined)
        assert p.mean_s == pytest.approx(sum(defined) / len(defined))
        assert p.max_s == pytest.approx(max(defined))


def test_latency_percentiles_match_numpy_linear_interpolation(data):
    """Percentiles use numpy's default linear interpolation."""
    for p in tel.latency_profiles(data, key="model"):
        defined = [
            r.latency_s for r in data.runs if r.model == p.label and r.latency_s is not None
        ]
        assert p.p50_s == pytest.approx(float(np.percentile(defined, 50)))
        assert p.p95_s == pytest.approx(float(np.percentile(defined, 95)))


def test_latency_is_none_not_zero_when_no_run_has_a_defined_value(data):
    undefined = tuple(r for r in data.runs if r.latency_s is None)
    assert undefined, "fixture assumption: some runs have no defined latency"
    for p in tel.latency_profiles(data, runs=undefined, key="model"):
        assert p.n_defined == 0
        assert p.mean_s is None
        assert p.p50_s is None
        assert p.p95_s is None
        assert p.max_s is None


def test_the_dollar_ranking_and_the_time_ranking_disagree(data):
    """DeepSeek is cheap and accurate but much slower than the other models."""
    by_label = {p.label: p for p in tel.latency_profiles(data, key="model")}
    deepseek = by_label["openrouter/deepseek/deepseek-v4-flash-0731"]
    nova = by_label["openrouter/amazon/nova-lite-v1"]
    gpt_oss = by_label["openrouter/openai/gpt-oss-120b"]
    assert deepseek.p50_s > gpt_oss.p50_s > nova.p50_s
    # Right-skewed, so the mean is well above the median.
    assert deepseek.mean_s > deepseek.p50_s * 2


def test_abstention_rate_matches_a_hand_count(data):
    for p in tel.abstention_profiles(data, key="model"):
        runs = [r for r in data.runs if r.model == p.label]
        n_abstained = sum(1 for r in runs if r.abstained)
        assert p.n_runs == len(runs)
        assert p.n_abstained == n_abstained
        assert p.rate == pytest.approx(n_abstained / len(runs))


def test_abstention_totals_match_the_failure_breakdown_count(data):
    """Abstention totals agree with `sensitivity.failure_breakdown`."""
    from grant_eval.analysis import sensitivity as sens

    total = sum(p.n_abstained for p in tel.abstention_profiles(data, key="model"))
    assert total == sens.failure_breakdown(data).n_abstained


def test_abstention_is_not_failure(data):
    """No run is both abstained and failed, so abstentions are not double-counted."""
    assert not any(r.abstained and r.run_failed for r in data.runs)


def test_abstention_rate_is_none_for_an_empty_group(data):
    assert tel.abstention_profiles(data, runs=(), key="model") == ()


def test_cost_per_true_positive_matches_a_hand_computation(data):
    for row in tel.cost_per_true_positive(data, key="config"):
        runs = [r for r in data.runs if r.config_label == row.label]
        cost = sum(r.cost_usd for r in runs if r.cost_usd is not None)
        tp = sum(r.tp for r in runs)
        assert row.cost_usd == pytest.approx(cost)
        assert row.n_true_positives == tp
        assert row.cost_per_tp == pytest.approx(cost / tp)


def test_cost_per_true_positive_carries_the_unbilled_exclusion_forward(data):
    """Unbilled runs are left out of the cost but counted in `n_cost_excluded`."""
    for row in tel.cost_per_true_positive(data, key="config"):
        runs = [r for r in data.runs if r.config_label == row.label]
        assert row.n_cost_excluded == sum(1 for r in runs if r.cost_usd is None)


def test_cost_per_true_positive_is_undefined_not_zero_when_nothing_was_recovered(data):
    """Cost per true positive is None, not 0 or infinity, when nothing was recovered."""
    barren = tuple(r for r in data.runs if r.tp == 0)
    assert barren, "fixture assumption: some runs recover nothing"
    rows = tel.cost_per_true_positive(data, runs=barren, key="config")
    assert rows
    for row in rows:
        assert row.n_true_positives == 0
        assert row.cost_per_tp is None


def test_cost_per_true_positive_rejects_an_unknown_key(data):
    with pytest.raises(ValueError, match="key must be one of"):
        tel.cost_per_true_positive(data, key="funder")
