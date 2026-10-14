"""Tests for the cost-effectiveness Pareto frontier."""
from __future__ import annotations

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import frontier


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


def test_pareto_frontier_on_a_hand_checked_set():
    """A and B are on the frontier; C costs more than B for the same F1, so B dominates it."""
    points = frontier.pareto_frontier({
        "A": (1.0, 0.80),
        "B": (0.2, 0.30),
        "C": (0.5, 0.30),
        "D": (2.0, 0.75),
    })
    on = {p.config_label for p in points if p.on_frontier}
    assert on == {"A", "B"}
    c = next(p for p in points if p.config_label == "C")
    assert "B" in c.dominated_by


def test_an_equal_point_does_not_dominate_its_twin():
    """Two configs with identical cost and F1 do not dominate each other."""
    points = frontier.pareto_frontier({"A": (1.0, 0.5), "B": (1.0, 0.5)})
    assert all(p.on_frontier for p in points)


def test_the_real_frontier_includes_the_cheapest_and_the_best(data):
    points = frontier.cost_effectiveness(data)
    on = [p for p in points if p.on_frontier]
    assert len(points) == 24
    best = max(points, key=lambda p: p.macro_f1)
    cheapest = min(points, key=lambda p: p.cost_usd)
    assert best.on_frontier and cheapest.on_frontier
    assert best.config_label.startswith("openrouter/deepseek")
    assert best.config_label.endswith("few_shot | t0.7")
    assert on  # sanity: there is a frontier at all


def test_frontier_membership_probability_is_reported(data):
    """Each config gets a frontier membership probability between 0 and 1."""
    points = frontier.cost_effectiveness(data, n_resamples=500)
    assert all(0.0 <= p.frontier_probability <= 1.0 for p in points)
    on = [p for p in points if p.on_frontier]
    assert max(p.frontier_probability for p in on) > 0.5


def test_frontier_membership_probability_varies_with_uncertainty(data):
    """At least one frontier config has a membership probability below 1.0.

    On the real grid deepseek few_shot t0.7 only just beats the cheaper deepseek minimal t0.7.
    """
    points = frontier.cost_effectiveness(data, n_resamples=500)
    on = [p for p in points if p.on_frontier]
    assert any(p.frontier_probability < 1.0 for p in on)


def test_runs_that_died_before_billing_do_not_read_as_free(data):
    """Runs with no reported cost are not counted as $0, which would favour nova-lite."""
    points = {p.config_label: p for p in frontier.cost_effectiveness(data)}
    total = sum(p.cost_usd for p in points.values())
    assert total == pytest.approx(25.21, abs=0.01)
    assert all(p.cost_usd > 0 for p in points.values())


def _record(**kw):
    base = dict(
        funder="A", model="m", prompt="p", temperature=0.0, replicate=1,
        config_label="X", config_hash="h", f1=0.5, precision=0.5, recall=0.5,
        tp=1, fp=0, fn=1, n_extracted=1, n_in_scope=2, hallucination_rate=0.0,
        n_fabricated=0, fp_by_reason={}, true_positive_ids=(), false_negative_ids=(),
        cost_usd=0.1, latency_s=1.0, run_failed=False, abstained=False, split="tune",
    )
    return ds.RunRecord(**{**base, **kw})


def test_macro_f1_averages_over_funder_means_not_runs():
    """Macro F1 is the mean of funder means, not the mean over runs.

    The real grid is balanced so both give the same answer; this uses an unbalanced synthetic set.
    """
    runs = (
        _record(config_label="X", funder="A", cost_usd=1.0, f1=0.0),
        _record(config_label="X", funder="B", cost_usd=1.0, f1=1.0),
        _record(config_label="X", funder="B", cost_usd=1.0, f1=1.0),
        _record(config_label="X", funder="B", cost_usd=1.0, f1=1.0),
    )
    data = ds.AnalysisDataset(runs=runs, manifest={})
    (point,) = frontier.cost_effectiveness(data, n_resamples=0)
    # mean over funder means: (0.0 + 1.0) / 2 = 0.5, NOT mean over runs (0.75)
    assert point.macro_f1 == pytest.approx(0.5)
    assert point.cost_usd == pytest.approx(4.0)


def test_null_cost_runs_are_counted_not_silently_dropped(data):
    """Runs with no cost are left out of the sum but counted per config."""
    points = {p.config_label: p for p in frontier.cost_effectiveness(data)}
    total_excluded = sum(p.n_cost_excluded for p in points.values())
    assert total_excluded == 98
    nova_labels = [label for label in points if "nova-lite" in label]
    assert sum(points[label].n_cost_excluded for label in nova_labels) == 76


def test_a_resample_missing_a_funder_reports_the_gap_not_a_silent_shrink():
    """Resamples that draw a funder a config has no runs for are counted as gaps.

    This can't happen on the full grid, so config Y here is missing funder B.
    """
    runs = (
        _record(config_label="X", funder="A", cost_usd=1.0, f1=0.2),
        _record(config_label="X", funder="B", cost_usd=1.0, f1=0.4),
        _record(config_label="Y", funder="A", cost_usd=1.0, f1=0.6),
        # Y has no funder="B" run at all.
    )
    data = ds.AnalysisDataset(runs=runs, manifest={})
    points = {p.config_label: p for p in frontier.cost_effectiveness(data, n_resamples=200)}
    assert points["X"].n_resample_funder_gaps == 0
    assert points["Y"].n_resample_funder_gaps > 0
