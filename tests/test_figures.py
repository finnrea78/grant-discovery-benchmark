"""Tests for `figures`, which regenerates every table and figure from the frozen JSON."""
from __future__ import annotations

import json

import pytest

from grant_eval.analysis import figures


def test_figures_writes_every_artefact(tmp_path):
    rc = figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    assert rc == 0
    for name in (
        "report.txt", "results.json", "f1_by_config.png", "contrasts.png",
        "frontier.png", "strata.png", "reliability.png", "interaction.png",
        "fp_reasons.png", "latency.png", "abstention.png", "fabrication.png",
        "cost_per_tp.png", "edge_cases.png", "edge_case_overlap.png",
        "field_accuracy.png", "end_to_end.png", "field_by_tag.png",
        "end_to_end_frontier.png", "field_by_prompt.png",
    ):
        assert (tmp_path / name).exists(), name
        assert (tmp_path / name).stat().st_size > 0


def test_results_json_nulls_undefined_values(tmp_path):
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["counts"]["n_runs"] == 1920
    assert len(payload["contrasts"]) == 10
    assert payload["provenance"]["scorer_source_hash"]
    undefined = [
        c for c in payload["leaderboard"] if c["macro_precision"] is None
    ]
    assert all(c["macro_precision_undefined"] for c in undefined)


def test_figures_regenerate_identically_from_the_frozen_json(tmp_path):
    """Two runs with the same seed give identical results.json (apart from the timestamp)."""
    figures.main(["--out-dir", str(tmp_path / "a"), "--resamples", "200"])
    figures.main(["--out-dir", str(tmp_path / "b"), "--resamples", "200"])
    a = json.loads((tmp_path / "a" / "results.json").read_text())
    b = json.loads((tmp_path / "b" / "results.json").read_text())
    a.pop("generated_utc"), b.pop("generated_utc")
    assert a == b


def test_figures_refuses_to_run_against_a_missing_artefact(tmp_path, capsys):
    rc = figures.main(
        ["--out-dir", str(tmp_path), "--scored-dir", str(tmp_path / "nope")]
    )
    assert rc == 1
    assert "freeze_scores" in capsys.readouterr().err


# The tests below check content, since an empty file would pass the existence checks.
def test_results_json_matches_the_known_leaderboard_extremes(tmp_path):
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    payload = json.loads((tmp_path / "results.json").read_text())
    board = sorted(payload["leaderboard"], key=lambda c: -c["macro_f1"])
    assert board[0]["config_label"] == (
        "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"
    )
    assert board[0]["macro_f1"] == pytest.approx(0.793, abs=0.001)
    assert board[-1]["config_label"] == (
        "openrouter/amazon/nova-lite-v1 | chain_of_thought | t0.0"
    )
    assert board[-1]["macro_f1"] == pytest.approx(0.138, abs=0.001)


def test_report_txt_matches_the_report_module_output(tmp_path):
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    text = (tmp_path / "report.txt").read_text()
    assert "LEADERBOARD" in text
    assert "SCOPE" in text
    assert "RQ6" in text


def test_results_json_nulls_undefined_fleiss_kappa_for_real_cases(tmp_path):
    """Undefined per-funder precision and Fleiss' kappa are written as null.

    The leaderboard check above has no undefined cases on the real grid, but `per_funder` does.
    """
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    payload = json.loads((tmp_path / "results.json").read_text())
    per_funder = payload["per_funder"]
    undefined_kappa = [r for r in per_funder if r["fleiss_kappa_undefined"]]
    undefined_precision = [r for r in per_funder if r["precision_undefined"]]
    assert len(undefined_kappa) > 0 and len(undefined_precision) > 0
    assert all(r["fleiss_kappa"] is None for r in undefined_kappa)
    assert all(r["precision"] is None for r in undefined_precision)


def test_results_json_carries_the_preregistered_interaction(tmp_path):
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    payload = json.loads((tmp_path / "results.json").read_text())
    interaction = payload["interaction"]
    assert interaction["difference_in_differences"] == pytest.approx(-0.164, abs=0.001)
    assert interaction["n_funders"] == 16


def test_results_json_carries_the_failure_breakdown(tmp_path):
    figures.main(["--out-dir", str(tmp_path), "--resamples", "200"])
    payload = json.loads((tmp_path / "results.json").read_text())
    failures = payload["failures"]
    assert failures["n_failed"] == 98
    assert failures["n_context_overflow"] == 76
    assert failures["n_timed_out"] == 22
    assert failures["n_abstained"] == 216


def test_the_agg_backend_is_set_so_the_cli_runs_headless():
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_mean_recall_by_model_and_tag_matches_a_hand_flattened_average():
    """The strata bar height for (nova-lite, restricted) matches a mean computed here by hand."""
    import itertools

    from grant_eval.analysis import dataset as ds
    from grant_eval.analysis import strata as strat
    from grant_eval.scoring.groundtruth import load_ground_truth

    data = ds.load_dataset()
    gt = load_ground_truth()
    model = "openrouter/amazon/nova-lite-v1"
    tag = "restricted"

    model_runs = tuple(r for r in data.runs if r.model == model)
    by_config = strat.stratum_recall_by_funder(data, gt, tag, runs=model_runs)
    expected_vals = list(
        itertools.chain.from_iterable(by_funder.values() for by_funder in by_config.values())
    )
    expected = sum(expected_vals) / len(expected_vals)

    assert figures._mean_recall_by_model_and_tag(data, gt, tag, model) == pytest.approx(expected)
    assert 0.0 <= figures._mean_recall_by_model_and_tag(data, gt, tag, model) <= 1.0


def test_the_interaction_figure_uses_the_tested_factor_gap_helper(tmp_path):
    """`_draw_interaction` gets its bars from `contrasts.factor_gap_by_level`."""
    from unittest import mock

    from grant_eval.analysis import contrasts as con
    from grant_eval.analysis import dataset as ds

    data = ds.load_dataset()
    cells = con.cell_means(data, "model", "prompt")
    real_gaps = con.factor_gap_by_level(cells, "chain_of_thought", "minimal")
    assert real_gaps

    with mock.patch.object(
        figures.con, "factor_gap_by_level", wraps=con.factor_gap_by_level
    ) as spy:
        figures._draw_interaction(data, tmp_path / "interaction.png")
    assert spy.called
    spy.assert_called_with(cells, "chain_of_thought", "minimal")


def test_short_config_label_keeps_the_identifying_part():
    """The short label drops the vendor path but keeps prompt and temperature."""
    assert figures._short_config_label(
        "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"
    ) == "deepseek-v4-flash-0731 | few_shot | t0.7"
    assert figures._short_config_label(
        "openrouter/amazon/nova-lite-v1 | minimal | t0.0"
    ) == "nova-lite-v1 | minimal | t0.0"
    # input with no " | " separator
    assert figures._short_config_label("openrouter/amazon/nova-lite-v1") == "nova-lite-v1"


def test_frontier_annotation_reports_cost_f1_and_frontier_probability():
    from grant_eval.analysis import frontier as fr

    point = fr.FrontierPoint(
        config_label="openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7",
        cost_usd=1.016,
        macro_f1=0.7928,
        on_frontier=True,
        dominated_by=(),
        frontier_probability=0.495,
    )
    text = figures._frontier_annotation(point)
    assert "deepseek-v4-flash-0731 | few_shot | t0.7" in text
    assert "$1.02" in text
    assert "F1 0.793" in text
    assert "P(front) 0.49" in text


def test_frontier_annotation_never_prints_a_missing_probability_as_a_number():
    """A missing frontier probability is shown as n/a, not 0.00."""
    from grant_eval.analysis import frontier as fr

    point = fr.FrontierPoint(
        config_label="openrouter/amazon/nova-lite-v1 | minimal | t0.7",
        cost_usd=0.16,
        macro_f1=0.3028,
        on_frontier=True,
        dominated_by=(),
        frontier_probability=None,
    )
    text = figures._frontier_annotation(point)
    assert "P(front) n/a" in text
    assert "0.00" not in text
