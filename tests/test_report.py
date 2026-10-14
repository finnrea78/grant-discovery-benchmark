"""Tests for `report.py`, which formats the analysis results as text tables.

It does no calculation of its own; the numbers come from the other analysis modules.
"""
from __future__ import annotations

import pytest

from grant_eval.analysis import contrasts as con
from grant_eval.analysis import dataset as ds
from grant_eval.analysis import frontier as fr
from grant_eval.analysis import funders as fm
from grant_eval.analysis import report
from grant_eval.analysis import sensitivity as sens
from grant_eval.scoring.groundtruth import load_ground_truth

BEST = "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


@pytest.fixture(scope="module")
def gt():
    return load_ground_truth()


@pytest.fixture(scope="module")
def profiles():
    return fm.load_profiles()


@pytest.fixture(scope="module")
def family(data):
    return con.confirmatory_family(data)


@pytest.fixture(scope="module")
def points(data):
    return fr.cost_effectiveness(data, n_resamples=200)


def test_the_leaderboard_lists_every_config_best_first(data):
    text = report.leaderboard(data)
    assert text.count("\n") >= 24
    assert "few_shot" in text.splitlines()[2]
    assert "macroF1" in text


def test_undefined_quantities_render_as_a_dash_never_a_zero(data):
    """Undefined values print as a dash, as in aggregate.format_aggregate_report.

    Uses Fleiss' kappa for Cara on the best config, which is undefined because all
    replicates agree. Printing 0.000 would wrongly suggest chance-level agreement.
    """
    text = report.per_funder_table(data, BEST)
    cara = next(line for line in text.splitlines() if line.startswith("  Cara"))
    assert cara.rstrip().endswith("—")
    assert "0.000" not in cara.split("Cara")[-1][-8:]


def test_the_contrast_table_shows_effect_sizes_beside_p_values(family):
    text = report.contrast_table(family)
    for header in ("median d", "rank-biserial", "p", "Holm p", "95% CI"):
        assert header in text
    assert "n=16 clusters" in text


def test_every_contrast_row_has_separable_fields_not_glued_together(family):
    """Every contrast row splits into separate fields, so long model names don't run together."""
    text = report.contrast_table(family)
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("factor"))
    data_lines = [l for l in lines[header_idx + 1 : header_idx + 1 + len(family)] if l.strip()]
    assert len(data_lines) == len(family)
    for line in data_lines:
        fields = line.split()
        # factor, level_a, level_b, mean d, median d, rank-biserial, p, Holm p,
        # "[lower,", "upper]", reject. Exact count, since two glued fields give 10.
        assert len(fields) == 11, f"row did not separate into 11 fields: {line!r}"
        assert not any(field.count("openrouter/") >= 2 for field in fields), (
            f"two identifiers glued together on one line: {line!r}"
        )


def test_the_stratum_table_marks_which_strata_are_inferential(data, gt):
    text = report.stratum_table(data, gt)
    assert "restricted" in text and "multi_deadline" in text
    assert "descriptive" in text
    assert "js_portal" in text and "no tune rows" in text


def test_scope_notes_state_what_is_out_of_scope(data):
    """The scope notes say what was left out of the analysis."""
    text = report.scope_notes(data)
    assert "RQ6" in text and "validation run has not happened yet" in text
    assert "Pivot-RP" in text
    assert "tune" in text


def test_scope_notes_record_the_provenance_of_the_numbers(data):
    text = report.scope_notes(data)
    assert data.manifest["scorer"]["source_hash"][:12] in text
    assert "1920" in text


def test_the_full_report_contains_every_section(data, gt, profiles, family, points):
    text = report.full_report(data, gt, profiles, family, points)
    for heading in (
        "LEADERBOARD", "CONFIRMATORY CONTRASTS", "STRATA", "FUNDER TYPE",
        "STRUCTURAL COMPLEXITY", "COST-EFFECTIVENESS", "RELIABILITY",
        "SENSITIVITY", "SCOPE",
    ):
        assert heading in text


def test_the_full_report_states_the_preregistered_interaction(data, gt, profiles, family, points):
    """report.txt states the pre-registered chain_of_thought x model interaction and its sign.

    It came out negative rather than positive as predicted, so the number is printed, not just plotted.
    """
    text = report.full_report(data, gt, profiles, family, points)
    assert "INTERACTION" in text
    assert "D-B" in text
    assert "-0.164" in text
    assert "negative" in text.lower()


def test_interaction_note_reports_the_clustered_interval(data):
    text = report.interaction_note(data)
    assert "n=16 clusters" in text or "16 clusters" in text
    assert "nova-lite" in text.lower() or "amazon" in text.lower()


def test_temperature_note_names_every_model_pair_and_reports_clustered_intervals(data, family):
    """The temperature note covers all three model pairs, and every interaction interval crosses zero."""
    text = report.temperature_note(data, family)
    assert "deepseek" in text.lower()
    assert "nova-lite" in text.lower()
    assert "gpt-oss" in text.lower()
    assert "16 clusters" in text
    assert text.count("crosses zero") >= 3
    assert "no evidence of a temperature x model interaction" in text.lower()


def test_temperature_note_states_there_is_no_main_effect(data, family):
    text = report.temperature_note(data, family)
    assert "no temperature main effect" in text.lower()


def test_temperature_note_scopes_its_conclusion_to_f1_and_flags_post_hoc(data, family):
    """The temperature conclusion is limited to F1, flagged as post hoc, and points to RELIABILITY.

    Temperature does affect run-to-run stability, so a blanket "no effect" claim would be wrong.
    """
    text = report.temperature_note(data, family)
    assert "post hoc" in text.lower()
    assert "uncorrected" in text.lower()
    assert "on f1" in text.lower()
    assert "reliability" in text.lower()
    assert "does not appear to matter" not in text.lower()


def test_the_full_report_includes_the_temperature_note(data, gt, profiles, family, points):
    text = report.full_report(data, gt, profiles, family, points)
    assert "TEMPERATURE" in text
    assert "crosses zero" in text


def test_the_leaderboard_puts_the_worst_config_last(data):
    """The worst config is on the last row of the leaderboard."""
    text = report.leaderboard(data)
    last_row = [line for line in text.splitlines() if line.strip()][-1]
    assert "nova-lite" in last_row
    assert "chain_of_thought" in last_row
    assert "t0.0" in last_row


def test_leaderboard_pins_macro_precision_and_recall_for_the_best_config(data):
    """The best config's F1, precision and recall are in the right columns (values from configs.json)."""
    text = report.leaderboard(data)
    line = next(l for l in text.splitlines() if BEST in l)
    f1, precision, recall, n_funders = line.split()[-4:]
    assert f1 == "0.793"
    assert precision == "0.847"
    assert recall == "0.777"
    assert n_funders == "16"


def test_funder_type_table_reports_all_three_groups_and_flags_exploratory(data, profiles):
    text = report.funder_type_table(data, profiles)
    for group in ("charity", "niche_trust", "membership_body"):
        assert group in text
    assert "exploratory" in text.lower()


def test_report_pairs_each_decision_letter_with_the_right_heading(data, profiles):
    """The structural complexity and funder type tables carry their own labels, not each other's."""
    funder_type_text = report.funder_type_table(data, profiles)
    complexity_text = report.complexity_table(data, profiles)
    assert "(D-F)" in funder_type_text
    assert "(D-E)" not in funder_type_text
    assert "(D-E)" in complexity_text
    assert "(D-F)" not in complexity_text

    scope_text = report.scope_notes(data)
    assert "Funder type (D-F)" in scope_text
    assert "structural complexity (D-E)" in scope_text


def test_complexity_table_flags_exactly_the_two_packed_funders(data, profiles):
    text = report.complexity_table(data, profiles)
    assert "Society of Antiquaries" in text
    assert "Henry Moore" in text
    # Only data rows are indented, so the title line can't match "packed".
    data_rows = [line for line in text.splitlines() if line.startswith("  ")]
    packed_rows = [line for line in data_rows if "packed" in line and "one_per_page" not in line]
    assert len(packed_rows) == 2
    assert "artefact" in text.lower()


def test_frontier_table_marks_the_best_and_cheapest_as_on_the_frontier(points):
    text = report.frontier_table(points)
    best = max(points, key=lambda p: p.macro_f1)
    cheapest = min(points, key=lambda p: p.cost_usd)
    best_line = next(l for l in text.splitlines() if best.config_label in l)
    cheap_line = next(l for l in text.splitlines() if cheapest.config_label in l)
    assert "yes" in best_line
    assert "yes" in cheap_line


def test_frontier_table_states_how_many_runs_were_excluded_from_each_cost_total(points):
    """The frontier table notes how many unbilled runs each frontier config leaves out of its cost."""
    text = report.frontier_table(points)
    note = text.splitlines()[-1]
    assert "unequal denominators" in note.lower() or "not directly comparable" in note.lower()
    on_frontier = [p for p in points if p.on_frontier]
    assert on_frontier
    for p in on_frontier:
        assert p.config_label in note
        assert f"excludes {p.n_cost_excluded} run(s)" in note


def test_reliability_table_lists_every_config(data):
    text = report.reliability_table(data)
    labels = {c for c in data.config_labels()}
    for label in labels:
        assert label in text


def test_reliability_table_pins_stable_rate_and_kappa_for_a_named_row(data):
    """Stable rate and kappa for one config are in the right columns (values from configs.json)."""
    text = report.reliability_table(data)
    label = "openrouter/amazon/nova-lite-v1 | chain_of_thought | t0.0"
    line = next(l for l in text.splitlines() if l.strip().startswith(label))
    stable_rate, kappa = line.split()[-2:]
    assert stable_rate == "0.117"
    assert kappa == "0.243"


def test_failure_summary_states_the_main_counts_and_that_abstention_is_not_failure(data):
    """The summary gives 98 failed runs (76 overflows, 22 timeouts) and 216 abstentions, which are not failures."""
    text = report.failure_summary(data)
    assert "98 failed" in text
    assert "216 abstained" in text
    assert "76" in text and "22" in text
    assert "not a failure" in text.lower() or "not failures" in text.lower()
    assert "nova-lite" in text.lower()
    assert "deepseek" in text.lower()


def test_the_full_report_states_the_failure_and_abstention_counts(data, gt, profiles, family, points):
    """The full report includes the failure and abstention counts.

    Checked by phrase because "216" and "98" also appear elsewhere in the text.
    """
    text = report.full_report(data, gt, profiles, family, points)
    assert "98 failed" in text
    assert "216 abstained" in text


def test_sensitivity_table_reports_a_nonzero_exclusion_for_a_config_with_failures(data):
    """The last field on each sensitivity row is the number of excluded runs."""
    comparisons = sens.compare_with_and_without_failures(data)
    text = report.sensitivity_table(comparisons)
    lossy = next(c for c in comparisons if c.n_excluded > 0)
    line = next(l for l in text.splitlines() if lossy.label in l)
    assert line.split()[-1] == str(lossy.n_excluded)
    # zero exclusions should print as 0, not a dash
    clean = next(c for c in comparisons if c.n_excluded == 0)
    clean_line = next(l for l in text.splitlines() if clean.label in l)
    assert clean_line.split()[-1] == "0"
