"""Tests for the pre-registered confirmatory contrasts, factor marginals and interaction."""
from __future__ import annotations

import pytest

from grant_eval.analysis import contrasts as con
from grant_eval.analysis import dataset as ds
from grant_eval.analysis import stats as st


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


@pytest.fixture(scope="module")
def family(data):
    return con.confirmatory_family(data)


def test_the_family_is_ten_within_factor_contrasts(family):
    """The family is 3 model pairs, 6 prompt pairs and 1 temperature pair.

    Not all 276 config pairs, which at n=16 would reject nothing after correction.
    """
    assert len(family) == con.EXPECTED_FAMILY_SIZE == 10
    counts = {}
    for c in family:
        counts[c.factor] = counts.get(c.factor, 0) + 1
    assert counts == {"model": 3, "prompt": 6, "temperature": 1}


def test_expected_family_size_is_derived_from_the_levels_not_hardcoded(data):
    """The expected family size comes from the factor levels.

    The validation split has 1 model, 2 prompts and 1 temperature, so only one contrast.
    """
    assert con.expected_family_size(con.factor_levels(data)) == con.EXPECTED_FAMILY_SIZE
    assert con.expected_family_size(
        {"model": ("m",), "prompt": ("few_shot", "minimal"), "temperature": ("0.7",)}
    ) == 1
    assert con.expected_family_size(
        {"model": ("a", "b", "c"), "prompt": ("p",), "temperature": ("0.0", "0.7")}
    ) == 4


def test_every_contrast_is_paired_on_all_sixteen_funders(family):
    for contrast in family:
        assert contrast.n_funders == 16
        assert contrast.excluded == ()


def test_holm_is_applied_across_the_whole_family_not_per_factor(family):
    """Holm correction is applied over all 10 contrasts, not separately per factor."""
    raw_p = [c.p_value for c in family]
    expected = st.holm_bonferroni(raw_p)
    for contrast, adj, rej in zip(family, expected.adjusted, expected.rejected):
        assert contrast.adjusted_p == pytest.approx(adj)
        assert contrast.rejected == rej
    assert all(c.wilcoxon.n_used <= 16 for c in family)


def test_the_model_contrasts_recover_the_known_ordering(data, family):
    """All three model contrasts are significant with large effect sizes.

    The models' F1 ranges (deepseek 0.71-0.79, gpt-oss 0.51-0.57, nova-lite 0.14-0.30) do not overlap.
    """
    model_contrasts = [c for c in family if c.factor == "model"]
    assert len(model_contrasts) == 3
    assert all(c.rejected for c in model_contrasts)
    # Real values are -1.000, -0.900 and +0.779 (deepseek vs gpt-oss is the weakest).
    assert all(abs(c.wilcoxon.rank_biserial) > 0.7 for c in model_contrasts)


def test_no_prompt_or_temperature_contrast_survives_correction(family):
    """No prompt or temperature contrast survives Holm correction.

    The smallest raw prompt p is 0.0443 and the temperature p is 0.348. The temperature
    pattern on deepseek is an interaction rather than a main effect.
    """
    others = [c for c in family if c.factor in ("prompt", "temperature")]
    assert len(others) == 7
    assert not any(c.rejected for c in others)


def test_each_contrast_carries_an_effect_size_and_an_interval(family):
    """Every contrast has a rank-biserial effect size and a funder-clustered CI."""
    for contrast in family:
        assert contrast.wilcoxon.rank_biserial is not None
        assert contrast.difference_ci.n_clusters == 16
        assert contrast.difference_ci.lower <= contrast.mean_difference
        assert contrast.mean_difference <= contrast.difference_ci.upper


def test_the_sign_test_is_reported_beside_wilcoxon(family):
    """The sign test is reported alongside Wilcoxon, as pre-registered."""
    for contrast in family:
        assert contrast.sign.n_used <= 16
        assert contrast.sign.p_value is not None


def test_marginal_means_average_over_the_other_factors(data):
    """A model's per-funder value is the mean over 40 runs (4 prompts x 2 temperatures x 5 replicates)."""
    means = con.marginal_means(data, "model")
    assert set(means) == set(con.factor_levels(data)["model"])
    for level in means:
        assert len(means[level]) == 16
    runs = [
        r for r in data.runs
        if r.model == "openrouter/amazon/nova-lite-v1" and r.funder == "Wolfson Foundation"
    ]
    assert len(runs) == 40
    assert means["openrouter/amazon/nova-lite-v1"]["Wolfson Foundation"] == pytest.approx(
        sum(r.f1 for r in runs) / 40
    )


def test_factor_levels_are_read_from_the_data_not_hardcoded(data):
    levels = con.factor_levels(data)
    assert len(levels["model"]) == 3
    assert len(levels["prompt"]) == 4
    assert levels["temperature"] == ("0.0", "0.7")


def _record(**kw):
    base = dict(
        funder="A", model="m", prompt="p", temperature=0.0, replicate=1,
        config_label="X", config_hash="h", f1=0.5, precision=0.5, recall=0.5,
        tp=1, fp=0, fn=1, n_extracted=1, n_in_scope=2, hallucination_rate=0.0,
        n_fabricated=0, fp_by_reason={}, true_positive_ids=(), false_negative_ids=(),
        cost_usd=0.1, latency_s=1.0, run_failed=False, abstained=False, split="tune",
    )
    return ds.RunRecord(**{**base, **kw})


def test_paired_levels_pairs_by_funder_name_not_dict_position():
    """Pairing is by funder name, not by position (funders are in a different order per level)."""
    data = ds.AnalysisDataset(
        runs=(
            _record(funder="A", model="m1", f1=1.0),
            _record(funder="B", model="m1", f1=2.0),
            _record(funder="B", model="m2", f1=20.0),
            _record(funder="A", model="m2", f1=10.0),
        ),
        manifest={},
    )
    funders, a, b, excluded = con._paired_levels(data, "model", "m1", "m2", "f1", None)
    assert excluded == ()
    by_funder = dict(zip(funders, zip(a, b)))
    assert by_funder["A"] == (1.0, 10.0)
    assert by_funder["B"] == (2.0, 20.0)


def test_a_contrast_over_a_value_with_undefined_cells_reports_its_exclusions(data):
    """With precision undefined for some runs, any funder dropped is listed as excluded."""
    family = con.confirmatory_family(data, value="precision")
    assert len(family) == 10
    for contrast in family:
        assert contrast.n_funders + len(contrast.excluded) == 16


NOVA = "openrouter/amazon/nova-lite-v1"
DEEPSEEK = "openrouter/deepseek/deepseek-v4-flash-0731"


def test_cell_means_cover_every_combination(data):
    cells = con.cell_means(data, "model", "prompt")
    assert len(cells.means) == 3 * 4
    for funders in cells.means.values():
        assert len(funders) == 16


def test_the_preregistered_cot_by_model_interaction_is_computed(data):
    """The pre-registered chain_of_thought vs minimal by model interaction.

    The gap was predicted to be largest on nova-lite. It is, but it is negative.
    """
    result = con.interaction(
        data, "model", NOVA, DEEPSEEK,
        "prompt", "chain_of_thought", "minimal",
        predicted_direction="largest on nova-lite",
    )
    assert result.n_funders == 16
    assert result.difference_in_differences < 0
    # Gaps are roughly nova -0.152, deepseek +0.012, gpt-oss +0.016.
    assert result.observed_direction == f"largest on {NOVA}, negative"
    assert result.excluded == ()


def test_observed_direction_is_derived_not_the_first_argument():
    """`observed_direction` names the model with the largest gap (m3), not the first argument."""
    data = ds.AnalysisDataset(
        runs=(
            _record(funder="A", model="m1", prompt="p1", f1=0.5),
            _record(funder="A", model="m1", prompt="p2", f1=0.6),
            _record(funder="A", model="m2", prompt="p1", f1=0.5),
            _record(funder="A", model="m2", prompt="p2", f1=0.45),
            _record(funder="A", model="m3", prompt="p1", f1=0.9),
            _record(funder="A", model="m3", prompt="p2", f1=0.4),
        ),
        manifest={},
    )
    result = con.interaction(data, "model", "m1", "m2", "prompt", "p1", "p2")
    assert result.observed_direction == "largest on m3, positive"


def test_factor_gap_by_level_is_the_ingredient_behind_observed_direction(data):
    """The per-model chain_of_thought minus minimal gaps match the known values."""
    cells = con.cell_means(data, "model", "prompt")
    gaps = con.factor_gap_by_level(cells, "chain_of_thought", "minimal")
    assert set(gaps) == {NOVA, DEEPSEEK, "openrouter/openai/gpt-oss-120b"}
    assert gaps[NOVA] == pytest.approx(-0.152, abs=0.001)
    assert gaps[DEEPSEEK] == pytest.approx(0.012, abs=0.001)
    assert gaps["openrouter/openai/gpt-oss-120b"] == pytest.approx(0.016, abs=0.001)


def test_the_interaction_is_a_difference_in_differences_per_funder(data):
    """The per-funder interaction matches a hand calculation from the cell means."""
    cells = con.cell_means(data, "model", "prompt")
    funder = "Wolfson Foundation"
    by_hand = (
        cells.means[(NOVA, "chain_of_thought")][funder]
        - cells.means[(NOVA, "minimal")][funder]
    ) - (
        cells.means[(DEEPSEEK, "chain_of_thought")][funder]
        - cells.means[(DEEPSEEK, "minimal")][funder]
    )
    result = con.interaction(
        data, "model", NOVA, DEEPSEEK, "prompt", "chain_of_thought", "minimal"
    )
    assert result.per_funder[funder] == pytest.approx(by_hand)


def test_the_interaction_carries_a_clustered_interval(data):
    result = con.interaction(
        data, "model", NOVA, DEEPSEEK, "prompt", "chain_of_thought", "minimal"
    )
    assert result.ci.n_clusters == 16
    assert result.ci.lower <= result.difference_in_differences <= result.ci.upper


def test_an_interaction_over_an_absent_level_raises(data):
    with pytest.raises(KeyError, match="no such level"):
        con.interaction(data, "model", NOVA, "not-a-model", "prompt", "minimal", "few_shot")


def test_an_interaction_names_a_funder_missing_from_one_cell():
    """A funder missing from any of the four cells is dropped and listed in `excluded` with the cell."""
    data = ds.AnalysisDataset(
        runs=(
            _record(funder="A", model="m1", prompt="p1", f1=1.0),
            _record(funder="A", model="m1", prompt="p2", f1=2.0),
            _record(funder="A", model="m2", prompt="p1", f1=3.0),
            _record(funder="A", model="m2", prompt="p2", f1=4.0),
            _record(funder="B", model="m1", prompt="p1", f1=10.0),
            # funder B has no (m1, p2) run
            _record(funder="B", model="m2", prompt="p1", f1=30.0),
            _record(funder="B", model="m2", prompt="p2", f1=40.0),
        ),
        manifest={},
    )
    result = con.interaction(data, "model", "m1", "m2", "prompt", "p1", "p2")
    assert result.n_funders == 1
    assert "B" not in result.per_funder
    assert result.excluded == (("B", "undefined f1 for model='m1', prompt='p2'"),)
