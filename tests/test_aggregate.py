"""Tests for the aggregator, built from hand-made RunScore fixtures.

The TP/FN lists hold `_Item` placeholders because aggregation only reads list lengths and
`opportunity_id`. CLI tests write their own small files into tmp_path.
"""
from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from grant_eval.schema import ExtractedOpportunity, ExtractionResult, Split
from grant_eval.scoring import aggregate as aggmod
from grant_eval.scoring.groundtruth import REQUIRED_COLUMNS
from grant_eval.scoring.score import DetectionScore, RunScore


def _result(
    funder,
    *,
    model="test/model",
    prompt_strategy="minimal",
    temperature=0.0,
    agent_config="standard",
    replicate=1,
    **kw,
) -> ExtractionResult:
    return ExtractionResult(
        snapshot_date=date(2026, 8, 21),
        funder=funder,
        split=Split.TUNE,
        model=model,
        prompt_strategy=prompt_strategy,
        temperature=temperature,
        agent_config=agent_config,
        replicate=replicate,
        config_hash="test",
        **kw,
    )


class _Item:
    """Stand-in for TruePositive/FalseNegative that only has `opportunity_id`."""

    def __init__(self, opportunity_id: str):
        self.opportunity_id = opportunity_id


def _detection(tp: int, fp: int, fn: int, n_in_scope: int, n_extracted: int | None = None) -> DetectionScore:
    d = DetectionScore(
        n_in_scope=n_in_scope, n_extracted=tp + fp if n_extracted is None else n_extracted
    )
    # Consistency aggregation reads `.opportunity_id` off TPs and FNs, so the
    # placeholders need one even when a test only cares about the counts.
    d.true_positives = [_Item(f"placeholder-tp-{i}") for i in range(tp)]
    d.false_positives = [object()] * fp
    d.false_negatives = [_Item(f"placeholder-fn-{i}") for i in range(fn)]
    return d


def _run(funder, tp, fp, fn, n_in_scope, *, n_extracted=None, result=None, **result_kw) -> RunScore:
    if result is None:
        result = _result(funder, **result_kw)
    return RunScore(
        funder=funder,
        result=result,
        detection=_detection(tp, fp, fn, n_in_scope, n_extracted),
        strata={},
        fields={},
    )


def _run_with_items(funder, tp_ids, fn_ids, *, result=None, **result_kw) -> RunScore:
    """Build a RunScore from ids: `tp_ids` were found in this replicate, `fn_ids` were not."""
    if result is None:
        result = _result(funder, **result_kw)
    d = DetectionScore(n_in_scope=len(tp_ids) + len(fn_ids), n_extracted=len(tp_ids))
    d.true_positives = [_Item(i) for i in tp_ids]
    d.false_negatives = [_Item(i) for i in fn_ids]
    return RunScore(funder=funder, result=result, detection=d, strata={}, fields={})


def test_two_replicates_of_one_funder_average_their_f1():
    # rep1: n_extracted=5, tp=2 -> precision 0.4; n_in_scope=5 -> recall 0.4 -> F1 0.4
    rep1 = _run("Funder A", tp=2, fp=3, fn=3, n_in_scope=5)
    # rep2: n_extracted=5, tp=4 -> precision 0.8; n_in_scope=5 -> recall 0.8 -> F1 0.8
    rep2 = _run("Funder A", tp=4, fp=1, fn=1, n_in_scope=5, result=_result("Funder A", replicate=2))

    [config] = aggmod.aggregate_runs([rep1, rep2])
    [funder_agg] = config.funders
    assert funder_agg.n_replicates == 2
    assert funder_agg.n_in_scope == 5
    assert round(funder_agg.f1, 4) == round((0.4 + 0.8) / 2, 4) == 0.6


def test_macro_and_micro_differ_on_unequal_n_in_scope():
    # Funder A: n_in_scope 10, tp 5, fp 5 -> precision .5 recall .5 f1 .5
    a = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    # Funder B: n_in_scope 100, tp 100, fp 0 -> precision 1 recall 1 f1 1
    b = _run("Funder B", tp=100, fp=0, fn=0, n_in_scope=100)

    [config] = aggmod.aggregate_runs([a, b])

    # macro: unweighted mean of per-funder F1
    assert round(config.macro_f1, 4) == 0.75
    assert round(config.macro_precision, 4) == 0.75
    assert round(config.macro_recall, 4) == 0.75

    # micro: summed TP/FP/FN across both funders' runs
    # total tp=105, fp=5, fn=5
    assert round(config.micro_precision, 4) == round(105 / 110, 4)
    assert round(config.micro_recall, 4) == round(105 / 110, 4)
    assert round(config.micro_f1, 4) == round(105 / 110, 4)

    # Confirm they actually differ from macro (unequal n_in_scope is why)
    assert round(config.macro_f1, 4) != round(config.micro_f1, 4)


def test_zero_positive_funder_is_excluded_from_macro_and_named():
    included = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    zero_positive = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)

    [config] = aggmod.aggregate_runs([included, zero_positive])

    assert config.excluded_funders == ["Funder Z"]
    assert config.n_funders == 1  # only the included funder counts in the macro mean
    # macro_f1 is just Funder A's F1 (.5), unaffected by the excluded funder
    assert round(config.macro_f1, 4) == 0.5

    [abstention] = config.abstentions
    assert abstention.funder == "Funder Z"
    assert abstention.n_runs == 1
    assert abstention.mean_extracted == 0.0
    assert abstention.abstention_rate == 1.0


def test_all_funders_zero_positive_makes_macro_f1_undefined():
    only_zero = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)

    [config] = aggmod.aggregate_runs([only_zero])

    assert config.excluded_funders == ["Funder Z"]
    assert config.n_funders == 0
    assert config.macro_f1_undefined is True
    assert config.macro_f1 == 0.0
    assert config.macro_precision == 0.0
    assert config.macro_recall == 0.0


def test_replicates_disagreeing_on_n_in_scope_raise_value_error_naming_the_funder():
    rep1 = _run("Funder D", tp=1, fp=0, fn=4, n_in_scope=5)
    rep2 = _run(
        "Funder D", tp=1, fp=0, fn=5, n_in_scope=6, result=_result("Funder D", replicate=2)
    )
    with pytest.raises(ValueError, match="Funder D"):
        aggmod.aggregate_runs([rep1, rep2])


def test_runs_of_different_configs_land_in_different_config_aggregates():
    r1 = _run("Funder A", tp=1, fp=1, fn=1, n_in_scope=2, model="model-1")
    r2 = _run("Funder A", tp=1, fp=1, fn=1, n_in_scope=2, model="model-2")

    configs = aggmod.aggregate_runs([r1, r2])
    assert len(configs) == 2
    assert {c.key.model for c in configs} == {"model-1", "model-2"}
    # sorted by ConfigKey.label
    assert [c.key.label for c in configs] == sorted(c.key.label for c in configs)


def test_config_key_label_format():
    key = aggmod.ConfigKey(
        model="deepseek-v4-flash-0731",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="shallow",
    )
    assert key.label == "deepseek-v4-flash-0731 | minimal | t0.0 | shallow"


def test_schema_valid_none_lands_in_n_unknown_and_leaves_the_rate_denominator():
    valid = _run("Funder A", tp=1, fp=0, fn=0, n_in_scope=1, schema_valid=True)
    unknown = _run(
        "Funder A", tp=1, fp=0, fn=0, n_in_scope=1, result=_result("Funder A", replicate=2, schema_valid=None)
    )

    [config] = aggmod.aggregate_runs([valid, unknown])
    conf = config.conformance
    assert conf.n_valid == 1
    assert conf.n_invalid == 0
    assert conf.n_unknown == 1
    assert conf.rate_undefined is False
    assert conf.rate == 1.0  # denominator is n_valid + n_invalid = 1, not 2


def test_conformance_rate_undefined_with_no_true_or_false_runs():
    unknown_only = _run(
        "Funder A", tp=0, fp=0, fn=1, n_in_scope=1, schema_valid=None
    )
    [config] = aggmod.aggregate_runs([unknown_only])
    conf = config.conformance
    assert conf.n_unknown == 1
    assert conf.rate_undefined is True
    assert conf.rate == 0.0


def test_abstaining_run_with_zero_opportunities_counts_as_conformant():
    abstainer = _run(
        "Funder A",
        tp=0,
        fp=0,
        fn=0,
        n_in_scope=0,
        n_extracted=0,
        schema_valid=True,
        abstained=True,
    )
    [config] = aggmod.aggregate_runs([abstainer])
    conf = config.conformance
    assert conf.n_valid == 1
    assert conf.rate == 1.0
    assert conf.n_abstained == 1


def test_conformance_counts_submit_failed_and_budget_exhausted():
    r = _run(
        "Funder A",
        tp=0,
        fp=1,
        fn=1,
        n_in_scope=1,
        schema_valid=False,
        submit_failed=True,
        step_budget_exhausted=True,
    )
    [config] = aggmod.aggregate_runs([r])
    conf = config.conformance
    assert conf.n_invalid == 1
    assert conf.n_submit_failed == 1
    assert conf.n_budget_exhausted == 1


def test_cost_per_tp_undefined_when_tp_totals_zero():
    r = _run(
        "Funder A",
        tp=0,
        fp=0,
        fn=0,
        n_in_scope=0,
        n_extracted=0,
        cost_usd=12.5,
        input_tokens=1_100_000,
        output_tokens=200,
        latency_s=30.0,
    )
    [config] = aggmod.aggregate_runs([r])
    cost = config.cost
    assert cost.cost_per_tp is None
    assert cost.cost_per_tp_undefined is True
    assert cost.total_cost_usd == 12.5
    assert cost.total_input_tokens == 1_100_000


def test_cost_aggregate_sums_and_reports_missing_fields():
    with_cost = _run(
        "Funder A", tp=1, fp=0, fn=0, n_in_scope=1, cost_usd=1.0, input_tokens=10,
        output_tokens=5, latency_s=2.0,
    )
    without_cost = _run(
        "Funder A", tp=1, fp=0, fn=0, n_in_scope=1,
        result=_result("Funder A", replicate=2),  # cost_usd/latency_s default None
    )
    [config] = aggmod.aggregate_runs([with_cost, without_cost])
    cost = config.cost
    assert cost.n_runs == 2
    assert cost.total_cost_usd == 1.0
    assert cost.n_missing_cost == 1
    assert cost.n_missing_latency == 1
    assert cost.mean_latency_s == 2.0
    # total_tp across both runs is 2
    assert cost.cost_per_tp == pytest.approx(0.5)
    assert cost.cost_per_tp_undefined is False


def test_micro_precision_undefined_when_the_whole_config_extracted_nothing():
    """If nothing was extracted, micro precision is undefined but micro recall is not."""
    r = _run("Funder A", tp=0, fp=0, fn=3, n_in_scope=3, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.micro_precision_undefined is True
    assert config.micro_precision == 0.0
    assert config.micro_recall == 0.0
    assert config.micro_f1 == 0.0
    # recall/F1 stay DEFINED here: this funder is included (n_in_scope=3), so the
    # summed n_in_scope denominator is positive even though nothing was extracted.
    assert config.micro_recall_undefined is False
    assert config.micro_f1_undefined is False


def test_micro_recall_and_f1_undefined_when_every_funder_is_zero_positive():
    """With every funder zero-positive, micro recall and F1 are undefined as well."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.macro_f1_undefined is True
    assert config.micro_precision_undefined is True
    assert config.micro_recall_undefined is True
    assert config.micro_f1_undefined is True
    # the value stays 0.0 on the dataclass, only the flag marks it as undefined
    assert config.micro_recall == 0.0
    assert config.micro_f1 == 0.0


def test_stable_rate_is_one_when_all_replicates_surface_the_same_opportunities():
    ids = ["opp-1", "opp-2", "opp-3"]
    runs = [
        _run_with_items("Funder A", tp_ids=ids, fn_ids=[], replicate=rep)
        for rep in range(1, 6)
    ]
    [config] = aggmod.aggregate_runs(runs)
    consistency = config.consistency["Funder A"]
    assert consistency.n_replicates == 5
    assert consistency.n_surfaced_any == 3
    assert consistency.n_stable == 3
    assert consistency.stable_rate == 1.0


def test_stable_rate_is_half_when_one_opportunity_surfaces_four_times_and_one_twice():
    # opp-A: surfaced (TP) in reps 1-4, missed (FN) in rep 5 -> surfaced 4/5
    # opp-B: surfaced (TP) in reps 1-2, missed (FN) in reps 3-5 -> surfaced 2/5
    runs = [
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=1),
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=2),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=3),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=4),
        _run_with_items("Funder A", tp_ids=[], fn_ids=["opp-A", "opp-B"], replicate=5),
    ]
    [config] = aggmod.aggregate_runs(runs)
    consistency = config.consistency["Funder A"]
    assert consistency.n_surfaced_any == 2
    assert consistency.n_stable == 1  # only opp-A clears the >=4-of-5 threshold
    assert consistency.stable_rate == 0.5


def test_single_replicate_sets_fleiss_kappa_undefined():
    runs = [_run_with_items("Funder A", tp_ids=["opp-1"], fn_ids=["opp-2"], replicate=1)]
    [config] = aggmod.aggregate_runs(runs)
    consistency = config.consistency["Funder A"]
    assert consistency.n_replicates == 1
    assert consistency.fleiss_kappa is None
    assert consistency.fleiss_kappa_undefined is True


def test_perfect_agreement_kappa_nan_surfaces_as_none_not_nan():
    # Every replicate surfaces exactly the same 3 opportunities -> statsmodels'
    # fleiss_kappa is degenerate (nan) because there is no variability at all in the
    # population: every item was unanimously "surfaced" by every rater.
    ids = ["opp-1", "opp-2", "opp-3"]
    runs = [
        _run_with_items("Funder A", tp_ids=ids, fn_ids=[], replicate=rep)
        for rep in range(1, 6)
    ]
    [config] = aggmod.aggregate_runs(runs)
    consistency = config.consistency["Funder A"]
    assert consistency.fleiss_kappa_undefined is True
    assert consistency.fleiss_kappa is None


def test_partial_agreement_produces_finite_kappa_strictly_between_minus_one_and_one():
    runs = [
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=1),
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=2),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=3),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=4),
        _run_with_items("Funder A", tp_ids=[], fn_ids=["opp-A", "opp-B"], replicate=5),
    ]
    [config] = aggmod.aggregate_runs(runs)
    consistency = config.consistency["Funder A"]
    assert consistency.fleiss_kappa_undefined is False
    assert consistency.fleiss_kappa is not None
    assert -1.0 < consistency.fleiss_kappa < 1.0

    # Check the exact value too, since a transposed table would also land in (-1, 1).
    # Worked by hand: 2 items, 5 raters, opp-A surfaced by 4 and opp-B by 2.
    #   P_A = (1^2 + 4^2 - 5) / (5*4) = 12/20 = 0.6
    #   P_B = (3^2 + 2^2 - 5) / (5*4) =  8/20 = 0.4      -> P-bar  = 0.5
    #   p(not surfaced) = (1+3)/10 = 0.4; p(surfaced) = (4+2)/10 = 0.6
    #   P-bar_e = 0.4^2 + 0.6^2 = 0.52
    #   kappa = (0.5 - 0.52) / (1 - 0.52) = -0.02/0.48 = -0.0416667
    assert consistency.fleiss_kappa == pytest.approx(-0.0416667)


def test_funder_that_surfaced_nothing_is_excluded_from_mean_stable_rate():
    included = [
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=1),
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=2),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=3),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=4),
        _run_with_items("Funder A", tp_ids=[], fn_ids=["opp-A", "opp-B"], replicate=5),
    ]
    # Funder B has in-scope rows but never surfaces either of them in any replicate.
    never_found = [
        _run_with_items("Funder B", tp_ids=[], fn_ids=["opp-X", "opp-Y"], replicate=rep)
        for rep in range(1, 6)
    ]
    [config] = aggmod.aggregate_runs(included + never_found)

    never_found_consistency = config.consistency["Funder B"]
    assert never_found_consistency.n_surfaced_any == 0
    assert never_found_consistency.stable_rate_undefined is True

    # mean_stable_rate is Funder A's 0.5 alone, not averaged with Funder B's undefined 0.0
    assert config.mean_stable_rate == 0.5
    assert config.mean_stable_rate_undefined is False


def test_zero_positive_funder_excluded_from_mean_fleiss_kappa():
    # Funder Z has no in-scope items, so kappa is undefined for it.
    with_kappa = [
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=1),
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=2),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=3),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=4),
        _run_with_items("Funder A", tp_ids=[], fn_ids=["opp-A", "opp-B"], replicate=5),
    ]
    zero_positive = [_run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)]

    [config] = aggmod.aggregate_runs(with_kappa + zero_positive)

    zero_positive_consistency = config.consistency["Funder Z"]
    assert zero_positive_consistency.fleiss_kappa_undefined is True
    assert zero_positive_consistency.fleiss_kappa is None

    # mean_fleiss_kappa is Funder A's value alone
    assert config.mean_fleiss_kappa == config.consistency["Funder A"].fleiss_kappa
    assert config.mean_fleiss_kappa_undefined is False


def test_aggregate_to_dict_carries_macro_f1_conformance_and_cost_per_tp():
    r = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10, schema_valid=True, cost_usd=1.0)
    [config] = aggmod.aggregate_runs([r])

    payload = aggmod.aggregate_to_dict(config)

    assert round(payload["macro_f1"], 4) == 0.5
    assert payload["conformance"]["rate"] == 1.0
    assert payload["cost"]["cost_per_tp"] == pytest.approx(0.2)  # 1.0 / tp=5
    # the payload must be JSON-serialisable as-is
    json.dumps(payload)


def test_aggregate_to_dict_serialises_censored_cost_per_tp_as_none():
    # tp totals zero across the whole config -> cost_per_tp is None, not 0.0
    r = _run(
        "Funder A", tp=0, fp=0, fn=1, n_in_scope=1, n_extracted=0, cost_usd=5.0
    )
    [config] = aggmod.aggregate_runs([r])

    payload = aggmod.aggregate_to_dict(config)
    assert payload["cost"]["cost_per_tp"] is None
    assert payload["cost"]["cost_per_tp_undefined"] is True

    # and it survives an actual JSON round trip as `null`, not `0.0`
    reloaded = json.loads(json.dumps(payload))
    assert reloaded["cost"]["cost_per_tp"] is None


def test_aggregate_to_dict_serialises_undefined_micro_recall_and_f1_as_none():
    """With every funder zero-positive, micro recall and F1 are null in the JSON."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])

    payload = aggmod.aggregate_to_dict(config)
    assert payload["micro_recall"] is None
    assert payload["micro_recall_undefined"] is True
    assert payload["micro_f1"] is None
    assert payload["micro_f1_undefined"] is True

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["micro_recall"] is None
    assert reloaded["micro_f1"] is None


def test_aggregate_to_dict_nulls_macro_fields_when_every_funder_is_zero_positive():
    """Macro precision, recall and F1 are null in the JSON when every funder is zero-positive."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.macro_f1_undefined is True  # sanity: the in-memory flag is set

    payload = aggmod.aggregate_to_dict(config)
    assert payload["macro_precision"] is None
    assert payload["macro_recall"] is None
    assert payload["macro_f1"] is None
    assert payload["macro_f1_undefined"] is True

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["macro_precision"] is None
    assert reloaded["macro_recall"] is None
    assert reloaded["macro_f1"] is None


def test_aggregate_to_dict_nulls_micro_precision_when_nothing_extracted():
    """micro_precision is null in the JSON when nothing was extracted; recall and F1 stay 0.0."""
    r = _run("Funder A", tp=0, fp=0, fn=3, n_in_scope=3, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.micro_precision_undefined is True
    assert config.micro_recall_undefined is False

    payload = aggmod.aggregate_to_dict(config)
    assert payload["micro_precision"] is None
    assert payload["micro_precision_undefined"] is True
    assert payload["micro_recall"] == 0.0
    assert payload["micro_f1"] == 0.0

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["micro_precision"] is None


def test_aggregate_to_dict_nulls_conformance_rate_when_only_unknown_runs():
    """The conformance rate is null in the JSON when every run has schema_valid=None."""
    unknown_only = _run("Funder A", tp=0, fp=0, fn=1, n_in_scope=1, schema_valid=None)
    [config] = aggmod.aggregate_runs([unknown_only])
    assert config.conformance.rate_undefined is True

    payload = aggmod.aggregate_to_dict(config)
    assert payload["conformance"]["rate"] is None
    assert payload["conformance"]["rate_undefined"] is True

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["conformance"]["rate"] is None


def test_aggregate_to_dict_nulls_mean_stable_rate_and_per_funder_stable_rate():
    """mean_stable_rate and the funder stable_rate are null in the JSON when undefined."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.mean_stable_rate_undefined is True
    assert config.consistency["Funder Z"].stable_rate_undefined is True

    payload = aggmod.aggregate_to_dict(config)
    assert payload["mean_stable_rate"] is None
    assert payload["mean_stable_rate_undefined"] is True
    assert payload["consistency"]["Funder Z"]["stable_rate"] is None
    assert payload["consistency"]["Funder Z"]["stable_rate_undefined"] is True

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["mean_stable_rate"] is None
    assert reloaded["consistency"]["Funder Z"]["stable_rate"] is None


def test_format_aggregate_report_renders_macro_and_micro_lines():
    r = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    [config] = aggmod.aggregate_runs([r])

    text = aggmod.format_aggregate_report(config)
    assert "detection (macro)" in text
    assert "precision 0.500" in text
    assert "detection (micro)" in text


def test_format_aggregate_report_renders_censored_cost_per_tp_as_dash_never_zero():
    r = _run("Funder A", tp=0, fp=0, fn=1, n_in_scope=1, n_extracted=0, cost_usd=5.0)
    [config] = aggmod.aggregate_runs([r])

    text = aggmod.format_aggregate_report(config)
    cost_line = next(line for line in text.splitlines() if "cost/TP" in line)
    assert "cost/TP —" in cost_line
    assert "0.000" not in cost_line


def test_format_aggregate_report_renders_micro_recall_and_f1_as_dash_when_all_zero_positive():
    """Micro recall and F1 print as dashes, not 0.000, when every funder is zero-positive."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])

    text = aggmod.format_aggregate_report(config)
    micro_line = next(line for line in text.splitlines() if line.startswith("detection (micro)"))
    assert "recall —" in micro_line
    assert "F1 —" in micro_line
    assert "0.000" not in micro_line


def test_format_aggregate_report_footer_names_excluded_funders_and_abstention():
    included = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    zero_positive = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([included, zero_positive])

    text = aggmod.format_aggregate_report(config)
    assert "Funder Z" in text
    assert "abstention rate 1.000" in text


def test_format_aggregate_report_by_funder_table_excludes_zero_positive_funder():
    """The by-funder table only lists included funders, zero-positive ones go in the footer.

    Otherwise a funder with nothing to detect would show a misleading 0.000 row.
    """
    included = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    zero_positive = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([included, zero_positive])

    without_table = aggmod.format_aggregate_report(config, by_funder=False)
    with_table = aggmod.format_aggregate_report(config, by_funder=True)

    assert "Funder A" not in without_table
    assert "Funder A" in with_table
    # the table itself must not carry the zero-positive funder as a detection row
    table_section = with_table.split("excluded (zero-positive)")[0]
    assert "Funder Z" not in table_section


def _write_ground_truth_csv(directory, slug, rows):
    """Write a fake ground-truth CSV named `<slug>_<date>.csv` so groundtruth.py accepts it."""
    path = directory / f"{slug}_2026-08-21.csv"
    fieldnames = sorted(REQUIRED_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def _gt_row(opportunity_id, funder, title, url, *, in_scope="yes"):
    return {
        "opportunity_id": opportunity_id,
        "funder": funder,
        "title": title,
        "url": url,
        "status": "open",
        "deadlines": "",
        "in_scope": in_scope,
        "edge_case_tags": "",
        "uncertain_fields": "",
    }


def _write_result(path, funder, opportunities, **kw):
    result = _result(funder, opportunities=opportunities, **kw)
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return path


def _extracted(funder, oid, title, url) -> ExtractedOpportunity:
    return ExtractedOpportunity(funder=funder, opportunity_id=oid, title=title, url=url)


def test_main_scores_a_directory_of_two_configs_and_prints_both(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "one.json", "Funder A", [hit], model="model-1")
    _write_result(results_dir / "two.json", "Funder A", [hit], model="model-2")

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model-1" in out
    assert "model-2" in out


def test_main_json_output_parses_and_carries_macro_f1_conformance_cost_per_tp(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "one.json", "Funder A", [hit], cost_usd=2.0)

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    [config] = payload
    assert config["macro_f1"] == 1.0
    assert "rate" in config["conformance"]
    assert config["cost"]["cost_per_tp"] == pytest.approx(2.0)


def test_main_cli_censored_cost_per_tp_renders_dash_in_text_and_null_in_json(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # the agent extracted nothing at all -> tp totals zero for this config -> censored
    _write_result(results_dir / "one.json", "Funder A", [], cost_usd=5.0)

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload[0]["cost"]["cost_per_tp"] is None

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    cost_line = next(line for line in out.splitlines() if "cost/TP" in line)
    assert "cost/TP —" in cost_line
    assert "0.000" not in cost_line


def test_main_returns_1_on_empty_directory(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    rc = aggmod.main([str(empty_dir), "--ground-truth", str(gt_dir)])
    assert rc == 1
    assert "no result files found" in capsys.readouterr().err


def test_main_skips_unlabelled_funder_and_continues(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "good.json", "Funder A", [hit])
    _write_result(results_dir / "bad.json", "Not A Labelled Funder", [])

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir)])
    err = capsys.readouterr().err
    assert rc == 0
    assert "no ground truth for funder" in err


def test_main_by_funder_flag_adds_per_funder_table(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "one.json", "Funder A", [hit])

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir)])
    without_table = capsys.readouterr().out
    assert rc == 0

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir), "--by-funder"])
    with_table = capsys.readouterr().out
    assert rc == 0

    assert len(with_table) > len(without_table)
    assert "Funder A" in with_table


# A run that extracted nothing has an undefined precision. It must not be averaged
# into the means as 0.0, or it looks like a real (bad) score.
def _abstaining_replicate(funder, ids, replicate):
    """A replicate that extracted nothing, so its precision is undefined (stored as 0.0)."""
    return _run_with_items(funder, tp_ids=[], fn_ids=list(ids), replicate=replicate)


def test_funder_precision_excludes_replicates_whose_precision_is_undefined():
    """Abstaining replicates are left out of the precision mean but still count for recall/F1.

    Four abstainers and one perfect run give precision 1.0 (not 0.2), and recall/F1 of 0.2.
    """
    ids = ["opp-1", "opp-2"]
    runs = [_abstaining_replicate("Funder A", ids, rep) for rep in range(1, 5)]
    runs.append(_run_with_items("Funder A", tp_ids=ids, fn_ids=[], replicate=5))

    [config] = aggmod.aggregate_runs(runs)
    [fa] = config.funders

    assert fa.n_replicates == 5
    assert fa.n_precision_undefined == 4
    assert fa.precision_undefined is False  # one replicate DID have a precision
    assert fa.precision == pytest.approx(1.0)
    assert fa.recall == pytest.approx(0.2)
    assert fa.f1 == pytest.approx(0.2)


def test_funder_precision_undefined_when_every_replicate_extracted_nothing():
    """If no replicate extracted anything, funder precision is undefined but macro F1 is not."""
    ids = ["opp-1", "opp-2"]
    runs = [_abstaining_replicate("Funder A", ids, rep) for rep in (1, 2)]

    [config] = aggmod.aggregate_runs(runs)
    [fa] = config.funders

    assert fa.n_precision_undefined == 2
    assert fa.precision_undefined is True
    assert fa.precision == 0.0
    assert fa.recall == 0.0
    assert fa.f1 == 0.0
    assert config.macro_f1_undefined is False


def test_funder_aggregate_carries_the_per_replicate_values_behind_its_means():
    """The funder aggregate keeps per-replicate values, with None for an undefined precision."""
    ids = ["opp-1", "opp-2"]
    runs = [_abstaining_replicate("Funder A", ids, rep) for rep in range(1, 5)]
    runs.append(_run_with_items("Funder A", tp_ids=ids, fn_ids=[], replicate=5))

    [config] = aggmod.aggregate_runs(runs)
    [fa] = config.funders

    assert fa.replicate_precision == [None, None, None, None, 1.0]
    assert fa.replicate_recall == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert fa.replicate_f1 == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_macro_precision_undefined_when_no_included_funder_has_a_defined_precision():
    """Macro precision is undefined when no included funder has a defined precision."""
    runs = [
        _abstaining_replicate("Funder A", ["a-1", "a-2"], 1),
        _abstaining_replicate("Funder B", ["b-1"], 1),
    ]
    [config] = aggmod.aggregate_runs(runs)

    assert config.macro_precision_undefined is True
    assert config.macro_precision == 0.0
    assert config.macro_f1_undefined is False
    assert config.macro_recall == 0.0
    assert config.macro_f1 == 0.0


def test_macro_precision_averages_only_the_funders_with_a_defined_precision():
    """Macro precision only averages funders with a defined precision (0.5, not 0.25)."""
    runs = [
        _abstaining_replicate("Funder A", ["a-1", "a-2"], 1),
        _run("Funder B", tp=1, fp=1, fn=1, n_in_scope=2),
    ]
    [config] = aggmod.aggregate_runs(runs)

    assert config.n_funders == 2
    assert config.macro_precision_undefined is False
    assert config.macro_precision == pytest.approx(0.5)
    # recall/F1 still average BOTH funders: A's 0.0 recall is real.
    assert config.macro_recall == pytest.approx(0.25)


def test_macro_precision_undefined_when_every_funder_is_zero_positive():
    """With no included funders, macro precision is flagged undefined along with macro F1."""
    r = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([r])
    assert config.macro_f1_undefined is True
    assert config.macro_precision_undefined is True


def test_report_macro_line_dashes_precision_when_no_funder_has_a_defined_one():
    """The macro line shows a dash for precision when no funder has one, but keeps recall/F1."""
    runs = [_abstaining_replicate("Funder A", ["a-1", "a-2"], 1)]
    [config] = aggmod.aggregate_runs(runs)

    text = aggmod.format_aggregate_report(config)
    macro_line = next(l for l in text.splitlines() if l.startswith("detection (macro)"))
    assert "precision —" in macro_line
    assert "precision 0.000" not in macro_line
    assert "recall 0.000" in macro_line
    assert "F1 0.000" in macro_line
    assert "precision undefined" in macro_line  # named, never silent


def test_report_by_funder_table_dashes_a_funder_with_no_defined_precision():
    """The by-funder table shows a dash for a funder with no defined precision."""
    runs = [
        _abstaining_replicate("Funder A", ["a-1", "a-2"], 1),
        _run("Funder B", tp=1, fp=1, fn=1, n_in_scope=2),
    ]
    [config] = aggmod.aggregate_runs(runs)

    text = aggmod.format_aggregate_report(config, by_funder=True)
    # Fixed-width row: 2 spaces + funder(40) + n(4) + reps(5) + prec(7).
    # Only check the precision cell, because A's recall and F1 really are 0.000.
    row_a = next(l for l in text.splitlines() if l.startswith("  Funder A"))
    row_b = next(l for l in text.splitlines() if l.startswith("  Funder B"))
    assert row_a[51:58].strip() == "—"     # no defined precision in any replicate
    assert row_b[51:58].strip() == "0.500"  # B's precision is real and still prints


def test_aggregate_to_dict_nulls_macro_precision_and_funder_precision_when_undefined():
    """Undefined macro and funder precision are null in the JSON, so pandas reads them as NaN."""
    runs = [_abstaining_replicate("Funder A", ["a-1", "a-2"], 1)]
    [config] = aggmod.aggregate_runs(runs)

    payload = aggmod.aggregate_to_dict(config)
    assert payload["macro_precision"] is None
    assert payload["macro_precision_undefined"] is True
    assert payload["macro_recall"] == 0.0  # defined, stays a real number
    [funder] = payload["funders"]
    assert funder["precision"] is None
    assert funder["precision_undefined"] is True
    assert funder["n_precision_undefined"] == 1

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["macro_precision"] is None
    assert reloaded["funders"][0]["precision"] is None


def test_aggregate_to_dict_emits_the_per_replicate_values_behind_each_funder_mean():
    """The JSON keeps per-replicate values, with null for an abstention rather than 0.0."""
    ids = ["opp-1", "opp-2"]
    runs = [_abstaining_replicate("Funder A", ids, rep) for rep in range(1, 5)]
    runs.append(_run_with_items("Funder A", tp_ids=ids, fn_ids=[], replicate=5))
    [config] = aggmod.aggregate_runs(runs)

    payload = aggmod.aggregate_to_dict(config)
    [funder] = payload["funders"]
    assert funder["replicate_precision"] == [None, None, None, None, 1.0]
    assert funder["replicate_recall"] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert funder["replicate_f1"] == [0.0, 0.0, 0.0, 0.0, 1.0]

    reloaded = json.loads(json.dumps(payload))
    assert reloaded["funders"][0]["replicate_precision"] == [None, None, None, None, 1.0]


# stable_rate is undefined, not 0.000, when there are fewer replicates than the
# threshold, because the >=4-of-5 bar cannot be reached.
def test_stable_rate_undefined_when_there_are_fewer_replicates_than_the_threshold():
    """With fewer replicates than the threshold, stable_rate is undefined rather than 0.0."""
    runs = [_run_with_items("Funder A", tp_ids=["opp-1"], fn_ids=[], replicate=1)]
    [config] = aggmod.aggregate_runs(runs)
    c = config.consistency["Funder A"]

    assert c.n_replicates == 1
    assert c.n_surfaced_any == 1
    assert c.stable_rate_undefined is True
    assert c.stable_rate == 0.0  # the 0.0 stays behind the flag


def test_stable_rate_is_defined_once_replicates_reach_the_threshold():
    """With exactly four replicates the stable rate is defined (the guard is `<`, not `<=`)."""
    runs = [
        _run_with_items("Funder A", tp_ids=["opp-1"], fn_ids=[], replicate=rep)
        for rep in range(1, 5)
    ]
    [config] = aggmod.aggregate_runs(runs)
    c = config.consistency["Funder A"]

    assert c.n_replicates == 4
    assert c.stable_rate_undefined is False
    assert c.stable_rate == 1.0


def test_mean_stable_rate_undefined_when_every_funder_has_too_few_replicates():
    """mean_stable_rate is undefined when every funder has too few replicates."""
    runs = [
        _run_with_items("Funder A", tp_ids=["a-1"], fn_ids=[], replicate=1),
        _run_with_items("Funder B", tp_ids=["b-1"], fn_ids=[], replicate=1),
    ]
    [config] = aggmod.aggregate_runs(runs)

    assert config.mean_stable_rate_undefined is True
    assert config.mean_stable_rate == 0.0


def test_report_and_json_dash_and_null_a_stable_rate_with_too_few_replicates():
    runs = [_run_with_items("Funder A", tp_ids=["opp-1"], fn_ids=[], replicate=1)]
    [config] = aggmod.aggregate_runs(runs)

    text = aggmod.format_aggregate_report(config, by_funder=True)
    rq3_line = next(l for l in text.splitlines() if l.startswith("consistency (RQ3)"))
    assert "mean stable-rate —" in rq3_line
    assert "0.000" not in rq3_line
    # columns: 2 + funder(40) + n(4) + reps(5) + prec(7) + rec(7) + F1(7) + stable(8)
    row = next(l for l in text.splitlines() if l.startswith("  Funder A"))
    assert row[72:80].strip() == "—"  # the per-funder stable column

    payload = aggmod.aggregate_to_dict(config)
    assert payload["mean_stable_rate"] is None
    assert payload["mean_stable_rate_undefined"] is True
    assert payload["consistency"]["Funder A"]["stable_rate"] is None
    assert json.loads(json.dumps(payload))["mean_stable_rate"] is None


def test_stable_threshold_is_a_parameter_defaulting_to_the_module_constant():
    """`threshold` defaults to STABLE_THRESHOLD, can be lowered, and is stored on the aggregate."""
    assert aggmod.STABLE_THRESHOLD == 4
    runs = [
        _run_with_items("Funder A", tp_ids=["opp-A", "opp-B"], fn_ids=[], replicate=1),
        _run_with_items("Funder A", tp_ids=["opp-A"], fn_ids=["opp-B"], replicate=2),
    ]

    [default_config] = aggmod.aggregate_runs(runs)
    assert default_config.consistency["Funder A"].stable_rate_undefined is True
    assert default_config.consistency["Funder A"].threshold == 4

    [lowered] = aggmod.aggregate_runs(runs, threshold=2)
    c = lowered.consistency["Funder A"]
    assert c.threshold == 2
    assert c.stable_rate_undefined is False
    assert c.n_surfaced_any == 2
    assert c.n_stable == 1  # only opp-A appears in both replicates
    assert c.stable_rate == 0.5
    assert aggmod.aggregate_to_dict(lowered)["consistency"]["Funder A"]["threshold"] == 2


# Missing cost data should show as unknown, not as $0.0000.
def test_total_cost_undefined_when_no_run_reported_a_cost():
    """Total cost is undefined when no run reported a cost, instead of summing to 0.0."""
    runs = [
        _run("Funder A", tp=1, fp=0, fn=1, n_in_scope=2),  # cost_usd defaults to None
        _run("Funder A", tp=1, fp=0, fn=1, n_in_scope=2, result=_result("Funder A", replicate=2)),
    ]
    [config] = aggmod.aggregate_runs(runs)
    cost = config.cost

    assert cost.n_missing_cost == 2
    assert cost.n_runs == 2
    assert cost.total_cost_undefined is True
    assert cost.total_cost_usd == 0.0  # coerced value stays behind the flag
    # cost/TP is total_cost / total_tp: with no cost data at all its numerator is
    # unknown, so the ratio is unknown too, not "free per true positive".
    assert cost.cost_per_tp is None
    assert cost.cost_per_tp_undefined is True


def test_a_genuine_zero_cost_is_not_flagged_undefined():
    """A reported cost of 0.0 is a real value, so it is not flagged undefined."""
    r = _run("Funder A", tp=1, fp=0, fn=0, n_in_scope=1, cost_usd=0.0, latency_s=0.02)
    [config] = aggmod.aggregate_runs([r])

    assert config.cost.n_missing_cost == 0
    assert config.cost.total_cost_undefined is False
    assert config.cost.total_cost_usd == 0.0
    assert config.cost.cost_per_tp == 0.0
    assert config.cost.cost_per_tp_undefined is False


def test_cost_line_renders_missing_counts_and_dashes_an_unknown_total():
    runs = [
        _run("Funder A", tp=1, fp=0, fn=1, n_in_scope=2),
        _run("Funder A", tp=1, fp=0, fn=1, n_in_scope=2, result=_result("Funder A", replicate=2)),
    ]
    [config] = aggmod.aggregate_runs(runs)

    line = next(l for l in aggmod.format_aggregate_report(config).splitlines()
                if l.startswith("cost"))
    assert "total —" in line
    assert "$0.0000" not in line
    assert "cost/TP —" in line
    assert "missing cost 2/2" in line
    assert "missing latency 2/2" in line


def test_cost_line_reports_partial_telemetry_gaps_beside_a_real_total():
    """With partial cost data the total still prints, along with how many runs are missing."""
    runs = [
        _run("Funder A", tp=1, fp=0, fn=0, n_in_scope=1, cost_usd=2.0, latency_s=4.0),
        _run("Funder A", tp=1, fp=0, fn=0, n_in_scope=1, result=_result("Funder A", replicate=2)),
    ]
    [config] = aggmod.aggregate_runs(runs)

    line = next(l for l in aggmod.format_aggregate_report(config).splitlines()
                if l.startswith("cost"))
    assert "total $2.0000" in line
    assert "missing cost 1/2" in line
    assert "missing latency 1/2" in line


def test_aggregate_to_dict_nulls_an_undefined_total_cost():
    runs = [_run("Funder A", tp=1, fp=0, fn=1, n_in_scope=2)]
    [config] = aggmod.aggregate_runs(runs)

    payload = aggmod.aggregate_to_dict(config)
    assert payload["cost"]["total_cost_usd"] is None
    assert payload["cost"]["total_cost_undefined"] is True
    assert json.loads(json.dumps(payload))["cost"]["total_cost_usd"] is None


def test_report_header_distinguishes_funders_in_the_macro_mean_from_total_runs():
    """The header separates funders in the macro mean from the total number of runs."""
    included = _run("Funder A", tp=5, fp=5, fn=5, n_in_scope=10)
    zero_positive = _run("Funder Z", tp=0, fp=0, fn=0, n_in_scope=0, n_extracted=0)
    [config] = aggmod.aggregate_runs([included, zero_positive])

    header = aggmod.format_aggregate_report(config).splitlines()[3]
    assert config.n_funders == 1 and config.n_runs == 2
    assert "1 funder(s) in the macro mean" in header
    assert "1 excluded (zero-positive)" in header
    assert "2 run(s) across all funders" in header


def test_main_show_rejected_lists_the_ground_truth_csvs_that_were_not_loaded(tmp_path, capsys):
    """--show-rejected lists ground-truth CSVs that were not loaded, like score.py does."""
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )
    # A CSV that does NOT follow <funder-slug>_<YYYY-MM-DD>.csv -> rejected.
    (gt_dir / "summary-table.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "one.json", "Funder A", [hit])

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir), "--show-rejected"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rejected summary-table.csv" in out
    assert "filename is not" in out
    assert "1 file(s)" in out  # the loaded ground-truth summary, as score.py prints it


def test_main_without_show_rejected_stays_quiet_about_them(tmp_path, capsys):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_ground_truth_csv(
        gt_dir, "funder-a", [_gt_row("a-1", "Funder A", "Grant A", "https://a.org/x")]
    )
    (gt_dir / "summary-table.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hit = _extracted("Funder A", "a-1", "Grant A", "https://a.org/x")
    _write_result(results_dir / "one.json", "Funder A", [hit])

    rc = aggmod.main([str(results_dir), "--ground-truth", str(gt_dir)])
    assert rc == 0
    assert "rejected" not in capsys.readouterr().out
