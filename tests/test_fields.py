"""Tests for structured-field accuracy and the end-to-end result (`analysis/fields.py`).

Everything here depends on rebuilding per-opportunity field outcomes from the frozen
results plus the ground truth's `uncertain_fields`; the first test checks that.

"""
from __future__ import annotations

import json

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import fields as fl
from grant_eval.analysis import frontier as fr
from grant_eval.analysis import telemetry as tel
from grant_eval.scoring.groundtruth import load_ground_truth


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


@pytest.fixture(scope="module")
def gt():
    return load_ground_truth()


@pytest.fixture(scope="module")
def raw_lines():
    rows = []
    with open(ds.DEFAULT_SCORED_DIR / "runs.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_reconstructed_outcomes_match_the_recorded_counts(data, gt, raw_lines):
    """evaluated = true positives minus uncertain-flagged; incorrect = mismatch ids.

    Checked against the counts the scorer recorded for every run and field, so the
    per-tag field accuracy and end-to-end rate are exact rather than estimates.

    """
    uncertain = fl._uncertain_index(gt)
    by_key = {(r.config_hash, r.funder, r.replicate): r for r in data.runs}

    checked = 0
    for payload in raw_lines:
        key = (payload["config_hash"], payload["funder"], payload["replicate"])
        run = by_key[key]
        for field in fl.FIELDS:
            block = payload["fields"][field]
            evaluated = frozenset(
                oid for oid in run.true_positive_ids
                if field not in uncertain.get(oid, frozenset())
            )
            mism = {m[0] for m in block.get("mismatches", [])}
            assert len(evaluated) == block["correct"] + block["incorrect"], (key, field)
            assert len(mism) == block["incorrect"], (key, field)
            assert mism <= evaluated, (key, field)
            checked += 1
    assert checked == len(raw_lines) * len(fl.FIELDS)


def test_excluded_no_target_is_zero_throughout(raw_lines):
    """The reconstruction assumes no rows are excluded for a missing target, so that is checked."""
    for payload in raw_lines:
        for field in fl.FIELDS:
            assert payload["fields"][field]["excluded_no_target"] == 0


def test_field_accuracy_matches_the_recorded_totals(data, gt, raw_lines):
    """Pooled accuracy must equal the scorer's own pooled correct/incorrect."""
    expected_correct: dict[tuple[str, str], int] = {}
    expected_incorrect: dict[tuple[str, str], int] = {}
    for payload in raw_lines:
        model = payload["model"]
        for field in fl.FIELDS:
            b = payload["fields"][field]
            expected_correct[(model, field)] = expected_correct.get((model, field), 0) + b["correct"]
            expected_incorrect[(model, field)] = (
                expected_incorrect.get((model, field), 0) + b["incorrect"]
            )
    for r in fl.field_accuracy(data, gt, key="model"):
        assert r.n_correct == expected_correct[(r.label, r.field)]
        assert r.n_incorrect == expected_incorrect[(r.label, r.field)]
        assert r.accuracy == pytest.approx(r.n_correct / r.n_evaluated)


def test_deadlines_is_the_weakest_field_for_every_model(data, gt):
    """Detection F1 is not extraction quality, and deadlines are the worst field."""
    by = {(r.label, r.field): r for r in fl.field_accuracy(data, gt, key="model")}
    for model in {label for label, _ in by}:
        deadlines = by[(model, "deadlines")].accuracy
        assert deadlines is not None
        assert deadlines < by[(model, "status")].accuracy


def test_field_accuracy_is_none_not_zero_when_nothing_was_evaluated(data, gt):
    for r in fl.field_accuracy(data, gt, runs=(), key="model"):
        assert r.n_evaluated == 0
        assert r.accuracy is None


def test_field_accuracy_rejects_an_unknown_key(data, gt):
    with pytest.raises(ValueError, match="key must be one of"):
        fl.field_accuracy(data, gt, key="funder")


def test_field_accuracy_raises_on_an_artefact_it_cannot_join(data, gt, tmp_path):
    """A mismatch between the dataset and scored_dir raises rather than quietly counting as zero."""
    (tmp_path / "runs.jsonl").write_text("")
    with pytest.raises(KeyError, match="no raw fields"):
        fl.field_accuracy(data, gt, key="model", scored_dir=tmp_path)


def test_factor_cells_partition_the_model_totals(data, gt):
    """The four prompt cells for a model add back up to the model's own evaluated total."""
    model_level = {
        (r.label, r.field): r for r in fl.field_accuracy(data, gt, key="model")
    }
    cells = fl.field_accuracy_by_factor(data, gt, "prompt")
    pooled: dict[tuple[str, str], list[int]] = {}
    for c in cells:
        acc = pooled.setdefault((c.model, c.field), [0, 0])
        acc[0] += c.n_evaluated
        acc[1] += c.n_correct
    for (model, field), (n_eval, n_correct) in pooled.items():
        assert n_eval == model_level[(model, field)].n_evaluated
        assert n_correct == model_level[(model, field)].n_correct


def test_temperature_is_also_a_valid_factor(data, gt):
    cells = fl.field_accuracy_by_factor(data, gt, "temperature")
    assert {c.level for c in cells} == {"t0.0", "t0.7"}


def test_field_accuracy_by_factor_rejects_an_unknown_factor(data, gt):
    with pytest.raises(ValueError, match="factor must be one of"):
        fl.field_accuracy_by_factor(data, gt, "funder")


def test_chain_of_thought_helps_deadlines_where_it_hurt_detection(data, gt):
    """On nova-lite, chain_of_thought is worst for finding opportunities but best for dating them.

    Same setting, opposite effects on the two measures, so a change that reverses this
    should fail here.

    """
    cells = {
        (c.model, c.level): c
        for c in fl.field_accuracy_by_factor(data, gt, "prompt")
        if c.field == "deadlines"
    }
    nova = "openrouter/amazon/nova-lite-v1"
    assert cells[(nova, "chain_of_thought")].accuracy > cells[(nova, "minimal")].accuracy
    assert cells[(nova, "chain_of_thought")].accuracy == max(
        c.accuracy for (m, _), c in cells.items() if m == nova
    )
    # nova-lite's detection F1 goes the other way: chain_of_thought is its worst.
    by_config = {r.config_label: r for r in data.runs}
    del by_config  # detection ranking is asserted in test_contrasts; not restated here

    deepseek = "openrouter/deepseek/deepseek-v4-flash-0731"
    assert cells[(deepseek, "chain_of_thought")].accuracy > cells[(deepseek, "minimal")].accuracy


def test_a_measured_zero_is_distinguishable_from_no_data(data, gt):
    """nova-lite's few_shot amount_max_gbp accuracy is 0.0 with n_evaluated > 0, not None.

    The figure labels this case because a zero-height bar looks like missing data.

    """
    cells = {
        (c.model, c.level, c.field): c
        for c in fl.field_accuracy_by_factor(data, gt, "prompt")
    }
    cell = cells[("openrouter/amazon/nova-lite-v1", "few_shot", "amount_max_gbp")]
    assert cell.n_evaluated > 0
    assert cell.accuracy == 0.0


def test_amount_max_errors_are_omissions_not_inventions(data):
    """No model ever gave a maximum amount the truth lacked, while all three left many out.

    Pooling both into one accuracy would hide that the field fails conservatively.

    """
    by = {(r.label, r.field): r for r in fl.field_directions(data, key="model")}
    for label, field in list(by):
        if field == "amount_max_gbp":
            assert by[(label, field)].asserted_where_blank == 0
            assert by[(label, field)].blank_where_value > 0


def test_deadline_errors_run_the_other_way_for_the_best_model(data):
    """DeepSeek adds deadlines that aren't there far more often than it misses real ones."""
    by = {(r.label, r.field): r for r in fl.field_directions(data, key="model")}
    d = by[("openrouter/deepseek/deepseek-v4-flash-0731", "deadlines")]
    assert d.asserted_where_blank > d.blank_where_value


# Field accuracy by edge-case tag.
def test_multi_deadline_costs_deadline_accuracy_severely(data, gt):
    """multi_deadline has a large effect on reading the deadline, though not on finding the opportunity.

    `edgecases.tag_contrast` finds no recall effect, but deadline accuracy drops a lot.

    """
    r = fl.field_accuracy_by_tag(data, gt, "deadlines", "multi_deadline", n_resamples=2000)
    assert r.accuracy_tagged < 0.30
    assert r.accuracy_untagged > 0.70
    assert r.difference.point < -0.4
    assert r.difference.upper < 0  # interval entirely below zero
    # Still descriptive under the paired-cluster rule, not a significance test.
    assert r.n_paired_clusters < 8
    assert r.reason


def test_the_effect_is_specific_to_the_deadline_field(data, gt):
    """multi_deadline does not lower every field (status is unaffected), so it is a deadline problem, not general difficulty."""
    deadlines = fl.field_accuracy_by_tag(data, gt, "deadlines", "multi_deadline", n_resamples=2000)
    status = fl.field_accuracy_by_tag(data, gt, "status", "multi_deadline", n_resamples=2000)
    assert deadlines.difference.point < status.difference.point - 0.4


def test_field_by_tag_is_undefined_when_no_funder_is_paired(data, gt):
    r = fl.field_accuracy_by_tag(data, gt, "deadlines", "multi_deadline", runs=(), n_resamples=200)
    assert r.n_paired_clusters == 0
    assert r.accuracy_tagged is None
    assert r.difference is None
    assert r.reason


def test_end_to_end_never_exceeds_detection(data, gt):
    """The end-to-end rate is at most the detection rate, since an opportunity must be found first."""
    for r in fl.end_to_end(data, gt, field="deadlines"):
        assert r.end_to_end_rate <= r.detection_rate + 1e-12


def test_requiring_all_fields_is_stricter_than_requiring_one(data, gt):
    single = {r.label: r for r in fl.end_to_end(data, gt, field="status")}
    every = {r.label: r for r in fl.end_to_end(data, gt, field="all")}
    for label, r in every.items():
        assert r.end_to_end_rate <= single[label].end_to_end_rate + 1e-12


def test_end_to_end_counts_missed_opportunities_as_failures(data, gt):
    """The end-to-end rate is below field accuracy for every config.

    Its denominator is every in-scope opportunity, not just the ones found.

    """
    e2e = {r.label: r for r in fl.end_to_end(data, gt, field="deadlines", key="config")}
    acc = {
        r.label: r
        for r in fl.field_accuracy(data, gt, key="config")
        if r.field == "deadlines"
    }
    for label, r in e2e.items():
        assert r.end_to_end_rate < acc[label].accuracy


def test_end_to_end_frontier_agrees_with_the_domain_free_pareto_function(data, gt):
    """The end-to-end frontier matches `frontier.pareto_frontier` given the same value axis."""
    rows = fl.end_to_end_frontier(fl.end_to_end(data, gt, field="all"))
    points = {r.label: (r.cost_usd, r.end_to_end_rate) for r in rows if r.end_to_end_rate is not None}
    expected = {p.config_label: p.on_frontier for p in fr.pareto_frontier(points)}
    for r in rows:
        assert r.on_frontier == expected[r.label]


def test_the_cheapest_frontier_config_is_the_one_the_f1_analysis_also_picks(data, gt):
    """`minimal | t0.7` comes out best on all three cost views.

    It is on the F1 frontier, the cheapest point on the end-to-end frontier, and the
    lowest cost per usable extraction among deployable configs.

    """
    rows = fl.end_to_end_frontier(fl.end_to_end(data, gt, field="all"))
    on = sorted((r for r in rows if r.on_frontier and r.end_to_end_rate > 0.1),
                key=lambda r: r.cost_usd)
    assert on[0].label == "openrouter/deepseek/deepseek-v4-flash-0731 | minimal | t0.7"
    deployable = [r for r in rows if r.cost_per_usable is not None and r.end_to_end_rate > 0.1]
    assert min(deployable, key=lambda r: r.cost_per_usable).label == on[0].label


def test_cost_per_usable_is_none_when_nothing_was_usable(data, gt):
    """nova-lite never gets all four fields right, so its unit cost is undefined, not zero or infinite."""
    rows = fl.end_to_end(data, gt, field="all")
    nova = [r for r in rows if "nova-lite" in r.label]
    assert nova
    assert any(r.end_to_end_rate == 0.0 and r.cost_per_usable is None for r in nova)


def test_runtime_health_matches_a_hand_count(raw_lines):
    health = tel.runtime_health()
    assert health.n_runs == len(raw_lines)
    assert health.n_schema_invalid == sum(
        1 for p in raw_lines if not p["telemetry"].get("schema_valid", True)
    )
    assert health.total_input_tokens == sum(
        p["telemetry"].get("input_tokens") or 0 for p in raw_lines
    )
    assert health.n_fp_unattributable == sum(p.get("fp_unattributable", 0) for p in raw_lines)


def test_schema_invalidity_concentrates_on_the_weakest_model():
    """About 8.4% of runs gave schema-invalid output, and most of those come from one model."""
    health = tel.runtime_health()
    assert health.n_schema_invalid > 0
    worst = max(health.by_model_schema_invalid.items(), key=lambda kv: kv[1])
    assert "nova-lite" in worst[0]
    assert worst[1] > health.n_schema_invalid / 2
