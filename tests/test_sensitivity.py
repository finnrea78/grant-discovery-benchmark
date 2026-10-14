"""Tests for run reliability and the sensitivity analysis that drops failed runs."""
from __future__ import annotations

import json

import pytest

from grant_eval.analysis import dataset as ds
from grant_eval.analysis import sensitivity as sens


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


def test_the_failure_breakdown_matches_the_recorded_counts(data):
    """There are 76 context overflows (all nova-lite) and 22 timeouts."""
    breakdown = sens.failure_breakdown(data)
    assert breakdown.n_runs == 1920
    assert breakdown.n_failed == 98
    assert breakdown.n_context_overflow == 76
    assert breakdown.n_timed_out == 22
    assert breakdown.by_model["openrouter/amazon/nova-lite-v1"] == 76
    assert breakdown.by_model["openrouter/deepseek/deepseek-v4-flash-0731"] == 22


def test_abstention_is_not_counted_as_a_failure(data):
    """The 216 abstentions are model behaviour, not failures."""
    breakdown = sens.failure_breakdown(data)
    assert breakdown.n_abstained == 216
    assert breakdown.n_failed == 98


def test_surviving_runs_drops_exactly_the_failures(data):
    surviving = sens.surviving_runs(data)
    assert len(surviving) == 1920 - 98
    assert not any(r.run_failed for r in surviving)


def test_the_sensitivity_analysis_reports_every_exclusion_per_config(data):
    """Each config reports how many runs it lost, and none loses all of them."""
    comparisons = sens.compare_with_and_without_failures(data)
    assert len(comparisons) == 24
    assert sum(c.n_excluded for c in comparisons) == 98
    n_undefined = sum(1 for c in comparisons if c.sensitivity is None)
    assert n_undefined == 0
    for c in comparisons:
        assert c.delta == pytest.approx(c.sensitivity - c.primary)


def test_excluding_failures_moves_nova_lite_most(data):
    """Dropping failed runs changes nova-lite's F1 more than DeepSeek's, since most failures are nova-lite's."""
    comparisons = {c.label: c for c in sens.compare_with_and_without_failures(data)}
    nova = [c for label, c in comparisons.items() if "nova-lite" in label]
    deepseek_t07 = [
        c for label, c in comparisons.items() if "deepseek" in label and "t0.7" in label
    ]
    assert max(c.delta for c in nova) > max(c.delta for c in deepseek_t07)


def test_a_config_losing_every_run_reports_an_undefined_sensitivity():
    """A config with every run failed has an undefined (None) sensitivity, not zero."""
    runs = tuple(
        _record(config_label="X", funder=f"F{i}", run_failed=True) for i in range(3)
    )
    data = ds.AnalysisDataset(runs=runs, manifest={})
    (comparison,) = sens.compare_with_and_without_failures(data)
    assert comparison.sensitivity is None
    assert comparison.n_excluded == 3


def test_temperature_stability_compares_replicate_spread(data):
    """RQ3: replicate spread at t0.0 and t0.7, paired within model, prompt and funder."""
    rows = sens.temperature_stability(data)
    assert len(rows) == 3 * 4 * 16
    assert all(r.delta == pytest.approx(r.replicate_sd_t07 - r.replicate_sd_t0) for r in rows)


def _record(**kw):
    base = dict(
        funder="A", model="m", prompt="p", temperature=0.0, replicate=1,
        config_label="X", config_hash="h", f1=0.5, precision=0.5, recall=0.5,
        tp=1, fp=0, fn=1, n_extracted=1, n_in_scope=2, hallucination_rate=0.0,
        n_fabricated=0, fp_by_reason={}, true_positive_ids=(), false_negative_ids=(),
        cost_usd=0.1, latency_s=1.0, run_failed=False, abstained=False, split="tune",
    )
    return ds.RunRecord(**{**base, **kw})


def test_temperature_stability_pairs_within_the_same_funder():
    """Both SDs in a row come from the same funder.

    Funder A is constant at t0.0 and varies at t0.7; funder B is the other way round,
    so pairing the wrong funders would show up.
    """
    runs = []
    b_variable = [0.0, 0.0, 0.0, 0.0, 1.0]  # population sd = 0.4
    for i in range(1, 6):
        # Funder A: constant at t0.0, variable at t0.7.
        runs.append(
            _record(funder="A", model="m", prompt="p", temperature=0.0,
                    replicate=i, config_label=ds.config_label("m", "p", 0.0), f1=0.5)
        )
        runs.append(
            _record(funder="A", model="m", prompt="p", temperature=0.7,
                    replicate=i, config_label=ds.config_label("m", "p", 0.7),
                    f1=b_variable[i - 1])
        )
        # Funder B: variable at t0.0, constant at t0.7.
        runs.append(
            _record(funder="B", model="m", prompt="p", temperature=0.0,
                    replicate=i, config_label=ds.config_label("m", "p", 0.0),
                    f1=b_variable[i - 1])
        )
        runs.append(
            _record(funder="B", model="m", prompt="p", temperature=0.7,
                    replicate=i, config_label=ds.config_label("m", "p", 0.7), f1=0.5)
        )
    data = ds.AnalysisDataset(runs=tuple(runs), manifest={})
    rows = {r.funder: r for r in sens.temperature_stability(data)}
    assert len(rows) == 2
    assert rows["A"].replicate_sd_t0 == pytest.approx(0.0)
    assert rows["A"].replicate_sd_t07 == pytest.approx(0.4)
    assert rows["B"].replicate_sd_t0 == pytest.approx(0.4)
    assert rows["B"].replicate_sd_t07 == pytest.approx(0.0)


def _write_runs_jsonl(dir_path, telemetry_rows):
    """Write a small `runs.jsonl` with just the fields `_raw_failure_flags` reads."""
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / "runs.jsonl").open("w") as handle:
        for row in telemetry_rows:
            handle.write(json.dumps(row) + "\n")


def _telemetry_row(config_hash, funder, replicate, timed_out, context_overflow):
    return {
        "config_hash": config_hash,
        "funder": funder,
        "replicate": replicate,
        "telemetry": {"timed_out": timed_out, "context_overflow": context_overflow},
    }


def test_failure_breakdown_reads_the_given_scored_dir_not_the_default(tmp_path):
    """`failure_breakdown` reads telemetry from `scored_dir`, not the default directory.

    The two directories share a key but record different failure types.
    """
    key = dict(config_hash="synthetic-hash-a", funder="Synthetic Funder", replicate=1)
    failed_run = _record(config_label="X", run_failed=True, **key)
    data = ds.AnalysisDataset(runs=(failed_run,), manifest={})

    dir_timeout = tmp_path / "scored_timeout"
    dir_overflow = tmp_path / "scored_overflow"
    _write_runs_jsonl(dir_timeout, [_telemetry_row(**key, timed_out=True, context_overflow=False)])
    _write_runs_jsonl(dir_overflow, [_telemetry_row(**key, timed_out=False, context_overflow=True)])

    breakdown_timeout = sens.failure_breakdown(data, scored_dir=dir_timeout)
    breakdown_overflow = sens.failure_breakdown(data, scored_dir=dir_overflow)

    assert (breakdown_timeout.n_timed_out, breakdown_timeout.n_context_overflow) == (1, 0)
    assert (breakdown_overflow.n_timed_out, breakdown_overflow.n_context_overflow) == (0, 1)


def test_a_scored_dir_missing_a_failed_runs_key_raises_naming_it(tmp_path):
    """A scored dir missing a failed run's key raises a KeyError naming the key."""
    key = dict(config_hash="synthetic-hash-missing", funder="Nobody Here", replicate=7)
    failed_run = _record(config_label="X", run_failed=True, **key)
    data = ds.AnalysisDataset(runs=(failed_run,), manifest={})

    empty_dir = tmp_path / "scored_empty"
    _write_runs_jsonl(empty_dir, [])

    with pytest.raises(KeyError) as exc_info:
        sens.failure_breakdown(data, scored_dir=empty_dir)
    message = str(exc_info.value)
    assert "synthetic-hash-missing" in message
    assert "Nobody Here" in message
    assert "7" in message


def test_paired_with_a_runs_filter_names_a_funder_dropped_entirely():
    """`paired(runs=...)` lists a funder with no surviving runs as excluded."""
    runs = (
        _record(config_label="X", funder="A", f1=0.2),
        _record(config_label="X", funder="B", f1=0.4, run_failed=True),
        _record(config_label="Y", funder="A", f1=0.6),
        _record(config_label="Y", funder="B", f1=0.8),
    )
    data = ds.AnalysisDataset(runs=runs, manifest={})
    survivors = sens.surviving_runs(data)
    # B's only run under X failed, so B has no X runs in `survivors`.
    assert not any(r.funder == "B" and r.config_label == "X" for r in survivors)

    pair = data.paired("X", "Y", value="f1", runs=survivors)
    assert pair.funders == ("A",)
    assert pair.excluded == (("B", "undefined f1 for config X"),)
