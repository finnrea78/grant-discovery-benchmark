"""Loading the frozen scored grid. Reads the committed artefact, so it is offline."""
from __future__ import annotations

import pytest

from grant_eval.analysis import dataset as ds

BEST = "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"
MELLON = "Paul Mellon Centre for Studies in British Art"


@pytest.fixture(scope="module")
def data():
    return ds.load_dataset()


def test_the_whole_grid_loads(data):
    assert len(data.runs) == 1920
    assert len(data.funders()) == 16
    assert len(data.config_labels()) == 24
    assert {r.split for r in data.runs} == {"tune"}


def test_every_cell_is_five_replicates(data):
    for label in data.config_labels():
        for funder in data.funders():
            assert len(data.cell(label, funder)) == 5


def test_an_undefined_precision_stays_none(data):
    """Runs that extracted nothing keep precision as None rather than 0.0.

    `score.py` writes 0.0 plus a `precision_undefined` flag, and the loader rebuilds None from the flag.
    """
    undefined = [r for r in data.runs if r.precision is None]
    assert len(undefined) == 377
    assert all(r.n_extracted == 0 for r in undefined)


def test_an_undefined_hallucination_rate_stays_none(data):
    assert len([r for r in data.runs if r.hallucination_rate is None]) == 377


def test_funder_means_average_the_five_replicates(data):
    reps = [r.f1 for r in data.cell(BEST, MELLON)]
    assert data.funder_means()[BEST][MELLON] == pytest.approx(sum(reps) / len(reps))
    # BEST/MELLON is all f1=1.0, so it can't tell a mean from `values[0]`.
    # Use a synthetic cell with different replicates as well.
    varied = ds.AnalysisDataset(
        runs=tuple(
            _record(replicate=i, f1=v)
            for i, v in enumerate([0.2, 0.4, 0.6, 0.8, 1.0], start=1)
        ),
        manifest={},
    )
    assert varied.funder_means()["X"]["A"] == pytest.approx(0.6)


def test_the_ninety_eight_failed_runs_are_flagged(data):
    assert sum(1 for r in data.runs if r.run_failed) == 98


def test_pairing_is_by_funder_and_reports_no_exclusions_on_f1(data):
    labels = data.config_labels()
    pair = data.paired(labels[0], labels[1])
    assert len(pair.funders) == len(pair.a) == len(pair.b) == 16
    assert pair.excluded == ()
    assert pair.differences == tuple(x - y for x, y in zip(pair.a, pair.b))


def _record(**kw):
    base = dict(
        funder="A", model="m", prompt="p", temperature=0.0, replicate=1,
        config_label="X", config_hash="h", f1=0.5, precision=0.5, recall=0.5,
        tp=1, fp=0, fn=1, n_extracted=1, n_in_scope=2, hallucination_rate=0.0,
        n_fabricated=0, fp_by_reason={}, true_positive_ids=(), false_negative_ids=(),
        cost_usd=0.1, latency_s=1.0, run_failed=False, abstained=False, split="tune",
    )
    return ds.RunRecord(**{**base, **kw})


def test_a_funder_undefined_on_either_side_is_excluded_and_named():
    """A funder with no value on one side is dropped from the pair and listed as excluded."""
    data = ds.AnalysisDataset(
        runs=(
            _record(config_label="X", precision=None, n_extracted=0, abstained=True),
            _record(config_label="Y", precision=0.6),
        ),
        manifest={},
    )
    pair = data.paired("X", "Y", value="precision")
    assert pair.funders == ()
    assert pair.excluded == (("A", "undefined precision for config X"),)


def test_a_cell_with_no_defined_replicate_has_no_mean_rather_than_zero():
    data = ds.AnalysisDataset(runs=(_record(precision=None),), manifest={})
    assert data.funder_means(value="precision") == {}


def test_the_manifest_travels_with_the_data(data):
    assert data.manifest["counts"]["n_scored"] == 1920
    assert data.manifest["mirror"]["resolved"] is True


def test_an_artefact_without_telemetry_is_refused(tmp_path):
    """An old freeze without cost and failure fields is rejected."""
    (tmp_path / "runs.jsonl").write_text('{"funder": "A", "detection": {}}\n')
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="no telemetry"):
        ds.load_dataset(tmp_path)


def test_a_missing_artefact_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze_scores"):
        ds.load_dataset(tmp_path / "nope")
