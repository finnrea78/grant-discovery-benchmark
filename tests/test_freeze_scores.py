"""Tests for `scoring/freeze_scores.py`, which freezes the scored grid with a provenance manifest.

The manifest hashes the inputs and the scorer source so later changes can be detected.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from grant_eval.schema import ExtractionResult, Split
from grant_eval.scoring import freeze_scores as freezemod

RESULTS = "tests/fixtures/skeleton"


@pytest.fixture
def frozen(tmp_path):
    return freezemod.freeze([RESULTS], tmp_path / "scored")


def test_freeze_writes_the_three_artefact_files(frozen):
    out = frozen.out_dir
    assert (out / "runs.jsonl").exists()
    assert (out / "configs.json").exists()
    assert (out / "manifest.json").exists()


def test_one_line_per_scored_run(frozen):
    lines = (frozen.out_dir / "runs.jsonl").read_text().splitlines()
    assert len(lines) == frozen.n_scored == 6
    assert all(json.loads(line)["funder"] for line in lines)


def test_configs_are_grouped_and_countable(frozen):
    configs = json.loads((frozen.out_dir / "configs.json").read_text())
    assert len(configs) == frozen.n_configs
    assert all("macro_f1" in c and "config" in c for c in configs)


def test_the_manifest_fingerprints_the_inputs_and_the_scorer(frozen):
    manifest = json.loads((frozen.out_dir / "manifest.json").read_text())
    assert manifest["inputs"]["n_files"] == 6
    assert len(manifest["inputs"]["content_hash"]) == 64
    assert len(manifest["scorer"]["source_hash"]) == 64
    assert "score.py" in manifest["scorer"]["modules"]
    assert "mirror.py" in manifest["scorer"]["modules"]


def test_the_input_fingerprint_moves_when_an_input_changes(tmp_path):
    """Editing a result file changes the input content hash."""
    src = sorted(__import__("pathlib").Path(RESULTS).glob("*.json"))
    inputs = tmp_path / "in"
    inputs.mkdir()
    for path in src:
        (inputs / path.name).write_bytes(path.read_bytes())

    before = freezemod.freeze([str(inputs)], tmp_path / "a").manifest["inputs"]["content_hash"]

    victim = sorted(inputs.glob("*.json"))[0]
    payload = json.loads(victim.read_text())
    payload["opportunities"] = []
    victim.write_text(json.dumps(payload))

    after = freezemod.freeze([str(inputs)], tmp_path / "b").manifest["inputs"]["content_hash"]
    assert before != after


def test_freezing_twice_over_unchanged_inputs_is_byte_identical(tmp_path):
    first = freezemod.freeze([RESULTS], tmp_path / "one").out_dir
    second = freezemod.freeze([RESULTS], tmp_path / "two").out_dir
    assert (first / "runs.jsonl").read_bytes() == (second / "runs.jsonl").read_bytes()
    assert (first / "configs.json").read_bytes() == (second / "configs.json").read_bytes()

    # The manifest carries a timestamp, so compare it field by field minus that.
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())
    a.pop("created_utc"), b.pop("created_utc")
    assert a == b


def test_an_unscorable_result_is_named_and_counted_not_dropped(tmp_path):
    """A result for a funder with no ground truth is listed as skipped in the manifest."""
    inputs = tmp_path / "in"
    inputs.mkdir()
    src = sorted(__import__("pathlib").Path(RESULTS).glob("*.json"))[0]
    (inputs / src.name).write_bytes(src.read_bytes())
    (inputs / "orphan.json").write_text(
        ExtractionResult(
            snapshot_date=date(2026, 8, 21),
            funder="Not A Labelled Funder",
            split=Split.TUNE,
            model="test/model",
            prompt_strategy="minimal",
            temperature=0.0,
            agent_config="standard",
            replicate=1,
            config_hash="test",
            opportunities=[],
        ).model_dump_json()
    )

    report = freezemod.freeze([str(inputs)], tmp_path / "out")
    assert report.n_scored == 1
    assert report.n_skipped == 1
    assert [name for name, _ in report.skipped] == ["orphan.json"]
    assert report.skipped[0][1]

    manifest = json.loads((report.out_dir / "manifest.json").read_text())
    assert manifest["counts"]["n_scored"] == 1
    assert manifest["counts"]["n_skipped"] == 1
    assert manifest["skipped"][0]["file"] == "orphan.json"


def test_the_manifest_records_the_mirror_it_resolved_against(frozen):
    manifest = json.loads((frozen.out_dir / "manifest.json").read_text())
    lev = manifest["mirror"]["snapshots"]["The Leverhulme Trust"]
    assert lev["snapshot_date"] == "2026-08-09"
    assert lev["content_hash"] == (
        "1dc0c98146f7f90e4e0139ff304908dfc6c33dd32b852d024b0022f9a970999a"
    )
    assert manifest["mirror"]["missing"] == []


def test_the_ground_truth_corpus_is_recorded_with_its_totals(frozen):
    gt = json.loads((frozen.out_dir / "manifest.json").read_text())["ground_truth"]
    assert gt["n_files"] == 24
    assert gt["n_rows"] == 292
    assert gt["n_in_scope"] == 167


def test_no_mirror_is_recorded_as_such_rather_than_looking_resolved(tmp_path):
    report = freezemod.freeze([RESULTS], tmp_path / "out", use_mirror=False)
    manifest = json.loads((report.out_dir / "manifest.json").read_text())
    assert manifest["mirror"]["resolved"] is False
    run = json.loads((report.out_dir / "runs.jsonl").read_text().splitlines()[0])
    assert run["detection"]["mirror_resolved"] is False
    assert run["detection"]["hallucination_rate"] is None


def test_a_grid_root_of_per_funder_directories_is_walked(tmp_path):
    """Passing the grid root finds results inside the per-funder subdirectories."""
    src = sorted(__import__("pathlib").Path(RESULTS).glob("*.json"))
    root = tmp_path / "grid"
    for i, path in enumerate(src):
        funder_dir = root / f"funder{i % 2}"
        funder_dir.mkdir(parents=True, exist_ok=True)
        (funder_dir / path.name).write_bytes(path.read_bytes())

    report = freezemod.freeze([str(root)], tmp_path / "out")
    assert report.n_scored == 6


def test_freezing_nothing_is_an_error_not_a_silent_empty_artefact(tmp_path):
    """Freezing an empty directory raises instead of writing empty files."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no result files"):
        freezemod.freeze([str(empty)], tmp_path / "out")


def test_cli_reports_finding_nothing_rather_than_writing_an_empty_artefact(
    tmp_path, capsys
):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = freezemod.main([str(empty), "--out-dir", str(tmp_path / "out")])
    assert rc == 1
    assert "no result files" in capsys.readouterr().err


def test_cli_freezes_and_reports_where_it_wrote(tmp_path, capsys):
    rc = freezemod.main([RESULTS, "--out-dir", str(tmp_path / "scored")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "6 run(s)" in out
    assert str(tmp_path / "scored") in out


def test_each_frozen_run_carries_its_telemetry(frozen):
    line = (frozen.out_dir / "runs.jsonl").read_text().splitlines()[0]
    telemetry = json.loads(line)["telemetry"]
    for key in (
        "split", "cost_usd", "latency_s", "input_tokens", "output_tokens",
        "timed_out", "context_overflow", "abstained", "schema_valid",
        "submit_failed", "step_budget_exhausted", "attempts", "provider",
    ):
        assert key in telemetry, key


def test_run_failed_flags_a_timeout_or_a_context_overflow(frozen):
    """`run_failed` is true exactly when a run timed out or overflowed its context."""
    runs = [json.loads(l) for l in (frozen.out_dir / "runs.jsonl").read_text().splitlines()]
    for run in runs:
        t = run["telemetry"]
        assert t["run_failed"] == bool(t["timed_out"] or t["context_overflow"])


def _frozen_from_one_fixture(tmp_path, mutate) -> dict:
    """Freeze one skeleton fixture after `mutate` edits its payload, and return its telemetry."""
    src = Path(RESULTS) / "leverhulme_haiku_minimal_t0_shallow_rep1.json"
    payload = json.loads(src.read_text())
    mutate(payload)
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / src.name).write_text(json.dumps(payload))
    report = freezemod.freeze([str(inputs)], tmp_path / "out")
    line = (report.out_dir / "runs.jsonl").read_text().splitlines()[0]
    return json.loads(line)["telemetry"]


def test_abstention_is_not_a_run_failure(tmp_path):
    """An abstained run is not marked as failed in the frozen output."""

    def mutate(payload):
        payload["abstained"] = True
        payload["timed_out"] = False
        payload["context_overflow"] = False

    telemetry = _frozen_from_one_fixture(tmp_path, mutate)
    assert telemetry["abstained"] is True
    assert telemetry["run_failed"] is False


def test_a_missing_cost_stays_null_rather_than_becoming_zero(tmp_path):
    """A run with no `cost_usd` keeps a null cost rather than 0."""
    telemetry = _frozen_from_one_fixture(tmp_path, lambda payload: payload.pop("cost_usd"))
    assert telemetry["cost_usd"] is None
