"""Tests for `lock_config`, which writes `eval/locked-config.json`.

`--split validation` will not run without this file. The ranking tests read the real
`eval/scored/grid` so they agree with the leaderboard in `eval/figures/report.txt`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grant_eval import run as run_cli
from grant_eval.grid import lock_config as lc

SCORED = Path("eval/scored/grid")

#: Top of the tune leaderboard by macro F1 (see `eval/figures/report.txt`).
BEST_LABEL = "openrouter/deepseek/deepseek-v4-flash-0731 | few_shot | t0.7"
BEST_F1 = 0.793
#: The tune leaderboard's #2, used for the two-config case.
SECOND_LABEL = "openrouter/deepseek/deepseek-v4-flash-0731 | minimal | t0.7"

#: 8 validation funders x 5 replicates.
CELLS_PER_CONFIG = 40


def test_ranking_puts_the_leaderboards_best_macro_f1_first():
    """Configs are ranked by macro F1, matching the leaderboard in `report.txt`."""
    ranked = lc.rank_configs(SCORED)
    assert ranked[0].label == BEST_LABEL
    assert ranked[0].macro_f1 == pytest.approx(BEST_F1, abs=0.001)
    assert ranked[1].label == SECOND_LABEL


def test_ranking_covers_every_config_and_is_monotonically_worse():
    ranked = lc.rank_configs(SCORED)
    assert len(ranked) == 24
    scores = [c.macro_f1 for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_the_lock_names_every_validation_cell_for_the_chosen_config():
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    assert len(lock["cells"]) == CELLS_PER_CONFIG
    assert len(set(lock["cells"])) == CELLS_PER_CONFIG


def test_the_lock_names_five_replicates_for_each_of_the_eight_validation_funders():
    """Each of the 8 validation funders gets exactly 5 replicates."""
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    by_funder: dict[str, int] = {}
    for cell_id in lock["cells"]:
        by_funder[cell_id.split("_", 1)[0]] = by_funder.get(cell_id.split("_", 1)[0], 0) + 1
    assert len(by_funder) == 8
    assert set(by_funder.values()) == {5}


def test_top_n_two_locks_both_configs_and_twice_the_cells():
    """`top_n=2` includes both top configs, since H4 is about the ranking of configs."""
    lock = lc.build_lock(scored_dir=SCORED, top_n=2)
    assert len(lock["cells"]) == 2 * CELLS_PER_CONFIG
    assert [c["label"] for c in lock["configs"]] == [BEST_LABEL, SECOND_LABEL]


def test_an_explicit_label_overrides_the_ranking():
    lock = lc.build_lock(scored_dir=SCORED, labels=[SECOND_LABEL])
    assert [c["label"] for c in lock["configs"]] == [SECOND_LABEL]


def test_an_unknown_label_is_refused_by_name():
    with pytest.raises(SystemExit) as exc:
        lc.build_lock(scored_dir=SCORED, labels=["not/a | real | t0.0"])
    assert "not/a | real | t0.0" in str(exc.value)


def test_tune_evidence_names_the_scored_dir_the_config_and_its_score():
    """`tune_evidence` names the scored dir, the chosen config and its F1."""
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    evidence = lock["tune_evidence"]
    assert "eval/scored/grid" in evidence
    assert BEST_LABEL in evidence
    assert "0.793" in evidence


def test_the_lock_records_the_scorer_hash_so_a_rescore_is_detectable():
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    manifest = json.loads((SCORED / "manifest.json").read_text())
    assert lock["tune_source_hash"] == manifest["scorer"]["source_hash"]


def test_locked_at_is_utc_iso8601():
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    assert lock["locked_at"].endswith("+00:00")


def test_the_emitted_lock_is_accepted_by_the_validation_guard(tmp_path, monkeypatch, capsys):
    """A lock file written by `lock_config` is accepted by `--split validation --dry-run`."""
    lock_path = tmp_path / "locked-config.json"
    lc.write_lock(lc.build_lock(scored_dir=SCORED, top_n=1), lock_path)
    monkeypatch.setattr(run_cli, "LOCKED_CONFIG_PATH", lock_path)

    assert run_cli.main(["--split", "validation", "--dry-run"]) == 0

    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[-1] == f"dry run: {CELLS_PER_CONFIG} cells (split=validation, seed=20260831)"


def test_it_refuses_to_overwrite_an_existing_lock(tmp_path):
    """The lock file is written once; a second write is refused without `force`."""
    lock_path = tmp_path / "locked-config.json"
    lock = lc.build_lock(scored_dir=SCORED, top_n=1)
    lc.write_lock(lock, lock_path)
    with pytest.raises(SystemExit) as exc:
        lc.write_lock(lock, lock_path)
    assert "already exists" in str(exc.value)


def test_force_overwrites_and_says_so(tmp_path):
    lock_path = tmp_path / "locked-config.json"
    lc.write_lock(lc.build_lock(scored_dir=SCORED, top_n=1), lock_path)
    lc.write_lock(lc.build_lock(scored_dir=SCORED, top_n=2), lock_path, force=True)
    assert len(json.loads(lock_path.read_text())["cells"]) == 2 * CELLS_PER_CONFIG


def test_it_refuses_to_lock_on_evidence_from_the_validation_split(tmp_path):
    """Scores from the validation split cannot be used to pick the config."""
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "configs.json").write_text((SCORED / "configs.json").read_text())
    (scored / "manifest.json").write_text((SCORED / "manifest.json").read_text())
    with (SCORED / "runs.jsonl").open() as src, (scored / "runs.jsonl").open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["telemetry"]["split"] = "validation"
            dst.write(json.dumps(payload) + "\n")

    with pytest.raises(SystemExit) as exc:
        lc.build_lock(scored_dir=scored, top_n=1)
    assert "validation" in str(exc.value)


def test_a_missing_scored_dir_names_the_command_that_makes_one(tmp_path):
    with pytest.raises(SystemExit) as exc:
        lc.build_lock(scored_dir=tmp_path / "nope", top_n=1)
    assert "freeze_scores" in str(exc.value)


def test_dry_run_writes_nothing(tmp_path, capsys):
    lock_path = tmp_path / "locked-config.json"
    assert lc.main(["--scored-dir", str(SCORED), "--out", str(lock_path), "--dry-run"]) == 0
    assert not lock_path.exists()
    assert BEST_LABEL in capsys.readouterr().out


def test_the_cli_writes_a_lock_the_guard_accepts(tmp_path):
    lock_path = tmp_path / "locked-config.json"
    assert lc.main(["--scored-dir", str(SCORED), "--out", str(lock_path)]) == 0
    payload = json.loads(lock_path.read_text())
    assert {"cells", "locked_at", "tune_evidence"} <= payload.keys()
    assert len(payload["cells"]) == CELLS_PER_CONFIG
