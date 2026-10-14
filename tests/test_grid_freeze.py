"""Tests for the result-affecting source hash and the grid run header.

Unchanged source resumes quietly; changed source halts unless the operator forces
a resume, and the override is recorded.

"""

import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from grant_eval.grid import freeze


@pytest.fixture(autouse=True)
def _git(request):
    """Tests that build their own git repo must not see the shared one."""
    name = request.node.name
    if "git_sha" not in name and "working_tree_clean" not in name:
        request.getfixturevalue("git_checkout")


def test_prompts_py_is_not_result_affecting():
    """prompts.py is not hashed, so adding strategies does not invalidate finished `minimal` cells."""
    assert not any("prompts.py" in p for p in freeze.RESULT_AFFECTING_PATHS)


def test_serve_py_is_result_affecting():
    """serve.py decides the bytes the agent reads, so it is hashed."""
    assert "src/grant_eval/dataset/serve.py" in freeze.RESULT_AFFECTING_PATHS


def test_snapshot_py_is_not_result_affecting():
    """Snapshot mirrors are pinned by content_hash in the registry instead."""
    assert not any("snapshot.py" in p for p in freeze.RESULT_AFFECTING_PATHS)


def test_result_sha_is_stable_and_64_hex():
    a = freeze.result_sha()
    assert a == freeze.result_sha()
    assert len(a) == 64
    int(a, 16)


def test_result_sha_moves_when_a_result_affecting_file_changes(tmp_path):
    root = tmp_path / "repo"
    for rel in freeze.RESULT_AFFECTING_PATHS:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n", encoding="utf-8")
    before = freeze.result_sha(root)
    (root / "src/grant_eval/agent/loop.py").write_text("changed\n", encoding="utf-8")
    assert freeze.result_sha(root) != before


def test_result_sha_ignores_a_non_result_affecting_file(tmp_path):
    root = tmp_path / "repo"
    for rel in freeze.RESULT_AFFECTING_PATHS:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n", encoding="utf-8")
    before = freeze.result_sha(root)
    prompts = root / "src/grant_eval/agent/prompts.py"
    prompts.write_text("a brand new strategy\n", encoding="utf-8")
    assert freeze.result_sha(root) == before


def test_grid_seed_is_not_in_result_sha(tmp_path, monkeypatch):
    from grant_eval import config

    before = freeze.result_sha()
    monkeypatch.setattr(config, "GRID_SEED", config.GRID_SEED + 1)
    assert freeze.result_sha() == before


def test_header_is_written_once_then_verified(tmp_path):
    header_path = tmp_path / "grid_run.json"
    first = freeze.verify_or_raise(header_path, force_resume=False)
    assert header_path.is_file()
    second = freeze.verify_or_raise(header_path, force_resume=False)
    assert second["result_sha"] == first["result_sha"]
    assert len(second["sessions"]) == 2


def test_resume_refuses_on_result_sha_mismatch(tmp_path):
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False)
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["result_sha"] = "0" * 64
    header_path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(freeze.FreezeViolation, match="result_sha"):
        freeze.verify_or_raise(header_path, force_resume=False)


def test_force_resume_records_both_shas(tmp_path):
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False)
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["result_sha"] = "0" * 64
    header_path.write_text(json.dumps(header), encoding="utf-8")
    out = freeze.verify_or_raise(header_path, force_resume=True)
    last = out["sessions"][-1]
    assert last["forced"] is True
    assert last["header_result_sha"] == "0" * 64
    assert last["current_result_sha"] == freeze.result_sha()


# The top-level `result_sha` is the record of what the run started from and never moves.


def test_forced_resume_never_mutates_the_headers_top_level_result_sha(tmp_path):
    """The top-level `result_sha` stays the same after an accepted forced resume.

    If it took the new hash, `git_sha` and `result_sha` would point at different
    commits with no sign of an override. Only the session log records the change.

    """
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False)

    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["result_sha"] = "0" * 64
    header_path.write_text(json.dumps(header), encoding="utf-8")

    forced = freeze.verify_or_raise(header_path, force_resume=True)
    assert forced["result_sha"] == "0" * 64, (
        "the top-level result_sha must stay pinned to whatever the header "
        "recorded before the forced resume, not move to the current hash"
    )

    # Re-read from disk in case the new hash was written but not returned.
    on_disk = json.loads(header_path.read_text(encoding="utf-8"))
    assert on_disk["result_sha"] == "0" * 64


def test_second_session_after_accepted_override_proceeds_without_forcing(tmp_path):
    """A later session on the same source does not need `--force-resume` again.

    The header's `result_sha` stays pinned, so a plain `recorded == current` check
    would keep raising for the rest of the run.

    """
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False)

    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["result_sha"] = "0" * 64
    header_path.write_text(json.dumps(header), encoding="utf-8")

    freeze.verify_or_raise(header_path, force_resume=True)

    # No --force-resume: the source hasn't changed since the accepted override.
    out = freeze.verify_or_raise(header_path, force_resume=False)
    assert out["result_sha"] == "0" * 64
    assert len(out["sessions"]) == 3


# The time, spend and concurrency caps affect results in practice, so resuming under
# a different cap would mix two regimes into one dataset.

_CAPS = {
    "cell_timeout_s": 5400.0,
    "iteration_timeout_s": 180.0,
    "max_rate_limit_requeues": 10,
    "max_spend_usd": 150.0,
    "per_arm_limit": 4,
}


def test_caps_are_exactly_the_cap_keys():
    """Checks the CAP_KEYS round trip both ways.

    A key in CAP_KEYS that callers don't stamp reads back as None and refuses every
    resume; a stamped key missing from CAP_KEYS is never compared.

    """
    assert set(_CAPS) == set(freeze.CAP_KEYS)


def test_the_caps_are_stamped_into_the_header_and_every_session(tmp_path):
    """Caps are written to both the header and each session entry.

    The header gives the regime the run was frozen under; the session entries show
    what each session actually ran under, which can differ after a forced resume.

    """
    header_path = tmp_path / "grid_run.json"
    out = freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    for key, value in _CAPS.items():
        assert out[key] == value, f"the header does not record {key}"
        assert out["sessions"][-1][key] == value, f"the session does not record {key}"


def test_a_resume_under_a_different_cap_is_refused(tmp_path):
    """A resume under a changed cap is refused.

    The caps are not in `config_hash`, `result_sha` or `grid_run.json`, but a
    wall-clock cap decides which cells come back `timed_out=True`, so it still
    affects results.

    """
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    drifted = dict(_CAPS, cell_timeout_s=1800.0)
    with pytest.raises(freeze.FreezeViolation, match="cell_timeout_s"):
        freeze.verify_or_raise(header_path, force_resume=False, caps=drifted)


def test_a_resume_under_a_different_requeue_cap_is_refused(tmp_path):
    """The requeue cap decides when a saturated arm fails a cell, so it changes which cells fail."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    drifted = dict(_CAPS, max_rate_limit_requeues=2)
    with pytest.raises(freeze.FreezeViolation, match="max_rate_limit_requeues"):
        freeze.verify_or_raise(header_path, force_resume=False, caps=drifted)


def test_a_header_predating_a_new_cap_key_needs_forcing_once(tmp_path):
    """Adding a key to CAP_KEYS is deliberately not backward compatible.

    An older header has no value for it, reads back as None and refuses until the
    operator forces once, after which the forced value stands.

    """
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    header = json.loads(header_path.read_text(encoding="utf-8"))
    del header["max_rate_limit_requeues"]
    header_path.write_text(json.dumps(header), encoding="utf-8")

    with pytest.raises(freeze.FreezeViolation, match="max_rate_limit_requeues"):
        freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    freeze.verify_or_raise(header_path, force_resume=True, caps=dict(_CAPS))
    out = freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    assert len(out["sessions"]) == 3


def test_an_unchanged_cap_resumes_quietly(tmp_path):
    """An unchanged cap resumes quietly."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    out = freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    assert len(out["sessions"]) == 2
    assert "forced" not in out["sessions"][-1]


def test_forcing_past_a_cap_change_records_both_values(tmp_path):
    """`--force-resume` records both cap values in the header."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    drifted = dict(_CAPS, cell_timeout_s=1800.0)
    out = freeze.verify_or_raise(header_path, force_resume=True, caps=drifted)
    last = out["sessions"][-1]
    assert last["forced"] is True
    assert last["header_caps"]["cell_timeout_s"] == 5400.0
    assert last["current_caps"]["cell_timeout_s"] == 1800.0
    # ...and the header's own record stays fixed, as for result_sha.
    assert out["cell_timeout_s"] == 5400.0


def test_a_session_after_an_accepted_cap_change_does_not_re_force(tmp_path):
    """A cap accepted once with `--force-resume` does not need forcing again on every resume."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, caps=dict(_CAPS))
    drifted = dict(_CAPS, cell_timeout_s=1800.0)
    freeze.verify_or_raise(header_path, force_resume=True, caps=drifted)
    out = freeze.verify_or_raise(header_path, force_resume=False, caps=drifted)
    assert len(out["sessions"]) == 3
    # ...but a new, unaccepted cap still refuses.
    with pytest.raises(freeze.FreezeViolation):
        freeze.verify_or_raise(
            header_path, force_resume=False, caps=dict(_CAPS, per_arm_limit=8)
        )


# Session outcome and progress counters.


def test_cells_pending_is_the_real_pending_count_not_the_plan_size(tmp_path):
    """`cells_pending` is the outstanding count, not the plan size."""
    header_path = tmp_path / "grid_run.json"
    out = freeze.verify_or_raise(
        header_path, force_resume=False, planned_cells=1920, cells_pending=37
    )
    assert out["planned_cells"] == 1920
    assert out["sessions"][-1]["cells_pending"] == 37


def test_record_outcome_writes_what_the_session_actually_did(tmp_path):
    """The session entry records completed/failed/skipped counts, halt kind and reason, and cost."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, planned_cells=10,
                           cells_pending=10)
    freeze.record_outcome(
        header_path,
        completed=7,
        failed=2,
        skipped=1,
        halted_kind="spend_cap",
        halted_reason="spend cap exceeded: $151.00 > $150.00",
        session_cost_usd=12.5,
        total_cost_usd=151.0,
    )
    header = json.loads(header_path.read_text(encoding="utf-8"))
    last = header["sessions"][-1]
    assert last["completed"] == 7
    assert last["failed"] == 2
    assert last["skipped"] == 1
    assert last["halted_kind"] == "spend_cap"
    assert "151.00" in last["halted_reason"]
    assert last["cost_usd"] == 12.5
    assert last["finished_at"] > last["started_at"]
    # Cumulative spend goes at the top level, where a reader would look for it.
    assert header["total_cost_usd"] == 151.0


def test_record_outcome_leaves_earlier_sessions_alone(tmp_path):
    """Updating a session only touches that session, so earlier sessions' counters are kept."""
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False, cells_pending=10)
    freeze.record_outcome(header_path, completed=3, failed=0, skipped=7,
                          halted_kind=None, halted_reason=None,
                          session_cost_usd=1.0, total_cost_usd=1.0)
    freeze.verify_or_raise(header_path, force_resume=False, cells_pending=7)
    freeze.record_outcome(header_path, completed=7, failed=0, skipped=0,
                          halted_kind=None, halted_reason=None,
                          session_cost_usd=2.0, total_cost_usd=3.0)
    header = json.loads(header_path.read_text(encoding="utf-8"))
    assert [s["completed"] for s in header["sessions"]] == [3, 7]
    assert header["total_cost_usd"] == 3.0


def _write_result_affecting_files(root: Path, content: str) -> None:
    for rel in freeze.RESULT_AFFECTING_PATHS:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_third_session_after_a_second_unrelated_change_still_refuses(tmp_path):
    """Acceptance only covers the specific hash that was force-resumed.

    Remembering just a boolean would let any later change through. The test moves the
    source through three states.

    """
    header_path = tmp_path / "grid_run.json"
    freeze.verify_or_raise(header_path, force_resume=False)

    root = tmp_path / "repo"
    _write_result_affecting_files(root, "v1\n")
    # This always mismatches the real project's pinned hash, so accept it.
    freeze.verify_or_raise(header_path, force_resume=True, repo_root=root)

    # Same source, no further edits: the accepted override should cover this.
    freeze.verify_or_raise(header_path, force_resume=False, repo_root=root)

    # A further, unaccepted change should still refuse without --force-resume.
    _write_result_affecting_files(root, "v2\n")
    with pytest.raises(freeze.FreezeViolation, match="result_sha"):
        freeze.verify_or_raise(header_path, force_resume=False, repo_root=root)


# Header writes must be atomic so a crash mid-write never leaves a truncated
# grid_run.json. These tests check the mechanism (temp file, fsync, os.replace)
# because a plain write_text would pass a write-and-read-back test.


def test_write_run_header_uses_os_replace_from_a_sibling_temp_file(tmp_path):
    """The bytes go to a temp file in the same directory and only appear via os.replace.

    Same directory means the rename stays on one filesystem and is atomic.

    """
    header_path = tmp_path / "grid_run.json"

    real_open = open
    opened_for_write = []

    def spy_open(file, mode="r", *args, **kwargs):
        if "w" in mode or "x" in mode or "a" in mode or "+" in mode:
            opened_for_write.append(Path(file))
        return real_open(file, mode, *args, **kwargs)

    captured = {}

    def fake_replace(src, dst):
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)
        # Not the real replace, so a direct write to header_path would show up.

    with mock.patch("builtins.open", side_effect=spy_open), \
         mock.patch("os.replace", side_effect=fake_replace):
        freeze.write_run_header(
            header_path, seed=1, git_sha="abc123", result_sha="0" * 64,
            split="tune", planned_cells=10, workers=1,
        )

    assert "src" in captured, "os.replace was never called, header write is not atomic"
    assert captured["dst"] == header_path
    assert captured["src"].parent == header_path.parent
    assert captured["src"] != header_path
    assert not header_path.exists()
    assert header_path not in opened_for_write


def test_write_run_header_cleans_up_temp_file_on_failure(tmp_path):
    """If the write fails before the replace, no temp file is left behind."""
    header_path = tmp_path / "grid_run.json"

    with mock.patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            freeze.write_run_header(
                header_path, seed=1, git_sha="abc123", result_sha="0" * 64,
                split="tune", planned_cells=10, workers=1,
            )

    assert not header_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_run_header_fsyncs_before_replace(tmp_path):
    """The temp file is fsynced before the rename.

    Removing the fsync would not change the final file, so this spies on both calls
    and checks the order.

    """
    header_path = tmp_path / "grid_run.json"
    calls = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    with mock.patch("os.fsync", side_effect=spy_fsync), \
         mock.patch("os.replace", side_effect=spy_replace):
        freeze.write_run_header(
            header_path, seed=1, git_sha="abc123", result_sha="0" * 64,
            split="tune", planned_cells=10, workers=1,
        )

    assert "fsync" in calls, "os.fsync was never called"
    assert calls.index("fsync") < calls.index("replace"), (
        f"fsync must happen before replace, got order {calls}"
    )


def test_append_session_uses_os_replace_from_a_sibling_temp_file(tmp_path):
    """`append_session` runs on every resume, so it gets the same atomicity check as the first write."""
    header_path = tmp_path / "grid_run.json"
    freeze.write_run_header(
        header_path, seed=1, git_sha="abc123", result_sha="0" * 64,
        split="tune", planned_cells=10, workers=1,
    )

    captured = {}

    def fake_replace(src, dst):
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)

    with mock.patch("os.replace", side_effect=fake_replace):
        freeze.append_session(header_path, started_at="2026-01-01T00:00:00+00:00", cells_pending=5)

    assert "src" in captured, "os.replace was never called, append_session is not atomic"
    assert captured["dst"] == header_path
    assert captured["src"].parent == header_path.parent
    assert captured["src"] != header_path
    # os.replace was faked out, so the header on disk should be unchanged.
    on_disk = json.loads(header_path.read_text(encoding="utf-8"))
    assert on_disk["sessions"] == []


def test_append_session_cleans_up_temp_file_on_failure(tmp_path):
    header_path = tmp_path / "grid_run.json"
    freeze.write_run_header(
        header_path, seed=1, git_sha="abc123", result_sha="0" * 64,
        split="tune", planned_cells=10, workers=1,
    )

    with mock.patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            freeze.append_session(header_path, started_at="2026-01-01T00:00:00+00:00", cells_pending=5)

    assert header_path.is_file(), "the original header must survive a failed append"
    on_disk = json.loads(header_path.read_text(encoding="utf-8"))
    assert on_disk["sessions"] == []
    assert {p.name for p in tmp_path.iterdir()} == {"grid_run.json"}, (
        "no stray temp file may survive beside the header"
    )


# A corrupted or half-written grid_run.json should give a clear error, not a bare
# JSONDecodeError or UnicodeDecodeError.


def test_read_run_header_raises_named_exception_on_truncated_json(tmp_path):
    header_path = tmp_path / "grid_run.json"
    header_path.write_text('{"seed": 1, "result_sha": "abc', encoding="utf-8")
    with pytest.raises(freeze.CorruptRunHeader):
        freeze.read_run_header(header_path)


def test_read_run_header_raises_named_exception_on_invalid_utf8_bytes(tmp_path):
    header_path = tmp_path / "grid_run.json"
    header_path.write_bytes(b'{"seed": 1, "result_sha": "\xff\xfe garbage"}')
    with pytest.raises(freeze.CorruptRunHeader):
        freeze.read_run_header(header_path)


def test_verify_or_raise_propagates_the_named_exception_not_a_bare_json_error(tmp_path):
    """`verify_or_raise` and `append_session` also handle a corrupted header cleanly."""
    header_path = tmp_path / "grid_run.json"
    header_path.write_text('{"seed": 1, "result_sha": "abc', encoding="utf-8")
    with pytest.raises(freeze.CorruptRunHeader):
        freeze.verify_or_raise(header_path, force_resume=False)


# git_sha and working_tree_clean are tested against throwaway repos under tmp_path,
# so the result does not depend on this checkout's own state.


def _init_git_repo(root: Path) -> None:
    """Create a one-commit git repo under `root`.

    The content depends on `root` so two repos made in the same second don't end up
    with the same commit hash.

    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "tracked.txt").write_text(f"hello from {root.name}\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def test_git_sha_reads_from_repo_root_not_a_fixed_location(tmp_path):
    """git_sha answers for the repo it is given, not the one pytest runs from."""
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _init_git_repo(repo_a)
    _init_git_repo(repo_b)

    expected_a = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_a, capture_output=True, text=True, check=True
    ).stdout.strip()
    expected_b = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_b, capture_output=True, text=True, check=True
    ).stdout.strip()

    assert freeze.git_sha(repo_a) == expected_a
    assert freeze.git_sha(repo_b) == expected_b
    assert freeze.git_sha(repo_a) != freeze.git_sha(repo_b)


def test_working_tree_clean_true_for_a_freshly_committed_repo(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    assert freeze.working_tree_clean(repo) is True


def test_working_tree_clean_false_for_an_uncommitted_tracked_modification(tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    assert freeze.working_tree_clean(repo) is False


def test_working_tree_clean_ignores_untracked_files(tmp_path):
    """Untracked files don't count as dirty.

    This repo always has untracked snapshot media and trace folders, so without
    `--untracked-files=no` a run could never start.

    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    (repo / "untracked_new_file.txt").write_text("never added to git\n", encoding="utf-8")
    assert freeze.working_tree_clean(repo) is True


def test_working_tree_clean_operates_on_repo_root_not_a_fixed_location(tmp_path):
    """A clean repo and a dirty repo each get their own answer."""
    clean_repo = tmp_path / "clean"
    dirty_repo = tmp_path / "dirty"
    _init_git_repo(clean_repo)
    _init_git_repo(dirty_repo)
    (dirty_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    assert freeze.working_tree_clean(clean_repo) is True
    assert freeze.working_tree_clean(dirty_repo) is False
