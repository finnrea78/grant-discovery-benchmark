"""Tests for the result store: atomic writes, resume checks, failure sentinels and
gzipped traces. A cell counts as done only if its file exists and parses as a valid
`ExtractionResult`, so a half-written file is never taken as finished.

"""
from __future__ import annotations

import gzip
import json
import os
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

from grant_eval.grid import store
from grant_eval.grid.cells import enumerate_cells
from grant_eval.dataset.registry import load_registry
from grant_eval.schema import ExtractionResult, Split

REG = load_registry(Path("data/funders.json"))


def _cell():
    return enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[0]


def _result(cell) -> ExtractionResult:
    return ExtractionResult(
        snapshot_date=cell.snapshot_date, funder=cell.display_name, split=cell.split,
        model=cell.model_slug, prompt_strategy=cell.prompt_strategy,
        temperature=cell.temperature, agent_config=cell.agent_config,
        replicate=cell.replicate, config_hash=cell.config_hash,
    )


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    cell = _cell()
    path = cell.result_path(tmp_path)
    store.write_result_atomic(path, _result(cell))
    assert path.is_file()
    assert list(path.parent.iterdir()) == [path]


def test_write_uses_os_replace_from_a_sibling_temp_file(tmp_path):
    """The bytes go to a temp file in the same directory and only appear via os.replace.

    A plain `open(path, "w").write(...)` passes the round-trip tests but not this one.

    """
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
        # Not the real replace: a direct write to `path` would make the assertions below fail.

    with mock.patch("builtins.open", side_effect=spy_open), \
         mock.patch("os.replace", side_effect=fake_replace):
        store.write_result_atomic(path, _result(cell))

    assert "src" in captured, "os.replace was never called, write is not atomic"
    assert captured["dst"] == path
    assert captured["src"].parent == path.parent
    assert captured["src"] != path
    # os.replace was faked out, so the destination should not exist yet.
    assert not path.exists()
    # The destination should never have been opened for writing, only the temp file.
    assert path not in opened_for_write


def test_write_cleans_up_temp_file_on_failure(tmp_path):
    """If the write fails before the replace, no temp file is left behind."""
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with mock.patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            store.write_result_atomic(path, _result(cell))

    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_write_fsyncs_before_replace(tmp_path):
    """The temp file is fsynced before the rename.

    Otherwise a crash could expose content the OS never wrote out. Removing the fsync
    would not change the final file, so this checks the call order.

    """
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
        store.write_result_atomic(path, _result(cell))

    assert "fsync" in calls, "os.fsync was never called"
    assert calls.index("fsync") < calls.index("replace"), (
        f"fsync must happen before replace, got order {calls}"
    )


def test_is_complete_true_after_a_write(tmp_path):
    cell = _cell()
    store.write_result_atomic(cell.result_path(tmp_path), _result(cell))
    assert store.is_complete(cell, tmp_path) is True


def test_is_complete_false_for_missing_file(tmp_path):
    cell = _cell()
    assert store.is_complete(cell, tmp_path) is False


def test_truncated_result_is_not_complete(tmp_path):
    """A file cut off mid-write does not count as a finished cell."""
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"snapshot_date": "2026-08-09", "fun', encoding="utf-8")
    assert store.is_complete(cell, tmp_path) is False


def test_schema_invalid_result_is_not_complete(tmp_path):
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"not": "an ExtractionResult"}), encoding="utf-8")
    assert store.is_complete(cell, tmp_path) is False


def test_invalid_utf8_bytes_is_not_complete(tmp_path):
    """Invalid UTF-8 is treated as corruption rather than raised.

    `UnicodeDecodeError` is not a JSONDecodeError or ValidationError, but it should not
    stop the resume scan over the other cells either.

    """
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00not valid utf-8")
    assert store.is_complete(cell, tmp_path) is False


def test_is_complete_false_when_result_path_is_a_directory(tmp_path):
    """A directory at `result_path` (IsADirectoryError) does not escape `is_complete`."""
    cell = _cell()
    path = cell.result_path(tmp_path)
    path.mkdir(parents=True)
    assert store.is_complete(cell, tmp_path) is False


def test_is_complete_false_when_result_path_is_unreadable(tmp_path):
    """A permission-denied result file does not escape `is_complete`."""
    cell = _cell()
    path = cell.result_path(tmp_path)
    store.write_result_atomic(path, _result(cell))
    path.chmod(0o000)
    try:
        assert store.is_complete(cell, tmp_path) is False
    finally:
        path.chmod(0o644)  # so tmp_path cleanup can remove it


def test_failure_sentinel_roundtrip(tmp_path):
    cell = _cell()
    store.write_failure(cell, tmp_path, exc=ValueError("boom"), attempts=2)
    assert store.is_failed(cell, tmp_path) is True
    rec = store.read_failure(cell, tmp_path)
    assert rec["exception_type"] == "ValueError"
    assert rec["message"] == "boom"
    assert rec["attempts"] == 2
    store.clear_failure(cell, tmp_path)
    assert store.is_failed(cell, tmp_path) is False


def test_read_failure_returns_none_when_absent(tmp_path):
    cell = _cell()
    assert store.read_failure(cell, tmp_path) is None


def test_read_failure_returns_none_for_invalid_utf8_bytes(tmp_path):
    """`UnicodeDecodeError` does not escape `read_failure` or `is_failed`."""
    cell = _cell()
    path = cell.failed_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00not valid utf-8")
    assert store.read_failure(cell, tmp_path) is None
    assert store.is_failed(cell, tmp_path) is False


def test_read_failure_returns_none_when_failed_path_is_a_directory(tmp_path):
    cell = _cell()
    path = cell.failed_path(tmp_path)
    path.mkdir(parents=True)
    assert store.read_failure(cell, tmp_path) is None


def test_read_failure_returns_none_when_failed_path_is_unreadable(tmp_path):
    cell = _cell()
    store.write_failure(cell, tmp_path, exc=RuntimeError("x"), attempts=1)
    path = cell.failed_path(tmp_path)
    path.chmod(0o000)
    try:
        assert store.read_failure(cell, tmp_path) is None
    finally:
        path.chmod(0o644)


def test_clear_failure_is_safe_when_absent(tmp_path):
    """Clearing a sentinel that was never written does not raise."""
    cell = _cell()
    store.clear_failure(cell, tmp_path)  # must not raise


def test_pending_skips_complete_and_failed(tmp_path):
    cells = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"])
    store.write_result_atomic(cells[0].result_path(tmp_path), _result(cells[0]))
    store.write_failure(cells[1], tmp_path, exc=RuntimeError("x"), attempts=2)
    remaining = store.pending(cells, tmp_path, retry_failed=False)
    ids = {c.cell_id for c in remaining}
    assert cells[0].cell_id not in ids
    assert cells[1].cell_id not in ids
    assert len(remaining) == len(cells) - 2


def test_pending_includes_failed_when_retry_failed(tmp_path):
    cells = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"])
    store.write_failure(cells[1], tmp_path, exc=RuntimeError("x"), attempts=2)
    ids = {c.cell_id for c in store.pending(cells, tmp_path, retry_failed=True)}
    assert cells[1].cell_id in ids


def test_pending_preserves_plan_order(tmp_path):
    cells = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"])
    store.write_result_atomic(cells[3].result_path(tmp_path), _result(cells[3]))
    remaining = store.pending(cells, tmp_path, retry_failed=False)
    expected = [c.cell_id for c in cells if c.cell_id != cells[3].cell_id]
    assert [c.cell_id for c in remaining] == expected


def test_gzip_traces(tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "iter1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (d / "iter2.jsonl").write_text('{"b": 2}\n', encoding="utf-8")
    assert store.gzip_traces(d) == 2
    assert not (d / "iter1.jsonl").exists()
    with gzip.open(d / "iter1.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert fh.read() == '{"a": 1}\n'


def test_gzip_traces_covers_every_jsonl_trace_not_just_the_iterations(tmp_path):
    """`misses.jsonl` is gzipped too, so every trace in the folder has the same format."""
    d = tmp_path / "cell"
    d.mkdir()
    (d / "iter1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (d / "misses.jsonl").write_text('{"event": "mirror_misses"}\n', encoding="utf-8")

    assert store.gzip_traces(d) == 2
    assert list(d.glob("*.jsonl")) == []
    with gzip.open(d / "misses.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert fh.read() == '{"event": "mirror_misses"}\n'


def test_sweep_traces_compresses_traces_a_crashed_cell_left_behind(tmp_path):
    """Leftover raw traces anywhere under the grid folder are gzipped.

    `gzip_traces` runs after the result write, so a crash in between leaves raw
    `*.jsonl` files that no resume will revisit.

    """
    orphan = tmp_path / "grid" / "cell-a"
    orphan.mkdir(parents=True)
    (orphan / "iter1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (orphan / "misses.jsonl").write_text('{"event": "mirror_misses"}\n', encoding="utf-8")
    already = tmp_path / "grid" / "cell-b"
    already.mkdir(parents=True)
    (already / "iter1.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))

    assert store.sweep_traces(tmp_path) == 2
    assert list(tmp_path.rglob("*.jsonl")) == []
    with gzip.open(orphan / "iter1.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert fh.read() == '{"a": 1}\n'
    # ...and a second sweep finds nothing, so it is safe to run every time.
    assert store.sweep_traces(tmp_path) == 0


def test_sweep_traces_does_not_reach_outside_the_grid_subtree(tmp_path):
    """The sweep leaves anything outside the `grid/` subtree alone.

    `eval/run-traces/skeleton/` holds git-tracked `.jsonl` files, and gzipping them
    would leave the working tree dirty, which then fails `working_tree_clean`.

    """
    grid_orphan = tmp_path / "grid" / "cell-a"
    grid_orphan.mkdir(parents=True)
    (grid_orphan / "iter1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    sibling = tmp_path / "skeleton" / "some-cell"
    sibling.mkdir(parents=True)
    (sibling / "iter1.jsonl").write_text('{"tracked": true}\n', encoding="utf-8")

    assert store.sweep_traces(tmp_path) == 1
    # The grid orphan was compressed...
    assert not (grid_orphan / "iter1.jsonl").exists()
    assert (grid_orphan / "iter1.jsonl.gz").exists()
    # ...but the file outside grid/ was not touched.
    assert (sibling / "iter1.jsonl").exists()
    assert (sibling / "iter1.jsonl").read_text(encoding="utf-8") == '{"tracked": true}\n'
    assert not (sibling / "iter1.jsonl.gz").exists()


def test_sweep_traces_tolerates_a_traces_root_that_does_not_exist(tmp_path):
    """A missing traces folder is fine, since a run that refused early may never have made it."""
    assert store.sweep_traces(tmp_path / "never-created") == 0


def test_gzip_traces_is_idempotent(tmp_path):
    d = tmp_path / "cell"
    d.mkdir()
    (d / "iter1.jsonl").write_text("{}\n", encoding="utf-8")
    store.gzip_traces(d)
    assert store.gzip_traces(d) == 0


def test_gzip_traces_fsyncs_gz_before_deleting_source(tmp_path):
    """The `.gz` is fsynced before the original is deleted.

    Closing a gzip file doesn't fsync, and removing the fsync would not change the
    output, so this spies on `os.fsync` and `Path.unlink` and checks the order.

    """
    d = tmp_path / "cell"
    d.mkdir()
    (d / "iter1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    calls = []
    real_fsync = os.fsync
    real_unlink = Path.unlink

    def spy_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def spy_unlink(self, *args, **kwargs):
        calls.append("unlink")
        return real_unlink(self, *args, **kwargs)

    with mock.patch("os.fsync", side_effect=spy_fsync), \
         mock.patch.object(Path, "unlink", spy_unlink):
        assert store.gzip_traces(d) == 1

    assert "fsync" in calls, "os.fsync was never called on the .gz file"
    assert calls.index("fsync") < calls.index("unlink"), (
        f"fsync must happen before the source is unlinked, got order {calls}"
    )
