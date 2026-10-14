"""Result store: atomic writes, failure records and resume checks.

A cell only counts as done if its result file parses as a valid `ExtractionResult`,
so a crash mid-write can never be mistaken for a finished cell. Failed cells are
recorded and skipped on resume unless `--retry-failed` is given.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from grant_eval.schema import ExtractionResult

if TYPE_CHECKING:
    from grant_eval.grid.cells import Cell


def write_result_atomic(path: Path, result: ExtractionResult) -> None:
    """Write `result` to `path` atomically, so readers never see a partial file.

    The temp file sits in the same directory so `os.replace` is a same-filesystem rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def is_complete(cell: "Cell", results_root: Path) -> bool:
    """True only if the result file exists and parses as a valid `ExtractionResult`.

    Never raises: one bad file must not stop a resume scan. JSON, pydantic and unicode
    errors are all `ValueError` subclasses.
    """
    path = cell.result_path(results_root)
    try:
        raw = path.read_text(encoding="utf-8")
        ExtractionResult.model_validate_json(raw)
        return True
    except (OSError, ValueError):
        return False


def write_failure(
    cell: "Cell", results_root: Path, *, exc: BaseException, attempts: int
) -> None:
    """Record that `cell` failed after `attempts` attempts, so resume can skip it.
    Written atomically like results."""
    path = cell.failed_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cell_id": cell.cell_id,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "attempts": attempts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_failure(cell: "Cell", results_root: Path) -> dict | None:
    """The failure record for `cell`, or `None` if absent or unreadable."""
    path = cell.failed_path(results_root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_failed(cell: "Cell", results_root: Path) -> bool:
    """Whether `cell` has a readable failure record."""
    return read_failure(cell, results_root) is not None


def clear_failure(cell: "Cell", results_root: Path) -> None:
    """Remove `cell`'s failure record, if any."""
    cell.failed_path(results_root).unlink(missing_ok=True)


def pending(
    plan: list["Cell"], results_root: Path, *, retry_failed: bool
) -> list["Cell"]:
    """The cells of `plan` still to run, in plan order. Failed cells are skipped unless
    `retry_failed`."""
    result = []
    for cell in plan:
        if is_complete(cell, results_root):
            continue
        if is_failed(cell, results_root) and not retry_failed:
            continue
        result.append(cell)
    return result


def gzip_traces(trace_dir: Path) -> int:
    """Gzip every `*.jsonl` in `trace_dir` in place and return how many were compressed.

    Each `.gz` is fsynced before the original is deleted, since a lost trace can only
    be recovered by paying for the model call again.
    """
    count = 0
    for src in sorted(trace_dir.glob("*.jsonl")):
        dst = src.with_suffix(src.suffix + ".gz")
        with src.open("rb") as fin, gzip.open(dst, "wb") as fout:
            fout.writelines(fin)
        fd = os.open(dst, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        src.unlink()
        count += 1
    return count


def sweep_traces(traces_root: Path) -> int:
    """Gzip every uncompressed `*.jsonl` under `traces_root / "grid"` and return the count.

    Catches traces left raw when a run crashed between writing a result and gzipping.
    Limited to the `grid` subtree so other trace folders are left alone. It can still
    gzip a trace that an abandoned worker thread is writing to; this is a known gap.
    """
    grid_root = Path(traces_root) / "grid"
    if not grid_root.is_dir():
        return 0
    directories = sorted({src.parent for src in grid_root.rglob("*.jsonl")})
    return sum(gzip_traces(directory) for directory in directories)
