"""Source hash and run header that stop a grid run resuming on changed code.

`result_sha` hashes the source files that can change a cell's result. It is saved in
`grid_run.json`, and resuming on a mismatch is refused unless `--force-resume` is
given, so results from two code versions never get mixed. The dispatcher itself,
`snapshot.py` (mirrors are pinned by `content_hash`), `agent/prompts.py` (prompt text
goes into `config_hash`) and `GRID_SEED` are deliberately left out.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Source files that can change a result. `agent/prompts.py` and the empty
# `__init__.py` are left out. Each file is hashed with its path, so order does not matter.
RESULT_AFFECTING_PATHS: tuple[str, ...] = (
    "src/grant_eval/config.py",
    "src/grant_eval/schema.py",
    "src/grant_eval/dataset/serve.py",
    "src/grant_eval/agent/extract.py",
    "src/grant_eval/agent/loop.py",
    "src/grant_eval/agent/ralph.py",
    "src/grant_eval/agent/router.py",
    "src/grant_eval/agent/tools.py",
)


# Run caps are not in `result_sha` or `config_hash` but still change results: the cell
# timeout decides which cells come back `timed_out=True`, which is a scored outcome.
# They are saved in the header and on every session entry, since a forced resume may
# change them. A new key reads back as `None` from older headers, so the first resume
# onto an older run needs `--force-resume` once.
CAP_KEYS: tuple[str, ...] = (
    "cell_timeout_s",
    "iteration_timeout_s",
    "max_rate_limit_requeues",
    "max_spend_usd",
    "per_arm_limit",
)


class CorruptRunHeader(RuntimeError):
    """`grid_run.json` exists but could not be parsed as JSON."""


class FreezeViolation(RuntimeError):
    """The run header's `result_sha` or caps no longer match and `--force-resume` was
    not passed."""


def result_sha(repo_root: Path = Path(".")) -> str:
    """SHA-256 over the files in `RESULT_AFFECTING_PATHS`.

    Raises `FileNotFoundError` if one is missing, so a rename gets noticed. Reads bytes
    rather than importing the modules, and hashes each file together with its path.
    """
    root = Path(repo_root)
    digest = hashlib.sha256()
    for rel in RESULT_AFFECTING_PATHS:
        file_bytes = (root / rel).read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        digest.update(f"{rel}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def git_sha(repo_root: Path = Path(".")) -> str:
    """The current commit hash, kept in the header for reference. Resume checks
    `result_sha`, not this."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def working_tree_clean(repo_root: Path = Path(".")) -> bool:
    """True if no tracked file has uncommitted changes.

    Untracked files are ignored: the repo always has untracked snapshot media and
    traces that cannot affect a result.
    """
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip() == ""


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write `payload` as JSON to `path` atomically, as `store.write_result_atomic` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_run_header(path: Path) -> dict | None:
    """The run header at `path`, or `None` if there is none yet.

    Raises `CorruptRunHeader` with a clear message if the file cannot be parsed.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptRunHeader(
            f"{path} exists but could not be parsed as the grid run header ({exc}). "
            "This usually means a crash interrupted a write to this file before it "
            "completed, or the file was corrupted or hand-edited. Restore it from a "
            "backup if you have one; otherwise the safest recovery is to move this "
            "file aside and let the next --force-resume start a fresh header, "
            "accepting that this run's session history before this point is lost."
        ) from exc


def write_run_header(
    path: Path,
    *,
    seed: int,
    git_sha: str,
    result_sha: str,
    split: str,
    planned_cells: int,
    workers: int,
    caps: dict | None = None,
) -> None:
    """Write a new run header with an empty session list, overwriting any existing one.

    `caps` are stored at the top level and never rewritten; a session that changes a
    cap records it in its own entry.
    """
    path = Path(path)
    header = {
        "seed": seed,
        "git_sha": git_sha,
        "result_sha": result_sha,
        "split": split,
        "planned_cells": planned_cells,
        "workers": workers,
        **{key: (caps or {}).get(key) for key in CAP_KEYS},
        "total_cost_usd": 0.0,
        "sessions": [],
    }
    _write_json_atomic(path, header)


def append_session(path: Path, *, started_at: str, cells_pending: int, **extra) -> None:
    """Append one session entry to the run header at `path`.

    `**extra` carries the force-resume details when a mismatch was overridden.
    """
    path = Path(path)
    header = read_run_header(path)
    header["sessions"].append({"started_at": started_at, "cells_pending": cells_pending, **extra})
    _write_json_atomic(path, header)


def record_outcome(
    path: Path,
    *,
    completed: int,
    failed: int,
    skipped: int,
    halted_kind: str | None,
    halted_reason: str | None,
    session_cost_usd: float,
    total_cost_usd: float,
    halted_cell: dict | None = None,
) -> None:
    """Record what this session did on its entry, and update the run's total spend.

    `halted_cell` is `Cell.identity()` for the cell that halted the run; it is the only
    place that cell is named, since a halt marks nothing failed. Only the last session
    entry is updated. Does nothing if there is no session entry yet, so it never hides
    the error that actually stopped the run.
    """
    path = Path(path)
    header = read_run_header(path)
    if not header or not header.get("sessions"):
        return
    header["sessions"][-1].update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "halted_kind": halted_kind,
            "halted_reason": halted_reason,
            "halted_cell": halted_cell,
            "cost_usd": session_cost_usd,
        }
    )
    header["total_cost_usd"] = total_cost_usd
    _write_json_atomic(path, header)


def _last_forced_current_sha(header: dict) -> str | None:
    """The `current_result_sha` accepted by the most recent forced session, or `None`."""
    for session in reversed(header.get("sessions", [])):
        if session.get("forced"):
            return session.get("current_result_sha")
    return None


def _last_forced_current_caps(header: dict) -> dict | None:
    """The `current_caps` accepted by the most recent forced session, or `None`.

    So an accepted cap change does not need `--force-resume` on every later session.
    """
    for session in reversed(header.get("sessions", [])):
        if session.get("forced") and session.get("current_caps") is not None:
            return session.get("current_caps")
    return None


def _header_caps(header: dict) -> dict:
    return {key: header.get(key) for key in CAP_KEYS}


def verify_or_raise(
    path: Path,
    *,
    force_resume: bool,
    repo_root: Path = Path("."),
    seed: int | None = None,
    split: str = "tune",
    planned_cells: int = 0,
    workers: int = 1,
    cells_pending: int | None = None,
    caps: dict | None = None,
) -> dict:
    """Start-up check: write the header on the first session, otherwise check that the
    source hash and caps still match it.

    On a mismatch without `force_resume`, raises `FreezeViolation`, unless the current
    value is the one the last forced session already accepted. With `force_resume`, the
    session entry records `forced: True` and both values. The header's own
    `result_sha` is never rewritten. `caps=None` skips the cap check (used in tests).
    """
    path = Path(path)
    current_result_sha = result_sha(repo_root)
    header = read_run_header(path)
    started_at = datetime.now(timezone.utc).isoformat()
    pending = planned_cells if cells_pending is None else cells_pending
    # `run.py` always passes caps; only direct test calls leave them out.
    cap_values = dict(caps) if caps is not None else None

    if header is None:
        from grant_eval import config

        write_run_header(
            path,
            seed=config.GRID_SEED if seed is None else seed,
            git_sha=git_sha(repo_root),
            result_sha=current_result_sha,
            split=split,
            planned_cells=planned_cells,
            workers=workers,
            caps=cap_values,
        )
        append_session(
            path, started_at=started_at, cells_pending=pending, **(cap_values or {})
        )
        return read_run_header(path)

    recorded_result_sha = header["result_sha"]
    sha_ok = recorded_result_sha == current_result_sha or (
        _last_forced_current_sha(header) == current_result_sha
    )
    recorded_caps = _header_caps(header)
    caps_ok = cap_values is None or (
        recorded_caps == cap_values or _last_forced_current_caps(header) == cap_values
    )

    if sha_ok and caps_ok:
        append_session(
            path, started_at=started_at, cells_pending=pending, **(cap_values or {})
        )
        return read_run_header(path)

    if not force_resume:
        if not sha_ok:
            raise FreezeViolation(
                "result_sha mismatch: the run header recorded "
                f"{recorded_result_sha}, but the current result-affecting source "
                f"hashes to {current_result_sha}. Something in config.py, schema.py, "
                "dataset/serve.py or agent/{extract,loop,ralph,router,tools}.py "
                "changed since this run started, which means cells banked before and "
                "after that change are not measurements of the same thing. Pass "
                "--force-resume to continue anyway; both values will be recorded in "
                "the session log."
            )
        drifted = ", ".join(
            f"{key}: header {recorded_caps[key]!r}, now {cap_values[key]!r}"
            for key in CAP_KEYS
            if recorded_caps[key] != cap_values[key]
        )
        raise FreezeViolation(
            f"run cap mismatch ({drifted}). This run was frozen under different "
            "wall-clock/spend/concurrency caps. A cap is not in result_sha and not in "
            "config_hash, but it decides which cells come back timed_out=True — and "
            "a timeout is a SCORED RESULT, not a failure — so resuming "
            "under a different one mixes two regimes into one dataset with nothing "
            "downstream able to tell them apart. Re-run with the recorded values, or "
            "pass --force-resume; both sets will be recorded in the session log."
        )

    extra: dict = {"forced": True}
    if not sha_ok:
        extra["header_result_sha"] = recorded_result_sha
        extra["current_result_sha"] = current_result_sha
    if not caps_ok:
        extra["header_caps"] = recorded_caps
        extra["current_caps"] = cap_values
    append_session(
        path,
        started_at=started_at,
        cells_pending=pending,
        **(cap_values or {}),
        **extra,
    )
    return read_run_header(path)
