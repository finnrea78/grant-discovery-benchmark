"""Tests for the dispatcher's logging setup (`grid/logs.py`).

INFO goes to stdout and DEBUG goes to a run log next to `grid_run.json`.
"""
from __future__ import annotations

import logging

import pytest

from grant_eval.grid import executor, logs


@pytest.fixture(autouse=True)
def restore_package_logger():
    """Restore the package logger after each test, since it is process-global."""
    logger = logs.package_logger()
    saved_handlers = list(logger.handlers)
    saved_level, saved_propagate = logger.level, logger.propagate
    yield
    added = [h for h in logger.handlers if h not in saved_handlers]
    logger.handlers = saved_handlers
    for handler in added:
        handler.close()
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


def test_the_run_log_lands_beside_the_run_header(tmp_path):
    path = logs.configure(tmp_path)
    assert path == tmp_path / logs.RUN_LOG_NAME
    logging.getLogger("grant_eval.grid.executor").info("completed: some_cell_id")
    assert "completed: some_cell_id" in path.read_text(encoding="utf-8")


def test_a_second_configure_appends_rather_than_overwriting(tmp_path):
    """Resuming a run into the same results root keeps the earlier log lines."""
    path = logs.configure(tmp_path)
    logging.getLogger("grant_eval.grid.executor").info("night one")
    logs.configure(tmp_path)
    logging.getLogger("grant_eval.grid.executor").info("night two")
    text = path.read_text(encoding="utf-8")
    assert "night one" in text and "night two" in text


def test_only_one_run_log_is_attached_per_process(tmp_path):
    """Pointing `configure` at a new results root stops writing to the old one."""
    first = logs.configure(tmp_path / "run-a")
    logging.getLogger("grant_eval.grid.executor").info("before the move")
    second = logs.configure(tmp_path / "run-b")
    logging.getLogger("grant_eval.grid.executor").info("after the move")
    assert "after the move" in second.read_text(encoding="utf-8")
    assert "after the move" not in first.read_text(encoding="utf-8")


def test_debug_reaches_the_run_log_and_not_the_console(tmp_path, capsys):
    """DEBUG lines go to the run log file but not to stdout."""
    path = logs.configure(tmp_path)
    router_log = logging.getLogger("grant_eval.agent.router")
    router_log.debug("completion request attempt=1 approx_chars=317000")
    router_log.info("completed: a_cell")

    out = capsys.readouterr().out
    assert "completed: a_cell" in out
    assert "approx_chars" not in out, (
        "DEBUG reached the console; a 1,920-cell run would be unreadable"
    )
    assert "approx_chars=317000" in path.read_text(encoding="utf-8")


def test_the_env_var_can_put_debug_on_the_console(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(logs.LEVEL_ENV_VAR, "DEBUG")
    logs.configure(tmp_path)
    logging.getLogger("grant_eval.agent.router").debug("approx_chars=1")
    assert "approx_chars=1" in capsys.readouterr().out


def test_an_unparseable_level_falls_back_rather_than_raising(monkeypatch):
    """A typo in the level env var falls back to INFO."""
    monkeypatch.setenv(logs.LEVEL_ENV_VAR, "VERBOSE-ISH")
    assert logs.level_from_env() == logging.INFO


def test_log_line_reaches_stdout_without_any_configure_call(capsys):
    """`executor.log_line` still prints when `configure` was never called."""
    executor.log_line("arm nova-lite saturated: returning a_cell to pending")
    assert "saturated" in capsys.readouterr().out
