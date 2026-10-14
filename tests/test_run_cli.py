"""Tests for the grid CLI in `run.py`.

Covers `--dry-run`, the validation split's lock file, and one full run through
preflight to `run_grid` with a mock backend.
"""
import json
import re
from pathlib import Path

import pytest

from grant_eval import run
from grant_eval.grid.executor import MAX_RATE_LIMIT_REQUEUES, GridOutcome

pytestmark = pytest.mark.usefixtures("git_checkout")


def test_dry_run_prints_1920_cells_in_seeded_order(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    rc = run.main(["--dry-run", "--prompt", "minimal"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "480 cells" in out  # 1 prompt x 3 models x 2 temps x 5 reps x 16 funders


def test_dry_run_full_grid_is_1920(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    rc = run.main(["--dry-run"])
    assert rc == 0
    assert "1920 cells" in capsys.readouterr().out


def test_dry_run_order_is_stable_across_invocations(capsys, monkeypatch):
    """Two identical dry runs print the same plan, so the shuffle is seeded."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    run.main(["--dry-run", "--prompt", "minimal", "--limit", "20"])
    first = capsys.readouterr().out
    run.main(["--dry-run", "--prompt", "minimal", "--limit", "20"])
    assert capsys.readouterr().out == first


def test_dry_run_executes_nothing(tmp_path, monkeypatch):
    """A dry run writes nothing to disk, checked on the filesystem rather than stdout."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    run.main(["--dry-run", "--prompt", "minimal",
              "--results-root", str(tmp_path)])
    assert not (tmp_path / "grid").exists()
    assert list(tmp_path.iterdir()) == []


# --replicate
def _replicates_in(out: str) -> set[int]:
    """The replicate numbers of every cell id printed by a --dry-run."""
    return {int(m) for m in re.findall(r"_rep(\d+)_", out)}


def test_replicate_narrows_the_plan_to_one_replicate(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert run.main(["--dry-run", "--funder", "leverhulme", "--replicate", "1"]) == 0
    out = capsys.readouterr().out
    assert "24 cells" in out  # 3 models x 4 prompts x 2 temps x 1 replicate
    assert _replicates_in(out) == {1}


def test_replicate_is_repeatable(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert run.main([
        "--dry-run", "--funder", "leverhulme", "--replicate", "1", "--replicate", "3",
    ]) == 0
    out = capsys.readouterr().out
    assert "48 cells" in out
    assert _replicates_in(out) == {1, 3}


def test_omitting_replicate_keeps_all_five(capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert run.main(["--dry-run", "--funder", "leverhulme"]) == 0
    out = capsys.readouterr().out
    assert "120 cells" in out
    assert _replicates_in(out) == {1, 2, 3, 4, 5}


def test_replicate_makes_the_run_scoped():
    p = run.build_parser()
    assert run.is_scoped(p.parse_args(["--replicate", "1"])) is True


def test_an_out_of_range_replicate_is_refused_rather_than_running_nothing(
    capsys, monkeypatch
):
    """`--replicate 6` is refused, since there are only 5 replicates."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with pytest.raises(SystemExit) as exc:
        run.main(["--dry-run", "--funder", "leverhulme", "--replicate", "6"])
    message = str(exc.value)
    assert "6" in message and "1..5" in message
    assert "0 cells" not in capsys.readouterr().out


def test_the_preregistered_dry_run_is_120_cells(capsys, monkeypatch):
    """The pre-registered dry run: 5 funders x 3 models x 4 prompts x 2 temps x 1 rep = 120."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert run.main([
        "--dry-run",
        "--funder", "cara", "--funder", "british-academy",
        "--funder", "carnegie-trust", "--funder", "leverhulme", "--funder", "nuffield",
        "--replicate", "1",
    ]) == 0
    out = capsys.readouterr().out
    assert "120 cells" in out
    assert _replicates_in(out) == {1}


def test_validation_refuses_without_a_lock_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(run, "LOCKED_CONFIG_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit) as exc:
        run.main(["--split", "validation", "--dry-run"])
    assert exc.value.code != 0


def test_validation_runs_only_the_locked_cells(capsys, tmp_path, monkeypatch):
    """Only the cells named in the lock file are run.

    The lock file uses the last three cells, so taking the first N cells would fail.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from grant_eval.dataset.registry import load_registry
    from grant_eval.grid.cells import enumerate_cells
    from grant_eval.schema import Split

    reg = load_registry(Path("data/funders.json"))
    cells = enumerate_cells(reg, Split.VALIDATION, models=["nova-lite"],
                            prompts=["structured"], replicates=[1])
    chosen_cells = cells[-3:]
    assert {c.funder for c in chosen_cells} != {cells[0].funder}, (
        "the tail slice must not accidentally land back on the head funder, "
        "or this stops being a non-prefix selection"
    )
    chosen = [c.cell_id for c in chosen_cells]
    lock = tmp_path / "locked-config.json"
    lock.write_text(json.dumps({
        "cells": chosen, "locked_at": "2026-09-05T12:00:00Z",
        "tune_evidence": "eval/results/phase1/tidy_20260905T120000Z.parquet",
    }), encoding="utf-8")
    monkeypatch.setattr(run, "LOCKED_CONFIG_PATH", lock)

    assert run.main(["--split", "validation", "--dry-run"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    printed_ids, summary_line = set(lines[:-1]), lines[-1]
    assert printed_ids == set(chosen)
    assert "3 cells" in summary_line


def test_malformed_lock_file_is_refused(tmp_path, monkeypatch):
    """A lock file missing required keys is refused."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    lock = tmp_path / "locked-config.json"
    lock.write_text(json.dumps({"cells": []}), encoding="utf-8")
    monkeypatch.setattr(run, "LOCKED_CONFIG_PATH", lock)
    with pytest.raises(SystemExit):
        run.main(["--split", "validation", "--dry-run"])


def test_lock_file_with_empty_cells_list_is_refused(tmp_path, monkeypatch):
    """A lock file with every key but an empty `cells` list is refused."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    lock = tmp_path / "locked-config.json"
    lock.write_text(json.dumps({
        "cells": [], "locked_at": "2026-09-05T12:00:00Z",
        "tune_evidence": "eval/results/phase1/tidy_20260905T120000Z.parquet",
    }), encoding="utf-8")
    monkeypatch.setattr(run, "LOCKED_CONFIG_PATH", lock)
    with pytest.raises(SystemExit):
        run.main(["--split", "validation", "--dry-run"])


def test_scoped_invocation_is_detected():
    p = run.build_parser()
    assert run.is_scoped(p.parse_args(["--funder", "leverhulme"])) is True
    assert run.is_scoped(p.parse_args(["--limit", "5"])) is True
    assert run.is_scoped(p.parse_args(["--replicate", "1"])) is True
    assert run.is_scoped(p.parse_args([])) is False


def test_end_to_end_production_path_runs_a_real_cell(tmp_path, monkeypatch):
    """A full run through preflight to `run_grid` writes the header and one result.

    Checks the files rather than the return code, which could be 0 even if nothing ran.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage

    def mock_backend_factory():
        return MockBackend(
            replies=[Reply(tool_calls=[
                ToolCall("c1", "submit_opportunities", {"opportunities": []})
            ])],
            usages=[Usage(100, 20, 0.001, "mock", "mock/model")],
        )

    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", mock_backend_factory
    )

    results_root = tmp_path / "results"
    traces_root = tmp_path / "traces"

    rc = run.main([
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
    ])
    assert rc == 0

    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["planned_cells"] == 1
    assert len(header["sessions"]) == 1

    result_files = [
        p for p in (results_root / "grid").rglob("*.json")
        if not p.name.endswith(".failed.json")
    ]
    assert len(result_files) == 1
    result = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert result["funder"] == "The Leverhulme Trust"
    assert result["model"] == "openrouter/amazon/nova-lite-v1"


# Defaults
def test_the_iteration_timeout_default_is_a_per_request_180s():
    """The iteration timeout defaults to 180s.

    litellm applies it per request, and one Ralph iteration can make up to
    `step_budget + 1` requests, so a large value adds up quickly.
    """
    assert run.build_parser().parse_args([]).iteration_timeout_s == 180.0


def test_the_traces_root_default_does_not_double_the_grid_segment():
    """`Cell.trace_dir` adds `grid/` itself, so the default root must not end in `grid`."""
    args = run.build_parser().parse_args([])
    assert args.traces_root == Path("eval/final_grid_traces")
    assert args.traces_root.name != "grid"


def _mock_backend_factory(cost=0.001):
    from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage

    def factory():
        return MockBackend(
            replies=[Reply(tool_calls=[
                ToolCall("c1", "submit_opportunities", {"opportunities": []})
            ])],
            usages=[Usage(100, 20, cost, "mock", "mock/model")],
        )

    return factory


def _run(monkeypatch, results_root, traces_root, *extra, cost=0.001):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", _mock_backend_factory(cost)
    )
    return run.main([
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
        *extra,
    ])


def test_traces_land_under_one_grid_segment(tmp_path, monkeypatch):
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0
    dirs = [p for p in (traces_root / "grid").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    assert not (dirs[0] / "grid").exists()
    assert list(dirs[0].glob("*.jsonl.gz"))


def test_the_run_header_records_the_caps_and_the_session_outcome(tmp_path, monkeypatch):
    """The run header records every cap, the pending count and what the session did."""
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "2") == 0

    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["cell_timeout_s"] == 5400.0
    assert header["iteration_timeout_s"] == 180.0
    assert header["max_rate_limit_requeues"] == 10
    assert header["max_spend_usd"] == 150.0
    assert header["per_arm_limit"] == 1
    session = header["sessions"][-1]
    assert session["cells_pending"] == 2
    assert session["completed"] == 2
    assert session["failed"] == 0
    assert session["halted_kind"] is None
    assert session["cost_usd"] == pytest.approx(0.002)
    assert "finished_at" in session
    assert header["total_cost_usd"] == pytest.approx(0.002)


def test_deferred_cells_make_the_process_exit_non_zero(tmp_path, monkeypatch):
    """Deferred cells give a non-zero exit code even without a halt.

    `run_grid` is stubbed, so this only checks the return code.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", _mock_backend_factory()
    )

    def fake_run_grid(plan, **kwargs):
        return GridOutcome(
            completed=0, failed=0, skipped=len(plan), deferred=len(plan),
            total_cost_usd=0.0, halted_kind=None,
        )

    monkeypatch.setattr(run, "run_grid", fake_run_grid)

    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    rc = run.main([
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
    ])
    assert rc == 1, "a run with deferred cells but no halt must not exit 0"


def test_a_resume_records_only_what_is_still_pending(tmp_path, monkeypatch):
    """A resumed session only counts the cells and spend left, while the total adds up.

    This also checks the spend guard starts from earlier spend rather than zero.
    """
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0
    assert _run(monkeypatch, results_root, traces_root, "--limit", "2") == 0

    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    first, second = header["sessions"]
    assert first["cells_pending"] == 1
    assert second["cells_pending"] == 1, (
        "the resumed session claimed the whole plan was outstanding"
    )
    assert second["cost_usd"] == pytest.approx(0.001)
    assert header["total_cost_usd"] == pytest.approx(0.002), (
        "the run's cumulative spend restarted at zero, so --max-spend-usd is a "
        "per-process cap and a six-night run has six times the ceiling it names"
    )


def test_a_resume_under_a_different_cap_is_refused_by_the_cli(tmp_path, monkeypatch):
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0
    with pytest.raises(SystemExit, match="cap mismatch"):
        _run(monkeypatch, results_root, traces_root, "--limit", "1",
             "--cell-timeout-s", "60")
    # --force-resume lets it through and records both sets of caps
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1",
                "--cell-timeout-s", "60", "--force-resume") == 0
    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["sessions"][-1]["header_caps"]["cell_timeout_s"] == 5400.0
    assert header["sessions"][-1]["current_caps"]["cell_timeout_s"] == 60.0


# --max-rate-limit-requeues
def test_the_requeue_cap_defaults_to_the_executor_constant():
    """The flag's default is the executor's own constant."""
    args = run.build_parser().parse_args([])
    assert args.max_rate_limit_requeues == MAX_RATE_LIMIT_REQUEUES


def test_the_requeue_cap_is_threaded_to_run_grid_and_stamped(tmp_path, monkeypatch):
    """The requeue cap is passed through to `run_grid` and written to the header."""
    seen: dict = {}

    def fake_run_grid(plan, **kwargs):
        seen.update(kwargs)
        return GridOutcome(skipped=len(plan), total_cost_usd=0.0)

    monkeypatch.setattr(run, "run_grid", fake_run_grid)
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1",
                "--max-rate-limit-requeues", "3") == 0
    assert seen["max_rate_limit_requeues"] == 3
    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["max_rate_limit_requeues"] == 3


def test_a_non_positive_requeue_cap_is_refused(monkeypatch):
    """A requeue cap below 1 is refused, since the first 429 would then fail the cell."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with pytest.raises(SystemExit, match="at least 1"):
        run.main(["--dry-run", "--max-rate-limit-requeues", "0"])


def test_retry_failed_re_runs_a_failed_cell_end_to_end(tmp_path, monkeypatch):
    """A failed cell is skipped on resume, then re-run and cleared with `--retry-failed`."""
    from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    fail = {"on": True}

    class _FlakyBackend(MockBackend):
        def complete(self, messages, model, temperature, tools):
            if fail["on"]:
                exc = ConnectionError("upstream connection reset")
                raise exc
            return super().complete(messages, model, temperature, tools)

    def factory():
        return _FlakyBackend(
            replies=[Reply(tool_calls=[
                ToolCall("c1", "submit_opportunities", {"opportunities": []})
            ])],
            usages=[Usage(100, 20, 0.001, "mock", "mock/model")],
        )

    monkeypatch.setattr("grant_eval.agent.router.OpenRouterBackend", factory)
    argv = [
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
    ]

    assert run.main(argv) == 0  # a retryable failure does not halt the run
    sentinels = list((results_root / "grid").rglob("*.failed.json"))
    assert len(sentinels) == 1
    assert json.loads(sentinels[0].read_text(encoding="utf-8"))["attempts"] == 2

    # without --retry-failed the failed cell is skipped
    fail["on"] = False
    assert run.main(argv) == 0
    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["sessions"][-1]["cells_pending"] == 0
    assert header["sessions"][-1]["completed"] == 0
    assert list((results_root / "grid").rglob("*.failed.json")) == sentinels

    # with it, the cell runs and the sentinel is removed
    assert run.main([*argv, "--retry-failed"]) == 0
    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    assert header["sessions"][-1]["cells_pending"] == 1
    assert header["sessions"][-1]["completed"] == 1
    assert list((results_root / "grid").rglob("*.failed.json")) == []
    results = [
        p for p in (results_root / "grid").rglob("*.json")
        if not p.name.endswith(".failed.json")
    ]
    assert len(results) == 1


def test_orphaned_traces_are_swept_at_the_end_of_a_run(tmp_path, monkeypatch):
    """Leftover `*.jsonl` traces from abandoned cells are gzipped at the end of a run."""
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    orphan = traces_root / "grid" / "some-abandoned-cell"
    orphan.mkdir(parents=True)
    (orphan / "iter1.jsonl").write_text('{"orphan": true}\n', encoding="utf-8")

    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0
    assert not (orphan / "iter1.jsonl").exists()
    assert (orphan / "iter1.jsonl.gz").is_file()


def test_each_per_cell_line_is_timestamped(tmp_path, monkeypatch, capsys):
    """Each per-cell line on stdout starts with a timestamp."""
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0
    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if "completed: " in line
    ]
    assert lines
    assert all(
        re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z ", line) for line in lines
    ), lines


def test_the_halting_cell_reaches_the_run_header(tmp_path, monkeypatch):
    """When a cell halts the run, the header and the log both name that cell.

    Uses a malformed-request 400, because a context overflow 400 no longer halts.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    class _MalformedRequest:
        name = "boom"

        def complete(self, messages, model, temperature, tools):
            exc = Exception('Input required: specify "prompt" or "messages"')
            exc.status_code = 400  # `policy.classify` treats this as HALT
            raise exc

    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", lambda: _MalformedRequest()
    )
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    rc = run.main([
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
    ])
    assert rc == 1

    session = json.loads(
        (results_root / "grid_run.json").read_text(encoding="utf-8")
    )["sessions"][-1]
    assert session["halted_kind"] == "error"
    assert session["failed"] == 0, "a halt still marks nothing failed"
    halted = session["halted_cell"]
    assert halted is not None, (
        "the header recorded a halt without naming the cell that caused it"
    )
    assert halted["funder"] == "leverhulme"
    assert halted["model"] == "nova-lite"
    assert halted["model_slug"] == "openrouter/amazon/nova-lite-v1"
    assert halted["prompt_strategy"] == "minimal"
    assert halted["temperature"] in (0.0, 0.7)
    assert halted["replicate"] in range(1, 6)
    assert halted["cell_id"].startswith("leverhulme_nova-lite-v1_minimal_")

    log_text = (results_root / "grid_run.log").read_text(encoding="utf-8")
    assert halted["cell_id"] in log_text
    assert "Input required" in log_text


def test_the_run_log_lands_beside_the_header_and_a_resume_appends_to_it(
    tmp_path, monkeypatch
):
    """The run log sits next to the header, and a resumed run appends to it."""
    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    assert _run(monkeypatch, results_root, traces_root, "--limit", "1") == 0

    log_path = results_root / "grid_run.log"
    assert log_path.is_file()
    first = log_path.read_text(encoding="utf-8")
    assert "completed: " in first

    assert _run(monkeypatch, results_root, traces_root, "--limit", "2") == 0
    second = log_path.read_text(encoding="utf-8")
    assert second.startswith(first), "the resumed session overwrote the first's log"
    assert len(second) > len(first)


def test_a_dry_run_writes_no_log_file(tmp_path, monkeypatch):
    """A dry run does not create a log file."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    run.main(["--dry-run", "--prompt", "minimal", "--results-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_an_unexpected_exception_still_records_the_session_outcome(tmp_path, monkeypatch):
    """The session outcome is still written when `run_grid` raises.

    Uses a plain exception, because `KeyboardInterrupt` is caught and would not test this.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", _mock_backend_factory()
    )

    def boom(plan, **kwargs):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(run, "run_grid", boom)

    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    with pytest.raises(RuntimeError, match="worker exploded"):
        run.main([
            "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
            "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
            "--results-root", str(results_root), "--traces-root", str(traces_root),
        ])

    header = json.loads((results_root / "grid_run.json").read_text(encoding="utf-8"))
    session = header["sessions"][-1]
    assert "finished_at" in session, "the session was left open by a crashing run"
    assert session["skipped"] == 1, "the unrun cell must be recorded as not run"


def test_an_interrupt_exits_non_zero_rather_than_raising(tmp_path, monkeypatch):
    """Ctrl-C makes `main()` return 1 instead of raising `KeyboardInterrupt`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", _mock_backend_factory()
    )

    def interrupt(plan, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "run_grid", interrupt)

    results_root, traces_root = tmp_path / "results", tmp_path / "traces"
    rc = run.main([
        "--funder", "leverhulme", "--model", "nova-lite", "--prompt", "minimal",
        "--limit", "1", "--workers", "1", "--per-arm-limit", "1",
        "--results-root", str(results_root), "--traces-root", str(traces_root),
    ])
    assert rc == 1, "an interrupted run must not exit 0"
