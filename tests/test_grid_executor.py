"""Tests for the threaded grid executor.

Each test runs the real server, Toolset, agent loop and Ralph loop and only fakes the
model. Many tests include a control so they cannot pass for the wrong reason.
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage
from grant_eval.dataset.registry import load_registry
from grant_eval.grid import policy, store
from grant_eval.grid.cells import enumerate_cells, ordered_plan
from grant_eval.grid import executor
from grant_eval.grid.executor import (
    RATE_LIMIT_BASE_BACKOFF_S,
    GridOutcome,
    run_cell,
    run_grid,
    serve_cell,
)
from grant_eval.grid.policy import CircuitBreaker, SpendGuard
from grant_eval.schema import ExtractionResult, Split

REG = load_registry(Path("data/funders.json"))

# Two real tune-split funders with real mirrors on disk: british-academy is the
# smallest tune manifest (157 files) and british-ecological-society one of the larger
# (856), so the per-cell `build_lookup` cost is exercised at both ends.
FUNDERS = ["british-academy", "british-ecological-society"]


def _plan(funders=FUNDERS, replicates=(1,), models=("nova-lite",), seed=1):
    """A small real plan, with both temperatures for each funder, replicate and model."""
    cells = enumerate_cells(
        REG,
        Split.TUNE,
        funders=list(funders),
        models=list(models),
        prompts=["minimal"],
        replicates=list(replicates),
    )
    return ordered_plan(cells, seed)


def _submit(opportunities=()) -> Reply:
    return Reply(
        tool_calls=[
            ToolCall("c1", "submit_opportunities", {"opportunities": list(opportunities)})
        ]
    )


class _Recorder:
    """Thread-safe count of backend calls.

    Every script here converges at iteration 1, so `iterations` is also the number of
    cell runs.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.iterations = 0
        self.models: list[str] = []
        self.live = 0
        self.peak = 0

    def enter(self, model: str) -> None:
        with self.lock:
            self.iterations += 1
            self.models.append(model)
            self.live += 1
            self.peak = max(self.peak, self.live)

    def leave(self) -> None:
        with self.lock:
            self.live -= 1


def _ok_factory(rec: _Recorder | None = None, cost: float = 0.001, hold: float = 0.0):
    """A backend that submits an empty set, so Ralph stops at iteration 1.

    `hold` keeps `complete()` busy for a while so overlapping calls can be seen.
    """
    recorder = rec if rec is not None else _Recorder()

    class _Backend(MockBackend):
        def complete(self, messages, model, temperature, tools):
            recorder.enter(model)
            try:
                if hold:
                    time.sleep(hold)
                return super().complete(messages, model, temperature, tools)
            finally:
                recorder.leave()

    def factory():
        return _Backend(
            replies=[_submit()], usages=[Usage(100, 20, cost, "mock", "mock/model")]
        )

    return factory


def _raising_factory(exc_maker, rec: _Recorder | None = None):
    """A backend whose `complete()` always raises `exc_maker()`."""
    recorder = rec if rec is not None else _Recorder()

    class _Boom:
        name = "boom"

        def complete(self, messages, model, temperature, tools):
            recorder.enter(model)
            try:
                raise exc_maker()
            finally:
                recorder.leave()

    return lambda: _Boom()


def _status_error(status: int, message: str) -> Exception:
    exc = Exception(message)
    exc.status_code = status
    return exc


def _context_overflow_error() -> Exception:
    """The context overflow error OpenRouter returns through litellm (a plain BadRequestError).

    litellm is imported lazily because a cold import takes a few seconds.
    """
    import litellm.exceptions as litellm_exc

    return litellm_exc.BadRequestError(
        message=(
            "OpenrouterException - This endpoint's maximum context length is 300000 "
            "tokens. However, you requested about 334283 tokens (334283 of text input)."
        ),
        model="openrouter/amazon/nova-lite-v1",
        llm_provider="openrouter",
    )


class _RecordingBreaker(CircuitBreaker):
    """A CircuitBreaker that remembers every `record()` call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.records: list[bool] = []

    def record(self, ok: bool) -> None:
        self.records.append(ok)
        super().record(ok)


def _record(sink: list):
    """A delay stand-in that records the requested delay.

    It accepts `interrupt` because the breaker pause passes one, and a TypeError there would
    hang the suite instead of failing it.
    """

    def delay(seconds: float, interrupt=None) -> None:
        sink.append(seconds)

    return delay


def _grid(plan, tmp_path, *, name="run", **kwargs):
    """`run_grid` with this module's defaults filled in."""
    kwargs.setdefault("workers", 2)
    kwargs.setdefault("per_arm_limit", 2)
    kwargs.setdefault("backend_factory", _ok_factory())
    kwargs.setdefault("breaker", CircuitBreaker())
    kwargs.setdefault("spend_guard", SpendGuard(cap_usd=None))
    kwargs.setdefault("iteration_timeout_s", 60)
    kwargs.setdefault("cell_timeout_s", 300)
    kwargs.setdefault("results_root", tmp_path / name / "results")
    kwargs.setdefault("traces_root", tmp_path / name / "traces")
    return run_grid(plan, **kwargs)


def test_serve_cell_yields_a_live_base_url_and_the_handler_s_own_misses_counter():
    """The yielded counter is the same one the handler writes misses into."""
    rec = REG["british-academy"]
    with serve_cell(rec.snapshot_dir()) as (base_url, misses):
        assert base_url.startswith("http://127.0.0.1:")
        assert misses == {}

        with urllib.request.urlopen(f"{base_url}/funding/", timeout=15) as resp:
            assert resp.status == 200

        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base_url}/not-in-the-mirror", timeout=15)
        assert misses["/not-in-the-mirror"] == 1

    # torn down on exit
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"{base_url}/funding/", timeout=2)


def test_run_grid_writes_one_valid_result_per_cell(tmp_path):
    """End to end through serve -> Toolset -> loop -> Ralph -> result file."""
    plan = _plan()
    rec = _Recorder()
    outcome = _grid(plan, tmp_path, backend_factory=_ok_factory(rec))

    assert isinstance(outcome, GridOutcome)
    assert outcome.completed == len(plan)
    assert outcome.failed == 0
    assert outcome.skipped == 0
    assert outcome.halted_reason is None
    assert outcome.total_cost_usd == pytest.approx(0.001 * len(plan))
    assert rec.iterations == len(plan)

    for cell in plan:
        assert store.is_complete(cell, tmp_path / "run" / "results")


def test_result_carries_the_display_name_while_the_path_carries_the_slug(tmp_path):
    """The result holds the funder display name while the file path uses the slug."""
    plan = _plan(funders=["british-academy"])
    _grid(plan, tmp_path)

    cell = plan[0]
    path = cell.result_path(tmp_path / "run" / "results")
    assert path.parent.name == "british-academy"
    result = ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert result.funder == "The British Academy"
    assert result.split is Split.TUNE
    assert result.snapshot_date == cell.snapshot_date
    assert result.config_hash == cell.config_hash
    assert result.timed_out is False


def test_results_are_written_through_the_atomic_store(tmp_path, monkeypatch):
    """Results are written through `store.write_result_atomic`, not a plain write."""
    calls: list[Path] = []
    real = store.write_result_atomic

    def spy(path, result):
        calls.append(Path(path))
        return real(path, result)

    monkeypatch.setattr(store, "write_result_atomic", spy)
    # the executor imports the module, not the name, so patching `store` is enough
    plan = _plan(funders=["british-academy"])
    _grid(plan, tmp_path)

    assert sorted(calls) == sorted(c.result_path(tmp_path / "run" / "results") for c in plan)


def test_resume_reruns_nothing_and_skips_nothing(tmp_path):
    """Resuming over the whole plan re-runs no finished cell and skips no pending one."""
    plan = _plan(replicates=(1, 2))
    results_root = tmp_path / "results"
    traces_root = tmp_path / "traces"
    first_half = plan[: len(plan) // 2]
    assert first_half and len(first_half) < len(plan)

    rec1 = _Recorder()
    run_grid(
        first_half,
        results_root=results_root,
        traces_root=traces_root,
        workers=2,
        per_arm_limit=2,
        backend_factory=_ok_factory(rec1),
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        iteration_timeout_s=60,
        cell_timeout_s=300,
    )
    assert rec1.iterations == len(first_half)
    before = {
        c.cell_id: (
            c.result_path(results_root).read_bytes(),
            c.result_path(results_root).stat().st_mtime_ns,
        )
        for c in first_half
    }

    remaining = store.pending(plan, results_root, retry_failed=False)
    assert {c.cell_id for c in remaining} == {
        c.cell_id for c in plan if c.cell_id not in before
    }

    # The resume pass gets the whole plan, so the executor has to drop the finished
    # half itself, as a real resume re-deriving the plan from GRID_SEED would.
    rec2 = _Recorder()
    outcome = run_grid(
        plan,
        results_root=results_root,
        traces_root=traces_root,
        workers=2,
        per_arm_limit=2,
        backend_factory=_ok_factory(rec2),
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        iteration_timeout_s=60,
        cell_timeout_s=300,
    )

    assert rec2.iterations == len(plan) - len(first_half)
    assert outcome.completed == len(plan) - len(first_half)
    assert outcome.skipped == len(first_half)
    for cell in first_half:
        path = cell.result_path(results_root)
        assert path.read_bytes() == before[cell.cell_id][0], f"re-ran {cell.cell_id}"
        assert path.stat().st_mtime_ns == before[cell.cell_id][1], f"re-ran {cell.cell_id}"

    for cell in plan:
        assert store.is_complete(cell, results_root), f"skipped {cell.cell_id}"


def test_resume_after_a_mid_run_halt_completes_every_cell_exactly_once(tmp_path):
    """Resume still works when the first run halted part-way rather than at a clean split.

    An executor resuming from a counter or plan index would pass the test above but not this.
    """
    plan = _plan(replicates=(1, 2))
    results_root = tmp_path / "results"
    traces_root = tmp_path / "traces"

    stop_after = 3
    rec1 = _Recorder()
    ok = _ok_factory(rec1)

    def halting_factory():
        with rec1.lock:
            n = rec1.iterations
        if n >= stop_after:
            # 401 is HALT-class: the grid stops, marking nothing failed.
            return _raising_factory(lambda: _status_error(401, "unauthorized"))()
        return ok()

    first = run_grid(
        plan,
        results_root=results_root,
        traces_root=traces_root,
        workers=1,
        per_arm_limit=1,
        backend_factory=halting_factory,
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        iteration_timeout_s=60,
        cell_timeout_s=300,
    )
    assert first.halted_reason is not None
    banked = [c for c in plan if store.is_complete(c, results_root)]
    assert 0 < len(banked) < len(plan)
    before = {c.cell_id: c.result_path(results_root).read_bytes() for c in banked}

    rec2 = _Recorder()
    outcome = run_grid(
        plan,
        results_root=results_root,
        traces_root=traces_root,
        workers=2,
        per_arm_limit=2,
        backend_factory=_ok_factory(rec2),
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        iteration_timeout_s=60,
        cell_timeout_s=300,
    )

    assert rec2.iterations == len(plan) - len(banked)
    assert outcome.completed == len(plan) - len(banked)
    assert outcome.skipped == len(banked)
    for cell in banked:
        assert cell.result_path(results_root).read_bytes() == before[cell.cell_id]
    for cell in plan:
        assert store.is_complete(cell, results_root)


def test_resume_skips_failed_cells_unless_retry_failed(tmp_path):
    """A resume skips failed cells unless `retry_failed` is set."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    results_root = tmp_path / "results"
    traces_root = tmp_path / "traces"

    rec = _Recorder()
    _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=traces_root,
        workers=1,
        backend_factory=_raising_factory(lambda: ConnectionError("reset"), rec),
    )
    assert all(store.is_failed(c, results_root) for c in plan)
    burned = rec.iterations

    rec2 = _Recorder()
    outcome = _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=traces_root,
        workers=1,
        backend_factory=_raising_factory(lambda: ConnectionError("reset"), rec2),
    )
    assert rec2.iterations == 0
    assert outcome.skipped == len(plan)

    rec3 = _Recorder()
    _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=traces_root,
        workers=1,
        retry_failed=True,
        backend_factory=_raising_factory(lambda: ConnectionError("reset"), rec3),
    )
    assert rec3.iterations == burned


def _peak_concurrency(tmp_path, name, *, per_arm_limit, workers=8, hold=0.25):
    plan = _plan(replicates=(1, 2))  # 8 cells, all on the same arm
    rec = _Recorder()
    _grid(
        plan,
        tmp_path,
        name=name,
        workers=workers,
        per_arm_limit=per_arm_limit,
        backend_factory=_ok_factory(rec, hold=hold),
    )
    return rec.peak, len(plan)


def test_per_arm_semaphore_caps_concurrency_and_the_workload_can_exceed_it(tmp_path):
    """The per-arm limit caps concurrency, with an uncapped run as a control.

    Each call is held open so cells really overlap, otherwise `peak <= limit` would pass
    trivially.
    """
    capped, n_cells = _peak_concurrency(tmp_path, "capped", per_arm_limit=2)
    assert capped <= 2, f"per-arm limit of 2 was exceeded (peak {capped})"
    assert capped == 2, f"the workload never contended (peak {capped}); test is vacuous"

    uncapped, _ = _peak_concurrency(tmp_path, "uncapped", per_arm_limit=n_cells)
    assert uncapped > 2, (
        f"control: the same workload under a limit of {n_cells} only reached "
        f"{uncapped} concurrent, so the capped result proves nothing"
    )


def test_arms_do_not_block_each_other_during_a_rate_limit_backoff(tmp_path):
    """A 429 backs off only that arm, so cells on other arms can still start.

    One rate-limited cell is followed by three on another arm with two workers, so other-arm
    cells are still queued when the backoff begins.
    """
    limited = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    others = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1, 2],
    )[:3]
    plan = limited + others
    limited_slug = limited[0].model_slug

    starts = {"other_arm": 0}
    lock = threading.Lock()
    hits = {"n": 0}
    observed: dict[str, bool] = {}
    other_arm_running = threading.Event()
    ok = _ok_factory()

    class _Sometimes429:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == limited_slug:
                with lock:
                    hits["n"] += 1
                    n = hits["n"]
                if n == 1:
                    # Wait until the other arm's cell is counted before backing off, otherwise it
                    # starts during the pause and looks like a new start.
                    other_arm_running.wait(5.0)
                    raise _status_error(429, "rate limited")
            else:
                with lock:
                    starts["other_arm"] += 1
                other_arm_running.set()
                # Long enough that one other-arm cell is always left to start during it.
                time.sleep(0.3)
            return ok().complete(messages, model, temperature, tools)

    def sleeper(seconds: float, interrupt=None) -> None:
        with lock:
            before = starts["other_arm"]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with lock:
                if starts["other_arm"] > before:
                    observed["other_arm_started"] = True
                    break
            time.sleep(0.02)
        else:
            observed["other_arm_started"] = False

    outcome = _grid(
        plan,
        tmp_path,
        workers=2,
        per_arm_limit=2,
        backend_factory=lambda: _Sometimes429(),
        sleep=sleeper,
    )

    assert outcome.completed == len(plan)
    assert observed.get("other_arm_started") is True, (
        "no cell on another arm was able to START while one arm backed off"
    )


def test_halt_class_error_stops_the_grid_early_and_marks_nothing_failed(tmp_path):
    """A halt-class error stops the grid after one attempt and marks nothing failed."""
    plan = _plan(replicates=(1, 2))
    rec = _Recorder()
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(lambda: _status_error(401, "unauthorized"), rec),
    )

    assert outcome.halted_reason is not None
    assert outcome.halted_kind == "error"
    assert outcome.failed == 0
    assert outcome.completed == 0
    assert rec.iterations == 1, f"the grid kept going after a HALT ({rec.iterations} attempts)"
    for cell in plan:
        assert not store.is_failed(cell, tmp_path / "run" / "results")
        assert not store.is_complete(cell, tmp_path / "run" / "results")


def test_a_halt_records_which_cell_caused_it(tmp_path):
    """A halt records the identity of the cell that caused it.

    Uses a malformed-request 400, which halts, rather than the context-window 400, which is
    recorded as a result.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(
            lambda: _status_error(400, 'Input required: specify "prompt" or "messages"')
        ),
    )

    assert outcome.halted_kind == "error"
    assert outcome.failed == 0
    assert outcome.halted_cell is not None, (
        "the grid halted without recording which cell did it"
    )
    halted = next(
        c for c in plan if c.cell_id == outcome.halted_cell["cell_id"]
    )
    assert outcome.halted_cell == halted.identity()
    assert outcome.halted_cell["funder"] == "british-academy"
    assert outcome.halted_cell["model"] == "nova-lite"
    assert outcome.halted_cell["prompt_strategy"] == "minimal"
    assert outcome.halted_cell["replicate"] == 1


def test_a_clean_run_names_no_halted_cell(tmp_path):
    """A run that never halted has no halted cell."""
    outcome = _grid(_plan(funders=["british-academy"], replicates=(1,)), tmp_path)
    assert outcome.halted_kind is None
    assert outcome.halted_cell is None


def test_retry_class_error_marks_failed_after_exactly_two_attempts(tmp_path):
    """A retry-class error marks the cell failed after MAX_ATTEMPTS and the grid carries on."""
    plan = _plan(replicates=(1,))
    rec = _Recorder()
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(lambda: ConnectionError("reset"), rec),
    )

    assert outcome.failed == len(plan)
    assert outcome.completed == 0
    assert outcome.halted_reason is None, "a failed cell must not stop the grid"
    assert rec.iterations == policy.MAX_ATTEMPTS * len(plan)
    record = store.read_failure(plan[0], tmp_path / "run" / "results")
    assert record["attempts"] == policy.MAX_ATTEMPTS
    assert record["exception_type"] == "ConnectionError"
    assert not plan[0].result_path(tmp_path / "run" / "results").exists()


def test_rate_limit_requeues_the_cell_without_consuming_an_attempt(tmp_path):
    """A 429 re-queues the cell without using up a retry attempt."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    rec = _Recorder()
    ok = _ok_factory(rec)
    state = {"n": 0}
    lock = threading.Lock()
    slept: list[float] = []

    class _RateLimited:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            with lock:
                state["n"] += 1
                n = state["n"]
            if n <= 3:
                raise _status_error(429, "rate limited")
            return ok().complete(messages, model, temperature, tools)

    breaker = _RecordingBreaker()
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=lambda: _RateLimited(),
        breaker=breaker,
        sleep=_record(slept),
    )

    assert outcome.completed == len(plan)
    assert outcome.failed == 0
    assert len(slept) == 3, "one backoff per 429"
    assert slept == sorted(slept), "the per-arm backoff must not shrink"
    # A re-queue is NOT a completed cell and must never reach the breaker.
    assert breaker.records == [True] * len(plan)


def test_a_cell_that_recovers_within_the_requeue_bound_still_completes(tmp_path):
    """A cell that gets past a few 429s within the re-queue limit still completes."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    rec = _Recorder()
    ok = _ok_factory(rec)
    state = {"n": 0}
    lock = threading.Lock()

    class _RateLimited:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            with lock:
                state["n"] += 1
                n = state["n"]
            if n <= 3:
                raise _status_error(429, "engine_overloaded")
            return ok().complete(messages, model, temperature, tools)

    breaker = _RecordingBreaker()
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=lambda: _RateLimited(),
        breaker=breaker,
        sleep=_record([]),
        max_rate_limit_requeues=3,
    )

    assert outcome.completed == len(plan)
    assert outcome.failed == 0
    assert breaker.records == [True] * len(plan)
    assert not store.is_failed(plan[0], tmp_path / "run" / "results")


def test_a_sustained_rate_limit_fails_the_cell_and_the_breaker_hears_it(tmp_path):
    """Past the re-queue limit a rate-limited cell fails and the breaker records it.

    Without a limit the grid could keep restarting the same cell with no failure or trip.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    rec = _Recorder()
    breaker = _RecordingBreaker()
    logged: list[str] = []
    results_root = tmp_path / "run" / "results"

    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(
            lambda: _status_error(429, "engine_overloaded"), rec
        ),
        breaker=breaker,
        sleep=_record([]),
        log=logged.append,
        max_rate_limit_requeues=4,
    )

    assert outcome.failed == len(plan)
    assert outcome.completed == 0
    assert outcome.halted_reason is None, "a failed cell must not stop the grid"
    assert breaker.records == [False] * len(plan)
    # The bound is a REQUEUE count, per cell: four requeues, then the fifth 429 fails it.
    assert rec.iterations == 5 * len(plan)
    record = store.read_failure(plan[0], results_root)
    assert record is not None, "a failed cell must leave a sentinel for resume"
    assert "429" in record["message"] or "engine_overloaded" in record["message"]
    # ...and the log says plainly that this is the requeue cap, naming it.
    assert any("4 requeues" in line and plan[0].cell_id in line for line in logged), logged


def test_breaker_records_once_per_completed_cell_not_once_per_attempt(tmp_path):
    """The breaker records once per finished cell, not once per attempt.

    Its threshold assumes each failure is a cell that already used all its attempts.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    rec = _Recorder()
    breaker = _RecordingBreaker()
    _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=_raising_factory(lambda: ConnectionError("reset"), rec),
    )

    assert rec.iterations == policy.MAX_ATTEMPTS * len(plan), "each cell tried twice"
    assert breaker.records == [False] * len(plan), (
        f"expected one record per completed cell, got {breaker.records}"
    )


def test_breaker_is_not_told_about_cells_a_halt_abandoned(tmp_path):
    """A halt says nothing about any model arm, so the breaker should not hear about it."""
    plan = _plan(replicates=(1, 2))
    breaker = _RecordingBreaker()
    _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=_raising_factory(lambda: _status_error(402, "no balance")),
    )
    assert breaker.records == []


def test_breaker_trip_pauses_new_cells_and_then_auto_resumes(tmp_path):
    """A breaker trip pauses new cells and then resumes on its own.

    The fake sleep takes real time and counts cells started during it, so a pause that only
    sleeps in one thread while the other workers carry on would be caught.
    """
    plan = _plan(funders=["british-academy"], replicates=(1, 2, 3, 4))
    rec = _Recorder()
    # A 2-cell window trips the breaker as soon as the first two failures are recorded.
    breaker = _RecordingBreaker(window=2, threshold=0.5, max_trips=3)
    fail_until = 2
    ok = _ok_factory(rec)

    def factory():
        with rec.lock:
            done = rec.iterations
        if done < fail_until * policy.MAX_ATTEMPTS:
            return _raising_factory(lambda: ConnectionError("reset"), rec)()
        return ok()

    paused_at: list[int] = []
    resumed_at: list[int] = []
    delays: list[float] = []

    def sleeper(seconds: float, interrupt=None) -> None:
        delays.append(seconds)
        with rec.lock:
            paused_at.append(rec.iterations)
        time.sleep(1.5)
        with rec.lock:
            resumed_at.append(rec.iterations)

    outcome = _grid(
        plan,
        tmp_path,
        workers=2,
        per_arm_limit=2,
        breaker=breaker,
        backend_factory=factory,
        sleep=sleeper,
    )

    assert delays, "a trip must pause"
    assert delays[0] == 60.0, "first trip backs off 60s"
    assert resumed_at[0] - paused_at[0] <= 1, (
        f"cells kept starting during the pause: {paused_at[0]} -> {resumed_at[0]}"
    )
    # ...and then auto-resumed: the rest of the grid ran to completion.
    assert outcome.halted_reason is None
    assert outcome.completed == len(plan) - fail_until
    assert outcome.failed == fail_until


def test_breaker_exhaustion_halts_the_grid(tmp_path):
    """Hard-exit after `max_trips` CONSECUTIVE trips, resumably."""
    plan = _plan(replicates=(1, 2))
    breaker = _RecordingBreaker(window=2, threshold=0.5, max_trips=1)
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=_raising_factory(lambda: ConnectionError("reset")),
        sleep=lambda s, interrupt=None: None,
    )

    assert outcome.halted_kind == "circuit_breaker_exhausted"
    assert "circuit breaker" in outcome.halted_reason.lower()
    assert outcome.completed == 0
    assert outcome.failed == 2, "the two cells that did fail are still recorded failed"
    assert outcome.skipped == len(plan) - 2


def test_spend_cap_halts_the_grid_and_does_not_auto_resume(tmp_path):
    """Going over the spend cap halts the grid and, unlike a breaker trip, never auto-resumes."""
    plan = _plan(replicates=(1, 2))
    rec = _Recorder()
    slept: list[float] = []
    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_ok_factory(rec, cost=0.001),
        spend_guard=SpendGuard(cap_usd=0.0015),
        sleep=_record(slept),
    )

    assert outcome.halted_reason is not None
    assert "spend" in outcome.halted_reason.lower()
    assert outcome.halted_kind == "spend_cap"
    assert slept == [], "a spend-cap halt must not pause and auto-resume"
    assert outcome.completed == 2, "halts on the cell that crosses the cap"
    assert outcome.completed < len(plan)
    assert rec.iterations == 2, "no further cell was dispatched after the cap"
    assert outcome.total_cost_usd == pytest.approx(0.002)


def test_run_cell_raises_cell_wall_clock_timeout_not_a_bare_timeouterror(tmp_path):
    """A cell over its wall-clock cap raises CellWallClockTimeout, not a bare TimeoutError.

    A bare TimeoutError would be classified HALT and stop the grid over one slow cell.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    with pytest.raises(policy.CellWallClockTimeout) as excinfo:
        run_cell(
            plan[0],
            results_root=tmp_path / "results",
            traces_root=tmp_path / "traces",
            backend_factory=_ok_factory(hold=5.0),
            iteration_timeout_s=60,
            cell_timeout_s=0.3,
        )
    assert policy.classify(excinfo.value) is policy.Disposition.RESULT
    assert not plan[0].result_path(tmp_path / "results").exists()


def test_timeout_is_a_result_recorded_and_never_retried(tmp_path):
    """A timeout is recorded as a result, never retried and never marked failed.

    Retrying slow cells would bias the results towards whatever finished in time.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    rec = _Recorder()
    results_root = tmp_path / "results"
    outcome = _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=tmp_path / "traces",
        workers=1,
        per_arm_limit=1,
        backend_factory=_ok_factory(rec, hold=5.0),
        cell_timeout_s=0.3,
    )

    assert outcome.halted_reason is None
    assert outcome.failed == 0
    assert outcome.completed == len(plan)
    assert rec.iterations == len(plan), "a timed-out cell must never be retried"

    for cell in plan:
        assert not store.is_failed(cell, results_root)
        assert store.is_complete(cell, results_root)
        result = ExtractionResult.model_validate_json(
            cell.result_path(results_root).read_text(encoding="utf-8")
        )
        assert result.timed_out is True
        assert result.funder == "The British Academy"
        assert result.config_hash == cell.config_hash


def test_a_timed_out_cell_is_not_overwritten_when_its_abandoned_run_finishes(tmp_path):
    """A timed-out result is not overwritten when its abandoned thread finishes later.

    Python cannot kill a thread, so the run keeps going in the background.
    """
    plan = _plan(funders=["british-academy"], replicates=(1,))
    results_root = tmp_path / "results"
    _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=tmp_path / "traces",
        workers=1,
        per_arm_limit=1,
        backend_factory=_ok_factory(hold=0.6),
        cell_timeout_s=0.2,
    )
    path = plan[0].result_path(results_root)
    assert ExtractionResult.model_validate_json(path.read_text(encoding="utf-8")).timed_out

    time.sleep(1.5)  # long enough for the abandoned run to finish and try to write
    assert ExtractionResult.model_validate_json(path.read_text(encoding="utf-8")).timed_out


def test_the_breaker_treats_a_timeout_as_a_completed_cell_not_a_failure(tmp_path):
    """A timeout counts as a finished cell for the breaker, not a failure."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    breaker = _RecordingBreaker()
    _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=_ok_factory(hold=5.0),
        cell_timeout_s=0.3,
    )
    assert breaker.records == [True] * len(plan)


def test_context_overflow_is_a_result_recorded_and_never_retried(tmp_path):
    """A context overflow is recorded as a result: no halt, no retry and no failure file."""
    plan = _plan(replicates=(1,))
    rec = _Recorder()
    results_root = tmp_path / "run" / "results"

    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(_context_overflow_error, rec),
    )

    assert outcome.halted_reason is None
    assert outcome.halted_cell is None, "no halt, so no cell may be blamed for one"
    assert outcome.failed == 0
    assert outcome.completed == len(plan), "the run must continue with the other cells"
    assert rec.iterations == len(plan), "an overflowing cell must never be retried"

    for cell in plan:
        assert not store.is_failed(cell, results_root)
        assert store.is_complete(cell, results_root)
        result = ExtractionResult.model_validate_json(
            cell.result_path(results_root).read_text(encoding="utf-8")
        )
        assert result.context_overflow is True
        assert result.timed_out is False, "an overflow is its own finding, not a timeout"
        assert result.opportunities == []
        assert result.config_hash == cell.config_hash


def test_the_breaker_hears_an_overflow_as_unhealthy_unlike_a_timeout(tmp_path):
    """Unlike a timeout, an overflow is recorded as unhealthy by the breaker.

    An arm where every cell overflows is broken, so the breaker needs to be able to trip.
    """
    plan = _plan(replicates=(1,))
    breaker = _RecordingBreaker()

    _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=_raising_factory(_context_overflow_error),
    )

    assert breaker.records == [False] * len(plan)


def test_an_overflow_is_logged_and_reported_under_its_own_status(tmp_path):
    """An overflow gets its own status in `on_cell` and the log instead of `completed`."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    seen: list[str] = []
    logged: list[str] = []

    _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(_context_overflow_error),
        on_cell=lambda cell, status, result: seen.append(status),
        log=logged.append,
    )

    assert seen == ["context_overflow"] * len(plan)
    assert any("context overflow" in line for line in logged), logged


def test_a_genuine_non_context_400_still_halts_the_grid(tmp_path):
    """A malformed-request 400 still halts, since it is our bug and would hit every cell."""
    plan = _plan(replicates=(1, 2))
    rec = _Recorder()
    results_root = tmp_path / "run" / "results"

    outcome = _grid(
        plan,
        tmp_path,
        workers=1,
        per_arm_limit=1,
        backend_factory=_raising_factory(
            lambda: _status_error(400, 'Input required: specify "prompt" or "messages"'),
            rec,
        ),
    )

    assert outcome.halted_reason is not None
    assert outcome.halted_kind == "error"
    assert outcome.completed == 0
    assert outcome.failed == 0
    assert rec.iterations == 1, f"the grid kept going after a HALT ({rec.iterations})"
    for cell in plan:
        assert not store.is_complete(cell, results_root)
        assert not store.is_failed(cell, results_root)


def test_iteration_timeout_is_pushed_onto_the_backend(tmp_path):
    """The iteration timeout is set on the backend's TIMEOUT_S; other backends still run."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    seen: list[float] = []
    ok = _ok_factory()

    class _Timed(MockBackend):
        TIMEOUT_S = 180.0

        def complete(self, messages, model, temperature, tools):
            seen.append(self.TIMEOUT_S)
            return super().complete(messages, model, temperature, tools)

    def factory():
        return _Timed(replies=[_submit()], usages=[Usage(1, 1, 0.0, "mock", "m")])

    _grid(plan, tmp_path, workers=1, backend_factory=factory, iteration_timeout_s=42.0)
    assert seen == [42.0] * len(plan)

    # a backend with no such seam still runs
    _grid(plan, tmp_path, name="plain", workers=1, backend_factory=ok, iteration_timeout_s=42.0)


def test_mirror_misses_are_written_into_the_trace(tmp_path):
    """A request for a URL missing from the mirror is written to the trace's misses file."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    bogus = "https://www.thebritishacademy.ac.uk/not-in-the-mirror/"

    def factory():
        return MockBackend(
            replies=[
                Reply(tool_calls=[ToolCall("c0", "fetch_page", {"url": bogus})]),
                _submit(),
            ],
            usages=[
                Usage(10, 2, 0.0, "mock", "m"),
                Usage(10, 2, 0.0, "mock", "m"),
            ],
        )

    _grid(plan, tmp_path, workers=1, backend_factory=factory)

    trace_dir = plan[0].trace_dir(tmp_path / "run" / "traces")
    path = trace_dir / "misses.jsonl.gz"
    assert path.is_file()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    record = json.loads(lines[-1])
    assert record["event"] == "mirror_misses"
    assert record["count"] == 1
    assert record["paths"] == {"/not-in-the-mirror/": 1}


def test_traces_are_gzipped(tmp_path):
    """Trace files are gzipped (raw traces would be 0.6-2 GB over the full grid)."""
    plan = _plan(funders=["british-academy"], replicates=(1,))
    _grid(plan, tmp_path, workers=1)

    trace_dir = plan[0].trace_dir(tmp_path / "run" / "traces")
    assert list(trace_dir.glob("*.jsonl")) == []
    names = {p.name for p in trace_dir.glob("*.jsonl.gz")}
    assert "iter1.jsonl.gz" in names
    assert "misses.jsonl.gz" in names


def test_on_cell_reports_every_finished_cell(tmp_path):
    """`on_cell` fires for every finished cell, failures included."""
    plan = _plan(replicates=(1,))
    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def on_cell(cell, status, result):
        with lock:
            seen.append((cell.cell_id, status))
        assert (result is None) == (status == "failed")

    _grid(plan, tmp_path, workers=1, backend_factory=_ok_factory(), on_cell=on_cell)
    assert sorted(seen) == sorted((c.cell_id, "completed") for c in plan)

    failing: list[tuple[str, str]] = []
    _grid(
        plan,
        tmp_path,
        name="failing",
        workers=1,
        backend_factory=_raising_factory(lambda: ConnectionError("reset")),
        on_cell=lambda cell, status, result: failing.append((cell.cell_id, status)),
    )
    assert sorted(failing) == sorted((c.cell_id, "failed") for c in plan)


def test_a_trip_landing_after_a_halt_does_not_strand_the_workers():
    """A breaker trip that lands after a halt still leaves the resume gate open.

    Drives `_State` directly: a cell finishing after the halt fills the window and trips
    the breaker.
    """
    from grant_eval.grid.executor import _State

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    state = _State(
        breaker=CircuitBreaker(window=4, threshold=0.5, max_trips=5),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )

    for _ in range(3):
        state.finish(cell, ok=False, status="failed", result=None, cost=None)
    assert state.breaker.trips == 0, "the window is not full yet, so nothing has tripped"

    state.halt("unauthorized", "error")
    # the in-flight cell fills the window and trips the breaker after the halt
    state.finish(cell, ok=True, status="completed", result=None, cost=None)
    assert state.breaker.trips == 1, "the trip this test exists for did not happen"

    assert state.resume_event.is_set(), (
        "the resume gate was left shut after a pause that overlapped a halt"
    )

    finished = threading.Event()

    def waiter():
        assert state.wait_until_runnable() is False
        finished.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert finished.wait(5.0), "a worker is stranded on the resume gate"


def test_a_timed_out_cell_keeps_holding_its_arm_permit(tmp_path):
    """A timed-out cell keeps its arm permit while its abandoned thread is still running.

    Each cell goes over a 0.2s cap while `complete()` runs for 1.0s, so the abandoned runs
    overlap.
    """
    plan = _plan(funders=["british-academy"], replicates=(1, 2))
    rec = _Recorder()
    outcome = _grid(
        plan,
        tmp_path,
        workers=4,
        per_arm_limit=1,
        backend_factory=_ok_factory(rec, hold=1.0),
        cell_timeout_s=0.2,
    )

    assert outcome.completed == len(plan), "every cell still records a timed-out result"
    assert rec.peak == 1, (
        f"{rec.peak} concurrent provider calls on an arm capped at 1: an abandoned "
        f"run is still calling the provider and must still hold its permit"
    )


def test_an_unexpected_worker_exception_cancels_the_rest_of_the_grid(tmp_path):
    """An exception escaping a worker cancels the rest of the grid instead of draining it.

    `on_cell` is used because it runs inside a worker with no `except` around it. Ctrl-C
    takes the same path.
    """
    plan = _plan(replicates=(1, 2))
    rec = _Recorder()

    def on_cell(cell, status, result):
        raise RuntimeError("logger exploded")

    with pytest.raises(RuntimeError, match="logger exploded"):
        _grid(
            plan,
            tmp_path,
            workers=1,
            per_arm_limit=1,
            backend_factory=_ok_factory(rec),
            on_cell=on_cell,
        )

    assert rec.iterations < len(plan), (
        f"the whole grid ({rec.iterations} of {len(plan)} cells) drained before the "
        f"exception surfaced"
    )


def test_a_cell_failing_after_a_halt_is_abandoned_not_marked_failed(tmp_path):
    """A cell in flight when a halt fires is abandoned rather than marked failed.

    Two arms so the backend can tell them apart: gpt-oss fails attempt 1, nova raises a 401,
    then gpt-oss fails its last attempt after the halt.
    """
    halting = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    retrying = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = halting + retrying
    halt_slug = halting[0].model_slug

    first_attempt_done = threading.Event()
    halt_raised = threading.Event()
    breaker = _RecordingBreaker()

    class _Staged:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == halt_slug:
                first_attempt_done.wait(5.0)
                halt_raised.set()
                raise _status_error(401, "unauthorized")
            if not first_attempt_done.is_set():
                first_attempt_done.set()
                raise ConnectionError("reset")  # attempt 1, before the halt
            halt_raised.wait(5.0)
            time.sleep(1.5)  # let the dispatcher actually register the halt
            raise ConnectionError("reset")  # attempt 2, after the halt

    outcome = _grid(
        plan,
        tmp_path,
        workers=2,
        per_arm_limit=1,
        breaker=breaker,
        backend_factory=lambda: _Staged(),
    )

    results_root = tmp_path / "run" / "results"
    assert outcome.halted_reason is not None
    assert outcome.failed == 0, "a HALT marks NOTHING failed"
    assert not store.is_failed(retrying[0], results_root), (
        "a cell abandoned by a halt was durably marked failed, so every later resume "
        "will skip it"
    )
    assert breaker.records == []
    # ...and the abandoned cell is still pending, which is the whole point.
    assert store.pending(plan, results_root, retry_failed=False) == plan


def test_concurrent_rate_limits_on_one_arm_share_a_single_backoff_window(tmp_path):
    """Several 429s on one arm at once share a single backoff.

    The fake sleep takes real time so the other three 429s arrive during it.
    """
    plan = _plan(funders=["british-academy"], replicates=(1, 2))
    seen = {"n": 0}
    lock = threading.Lock()
    slept: list[float] = []
    ok = _ok_factory()

    class _OneRateLimitEach:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            with lock:
                seen["n"] += 1
                first_round = seen["n"] <= len(plan)
            if first_round:
                raise _status_error(429, "rate limited")
            return ok().complete(messages, model, temperature, tools)

    def sleeper(seconds: float, interrupt=None) -> None:
        slept.append(seconds)
        time.sleep(0.5)

    outcome = _grid(
        plan,
        tmp_path,
        workers=4,
        per_arm_limit=4,
        backend_factory=lambda: _OneRateLimitEach(),
        sleep=sleeper,
    )

    assert outcome.completed == len(plan)
    assert slept == [RATE_LIMIT_BASE_BACKOFF_S], (
        f"{len(slept)} backoffs were slept for one rate-limit incident on one arm: "
        f"{slept}"
    )


def test_a_backoff_in_flight_does_not_block_a_healthy_cell_on_the_same_arm():
    """A healthy cell finishing on an arm that is backing off does not wait for the backoff."""
    from grant_eval.grid.executor import _Arm

    arm = _Arm(limit=2)
    backing_off = threading.Event()

    def hold(seconds: float, interrupt=None) -> None:
        backing_off.set()
        time.sleep(1.0)

    threading.Thread(target=lambda: arm.back_off(hold), daemon=True).start()
    assert backing_off.wait(5.0)

    started = time.monotonic()
    arm.succeeded()
    assert time.monotonic() - started < 0.5, (
        "a healthy completing cell was blocked behind its arm's backoff"
    )


def test_a_halt_does_not_wait_for_an_abandoned_run_to_release_its_permit(tmp_path):
    """A halt does not wait for an abandoned run to give back its arm permit.

    The cap is 1s so the halting cell reaches its provider call before its own cap runs out.
    `rec.live`, read afterwards, is the control.
    """
    nova = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:2]
    halting = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = [nova[0], nova[1], halting[0]]
    halt_slug = halting[0].model_slug

    rec = _Recorder()
    slow = _ok_factory(rec, hold=6.0)

    class _SlowArmPlusAHalt:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == halt_slug:
                raise _status_error(401, "unauthorized")
            return slow().complete(messages, model, temperature, tools)

    started = time.monotonic()
    outcome = _grid(
        plan,
        tmp_path,
        workers=3,
        per_arm_limit=1,
        backend_factory=lambda: _SlowArmPlusAHalt(),
        cell_timeout_s=1.0,
    )
    elapsed = time.monotonic() - started

    assert outcome.halted_reason is not None
    assert rec.live >= 1, (
        "control failed: the abandoned run had already finished, so this test never "
        "measured a halt waiting on a held permit"
    )
    assert elapsed < 2.5, (
        f"the halt took {elapsed:.2f}s to return, gated by a 6s abandoned run holding "
        f"the arm's only permit"
    )


def test_a_worker_that_wins_a_permit_does_not_start_a_cell_on_a_halted_grid(tmp_path):
    """A worker that gets a permit after the grid halted does not start the cell.

    The gap is tiny, so the arm sets the halt exactly as it hands over the permit.
    """
    from grant_eval.grid.executor import _Arm, _State, _run_one

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    rec = _Recorder()

    class _HaltsAsItGrants(_Arm):
        def acquire(self, halt_event, poll=0.25, timeout=executor.ARM_SATURATION_TIMEOUT_S):
            granted = super().acquire(halt_event, poll, timeout)
            halt_event.set()  # the halt lands as the permit is handed over
            return granted

    arm = _HaltsAsItGrants(limit=1)
    state = _State(
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )

    _run_one(
        cell,
        state=state,
        arms={cell.model: arm},
        results_root=tmp_path / "results",
        traces_root=tmp_path / "traces",
        backend_factory=_ok_factory(rec),
        iteration_timeout_s=60,
        cell_timeout_s=300,
        snapshots_root=Path("data/snapshots"),
    )

    assert rec.iterations == 0, "a cell started its provider work on a halted grid"
    assert not store.is_complete(cell, tmp_path / "results")
    assert arm.semaphore.acquire(blocking=False), (
        "the permit was carried off by a cell that never ran, so the arm lost a permit"
    )


def test_a_breaker_pause_is_not_served_out_on_a_halted_grid():
    """A breaker pause is skipped on a halted grid, but the gate is still re-opened."""
    from grant_eval.grid.executor import _State

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    slept: list[float] = []
    state = _State(
        breaker=CircuitBreaker(window=4, threshold=0.5, max_trips=5),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=_record(slept),
    )

    for _ in range(3):
        state.finish(cell, ok=False, status="failed", result=None, cost=None)
    state.halt("unauthorized", "error")
    state.finish(cell, ok=True, status="completed", result=None, cost=None)

    assert state.breaker.trips == 1, "the trip this test exists for did not happen"
    assert slept == [], (
        f"a halted grid served out a {slept} backoff before stopping"
    )
    assert state.resume_event.is_set(), "the resume gate was left shut"


def _halt_latency(tmp_path, name, cap_s):
    """Time from a halt to `run_grid` returning, with one cell stuck.

    The returned `still_running` flag is the control: without it a fast return could just mean
    the stuck cell had already finished.
    """
    wedged = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    halting = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = wedged + halting
    halt_slug = halting[0].model_slug

    rec = _Recorder()
    wedge = _ok_factory(rec, hold=cap_s + 8.0)
    halted_at: list[float] = []

    class _WedgePlusAHalt:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == halt_slug:
                halted_at.append(time.monotonic())
                raise _status_error(401, "unauthorized")
            return wedge().complete(messages, model, temperature, tools)

    started = time.monotonic()
    outcome = _grid(
        plan,
        tmp_path,
        name=name,
        workers=2,
        per_arm_limit=2,
        backend_factory=lambda: _WedgePlusAHalt(),
        cell_timeout_s=cap_s,
    )
    returned = time.monotonic()
    assert halted_at, "the halting cell never reached its provider call"
    return returned - halted_at[0], outcome, rec.live >= 1


def test_a_halt_does_not_wait_out_the_per_cell_wall_clock_cap(tmp_path):
    """A halt does not wait out the per-cell wall-clock cap.

    Measured at two caps, since one value cannot tell a fast halt from one that scales with
    the cap.
    """
    fast, outcome_fast, wedged_fast = _halt_latency(tmp_path, "cap2", cap_s=2.0)
    slow, outcome_slow, wedged_slow = _halt_latency(tmp_path, "cap6", cap_s=6.0)

    assert outcome_fast.halted_reason is not None
    assert outcome_slow.halted_reason is not None
    assert wedged_fast and wedged_slow, (
        "control failed: the wedged run had already finished, so this measured nothing"
    )
    assert fast < 1.5, f"halt took {fast:.2f}s at a 2s cap"
    assert slow < 1.5, f"halt took {slow:.2f}s at a 6s cap"
    assert abs(slow - fast) < 1.0, (
        f"halt latency tracks the cap ({fast:.2f}s at 2s, {slow:.2f}s at 6s) instead "
        f"of the poll interval"
    )


def test_a_cell_abandoned_by_a_halt_writes_nothing_and_stays_pending(tmp_path):
    """A cell abandoned by a halt writes no result, timeout or failure, and stays pending."""
    wedged = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    halting = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = wedged + halting
    halt_slug = halting[0].model_slug
    wedge = _ok_factory(hold=4.0)

    class _WedgePlusAHalt:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == halt_slug:
                raise _status_error(401, "unauthorized")
            return wedge().complete(messages, model, temperature, tools)

    results_root = tmp_path / "results"
    outcome = _grid(
        plan,
        tmp_path,
        results_root=results_root,
        traces_root=tmp_path / "traces",
        workers=2,
        per_arm_limit=2,
        backend_factory=lambda: _WedgePlusAHalt(),
        cell_timeout_s=30.0,
    )

    assert outcome.halted_reason is not None
    assert outcome.completed == 0
    assert outcome.failed == 0
    assert not store.is_complete(wedged[0], results_root)
    assert not store.is_failed(wedged[0], results_root)
    assert store.pending(plan, results_root, retry_failed=False) == plan

    # ...and it stays that way once the abandoned run finishes and unwinds.
    time.sleep(5.0)
    assert not store.is_complete(wedged[0], results_root), (
        "the abandoned run wrote its result after run_grid had already reported the "
        "cell as not run"
    )


def test_a_halt_during_a_breaker_pause_ends_the_pause(tmp_path):
    """A halt during a breaker pause ends the pause early.

    Uses the real delay function, so a real 60s pause has to end within a couple of seconds.
    """
    from grant_eval.grid.executor import _State

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    state = _State(
        breaker=CircuitBreaker(window=2, threshold=0.5, max_trips=5),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=executor._delay,  # the real one
    )
    finished = threading.Event()

    def worker():
        state.finish(cell, ok=False, status="failed", result=None, cost=None)
        state.finish(cell, ok=False, status="failed", result=None, cost=None)
        finished.set()

    threading.Thread(target=worker, daemon=True).start()

    # Wait for the pause to actually be in flight (the gate is shut during it).
    deadline = time.monotonic() + 5.0
    while state.resume_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not state.resume_event.is_set(), "the breaker never paused"
    assert state.breaker.trips == 1

    started = time.monotonic()
    state.halt("unauthorized", "error")
    assert finished.wait(5.0), (
        "the paused worker never returned: a 60s backoff was served out in full on a "
        "grid that had already halted"
    )
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"the pause took {elapsed:.2f}s to notice the halt"
    assert state.resume_event.is_set(), "the resume gate was left shut"


def test_a_worker_that_wins_a_permit_does_not_start_a_cell_during_a_pause(tmp_path):
    """A worker given a permit during a breaker pause hands it back and runs the cell later."""
    from grant_eval.grid.executor import _Arm, _State, _run_one

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    rec = _Recorder()
    paused_calls: list[int] = []

    class _PausesAsItGrants(_Arm):
        def __init__(self, limit):
            super().__init__(limit)
            self.state = None
            self.granted = 0

        def acquire(self, halt_event, poll=0.25, timeout=executor.ARM_SATURATION_TIMEOUT_S):
            granted = super().acquire(halt_event, poll, timeout)
            self.granted += 1
            if self.granted == 1:
                # The breaker pauses exactly as the permit is handed over.
                self.state.resume_event.clear()
                threading.Timer(0.4, self._resume).start()
            return granted

        def _resume(self):
            with rec.lock:
                paused_calls.append(rec.iterations)
            self.state.resume_event.set()

    arm = _PausesAsItGrants(limit=1)
    state = _State(
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )
    arm.state = state

    _run_one(
        cell,
        state=state,
        arms={cell.model: arm},
        results_root=tmp_path / "results",
        traces_root=tmp_path / "traces",
        backend_factory=_ok_factory(rec),
        iteration_timeout_s=60,
        cell_timeout_s=300,
        snapshots_root=Path("data/snapshots"),
    )

    assert paused_calls == [0], (
        f"{paused_calls[0] if paused_calls else '?'} provider calls were made during "
        f"the pause"
    )
    # ...and the cell was not dropped: it ran once the grid resumed.
    assert rec.iterations == 1
    assert store.is_complete(cell, tmp_path / "results")
    assert arm.semaphore.acquire(blocking=False), "the permit was never handed back"


def test_an_exception_in_the_pause_still_re_opens_the_gate():
    """If the delay function raises during a pause, the gate still re-opens."""
    from grant_eval.grid.executor import _State

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]

    def explodes(seconds, interrupt=None):
        raise RuntimeError("the delay primitive is broken")

    state = _State(
        breaker=CircuitBreaker(window=2, threshold=0.5, max_trips=5),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=explodes,
    )

    state.finish(cell, ok=False, status="failed", result=None, cost=None)
    with pytest.raises(RuntimeError, match="broken"):
        state.finish(cell, ok=False, status="failed", result=None, cost=None)

    assert state.breaker.trips == 1, "the trip this test exists for did not happen"
    assert state.resume_event.is_set(), (
        "the resume gate was left shut by an exception in the pause"
    )


def test_finish_refuses_to_record_a_failure_once_the_grid_has_halted():
    """`finish` will not record a failure once the grid has halted.

    Tests the guard directly, since the end-to-end test now abandons the cell earlier, in
    `run_cell`.
    """
    from grant_eval.grid.executor import _State

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    state = _State(
        breaker=_RecordingBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )
    wrote: list[int] = []

    state.halt("unauthorized", "error")
    recorded = state.finish(
        cell,
        ok=False,
        status="failed",
        result=None,
        cost=None,
        on_record=lambda: wrote.append(1),
    )

    assert recorded is False, "a halted grid recorded a cell as failed"
    assert wrote == [], "a failure sentinel was written on a halted grid"
    assert state.failed == 0
    assert state.breaker.records == []


def _interrupt_latency(tmp_path, name, cap_s):
    """Time from an unexpected worker exception to `run_grid` returning, with one cell stuck.

    Like `_halt_latency`, but this path is not a halt: other workers only see it through
    `pool.shutdown(wait=True)`, the same as Ctrl-C.
    """
    wedged = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    quick = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = wedged + quick
    wedged_slug = wedged[0].model_slug

    rec = _Recorder()
    wedge = _ok_factory(rec, hold=cap_s + 8.0)
    fast = _ok_factory()
    interrupted_at: list[float] = []

    class _WedgeAndOne:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == wedged_slug:
                return wedge().complete(messages, model, temperature, tools)
            return fast().complete(messages, model, temperature, tools)

    def on_cell(cell, status, result):
        interrupted_at.append(time.monotonic())
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _grid(
            plan,
            tmp_path,
            name=name,
            workers=2,
            per_arm_limit=2,
            backend_factory=lambda: _WedgeAndOne(),
            cell_timeout_s=cap_s,
            on_cell=on_cell,
        )
    returned = time.monotonic()
    assert interrupted_at, "the quick cell never completed, so nothing interrupted"
    return returned - interrupted_at[0], rec.live >= 1


def test_an_interrupt_does_not_wait_out_the_per_cell_cap(tmp_path):
    """An interrupt (Ctrl-C or a worker exception) does not wait out the per-cell cap.

    Neither sets `halt_event`, so the cap wait also has to watch for a stop. Two caps, as in
    the halt test.
    """
    fast, wedged_fast = _interrupt_latency(tmp_path, "int2", cap_s=2.0)
    slow, wedged_slow = _interrupt_latency(tmp_path, "int6", cap_s=6.0)

    assert wedged_fast and wedged_slow, (
        "control failed: the wedged run had already finished, so this measured nothing"
    )
    assert fast < 1.5, f"interrupt took {fast:.2f}s at a 2s cap"
    assert slow < 1.5, f"interrupt took {slow:.2f}s at a 6s cap"
    assert abs(slow - fast) < 1.0, (
        f"interrupt latency tracks the cap ({fast:.2f}s at 2s, {slow:.2f}s at 6s) "
        f"instead of the poll interval"
    )


def test_a_stop_that_is_not_a_halt_wakes_every_wait_without_claiming_a_halt():
    """`stop()` wakes every wait but does not record itself as a halt.

    A Ctrl-C has no policy reason, so `halted_reason` stays None and the cells stay pending.
    """
    from grant_eval.grid.executor import _Arm, _State

    state = _State(
        breaker=CircuitBreaker(),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )
    arm = _Arm(limit=1)
    assert arm.acquire(state.stop_event) == executor.ARM_ACQUIRED  # the only permit

    parked = threading.Event()
    gave_up = threading.Event()

    def waiter():
        parked.set()
        # A saturation timeout far past the assertions below, so the only thing
        # that can free this waiter is the stop, not the saturation bound.
        if arm.acquire(state.stop_event, timeout=3600.0) == executor.ARM_STOPPED:
            gave_up.set()

    threading.Thread(target=waiter, daemon=True).start()
    assert parked.wait(5.0)
    state.resume_event.clear()  # as a breaker pause would leave it

    state.stop()

    assert gave_up.wait(5.0), "a worker parked on a permit was not woken by the stop"
    assert state.wait_until_runnable() is False, "the gate did not release on a stop"
    assert state.halted_reason is None, "a Ctrl-C recorded itself as a policy halt"
    assert state.halted_kind is None
    assert not state.halt_event.is_set(), (
        "a stop set halt_event, so an interrupted run would start refusing to record "
        "genuine failures and would report a halt it never had"
    )


def test_the_gate_re_opens_even_if_clearing_the_breaker_window_raises():
    """`_end_pause` re-opens the gate before resetting the breaker window, in case that raises."""
    from grant_eval.grid.executor import _State

    class _BadReset(CircuitBreaker):
        def reset_window(self):
            raise RuntimeError("reset_window is broken")

    cell = _plan(funders=["british-academy"], replicates=(1,))[0]
    state = _State(
        breaker=_BadReset(window=2, threshold=0.5, max_trips=5),
        spend_guard=SpendGuard(cap_usd=None),
        sleep=lambda seconds, interrupt=None: None,
    )

    state.finish(cell, ok=False, status="failed", result=None, cost=None)
    with pytest.raises(RuntimeError, match="reset_window is broken"):
        state.finish(cell, ok=False, status="failed", result=None, cost=None)

    assert state.resume_event.is_set(), (
        "the resume gate was left shut because clearing the breaker window raised "
        "first, stranding every worker"
    )


def test_serve_cell_tears_down_promptly():
    """`serve_cell` tears down quickly rather than waiting on the default 0.5s poll."""
    rec = REG["british-academy"]
    with serve_cell(rec.snapshot_dir()) as (base_url, _misses):
        with urllib.request.urlopen(f"{base_url}/funding/", timeout=15) as resp:
            assert resp.status == 200
        started = time.monotonic()
    teardown = time.monotonic() - started

    assert teardown < 0.25, (
        f"serve_cell took {teardown * 1000:.0f}ms to tear down; the stdlib default "
        f"poll (0.5s) is back"
    )


def test_an_interrupt_does_not_wait_for_a_worker_parked_on_a_permit(tmp_path):
    """An interrupt does not wait for a worker parked on an arm permit.

    Three cells, one permit: the first gets stuck holding it, the second waits for it and the
    third raises from `on_cell`.
    """
    nova = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[:2]
    quick = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )[:1]
    plan = [nova[0], nova[1], quick[0]]
    quick_slug = quick[0].model_slug

    rec = _Recorder()
    wedge = _ok_factory(rec, hold=8.0)
    fast = _ok_factory()
    interrupted_at: list[float] = []

    class _WedgeAndOne:
        name = "mock"

        def complete(self, messages, model, temperature, tools):
            if model == quick_slug:
                return fast().complete(messages, model, temperature, tools)
            return wedge().complete(messages, model, temperature, tools)

    def on_cell(cell, status, result):
        interrupted_at.append(time.monotonic())
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _grid(
            plan,
            tmp_path,
            workers=3,
            per_arm_limit=1,
            backend_factory=lambda: _WedgeAndOne(),
            cell_timeout_s=0.5,
            on_cell=on_cell,
        )
    elapsed = time.monotonic() - interrupted_at[0]

    assert rec.live >= 1, (
        "control failed: the wedged run had already released the permit, so nothing "
        "was ever parked waiting for it"
    )
    assert elapsed < 2.0, (
        f"the interrupt waited {elapsed:.2f}s for a worker parked on a permit held by "
        f"an abandoned run"
    )


def test_an_interrupt_cancels_queued_cells_rather_than_dequeuing_them_all(
    tmp_path, monkeypatch
):
    """An interrupt cancels queued cells rather than starting each one.

    Counts entries into `_run_one`, because provider calls stay low either way.
    """
    from grant_eval.grid import executor as _executor

    plan = _plan(replicates=(1, 2, 3, 4, 5))
    assert len(plan) == 20
    entries: list[str] = []
    real = _executor._run_one

    def counting(cell, **kwargs):
        entries.append(cell.cell_id)
        return real(cell, **kwargs)

    monkeypatch.setattr(_executor, "_run_one", counting)

    def on_cell(cell, status, result):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _grid(
            plan,
            tmp_path,
            workers=1,
            per_arm_limit=1,
            backend_factory=_ok_factory(),
            on_cell=on_cell,
        )

    assert len(entries) < 5, (
        f"{len(entries)} of {len(plan)} queued cells were dequeued and started after "
        f"the interrupt instead of being cancelled"
    )


def test_a_saturated_arm_does_not_park_the_pool_on_the_healthy_arms_work(
    tmp_path, monkeypatch
):
    """A saturated arm does not stop the pool working through the healthy arm's cells.

    The eight saturating cells are queued first: four take the permits and get stuck, the
    other four wait for one. The saturation timeout is patched down, and the check reads
    result files because the failure mode is that `run_grid` never returns.
    """
    monkeypatch.setattr(executor, "ARM_SATURATION_TIMEOUT_S", 0.3)

    dead = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1, 2, 3, 4],
    )
    healthy = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1, 2, 3, 4, 5],
    )
    assert len(dead) == 8 and len(healthy) == 10
    dead_slug = dead[0].model_slug
    plan = dead + healthy

    release = threading.Event()

    class _Wedging(MockBackend):
        def complete(self, messages, model, temperature, tools):
            if model == dead_slug:
                release.wait(90.0)
            return super().complete(messages, model, temperature, tools)

    def factory():
        return _Wedging(
            replies=[_submit()], usages=[Usage(100, 20, 0.001, "mock", "mock/model")]
        )

    results_root = tmp_path / "results"
    logged: list[str] = []
    log_lock = threading.Lock()

    def log(message: str) -> None:
        with log_lock:
            logged.append(message)

    def go():
        run_grid(
            plan,
            results_root=results_root,
            traces_root=tmp_path / "traces",
            workers=8,
            per_arm_limit=4,
            backend_factory=factory,
            breaker=CircuitBreaker(),
            spend_guard=SpendGuard(cap_usd=None),
            iteration_timeout_s=60,
            cell_timeout_s=600,
            log=log,
        )

    runner = threading.Thread(target=go, daemon=True)
    runner.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if all(store.is_complete(c, results_root) for c in healthy):
                break
            time.sleep(0.2)
        completed_healthy = sum(
            1 for c in healthy if store.is_complete(c, results_root)
        )
    finally:
        release.set()
        runner.join(120.0)

    assert completed_healthy == len(healthy), (
        f"only {completed_healthy} of {len(healthy)} cells on the HEALTHY arm "
        f"completed while another arm was saturated: the pool is parked"
    )
    with log_lock:
        assert any("saturated" in line for line in logged), (
            "the pool gave up on the saturated arm without saying so; an operator "
            f"watching stdout sees nothing at all. Logged: {logged}"
        )


def test_a_saturated_arm_is_counted_as_deferred_not_silently_absorbed(tmp_path, monkeypatch):
    """Cells deferred by a saturated arm are counted in `GridOutcome.deferred`.

    Otherwise they just fall into `skipped` and a run with deferrals looks like a clean one.
    """
    monkeypatch.setattr(executor, "ARM_SATURATION_TIMEOUT_S", 0.2)

    dead = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1, 2, 3],
    )
    healthy = enumerate_cells(
        REG, Split.TUNE, funders=["british-academy"], models=["gpt-oss-120b"],
        prompts=["minimal"], replicates=[1],
    )
    assert len(dead) == 6, "expected both pre-registered temperatures x 3 replicates"
    dead_slug = dead[0].model_slug
    plan = dead + healthy
    per_arm_limit = 2  # 2 of the 6 dead cells acquire the arm and wedge; 4 defer

    release = threading.Event()

    class _Wedging(MockBackend):
        def complete(self, messages, model, temperature, tools):
            if model == dead_slug:
                release.wait(90.0)
            return super().complete(messages, model, temperature, tools)

    def factory():
        return _Wedging(
            replies=[_submit()], usages=[Usage(100, 20, 0.001, "mock", "mock/model")]
        )

    results_root = tmp_path / "results"
    outcome_holder: dict[str, GridOutcome] = {}

    def go():
        outcome_holder["outcome"] = run_grid(
            plan,
            results_root=results_root,
            traces_root=tmp_path / "traces",
            workers=8,
            per_arm_limit=per_arm_limit,
            backend_factory=factory,
            breaker=CircuitBreaker(),
            spend_guard=SpendGuard(cap_usd=None),
            iteration_timeout_s=60,
            cell_timeout_s=600,
        )

    runner = threading.Thread(target=go, daemon=True)
    runner.start()
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if all(store.is_complete(c, results_root) for c in healthy):
                break
            time.sleep(0.05)
        # Margin past the monkeypatched 0.2s saturation timeout: the 4 waiting
        # dead-arm workers must TIME OUT and defer before the permit is freed.
        time.sleep(1.0)
    finally:
        release.set()
        runner.join(30.0)

    outcome = outcome_holder["outcome"]
    assert outcome.deferred == 4, (
        f"expected 4 of the 6 dead-arm cells to be deferred (per_arm_limit="
        f"{per_arm_limit}), got {outcome.deferred}"
    )
    # Deferred cells are a SUBSET of skipped, not additional to it.
    assert outcome.completed + outcome.failed + outcome.skipped == len(plan)
    assert outcome.skipped == outcome.deferred
    assert outcome.halted_kind is None


def test_a_cell_that_finishes_as_its_deadline_passes_keeps_its_real_result(tmp_path):
    """A cell that finishes right as its deadline passes keeps its real result.

    The race is very narrow, so `_completion_event` is swapped for an Event that waits for
    the work to finish and then returns False anyway.
    """
    cell = _plan(funders=["british-academy"], replicates=(1,))[0]

    class _FinishesJustTooLate(threading.Event):
        def wait(self, timeout=None):
            super().wait(60.0)  # the work really has finished by the time this returns
            return False        # ...and the caller is told it has not

    saved = executor._completion_event
    try:
        executor._completion_event = _FinishesJustTooLate
        result = run_cell(
            cell,
            results_root=tmp_path / "results",
            traces_root=tmp_path / "traces",
            backend_factory=_ok_factory(),
            iteration_timeout_s=60,
            cell_timeout_s=0.0,
            snapshots_root=Path("data/snapshots"),
        )
    finally:
        executor._completion_event = saved

    assert result.timed_out is not True, (
        "a cell that had genuinely finished was recorded as a wall-clock timeout"
    )


def test_every_rate_limit_backoff_is_logged(tmp_path):
    """Every rate-limit backoff is logged, since a 429 does not fail a cell or trip the breaker."""
    plan = _plan(funders=["british-academy"], replicates=(1,))[:1]
    calls = {"n": 0}

    class _RateLimited(MockBackend):
        def complete(self, messages, model, temperature, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _status_error(429, "rate limited")
            return super().complete(messages, model, temperature, tools)

    def factory():
        return _RateLimited(
            replies=[_submit()], usages=[Usage(100, 20, 0.001, "mock", "mock/model")]
        )

    logged: list[str] = []
    outcome = _grid(
        plan,
        tmp_path,
        backend_factory=factory,
        sleep=lambda seconds, interrupt=None: None,
        log=logged.append,
    )

    assert outcome.completed == 1
    assert any("rate limit" in line.lower() for line in logged), (
        f"a rate-limit backoff was served silently. Logged: {logged}"
    )


def test_the_log_line_carries_a_utc_timestamp(capsys):
    """The per-cell log line starts with a UTC timestamp."""
    executor.log_line("completed: some_cell_id")
    out = capsys.readouterr().out.strip()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z ", out), (
        f"the log line carries no UTC timestamp: {out!r}"
    )
    assert out.endswith("completed: some_cell_id")


def test_arm_saturation_timeout_is_120s_not_5s():
    """ARM_SATURATION_TIMEOUT_S is 120s.

    With 5s, simulations deferred 8-15% of cells on the first pass because contention on a
    healthy arm clears on the scale of a whole cell. 120s brought that down to about 1.7%.
    """
    assert executor.ARM_SATURATION_TIMEOUT_S == 120.0
