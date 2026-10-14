"""The grid dispatcher: runs a plan of cells on a thread pool.

Each cell gets its own mirror server, and each model arm has a semaphore limiting its
concurrent cells. Resume is based on which result files exist, so `run_grid` can be
given the whole grid and will only run what is missing. A wall-clock timeout or a
context overflow is recorded as a result, not a failure. Every wait that could delay
a stop polls `_State.stop_event`, so a halt or Ctrl-C takes effect quickly.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from grant_eval.agent.extract import RunConfig, run_extraction
from grant_eval.dataset.registry import DEFAULT_SNAPSHOT_ROOT
from grant_eval.dataset.serve import build_lookup, make_handler
from grant_eval.grid import logs, policy, store
from grant_eval.grid.cells import Cell
from grant_eval.grid.policy import (
    CellWallClockTimeout,
    CircuitBreaker,
    Disposition,
    SpendCapExceeded,
    SpendGuard,
)
from grant_eval.schema import ExtractionResult

_LOG = logging.getLogger(__name__)

# Rate-limit backoff for one arm, reset when it completes a cell.
RATE_LIMIT_BASE_BACKOFF_S: float = 5.0
RATE_LIMIT_MAX_BACKOFF_S: float = 60.0

# How many times one cell may be re-queued on 429s before it is failed. Without a
# limit a saturated provider re-queued one cell forever with no failure or trip.
# Set by `--max-rate-limit-requeues` and recorded in `freeze.CAP_KEYS`.
MAX_RATE_LIMIT_REQUEUES: int = 10

# How often a worker waiting for a permit checks whether the run has stopped.
PERMIT_POLL_S: float = 0.25

# How long a worker waits for an arm permit before handing the cell back to pending.
# Waiting forever tied up pool workers one by one until the grid stalled.
ARM_SATURATION_TIMEOUT_S: float = 120.0

# `_Arm.acquire`'s three outcomes. "Saturated" is logged, "stopped" is not.
ARM_ACQUIRED = "acquired"
ARM_STOPPED = "stopped"
ARM_SATURATED = "saturated"

# Indirected so tests can substitute their own Event.
_completion_event: Callable[[], threading.Event] = threading.Event


def log_line(message: str) -> None:
    """Log one operational line at INFO, to stdout and the run log if attached."""
    logs.ensure_console()
    _LOG.info(message)

# How often each cell's mirror server checks whether to stop. The 0.5s default
# added about 0.45s to every teardown.
SERVE_POLL_S: float = 0.05


def _delay(seconds: float, interrupt: threading.Event | None = None) -> None:
    """Wait `seconds`, returning early if `interrupt` is set."""
    if interrupt is None:
        time.sleep(seconds)
    else:
        interrupt.wait(seconds)


class CellAbandoned(Exception):
    """`run_cell` stopped waiting because the run is stopping.

    Not a failure or a result: nothing is written and the cell stays pending.
    """


@contextmanager
def serve_cell(snapshot_dir):
    """Serve one snapshot mirror for one cell, yielding `(base_url, misses)`.

    `misses` counts requests for URLs not in the mirror.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    lookup = build_lookup(snapshot_dir, manifest)
    misses: Counter = Counter()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lookup, None, misses))
    port = server.server_address[1]
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=SERVE_POLL_S), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", misses
    finally:
        # `server_close()` joins handler threads with no timeout, so this can block
        # if a request is still in progress.
        server.shutdown()
        server.server_close()


def _iteration_capped(backend_factory, iteration_timeout_s: float):
    """Wrap `backend_factory` so each new backend gets the per-request timeout.

    Despite the name this is per model request, not per Ralph iteration; the per-cell
    cap in `run_cell` is what really bounds a cell.
    """

    def make():
        backend = backend_factory()
        if hasattr(backend, "TIMEOUT_S"):
            backend.TIMEOUT_S = float(iteration_timeout_s)
        return backend

    return make


def _write_misses(trace_dir: Path, misses: Counter) -> None:
    """Append the mirror-miss count to `<trace_dir>/misses.jsonl`.

    Written even when zero, so a missing file means the cell crashed first.
    """
    record = {
        "event": "mirror_misses",
        "count": int(sum(misses.values())),
        "paths": {path: int(n) for path, n in sorted(misses.items())},
    }
    with (trace_dir / "misses.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def timed_out_result(cell: Cell, elapsed_s: float) -> ExtractionResult:
    """The empty `timed_out` result recorded for a cell that hit its wall-clock cap.

    The partial run is lost in the abandoned thread. `cost_usd` stays None, not 0.0,
    because money was spent but the amount is unknown.
    """
    return ExtractionResult(
        snapshot_date=cell.snapshot_date,
        funder=cell.display_name,
        split=cell.split,
        model=cell.model_slug,
        prompt_strategy=cell.prompt_strategy,
        temperature=cell.temperature,
        agent_config=cell.agent_config,
        replicate=cell.replicate,
        config_hash=cell.config_hash,
        opportunities=[],
        latency_s=elapsed_s,
        timed_out=True,
    )


def context_overflow_result(cell: Cell) -> ExtractionResult:
    """The empty `context_overflow` result for a cell whose request outgrew the context
    window. Not retried, since a retry would build the same oversized request.
    `latency_s` and `cost_usd` stay None."""
    return ExtractionResult(
        snapshot_date=cell.snapshot_date,
        funder=cell.display_name,
        split=cell.split,
        model=cell.model_slug,
        prompt_strategy=cell.prompt_strategy,
        temperature=cell.temperature,
        agent_config=cell.agent_config,
        replicate=cell.replicate,
        config_hash=cell.config_hash,
        opportunities=[],
        context_overflow=True,
    )


def _execute_cell(
    cell: Cell,
    *,
    results_root: Path,
    traces_root: Path,
    backend_factory,
    iteration_timeout_s: float,
    snapshots_root: Path,
    cancelled: threading.Event,
) -> ExtractionResult:
    """One cell's work: serve the mirror, run the extraction, save misses, result, gzip.

    `cancelled` is checked before the result write so a run abandoned at the cap
    cannot overwrite its `timed_out` result later. Traces are still kept.
    """
    snapshot_dir = Path(snapshots_root) / cell.funder / cell.snapshot_date.isoformat()
    trace_dir = cell.trace_dir(traces_root)
    trace_dir.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig(
        # The display name, which the prompt and ground truth use, not the slug.
        funder=cell.display_name,
        split=cell.split,
        snapshot_date=cell.snapshot_date,
        # The full LiteLLM slug, not the short grid label.
        model=cell.model_slug,
        prompt_strategy=cell.prompt_strategy,
        temperature=cell.temperature,
        agent_config=cell.agent_config,
        replicate=cell.replicate,
    )

    with serve_cell(snapshot_dir) as (base_url, misses):
        result = run_extraction(
            run_config=run_config,
            base_url=base_url,
            backend_factory=_iteration_capped(backend_factory, iteration_timeout_s),
            # The URL the mirror was crawled from, not the homepage.
            entry_url=cell.entry_url,
            trace_dir=str(trace_dir),
            # Saved below with an atomic write instead, so a crash cannot leave a
            # truncated result.
            out_path=None,
        )
        _write_misses(trace_dir, misses)

    if not cancelled.is_set():
        store.write_result_atomic(cell.result_path(results_root), result)
    store.gzip_traces(trace_dir)
    return result


def run_cell(
    cell: Cell,
    *,
    results_root,
    traces_root,
    backend_factory,
    iteration_timeout_s: float,
    cell_timeout_s: float,
    snapshots_root=DEFAULT_SNAPSHOT_ROOT,
    held_permit: threading.Semaphore | None = None,
    abort: threading.Event | None = None,
) -> ExtractionResult:
    """Run one cell under its per-cell wall-clock cap.

    The work runs in a daemon thread. Raises `CellWallClockTimeout` if the cap is hit
    and `CellAbandoned` if `abort` (the run's `stop_event`) is set.

    `held_permit` is the arm semaphore, already acquired by the caller. The work thread
    releases it when it really finishes, not when this function gives up, because an
    abandoned run is still calling the provider.
    """
    results_root = Path(results_root)
    traces_root = Path(traces_root)
    cancelled = threading.Event()
    box: dict[str, object] = {}
    done = _completion_event()  # a test seam; see `_completion_event`
    started = time.perf_counter()

    def target() -> None:
        try:
            box["value"] = _execute_cell(
                cell,
                results_root=results_root,
                traces_root=traces_root,
                backend_factory=backend_factory,
                iteration_timeout_s=iteration_timeout_s,
                snapshots_root=Path(snapshots_root),
                cancelled=cancelled,
            )
        except BaseException as exc:  # re-raised in the caller's thread below
            box["error"] = exc
        finally:
            # Released by the work thread itself; see the docstring.
            if held_permit is not None:
                held_permit.release()
            done.set()

    worker = threading.Thread(target=target, name=f"cell-{cell.cell_id}", daemon=True)
    try:
        worker.start()
    except BaseException:
        if held_permit is not None:
            held_permit.release()  # nothing is running, so nothing is holding the arm
        raise

    # Wait for the cell, its cap, or the run stopping. Polls so a stop is not delayed
    # until the cap.
    poll = PERMIT_POLL_S if abort is not None else cell_timeout_s
    deadline = time.monotonic() + cell_timeout_s
    while not done.wait(min(poll, max(0.0, deadline - time.monotonic()))):
        if abort is not None and abort.is_set():
            # The run stopped: abandon the cell and write nothing.
            cancelled.set()
            raise CellAbandoned(f"{cell.cell_id} abandoned: the run stopped")
        if done.is_set():
            # The cell may have finished just after `wait()` returned.
            break
        if time.monotonic() >= deadline:
            cancelled.set()
            raise CellWallClockTimeout(
                f"{cell.cell_id} exceeded its {cell_timeout_s}s wall-clock cap"
            )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


@dataclass(frozen=True)
class GridOutcome:
    """What one `run_grid` call did.

    `completed + failed + skipped == len(plan)`. `deferred` is the part of `skipped`
    handed back because an arm was saturated. `halted_kind` is `"error"`,
    `"circuit_breaker_exhausted"` or `"spend_cap"`, and `halted_cell` is the identity
    of the cell behind the halt, if any.
    """

    completed: int = 0
    failed: int = 0
    skipped: int = 0
    deferred: int = 0
    halted_reason: str | None = None
    total_cost_usd: float = 0.0
    halted_kind: str | None = None
    halted_cell: dict | None = None


class _Arm:
    """One model arm's concurrency permits and rate-limit backoff.

    Keyed by model label. The pinned arms use a single provider, so this is in effect
    per provider.
    """

    def __init__(self, limit: int) -> None:
        self.semaphore = threading.Semaphore(limit)
        # Held only during a backoff, so only one worker sleeps per arm.
        self._backoff_lock = threading.Lock()
        # Guards `_hits` alone, so a healthy completion never waits out a backoff.
        self._hits_lock = threading.Lock()
        self._hits = 0

    def acquire(
        self,
        stop_event: threading.Event,
        poll: float = PERMIT_POLL_S,
        timeout: float = ARM_SATURATION_TIMEOUT_S,
    ) -> str:
        """Wait for one of this arm's permits.

        Returns `ARM_ACQUIRED` (holding a permit), `ARM_STOPPED` if the run is stopping,
        or `ARM_SATURATED` if `timeout` passed. In the last two cases the caller must
        not start the cell. Polls `stop_event` so a halt is not held up here.
        """
        deadline = time.monotonic() + timeout
        while not self.semaphore.acquire(timeout=poll):
            if stop_event.is_set():
                return ARM_STOPPED
            if time.monotonic() >= deadline:
                return ARM_SATURATED
        return ARM_ACQUIRED

    def back_off(self, sleep: Callable[[float], None]) -> float:
        """Back this arm off and return how long this caller slept.

        Only the first worker sleeps; others just wait for that backoff to finish, so
        sleeps do not stack up.
        """
        if self._backoff_lock.acquire(blocking=False):
            try:
                with self._hits_lock:
                    self._hits += 1
                    delay = min(
                        RATE_LIMIT_MAX_BACKOFF_S,
                        RATE_LIMIT_BASE_BACKOFF_S * 2 ** (self._hits - 1),
                    )
                sleep(delay)
            finally:
                self._backoff_lock.release()
            return delay

        # Another worker is already backing off this arm: wait for it, then retry.
        with self._backoff_lock:
            pass
        return 0.0

    def succeeded(self) -> None:
        with self._hits_lock:
            self._hits = 0


@dataclass
class _State:
    """Shared run state: counters, breaker, spend guard and the pause/stop gates.

    All accounting happens under one lock in the worker that finished the cell, so a
    halt takes effect before that worker picks up another cell.
    """

    breaker: CircuitBreaker
    spend_guard: SpendGuard
    sleep: Callable[..., None]  # (seconds, interrupt=None) -> None; see `_delay`
    on_cell: Callable[[Cell, str, ExtractionResult | None], None] | None = None
    # Logs events that are not completed cells (rate-limit backoffs, saturated arms).
    log: Callable[[str], None] = log_line
    lock: threading.Lock = field(default_factory=threading.Lock)
    halt_event: threading.Event = field(default_factory=threading.Event)
    # Set when the run is stopping for any reason: a halt, Ctrl-C or an unexpected
    # error. Every wait that could delay a stop polls this.
    stop_event: threading.Event = field(default_factory=threading.Event)
    resume_event: threading.Event = field(default_factory=threading.Event)
    completed: int = 0
    failed: int = 0
    deferred: int = 0
    halted_reason: str | None = None
    halted_kind: str | None = None
    halted_cell: dict | None = None

    def __post_init__(self) -> None:
        self.resume_event.set()

    def wait_until_runnable(self) -> bool:
        """Block while the grid is paused; return False once it has stopped."""
        self.resume_event.wait()
        return not self.stop_event.is_set()

    def _end_pause(self) -> None:
        """End a breaker pause: re-open the gate and clear the health window.

        Always re-opens, even after a halt, since a shut gate would leave workers stuck.
        """
        with self.lock:
            # Open the gate first in case `reset_window` raises.
            self.resume_event.set()
            self.breaker.reset_window()

    def stop(self) -> None:
        """Tell every wait in the run to give up, without recording a halt."""
        self.stop_event.set()
        self.resume_event.set()

    def is_runnable(self) -> bool:
        """Whether a worker may start work now: not stopped and not paused."""
        return not self.stop_event.is_set() and self.resume_event.is_set()

    def _halt_locked(self, reason: str, kind: str, cell: Cell | None = None) -> None:
        if self.halted_reason is None:
            self.halted_reason = reason
            self.halted_kind = kind
            # Only the first halt is recorded.
            self.halted_cell = cell.identity() if cell is not None else None
            _LOG.error(
                "halt kind=%s cell=%s reason=%s",
                kind,
                json.dumps(self.halted_cell) if self.halted_cell else "unknown",
                reason,
            )
        self.halt_event.set()
        self.stop_event.set()
        # Release any worker waiting on a pause.
        self.resume_event.set()

    def halt(self, reason: str, kind: str = "error", cell: Cell | None = None) -> None:
        with self.lock:
            self._halt_locked(reason, kind, cell)

    def record_deferral(self) -> None:
        """Count one cell handed back to pending because its arm was saturated."""
        with self.lock:
            self.deferred += 1

    def finish(
        self,
        cell: Cell,
        *,
        ok: bool,
        status: str,
        result: ExtractionResult | None,
        cost: float | None,
        healthy: bool | None = None,
        on_record: Callable[[], None] | None = None,
    ) -> bool:
        """Record one completed cell and act on what it means for the run.

        Returns False, recording nothing, if the cell failed after the grid had already
        halted. `on_record` (the failure write) runs under the lock for the same reason.
        This is the only place the breaker is fed, once per cell after retries.
        `healthy` (defaults to `ok`) is what the breaker sees: a context overflow counts
        as completed but unhealthy, so an arm that always overflows still trips it.
        """
        tripped = False
        delay = 0.0
        with self.lock:
            if not ok and self.halt_event.is_set():
                # A halt marks nothing failed; otherwise resume would skip this cell.
                return False

            if on_record is not None:
                on_record()

            if ok:
                self.completed += 1
            else:
                self.failed += 1

            before = self.breaker.trips
            self.breaker.record(ok if healthy is None else healthy)

            spend_exc: SpendCapExceeded | None = None
            try:
                self.spend_guard.add(cost)
            except SpendCapExceeded as exc:
                spend_exc = exc

            if spend_exc is not None:
                # Halt and never auto-resume. The cell recorded is just the one whose
                # cost crossed the cap.
                self._halt_locked(str(spend_exc), "spend_cap", cell)
            elif self.breaker.exhausted:
                self._halt_locked(
                    f"circuit breaker exhausted after {self.breaker.trips} "
                    f"consecutive trips",
                    "circuit_breaker_exhausted",
                    cell,
                )
            elif self.breaker.trips > before:
                # A trip: pause rather than exit, since runs are unattended.
                tripped = True
                delay = self.breaker.backoff_seconds()
                self.resume_event.clear()

        # Outside the lock, so cells already in flight can still be recorded; the gate,
        # not the lock, stops new cells from starting.
        if tripped:
            # Skip the pause if already halted, and cut it short if a stop arrives.
            try:
                if not self.halt_event.is_set():
                    self.sleep(delay, interrupt=self.stop_event)
            finally:
                # Always re-open the gate, even if the sleep raised.
                self._end_pause()

        if self.on_cell is not None:
            self.on_cell(cell, status, result)
        return True


def _run_one(
    cell: Cell,
    *,
    state: _State,
    arms: dict[str, _Arm],
    results_root: Path,
    traces_root: Path,
    backend_factory,
    iteration_timeout_s: float,
    cell_timeout_s: float,
    snapshots_root: Path,
    max_rate_limit_requeues: int = MAX_RATE_LIMIT_REQUEUES,
) -> None:
    """Run one cell in a worker: attempts, classification and accounting.

    Retries and backoff happen here, so a backoff never blocks other arms.
    """
    attempts = 0
    requeues = 0
    arm = arms[cell.model]

    while True:
        if not state.wait_until_runnable():
            return  # halted: this cell was never attempted, and is not failed

        # `timeout` by name, not `acquire`'s default, which binds at import.
        granted = arm.acquire(state.stop_event, timeout=ARM_SATURATION_TIMEOUT_S)
        if granted == ARM_STOPPED:
            return  # stopped while queueing for the arm: this cell never started
        if granted == ARM_SATURATED:
            # Free the worker rather than keep it queued; the cell runs on the next
            # resume.
            state.log(
                f"arm {cell.model} saturated: returning {cell.cell_id} to pending"
            )
            state.record_deferral()
            return

        if not state.is_runnable():
            # The grid stopped or paused while we queued. Give the permit back; on a
            # pause, go back and wait at the gate.
            arm.semaphore.release()
            if state.stop_event.is_set():
                return
            continue

        try:
            result = run_cell(
                cell,
                results_root=results_root,
                traces_root=traces_root,
                backend_factory=backend_factory,
                iteration_timeout_s=iteration_timeout_s,
                cell_timeout_s=cell_timeout_s,
                snapshots_root=snapshots_root,
                # Released by the work thread, so an abandoned cell keeps its permit.
                held_permit=arm.semaphore,
                abort=state.stop_event,
            )
        except CellAbandoned:
            # The run already stopped; the cell stays pending.
            return
        except Exception as exc:
            disposition = policy.classify(exc)

            if disposition is Disposition.RATE_LIMIT:
                requeues += 1
                if requeues > max_rate_limit_requeues:
                    # Fail the cell like any other failure so the breaker can trip.
                    state.log(
                        f"rate limit: failing {cell.cell_id} after "
                        f"{max_rate_limit_requeues} requeues on arm {cell.model}"
                    )
                    state.finish(
                        cell,
                        ok=False,
                        status="failed",
                        result=None,
                        cost=None,
                        on_record=lambda: store.write_failure(
                            cell, results_root, exc=exc, attempts=attempts
                        ),
                    )
                    return
                # A 429 backs off this arm and re-queues the cell without using an
                # attempt. The breaker does not see it.
                slept = arm.back_off(state.sleep)
                state.log(
                    f"rate limit: arm {cell.model} backing off "
                    f"{slept:.1f}s for {cell.cell_id}"
                    if slept
                    else f"rate limit: arm {cell.model} waiting out an in-flight "
                    f"backoff for {cell.cell_id}"
                )
                continue

            if disposition is Disposition.RESULT:
                if isinstance(exc, CellWallClockTimeout):
                    # A result, never retried: retrying slow arms would bias the outcome.
                    result = timed_out_result(cell, cell_timeout_s)
                    store.write_result_atomic(cell.result_path(results_root), result)
                    store.clear_failure(cell, results_root)
                    state.finish(
                        cell, ok=True, status="timed_out", result=result, cost=None
                    )
                    return
                if policy.is_context_window_overflow(exc):
                    # Also a result and not retried, but reported as unhealthy to the
                    # breaker because it is systematic.
                    result = context_overflow_result(cell)
                    store.write_result_atomic(cell.result_path(results_root), result)
                    store.clear_failure(cell, results_root)
                    state.log(
                        f"context overflow: recording {cell.cell_id} on arm "
                        f"{cell.model} as a result — {exc}"
                    )
                    state.finish(
                        cell,
                        ok=True,
                        status="context_overflow",
                        result=result,
                        cost=None,
                        healthy=False,
                    )
                    return
                # `classify` only returns RESULT for the two cases above.
                state.halt(
                    f"unhandled RESULT-class exception {type(exc).__name__}: {exc}",
                    cell=cell,
                )
                return

            if disposition is Disposition.RETRY:
                attempts += 1
                if attempts < policy.MAX_ATTEMPTS:
                    continue
                state.finish(
                    cell,
                    ok=False,
                    status="failed",
                    result=None,
                    cost=None,
                    on_record=lambda: store.write_failure(
                        cell, results_root, exc=exc, attempts=attempts
                    ),
                )
                return

            # HALT: stop now, marking nothing failed. Log the traceback and the cell,
            # since a halt writes no failure file.
            _LOG.error(
                "HALT on %s (funder=%s model=%s prompt=%s t=%s rep=%d): %s: %s",
                cell.cell_id, cell.funder, cell.model_slug, cell.prompt_strategy,
                cell.temperature, cell.replicate, type(exc).__name__, exc,
                exc_info=True,
            )
            state.halt(f"{type(exc).__name__}: {exc}", cell=cell)
            return

        # Clear any old failure record from an earlier `--retry-failed` pass.
        arm.succeeded()
        store.clear_failure(cell, results_root)
        state.finish(
            cell, ok=True, status="completed", result=result, cost=result.cost_usd
        )
        return


def run_grid(
    plan: list[Cell],
    *,
    results_root,
    traces_root,
    workers: int,
    per_arm_limit: int,
    backend_factory,
    breaker: CircuitBreaker,
    spend_guard: SpendGuard,
    iteration_timeout_s: float,
    cell_timeout_s: float,
    max_rate_limit_requeues: int = MAX_RATE_LIMIT_REQUEUES,
    on_cell=None,
    retry_failed: bool = False,
    snapshots_root=DEFAULT_SNAPSHOT_ROOT,
    sleep: Callable[..., None] = _delay,
    log: Callable[[str], None] = log_line,
) -> GridOutcome:
    """Run `plan` until done or halted, and report what happened.

    `plan` may be the whole grid on every resume; it is filtered through
    `store.pending` here and run in plan order. Does not raise on a halt, so the counts
    of finished cells are kept.

    Args:
        workers: pool size; per_arm_limit: concurrent cells per model arm.
        backend_factory: zero-arg callable giving a new backend per Ralph iteration.
        breaker / spend_guard: not thread-safe, only used under `_State.lock`.
        on_cell: optional `(cell, status, result)` callback per finished cell,
            failures included.
        sleep: `(seconds, interrupt=None) -> None`; must honour `interrupt`.
        max_rate_limit_requeues: 429 re-queues one cell may take before it fails.
        log: callback for backoffs and saturated arms.
    """
    results_root = Path(results_root)
    traces_root = Path(traces_root)
    todo = store.pending(plan, results_root, retry_failed=retry_failed)
    state = _State(
        breaker=breaker, spend_guard=spend_guard, sleep=sleep, on_cell=on_cell, log=log
    )

    if todo:
        arms = {cell.model: _Arm(per_arm_limit) for cell in todo}
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grid")
        futures: list = []
        try:
            futures = [
                pool.submit(
                    _run_one,
                    cell,
                    state=state,
                    arms=arms,
                    results_root=results_root,
                    traces_root=traces_root,
                    backend_factory=backend_factory,
                    iteration_timeout_s=iteration_timeout_s,
                    cell_timeout_s=cell_timeout_s,
                    max_rate_limit_requeues=max_rate_limit_requeues,
                    snapshots_root=Path(snapshots_root),
                )
                for cell in todo
            ]
            cancelled = False
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                future.result()  # re-raise a bug in the worker
                if state.halt_event.is_set() and not cancelled:
                    # Drop queued and running cells. Some paid-for work is lost, but
                    # waiting for it would make a halt as slow as the per-cell cap.
                    for pending_future in futures:
                        pending_future.cancel()
                    cancelled = True
        finally:
            # Stop every wait first, or shutdown would wait out the per-cell cap.
            state.stop()
            # Without `cancel_futures` the pool would still run every queued task.
            pool.shutdown(wait=True, cancel_futures=True)

    return GridOutcome(
        completed=state.completed,
        failed=state.failed,
        skipped=len(plan) - state.completed - state.failed,
        deferred=state.deferred,
        halted_reason=state.halted_reason,
        total_cost_usd=spend_guard.total,
        halted_kind=state.halted_kind,
        halted_cell=state.halted_cell,
    )
