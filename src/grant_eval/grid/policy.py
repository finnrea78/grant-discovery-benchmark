"""Failure policy for grid cells: what to do with an exception raised in a cell.

`classify()` returns RETRY (transient, uses one of `MAX_ATTEMPTS`), RATE_LIMIT (429,
backs off without using a retry), HALT (stop the run so it can be resumed) or RESULT
(the cell is scored as it stands, e.g. a timeout or context overflow). Also holds the
circuit breaker and spend guard. litellm is imported lazily because it is slow to load.
"""
from __future__ import annotations

import re
from collections import deque
from enum import Enum
from functools import lru_cache
from types import ModuleType

MAX_ATTEMPTS: int = 2
"""Attempts per cell under RETRY before the cell is marked failed."""


class Disposition(str, Enum):
    """What `classify()` says to do with an exception; see the module docstring."""

    RETRY = "retry"
    HALT = "halt"
    RESULT = "result"
    RATE_LIMIT = "rate_limit"


class HaltGrid(RuntimeError):
    """Stop the run immediately; `.reason` is saved in the run header."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SpendCapExceeded(HaltGrid):
    """Raised by `SpendGuard.add()` once total cost goes past `cap_usd`. Its own class
    so `run.py` can stop a spend-cap halt from auto-resuming."""

    def __init__(self, total: float, cap_usd: float) -> None:
        self.total = total
        self.cap_usd = cap_usd
        super().__init__(f"spend cap exceeded: ${total:.2f} > ${cap_usd:.2f}")


class ProviderPinMismatch(Exception):
    """The served provider or quantisation did not match the pin in `MODEL_ROUTING`.

    A HALT, since cells run on the wrong provider would be wasted. Defined here because
    `classify()` needs it; `preflight.py` re-exports it.
    """

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"provider pin mismatch: expected {expected!r}, got {actual!r}")


class CellWallClockTimeout(Exception):
    """Raised when a cell exceeds its wall-clock cap. A RESULT: scored, never retried.

    Not a `TimeoutError`, because socket and asyncio timeouts share that class and
    would then look the same as this cap in `classify()`.
    """


@lru_cache(maxsize=1)
def _litellm_exceptions() -> ModuleType | None:
    """Import `litellm.exceptions` once, or return `None` if litellm is unavailable."""
    try:
        import litellm.exceptions as litellm_exc
    except ImportError:
        return None
    return litellm_exc


# Bad key, no balance, model not found: retrying would just fail again.
_HALT_STATUS_CODES = frozenset({401, 402, 403, 404})


# litellm raises a plain `BadRequestError` for every OpenRouter 400 instead of
# `ContextWindowExceededError` (checked against litellm 1.96.0), so the provider's
# message is matched instead, e.g. "This endpoint's maximum context length is 300000
# tokens". The `\d+ tokens` part stops it catching unrelated 400s, which should halt.
_CONTEXT_OVERFLOW_MESSAGE = re.compile(
    r"maximum context (?:length|limit) is\s+\d+\s+tokens", re.IGNORECASE
)


def is_context_window_overflow(exc: BaseException) -> bool:
    """Whether `exc` is the provider rejecting a request that is too long for its
    context window, either as litellm's own class or via the message match above."""
    litellm_exc = _litellm_exceptions()
    if litellm_exc is not None and isinstance(exc, litellm_exc.ContextWindowExceededError):
        return True
    return bool(_CONTEXT_OVERFLOW_MESSAGE.search(str(exc)))


def _halt_types() -> tuple[type[BaseException], ...]:
    """Types that HALT regardless of any `status_code` they also carry."""
    types: tuple[type[BaseException], ...] = (
        ProviderPinMismatch,
        FileNotFoundError,
        KeyError,
    )
    litellm_exc = _litellm_exceptions()
    if litellm_exc is not None:
        types += (litellm_exc.BudgetExceededError, litellm_exc.JSONSchemaValidationError)
    return types


def _retry_types() -> tuple[type[BaseException], ...]:
    """Types worth retrying when there is no `status_code`.

    Bare `OSError` and `TimeoutError` are left out, since they would also catch
    `FileNotFoundError` and the cell timeout.
    """
    types: tuple[type[BaseException], ...] = (ConnectionError,)
    litellm_exc = _litellm_exceptions()
    if litellm_exc is not None:
        types += (
            litellm_exc.APIConnectionError,
            litellm_exc.Timeout,
            litellm_exc.ServiceUnavailableError,
            litellm_exc.InternalServerError,
        )
    return types


def classify(exc: BaseException) -> Disposition:
    """Classify an exception raised while running a cell. Rules, in order:

    0. `CellWallClockTimeout` -> RESULT.
    1. `_halt_types()` -> HALT. Checked early because some of these carry misleading
       status codes (`BudgetExceededError` has 429).
    2. Context window overflow -> RESULT: a finding about the arm, not a reason to stop.
    3. `status_code`: 401/402/403/404 -> HALT, 429 -> RATE_LIMIT, >=500 -> RETRY.
    4. `_retry_types()` -> RETRY.
    5. Anything else -> HALT, so unknown errors are looked at rather than lost.
    """
    if isinstance(exc, CellWallClockTimeout):
        return Disposition.RESULT

    if isinstance(exc, _halt_types()):
        return Disposition.HALT

    if is_context_window_overflow(exc):
        return Disposition.RESULT

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if status_code in _HALT_STATUS_CODES:
            return Disposition.HALT
        if status_code == 429:
            return Disposition.RATE_LIMIT
        if status_code >= 500:
            return Disposition.RETRY

    if isinstance(exc, _retry_types()):
        return Disposition.RETRY

    return Disposition.HALT


class CircuitBreaker:
    """Trips when more than `threshold` of the last `window` completed cells failed.

    A trip pauses with exponential backoff and then resumes; the run stops only after
    `max_trips` trips in a row. `record()` takes post-retry outcomes. A trip is cleared
    after `window` successes since it happened.

    In simulation (1,920 cells, 50 seeds per rate) post-retry failure rates of
    25/30/35/40/55% exhausted the breaker in 0/1/11/46/50 runs.
    """

    def __init__(self, window: int = 20, threshold: float = 0.5, max_trips: int = 3) -> None:
        self._window = window
        self._threshold = threshold
        self.max_trips = max_trips
        self._results: deque[bool] = deque(maxlen=window)
        self.trips = 0
        self._was_tripped = False
        # Successful cells since the most recent trip. A failure does not reset it.
        self._successes_since_trip = 0

    @property
    def tripped(self) -> bool:
        """True once a full window has been recorded and more than `threshold` failed."""
        if len(self._results) < self._window:
            return False
        failures = sum(1 for ok in self._results if not ok)
        return (failures / self._window) > self._threshold

    def record(self, ok: bool) -> None:
        """Record one completed cell's outcome and update the trip count.

        A new trip clears the window. Enough successes since the last trip reset `trips`.
        """
        self._results.append(ok)
        now_tripped = self.tripped
        if now_tripped and not self._was_tripped:
            self.trips += 1
            self._successes_since_trip = 0
            self._results.clear()
            now_tripped = False  # the window is now empty; nothing is "tripped"
        elif ok:
            self._successes_since_trip += 1
            if self.trips > 0 and self._successes_since_trip >= self._window:
                self.trips = 0
        self._was_tripped = now_tripped

    def backoff_seconds(self) -> float:
        """Backoff for the current trip count: 60, 120, 240, ... seconds, capped at 1800."""
        return float(min(1800, 60 * 2 ** (self.trips - 1))) if self.trips > 0 else 0.0

    def reset_window(self) -> None:
        """Clear the window for a resumed run. Leaves `trips` alone, so recovery must
        still be earned."""
        self._results.clear()
        self._was_tripped = False

    @property
    def exhausted(self) -> bool:
        """True once `trips` reaches `max_trips`: stop the run instead of pausing again."""
        return self.trips >= self.max_trips


class SpendGuard:
    """Adds up `cost_usd` over the run and halts once it goes past `cap_usd`.

    Mainly a guard against a single runaway cell. `total` is seeded with what earlier
    sessions spent (from `grid.status.summarise`), so the cap covers the whole run.
    """

    def __init__(self, cap_usd: float | None, total: float = 0.0) -> None:
        self.cap_usd = cap_usd
        self.total = total

    def add(self, cost: float | None) -> None:
        """Add one cell's cost; `None` is ignored. Raises `SpendCapExceeded` once
        `total` is strictly above `cap_usd`. `cap_usd=None` disables the guard."""
        if cost is None:
            return
        self.total += cost
        if self.cap_usd is not None and self.total > self.cap_usd:
            raise SpendCapExceeded(self.total, self.cap_usd)
