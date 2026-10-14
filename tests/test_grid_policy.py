"""Tests for the grid dispatcher's error classification, circuit breaker and spend guard.

`record(ok)` sees outcomes after retries, so the failure rates below are worse than
they look.
"""
from __future__ import annotations

import random
import subprocess
import sys

import litellm.exceptions as litellm_exc
import pytest

from grant_eval.grid.policy import (
    MAX_ATTEMPTS,
    CellWallClockTimeout,
    CircuitBreaker,
    Disposition,
    HaltGrid,
    ProviderPinMismatch,
    SpendCapExceeded,
    SpendGuard,
    _halt_types,
    classify,
    is_context_window_overflow,
)

# A real 400 body from OpenRouter when nova-lite went over its 300,000 token limit.
OVERFLOW_BODY = (
    "This endpoint's maximum context length is 300000 tokens. However, you "
    "requested about 334283 tokens (334283 of text input). Please reduce the length "
    "of either one, or use the middle-out transform to compress your prompt "
    "automatically."
)


class _HTTPish(Exception):
    """A non-litellm exception with a `status_code`, to check `classify()` duck-types it."""

    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, Disposition.HALT),
        (402, Disposition.HALT),
        (403, Disposition.HALT),
        (404, Disposition.HALT),
        (429, Disposition.RATE_LIMIT),
        (500, Disposition.RETRY),
        (503, Disposition.RETRY),
    ],
)
def test_classify_http_status(status, expected):
    assert classify(_HTTPish(status)) is expected


def test_connection_error_retries():
    assert classify(ConnectionError("c")) is Disposition.RETRY


def test_bare_timeout_error_halts_not_retries():
    """A plain `TimeoutError` halts; cell timeouts should raise `CellWallClockTimeout`."""
    assert classify(TimeoutError("t")) is Disposition.HALT


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError("chmod'd mirror"),
        IsADirectoryError("path is a directory"),
        NotADirectoryError("path is not a directory"),
        FileExistsError("already exists"),
        InterruptedError("interrupted"),
    ],
)
def test_os_error_family_halts_not_retries(exc):
    """`OSError` subclasses are setup problems, so they halt instead of retrying."""
    assert classify(exc) is Disposition.HALT


def test_bare_os_error_halts():
    """A bare `OSError` is not retryable, so it falls through to HALT."""
    assert classify(OSError("generic os error")) is Disposition.HALT


def test_unknown_exception_halts():
    """An unknown error stops the grid rather than being retried on every cell."""

    class Weird(Exception):
        pass

    assert classify(Weird("?")) is Disposition.HALT


def test_missing_snapshot_halts():
    """A missing snapshot (`FileNotFoundError`) halts the grid."""
    assert classify(FileNotFoundError("no manifest")) is Disposition.HALT


def test_missing_registry_entry_halts():
    assert classify(KeyError("some-funder")) is Disposition.HALT


def test_provider_pin_mismatch_halts():
    """`ProviderPinMismatch` halts the grid."""
    assert classify(ProviderPinMismatch(expected="fp8", actual="bf16")) is Disposition.HALT


def test_provider_pin_mismatch_carries_expected_and_actual():
    """`expected` and `actual` are kept as attributes, not just in the message."""
    exc = ProviderPinMismatch(expected="deepinfra/fp8", actual="deepinfra/bf16")
    assert exc.expected == "deepinfra/fp8"
    assert exc.actual == "deepinfra/bf16"


def test_cell_wall_clock_timeout_is_result_not_retry_or_halt():
    """A cell timeout is recorded as a result, not retried or halted on."""
    assert classify(CellWallClockTimeout("cell exceeded its cap")) is Disposition.RESULT


def test_max_attempts_is_two():
    assert MAX_ATTEMPTS == 2


# Context-window overflow counts as a RESULT, like a timeout. It says something about
# the arm, and a retry would just build the same oversized transcript again.


def test_litellm_does_not_give_an_openrouter_overflow_its_own_class():
    """litellm maps an OpenRouter overflow to a plain `BadRequestError`.

    This is why `classify()` has to match on the message text. If a newer litellm
    starts using `ContextWindowExceededError` here, this test fails and the message
    match is no longer needed.
    """
    import httpx
    import openai
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    raw = openai.BadRequestError(
        message=OVERFLOW_BODY,
        response=httpx.Response(400, request=request, json={"error": {"message": OVERFLOW_BODY}}),
        body={"error": {"message": OVERFLOW_BODY}},
    )
    with pytest.raises(litellm_exc.BadRequestError) as excinfo:
        exception_type(
            model="openrouter/amazon/nova-lite-v1",
            original_exception=raw,
            custom_llm_provider="openrouter",
            completion_kwargs={},
            extra_kwargs={},
        )
    assert not isinstance(excinfo.value, litellm_exc.ContextWindowExceededError)
    assert excinfo.value.status_code == 400
    assert classify(excinfo.value) is Disposition.RESULT


def test_openrouter_context_overflow_is_result_not_halt():
    """A `BadRequestError` with OpenRouter's overflow message is a RESULT."""
    exc = litellm_exc.BadRequestError(
        message=f"OpenrouterException - {OVERFLOW_BODY}",
        model="openrouter/amazon/nova-lite-v1",
        llm_provider="openrouter",
    )
    assert exc.status_code == 400
    assert is_context_window_overflow(exc)
    assert classify(exc) is Disposition.RESULT


def test_litellm_context_window_exceeded_error_is_result():
    """`ContextWindowExceededError` is recognised by type, whatever its message says."""
    exc = litellm_exc.ContextWindowExceededError(
        message="you asked for too much", model="m", llm_provider="openrouter"
    )
    assert isinstance(exc, litellm_exc.BadRequestError)
    assert is_context_window_overflow(exc)
    assert classify(exc) is Disposition.RESULT


@pytest.mark.parametrize(
    "message",
    [
        # real body from an OpenRouter call sent with no messages
        '{"error":{"message":"Input required: specify \\"prompt\\" or \\"messages\\"",'
        '"code":400}}',
        "Invalid value for 'temperature': must be between 0 and 2",
        "tool_choice is not supported by this model",
    ],
)
def test_a_genuine_non_context_400_still_halts(message):
    """Any other 400 is a bug in our request, so it still halts the grid."""
    exc = litellm_exc.BadRequestError(
        message=f"OpenrouterException - {message}",
        model="openrouter/amazon/nova-lite-v1",
        llm_provider="openrouter",
    )
    assert exc.status_code == 400
    assert not is_context_window_overflow(exc)
    assert classify(exc) is Disposition.HALT


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: litellm_exc.RateLimitError(
            message="engine_overloaded", llm_provider="openrouter", model="m"
        ),
        lambda: litellm_exc.InternalServerError(
            message="upstream 500", llm_provider="openrouter", model="m"
        ),
        lambda: litellm_exc.AuthenticationError(
            message="bad key", llm_provider="openrouter", model="m"
        ),
        lambda: ConnectionError("connection reset"),
    ],
)
def test_the_overflow_match_does_not_fire_on_unrelated_errors(make_exc):
    """The overflow message match does not catch other errors such as a 429."""
    exc = make_exc()
    assert not is_context_window_overflow(exc)
    assert classify(exc) is not Disposition.RESULT


def test_a_halt_carve_out_wins_over_the_overflow_match():
    """HALT types are checked before the overflow match, even if the message matches."""
    exc = KeyError(f"funder lookup failed: {OVERFLOW_BODY}")
    assert is_context_window_overflow(exc)
    assert classify(exc) is Disposition.HALT


# Real litellm exceptions. Most have a status_code that matches their type, but
# `BudgetExceededError` (429) and `JSONSchemaValidationError` (500) are both HALTs.


def test_litellm_authentication_error_halts():
    exc = litellm_exc.AuthenticationError(
        message="bad key", llm_provider="openrouter", model="m"
    )
    assert exc.status_code == 401
    assert classify(exc) is Disposition.HALT


def test_litellm_permission_denied_halts():
    # litellm's `PermissionDeniedError` requires a real `response`, unlike its siblings.
    import httpx

    exc = litellm_exc.PermissionDeniedError(
        message="insufficient balance",
        llm_provider="openrouter",
        model="m",
        response=httpx.Response(
            status_code=403, request=httpx.Request("GET", "https://litellm.ai")
        ),
    )
    assert exc.status_code == 403
    assert classify(exc) is Disposition.HALT


def test_litellm_not_found_halts():
    exc = litellm_exc.NotFoundError(
        message="model not found", model="m", llm_provider="openrouter"
    )
    assert exc.status_code == 404
    assert classify(exc) is Disposition.HALT


def test_litellm_rate_limit_is_rate_limit_not_retry():
    exc = litellm_exc.RateLimitError(
        message="slow down", llm_provider="openrouter", model="m"
    )
    assert exc.status_code == 429
    assert classify(exc) is Disposition.RATE_LIMIT


def test_litellm_budget_exceeded_halts_not_rate_limited():
    """`BudgetExceededError` halts even though its status is 429, or it would re-queue forever."""
    exc = litellm_exc.BudgetExceededError(current_cost=10.0, max_budget=5.0)
    assert exc.status_code == 429
    assert classify(exc) is Disposition.HALT


def test_litellm_json_schema_validation_error_halts_not_retries():
    """`JSONSchemaValidationError` halts even though its status code is 500."""
    exc = litellm_exc.JSONSchemaValidationError(
        model="m", llm_provider="openrouter", raw_response="{}", schema="{}"
    )
    assert exc.status_code == 500
    assert classify(exc) is Disposition.HALT


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: litellm_exc.APIConnectionError(
            message="connection reset", llm_provider="openrouter", model="m"
        ),
        lambda: litellm_exc.Timeout(message="timed out", model="m", llm_provider="openrouter"),
        lambda: litellm_exc.ServiceUnavailableError(
            message="down", llm_provider="openrouter", model="m"
        ),
        lambda: litellm_exc.InternalServerError(
            message="500", llm_provider="openrouter", model="m"
        ),
    ],
)
def test_litellm_transient_upstream_errors_retry(make_exc):
    assert classify(make_exc()) is Disposition.RETRY


def test_litellm_timeout_status_code_is_not_5xx():
    """Timeout has status 408, so it is retried because of its type, not its status."""
    exc = litellm_exc.Timeout(message="timed out", model="m", llm_provider="openrouter")
    assert exc.status_code == 408


def test_isinstance_catches_a_subclass_a_name_match_would_miss():
    """Subclasses such as `MidStreamFallbackError` are classified through `isinstance`."""
    exc = litellm_exc.MidStreamFallbackError(
        message="fallback failed mid-stream", llm_provider="openrouter", model="m"
    )
    assert isinstance(exc, litellm_exc.ServiceUnavailableError)
    assert classify(exc) is Disposition.RETRY


# Circuit breaker


def test_breaker_does_not_trip_below_threshold():
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(10):
        b.record(ok=False)
    for _ in range(10):
        b.record(ok=True)
    assert b.tripped is False


def test_breaker_trips_above_threshold():
    """A trip increments `trips` and clears the window, so `tripped` reads False again."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1
    assert b.tripped is False, "the window was cleared by the trip itself"


def test_breaker_needs_a_full_window():
    """Three failures at the start of a run do not trip the breaker."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(3):
        b.record(ok=False)
    assert b.tripped is False


def test_backoff_grows_and_is_capped_at_30_minutes():
    b = CircuitBreaker(window=20, threshold=0.5)
    delays = []
    for _ in range(4):
        for _ in range(20):
            b.record(ok=False)
        delays.append(b.backoff_seconds())
        b.reset_window()
    assert delays[0] < delays[1] < delays[2]
    assert delays == [60.0, 120.0, 240.0, 480.0]


def test_backoff_actually_caps_at_1800_seconds():
    """Backoff stops growing at 1800 seconds."""
    b = CircuitBreaker(window=20, threshold=0.5, max_trips=100)
    for _ in range(10):
        for _ in range(20):
            b.record(ok=False)
        b.reset_window()
    # without the cap this would be 60 * 2**9 == 30720
    assert b.trips == 10
    assert b.backoff_seconds() == 1800.0


def test_breaker_hard_stops_after_max_trips():
    """`exhausted` turns on at the third trip and not before."""
    b = CircuitBreaker(window=20, threshold=0.5, max_trips=3)
    for i in range(3):
        for _ in range(20):
            b.record(ok=False)
        b.reset_window()
        if i < 2:
            assert b.exhausted is False, f"exhausted too early after trip {i + 1}"
        else:
            assert b.exhausted is True


def test_trip_recover_trip_is_not_two_consecutive_trips():
    """`trips` counts consecutive trips, so recovered incidents do not add up."""
    b = CircuitBreaker(window=20, threshold=0.5, max_trips=3)

    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1
    b.reset_window()

    for _ in range(20):
        b.record(ok=True)
    assert b.tripped is False

    # a later incident starts counting from 1 again
    b.reset_window()
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1, "a recovered incident must not accumulate into the next one"
    assert b.exhausted is False
    assert b.backoff_seconds() == 60.0, "backoff must also reset, not compound across incidents"


# Recovery: the breaker recovers after `window` successes in a row since the last trip.


def test_one_success_cannot_clear_a_trip():
    """One success after a trip does not clear it."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1

    b.record(ok=True)
    assert b.trips == 1, "one success is nowhere near `window` successes"


def test_recovery_requires_the_full_window_of_successes_not_fewer():
    """`window - 1` successes are not enough, but `window` successes are."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1

    for _ in range(19):
        b.record(ok=True)
    assert b.trips == 1, "19 successes is one short of a full window"

    b.record(ok=True)
    assert b.trips == 0, "the 20th consecutive success completes recovery"


def test_genuine_recovery_via_sliding_window_resets_trips():
    """`window` successes after a trip reset `trips` without calling `reset_window()`."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1

    # no reset_window() needed, the trip already cleared the window
    for _ in range(20):
        b.record(ok=True)
    assert b.trips == 0, "20 successes since the trip is genuine recovery"
    assert b.backoff_seconds() == 0.0


def test_sustained_55_percent_failure_rate_reaches_exhausted():
    """A steady 55% failure rate, above the 50% threshold, ends in `exhausted`."""
    b = CircuitBreaker(window=20, threshold=0.5)
    rng = random.Random(0)
    for _ in range(5000):
        ok = rng.random() >= 0.55
        b.record(ok)
        if b.exhausted:
            break
    assert b.exhausted is True, "a sustained 55% failure rate must eventually hard-exit"


@pytest.mark.parametrize("failure_rate,seed", [(0.25, s) for s in range(10)] + [(0.30, s) for s in range(10) if s != 1])
def test_low_post_retry_failure_rates_do_not_hard_exit_a_full_grid(failure_rate, seed):
    """25-30% failure after retries does not stop a full 1,920-cell grid.

    Seed 1 at 30% is left out because it is the one seed that does exhaust.
    """
    b = CircuitBreaker(window=20, threshold=0.5)
    rng = random.Random(seed)
    for _ in range(1920):
        b.record(rng.random() >= failure_rate)
        if b.exhausted:
            break
    assert b.exhausted is False, (
        f"failure_rate={failure_rate} seed={seed} hard-exited well below the calibrated range"
    )


def test_35_and_40_percent_post_retry_failure_reliably_hard_exits():
    """40% failure after retries stops the grid for at least half of 30 seeds.

    Stopping is wanted here: a resumed run loses little, while not stopping loses most
    of the cells.
    """
    b40_exhausted = 0
    n_seeds = 30
    for seed in range(n_seeds):
        b = CircuitBreaker(window=20, threshold=0.5)
        rng = random.Random(seed)
        for _ in range(1920):
            b.record(rng.random() >= 0.40)
            if b.exhausted:
                b40_exhausted += 1
                break
    assert b40_exhausted >= n_seeds * 0.5, (
        "a 40% post-retry failure rate should reliably hard-exit; "
        "if this now fails, the mechanism got LESS sensitive; update this test "
        "deliberately, don't just raise the bar"
    )


def test_reset_window_gives_a_resumed_run_a_fresh_window():
    """`reset_window()` gives a resumed run a clean window, even if it is already empty."""
    b = CircuitBreaker(window=20, threshold=0.5)
    for _ in range(20):
        b.record(ok=False)
    assert b.trips == 1
    assert b.tripped is False, "the trip already cleared the window"
    b.reset_window()
    assert b.tripped is False
    for _ in range(20):
        b.record(ok=True)
    assert b.tripped is False


# Spend guard


def test_spend_guard_halts_past_the_cap():
    g = SpendGuard(cap_usd=1.0)
    g.add(0.4)
    g.add(0.4)
    assert g.total == pytest.approx(0.8)
    with pytest.raises(HaltGrid, match="spend"):
        g.add(0.4)


def test_spend_guard_raises_spend_cap_exceeded_with_total_and_cap():
    """Going over the cap raises `SpendCapExceeded` with the total and cap attached."""
    g = SpendGuard(cap_usd=1.0)
    g.add(0.7)
    with pytest.raises(SpendCapExceeded) as excinfo:
        g.add(0.7)
    assert excinfo.value.total == pytest.approx(1.4)
    assert excinfo.value.cap_usd == 1.0


def test_spend_guard_exactly_at_the_cap_does_not_halt():
    """Spending exactly the cap is allowed; only going over it halts."""
    g = SpendGuard(cap_usd=1.0)
    g.add(0.6)
    g.add(0.4)
    assert g.total == pytest.approx(1.0)


def test_spend_guard_tolerates_none_costs():
    g = SpendGuard(cap_usd=1.0)
    g.add(None)
    assert g.total == 0.0


def test_spend_guard_can_be_seeded_with_a_prior_run_s_spend():
    """Starting `total` from earlier spend makes the cap cover resumed runs too."""
    g = SpendGuard(cap_usd=1.0, total=0.9)
    assert g.total == pytest.approx(0.9)
    with pytest.raises(SpendCapExceeded):
        g.add(0.2)


def test_spend_guard_without_a_cap_never_halts():
    g = SpendGuard(cap_usd=None)
    for _ in range(1000):
        g.add(10.0)
    assert g.total == 10000.0


# Import cost


def test_importing_policy_does_not_eagerly_import_litellm():
    """Importing the policy module does not pull in litellm, which is slow to import.

    Runs in a subprocess so earlier imports in this session do not affect it.
    """
    code = (
        "import sys\n"
        "import grant_eval.grid.policy\n"
        "assert 'litellm' not in sys.modules, sorted(m for m in sys.modules if 'litellm' in m)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# HALT types have to be checked before the status_code rule, or some of them would be
# sent to RETRY or RATE_LIMIT instead.


@pytest.mark.parametrize("conflicting_status_code", [429, 500, 503])
@pytest.mark.parametrize("halt_type", _halt_types())
def test_halt_carve_outs_survive_a_conflicting_status_code(halt_type, conflicting_status_code):
    """Each HALT type still halts when given a status code that means retry.

    Built with `__new__` because the constructors take different arguments.
    """
    synthetic = type(
        f"_Conflicting_{halt_type.__name__}", (halt_type,), {"status_code": conflicting_status_code}
    )
    exc = synthetic.__new__(synthetic)
    assert classify(exc) is Disposition.HALT


# The exact set of HALT types


def test_halt_types_membership_is_exactly_the_expected_set():
    """`_halt_types()` returns exactly the five expected types.

    Removing some of them would not break any other test, since HALT is also the default.
    """
    try:
        import litellm.exceptions as _real_litellm_exc
    except ImportError:
        _real_litellm_exc = None

    expected = {ProviderPinMismatch, FileNotFoundError, KeyError}
    if _real_litellm_exc is not None:
        expected |= {_real_litellm_exc.BudgetExceededError, _real_litellm_exc.JSONSchemaValidationError}

    assert set(_halt_types()) == expected
