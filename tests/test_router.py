"""Tests for the router, which turns provider responses into `Reply` and `Usage`.

No network: `OpenRouterBackend` is tested with a monkeypatched `litellm.completion`.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager

import pytest

from grant_eval.agent.router import (
    MockBackend,
    OpenRouterBackend,
    Reply,
    ToolCall,
    Usage,
)


# MockBackend
def test_mock_backend_returns_scripted_replies_in_order():
    script = [
        Reply(text=None, tool_calls=[ToolCall("c1", "fetch_page", {"url": "https://x"})]),
        Reply(
            text=None,
            tool_calls=[
                ToolCall("c2", "submit_opportunities", {"opportunities": [{"id": "a"}]})
            ],
        ),
    ]
    backend = MockBackend(script)
    assert backend.name == "mock"

    reply1, usage1 = backend.complete(
        messages=[{"role": "user", "content": "go"}],
        model="openrouter/anthropic/claude-haiku-4.5",
        temperature=0.0,
        tools=[{"type": "function"}],  # accepted and ignored
    )
    reply2, usage2 = backend.complete(
        messages=[], model="openrouter/anthropic/claude-haiku-4.5", temperature=0.0, tools=[]
    )

    assert reply1.has_tool_calls
    assert reply1.tool_calls[0].id == "c1"
    assert reply1.tool_calls[0].name == "fetch_page"
    assert reply1.tool_calls[0].arguments == {"url": "https://x"}

    assert reply2.tool_calls[0].name == "submit_opportunities"
    assert reply2.tool_calls[0].arguments == {"opportunities": [{"id": "a"}]}

    # usage is tagged with the mock backend and the model string passed in
    assert usage1.backend == "mock"
    assert usage1.model == "openrouter/anthropic/claude-haiku-4.5"

    assert backend.calls == 2


def test_mock_backend_usage_accrues():
    u1 = Usage(input_tokens=10, output_tokens=5, cost_usd=0.0, backend="mock", model="m")
    u2 = Usage(input_tokens=7, output_tokens=3, cost_usd=0.0, backend="mock", model="m")
    total = u1 + u2
    assert total.input_tokens == 17
    assert total.output_tokens == 8
    assert total.cost_usd == 0.0


def test_mock_backend_single_empty_submit():
    backend = MockBackend([Reply(text="done", tool_calls=[])])
    reply, _ = backend.complete(messages=[], model="m", temperature=0.0, tools=[])
    assert reply.text == "done"
    assert not reply.has_tool_calls


def test_mock_backend_n_tool_calls_never_submit():
    script = [
        Reply(text=None, tool_calls=[ToolCall(f"c{i}", "fetch_page", {"url": str(i)})])
        for i in range(3)
    ]
    backend = MockBackend(script)
    for i in range(3):
        reply, _ = backend.complete(messages=[], model="m", temperature=0.0, tools=[])
        assert reply.tool_calls[0].name == "fetch_page"
    assert backend.calls == 3


def test_mock_backend_exhausted_raises_clear_error():
    backend = MockBackend([Reply(text="only one", tool_calls=[])])
    backend.complete(messages=[], model="m", temperature=0.0, tools=[])
    with pytest.raises(Exception) as exc:
        backend.complete(messages=[], model="m", temperature=0.0, tools=[])
    assert "script" in str(exc.value).lower() or "exhaust" in str(exc.value).lower()


# Fake LiteLLM-style response objects
class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, as LiteLLM/OpenAI returns


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content, tool_calls, prompt_tokens, completion_tokens):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls))]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {}


# OpenRouter response mapping
def test_openrouter_maps_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    fake = _FakeResponse(
        content="hello",
        tool_calls=[
            _FakeToolCall("call_1", "fetch_page", json.dumps({"url": "https://x"}))
        ],
        prompt_tokens=42,
        completion_tokens=7,
    )

    import grant_eval.agent.router as router

    def fake_completion(**kwargs):
        assert kwargs["model"] == "openrouter/openai/gpt-4o"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.0
        assert kwargs["tools"] == [{"type": "function"}]
        assert kwargs["tool_choice"] == "auto"
        return fake

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.00123)

    backend = OpenRouterBackend()
    reply, usage = backend.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="openrouter/openai/gpt-4o",
        temperature=0.0,
        tools=[{"type": "function"}],
    )

    assert reply.text == "hello"
    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "fetch_page"
    assert tc.arguments == {"url": "https://x"}  # JSON-decoded to dict

    assert usage.input_tokens == 42
    assert usage.output_tokens == 7
    assert usage.cost_usd == pytest.approx(0.00123)
    assert usage.backend == "openrouter"
    assert usage.model == "openrouter/openai/gpt-4o"


def test_openrouter_malformed_tool_args_degrade_to_sentinel(monkeypatch):
    # Bad tool-argument JSON becomes a sentinel dict so the loop does not crash.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    bad_args = "{not valid json"
    fake = _FakeResponse(
        content=None,
        tool_calls=[_FakeToolCall("call_1", "submit_opportunities", bad_args)],
        prompt_tokens=5,
        completion_tokens=6,
    )

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    reply, _ = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[{"t": 1}]
    )

    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert tc.name == "submit_opportunities"
    assert tc.arguments["_raw"] == bad_args
    assert "_parse_error" in tc.arguments


def test_openrouter_tool_call_only_reply_null_content(monkeypatch):
    # The first reply is often content=None with a single tool call.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse(
        content=None,
        tool_calls=[_FakeToolCall("call_1", "fetch_page", json.dumps({"url": "https://x"}))],
        prompt_tokens=5,
        completion_tokens=6,
    )

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    reply, _ = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[{"t": 1}]
    )

    assert reply.text is None
    assert len(reply.tool_calls) == 1
    tc = reply.tool_calls[0]
    assert tc.name == "fetch_page"
    assert tc.arguments == {"url": "https://x"}


def test_openrouter_no_tools_omits_tool_kwargs(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        return _FakeResponse("txt", None, 1, 1)

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    reply, usage = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.2, tools=[]
    )
    assert "tools" not in seen
    assert "tool_choice" not in seen
    assert reply.text == "txt"
    assert reply.tool_calls == []


def test_openrouter_missing_cost_degrades_to_zero(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    def fake_completion(**kwargs):
        return _FakeResponse("txt", None, 3, 4)

    def raising_cost(response):
        raise Exception("no cost available")

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", raising_cost)

    backend = OpenRouterBackend()
    _, usage = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[]
    )
    assert usage.cost_usd == 0.0


# Served provider
def test_openrouter_records_the_served_provider_top_level(monkeypatch):
    # The grid dispatcher reads the provider off `Usage` to check the provider pin.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("txt", None, 3, 4)
    fake.provider = "DeepInfra"

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    _, usage = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[]
    )
    assert usage.provider == "DeepInfra"


def test_openrouter_provider_falls_back_to_provider_specific_fields(monkeypatch):
    # LiteLLM sometimes nests the provider instead of setting it at the top level.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("txt", None, 3, 4)
    fake.provider_specific_fields = {"provider": "Together"}

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    _, usage = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[]
    )
    assert usage.provider == "Together"


def test_openrouter_missing_provider_reads_back_as_none(monkeypatch):
    # No provider reported reads back as None, not an empty string or a KeyError.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("txt", None, 3, 4)

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    _, usage = backend.complete(
        messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[]
    )
    assert usage.provider is None


def test_usage_positional_construction_still_works_without_provider():
    # `provider` comes last with a default, so five positional arguments still work.
    u = Usage(100, 20, 0.001, "mock", "mock/model")
    assert u.provider is None


# The provider pin is checked on every completion, not just in preflight, because
# OpenRouter can switch provider on any request.
def test_openrouter_raises_on_a_pinned_arm_served_by_a_different_provider(monkeypatch):
    # nova-lite (config.MODELS) is pinned to amazon-bedrock (config.MODEL_ROUTING).
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router
    from grant_eval.grid.preflight import ProviderPinMismatch

    fake = _FakeResponse("txt", None, 3, 4)
    fake.provider = "DeepInfra"

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    with pytest.raises(ProviderPinMismatch, match="nova-lite"):
        backend.complete(
            messages=[], model="openrouter/amazon/nova-lite-v1", temperature=0.0, tools=[]
        )


def test_openrouter_passes_through_when_the_served_provider_matches_the_pin(monkeypatch):
    # Counterpart to the test above, so a check that always raises would fail here.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("txt", None, 3, 4)
    fake.provider = "Amazon Bedrock"  # compared ignoring case and separators

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    reply, usage = backend.complete(
        messages=[], model="openrouter/amazon/nova-lite-v1", temperature=0.0, tools=[]
    )
    assert reply.text == "txt"
    assert usage.provider == "Amazon Bedrock"


def test_openrouter_skips_pin_verification_for_a_model_string_not_in_the_grid(monkeypatch):
    # A model not in `config.MODELS` has no pin, so any provider is accepted.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("txt", None, 3, 4)
    fake.provider = "SomeRandomProvider"

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: fake)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterBackend()
    reply, usage = backend.complete(
        messages=[], model="openrouter/not-a-grid-arm/probe", temperature=0.0, tools=[]
    )
    assert reply.text == "txt"
    assert usage.provider == "SomeRandomProvider"


def _no_key(monkeypatch) -> None:
    """Remove the API key, and stop `load_dotenv()` reading it back from a real `.env`."""
    import grant_eval.agent.router as router

    monkeypatch.setattr(router, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_openrouter_construct_without_key_does_not_raise(monkeypatch):
    _no_key(monkeypatch)
    OpenRouterBackend()


def test_openrouter_complete_without_key_raises_clear_error(monkeypatch):
    _no_key(monkeypatch)
    backend = OpenRouterBackend()
    with pytest.raises(Exception) as exc:
        backend.complete(
            messages=[], model="openrouter/openai/gpt-4o", temperature=0.0, tools=[]
        )
    assert "OPENROUTER_API_KEY" in str(exc.value)


# 429 retries. OpenRouter returns `engine_overloaded` 429s in bursts, and if they
# reach the grid worker the whole run restarts from step 0, so they are retried here.
# Only 429s are retried: auth and billing errors should still fail straight away.
def _status_error(status: int, message: str = "boom") -> Exception:
    exc = Exception(message)
    exc.status_code = status
    return exc


def _retrying_backend(monkeypatch, errors, sleeps):
    """Build a backend that raises `errors` in turn, then succeeds.

    Sleeps are recorded instead of taken. Returns the backend and the kwargs of each call.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    fake = _FakeResponse("ok", None, 1, 1)
    queue = list(errors)
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if queue:
            raise queue.pop(0)
        return fake

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)
    backend = OpenRouterBackend()
    monkeypatch.setattr(backend, "_sleep", sleeps.append)
    return backend, calls


def _complete(backend):
    return backend.complete(
        messages=[], model="openrouter/not-a-grid-arm/probe", temperature=0.0, tools=[]
    )


def test_a_429_is_retried_and_the_call_succeeds(monkeypatch):
    sleeps: list[float] = []
    backend, calls = _retrying_backend(
        monkeypatch, [_status_error(429, "engine_overloaded")] * 2, sleeps
    )

    reply, _ = _complete(backend)

    assert reply.text == "ok"
    assert len(calls) == 3, "two 429s must cost two retries, not one call"
    assert len(sleeps) == 2, "every retry waits before re-issuing"


def test_the_real_litellm_rate_limit_class_is_retried(monkeypatch):
    """litellm's own `RateLimitError` is retried, not just a bare 429 status code."""
    from litellm.exceptions import RateLimitError

    sleeps: list[float] = []
    backend, calls = _retrying_backend(
        monkeypatch,
        [RateLimitError(message="engine_overloaded", llm_provider="openrouter", model="m")],
        sleeps,
    )

    reply, _ = _complete(backend)
    assert reply.text == "ok"
    assert len(calls) == 2


def test_the_retry_bound_is_respected_and_the_429_still_propagates(monkeypatch):
    """After MAX_RETRIES the last 429 is raised unchanged for the grid worker to handle."""
    sleeps: list[float] = []
    backend, calls = _retrying_backend(
        monkeypatch, [_status_error(429, "engine_overloaded")] * 20, sleeps
    )

    with pytest.raises(Exception) as exc:
        _complete(backend)

    assert getattr(exc.value, "status_code", None) == 429
    assert len(calls) == OpenRouterBackend.MAX_RETRIES + 1
    assert len(sleeps) == OpenRouterBackend.MAX_RETRIES


def test_the_backoff_grows_exponentially_and_is_jittered(monkeypatch):
    """Each wait falls in the upper half of its exponential window.

    litellm's own `num_retries` is not used because it retries with no delay.
    """
    sleeps: list[float] = []
    backend, _ = _retrying_backend(
        monkeypatch, [_status_error(429, "engine_overloaded")] * 20, sleeps
    )

    with pytest.raises(Exception):
        _complete(backend)

    base = OpenRouterBackend.RETRY_BASE_S
    for i, slept in enumerate(sleeps):
        window = base * 2 ** i
        assert window / 2 <= slept <= window, f"wait {i} outside its band: {slept}"
    assert sleeps == sorted(sleeps)


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_an_auth_or_billing_error_is_never_retried(monkeypatch, status):
    """Auth and billing errors are not retried, since retrying gives the same error."""
    sleeps: list[float] = []
    backend, calls = _retrying_backend(monkeypatch, [_status_error(status)] * 20, sleeps)

    with pytest.raises(Exception):
        _complete(backend)

    assert len(calls) == 1
    assert sleeps == []


def test_a_server_error_is_not_retried_here(monkeypatch):
    """5xx errors are left to the grid worker, so they are not retried twice."""
    sleeps: list[float] = []
    backend, calls = _retrying_backend(monkeypatch, [_status_error(503)] * 20, sleeps)

    with pytest.raises(Exception):
        _complete(backend)

    assert len(calls) == 1


def test_a_budget_exceeded_error_is_not_retried_despite_its_429(monkeypatch):
    """`BudgetExceededError` has status 429 but means the spend cap is hit, so no retry."""
    from litellm.exceptions import BudgetExceededError

    sleeps: list[float] = []
    backend, calls = _retrying_backend(
        monkeypatch, [BudgetExceededError(current_cost=2.0, max_budget=1.0)] * 5, sleeps
    )

    with pytest.raises(BudgetExceededError):
        _complete(backend)

    assert len(calls) == 1
    assert sleeps == []


def test_the_retry_never_touches_litellm_s_module_level_num_retries(monkeypatch):
    """The retry does not set litellm's global `num_retries`, which litellm itself resets."""
    import grant_eval.agent.router as router

    sleeps: list[float] = []
    backend, calls = _retrying_backend(
        monkeypatch, [_status_error(429, "engine_overloaded")], sleeps
    )
    monkeypatch.setattr(router.litellm, "num_retries", None, raising=False)

    _complete(backend)

    assert router.litellm.num_retries is None
    assert all("num_retries" not in kwargs for kwargs in calls)


# Logging the request size before the call
class _CapturedLogs(logging.Handler):
    """Collects router log records.

    `caplog` cannot be used because the package logger has `propagate=False`.
    """

    def __init__(self, order: list) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self._order = order

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self._order.append("log")


@contextmanager
def _capturing_router_logs(order: list):
    logger = logging.getLogger("grant_eval.agent.router")
    handler = _CapturedLogs(order)
    saved_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)


def test_the_outgoing_request_is_sized_before_the_call(monkeypatch):
    """The request size is logged before the call, so a context overflow still leaves a record."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    order: list[str] = []

    def fake_completion(**kwargs):
        order.append("call")
        raise _status_error(400, "maximum context length is 300000 tokens")

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    messages = [{"role": "user", "content": "x" * 500}]
    with _capturing_router_logs(order) as captured:
        backend = OpenRouterBackend()
        with pytest.raises(Exception):
            backend.complete(
                messages=messages,
                model="openrouter/not-a-grid-arm/probe",
                temperature=0.0,
                tools=[],
            )

    assert order[:2] == ["log", "call"], (
        f"the request was not described before it was issued: {order}"
    )
    message = captured.records[0].getMessage()
    assert "messages=1" in message
    assert "approx_chars=" in message
    approx = int(re.search(r"approx_chars=(\d+)", message).group(1))
    assert approx >= 500, f"the size is not the size of the messages: {message}"


def test_the_request_size_line_never_carries_the_api_key(monkeypatch):
    """The size log line must not include the API key from `kwargs`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-must-not-be-logged")
    import grant_eval.agent.router as router

    order: list[str] = []
    monkeypatch.setattr(
        router.litellm, "completion", lambda **kwargs: _FakeResponse("ok", None, 1, 1)
    )
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    with _capturing_router_logs(order) as captured:
        OpenRouterBackend().complete(
            messages=[{"role": "user", "content": "hi"}],
            model="openrouter/not-a-grid-arm/probe",
            temperature=0.0,
            tools=[],
        )

    assert captured.records, "nothing was logged before the call"
    assert not any(
        "sk-secret-must-not-be-logged" in r.getMessage() for r in captured.records
    )


def test_the_size_is_not_computed_when_debug_is_off(monkeypatch):
    """The size is only worked out when DEBUG logging is on, as it is slow on long transcripts."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    logger = logging.getLogger("grant_eval.agent.router")
    saved = logger.level
    logger.setLevel(logging.INFO)

    class _Exploding(list):
        def __len__(self) -> int:
            raise AssertionError("the messages were inspected with DEBUG off")

    try:
        router.OpenRouterBackend._log_request_size(
            {"model": "m", "messages": _Exploding(), "tools": []}, 0
        )
    finally:
        logger.setLevel(saved)
