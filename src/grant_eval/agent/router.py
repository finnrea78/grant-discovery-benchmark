"""Model transport for the grant-discovery agent.

The only module that talks to a model provider. The loop, tools and Ralph driver are
separate, so between models only the model string changes. Two backends share the
`RouterBackend` protocol:

* `MockBackend`: replays scripted replies offline, for tests.
* `OpenRouterBackend`: one `litellm.completion()` call against OpenRouter.

Responses are converted into small dataclasses so nothing else depends on the
LiteLLM response shape.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, replace
from typing import Protocol

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True

_LOG = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """One tool call the model asked for, with `arguments` already decoded from JSON."""

    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    """One model turn: `text` and/or `tool_calls`, either of which may be empty.

    `reasoning` holds a reasoning model's visible thinking. It is kept apart from
    `text` so the loop decides whether to feed it back, since only some models
    produce it.
    """

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_empty(self) -> bool:
        """No text, no reasoning and no tool calls."""
        return not (self.text or "").strip() and not self.reasoning and not self.tool_calls


@dataclass
class Usage:
    """Tokens and cost for one or more `complete()` calls, with the backend and model.

    Adding two `Usage`s sums tokens and cost and keeps the right-hand side's backend,
    model and provider, so a running total is tagged with the latest call.
    `provider` is the provider OpenRouter says served the call, used to check the
    provider pin. It is `None` for `MockBackend` or when the response does not say."""

    input_tokens: int
    output_tokens: int
    cost_usd: float
    backend: str
    model: str
    provider: str | None = None

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            backend=other.backend,
            model=other.model,
            provider=other.provider,
        )


class RouterBackend(Protocol):
    """The interface the agent loop calls. Implementations return `(Reply, Usage)`."""

    name: str

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict],
    ) -> tuple[Reply, Usage]:
        ...


# Token counts for scripted replies with no Usage given. Non-zero so tests of token
# totals mean something.
_MOCK_INPUT_TOKENS = 100
_MOCK_OUTPUT_TOKENS = 20


class MockBackend:
    """Replays a scripted list of `Reply` objects, one per `complete()` call.

    An optional matching list of `Usage` objects can be given; otherwise a fixed usage
    is used. `tools` is ignored."""

    name = "mock"

    def __init__(
        self,
        replies: list[Reply],
        usages: list[Usage] | None = None,
    ) -> None:
        self._replies = list(replies)
        self._usages = list(usages) if usages is not None else None
        if self._usages is not None and len(self._usages) != len(self._replies):
            raise ValueError(
                "MockBackend: `usages` must be the same length as `replies` "
                f"({len(self._usages)} vs {len(self._replies)})"
            )
        self._cursor = 0

    @property
    def calls(self) -> int:
        """How many `complete()` calls have been served."""
        return self._cursor

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict],
    ) -> tuple[Reply, Usage]:
        if self._cursor >= len(self._replies):
            raise RuntimeError(
                f"MockBackend script exhausted after {self._cursor} call(s): "
                f"no scripted Reply for call #{self._cursor + 1}. Add more replies "
                "to the script."
            )
        reply = self._replies[self._cursor]
        if self._usages is not None:
            # tag the scripted usage with the actual model string for this call
            usage = replace(self._usages[self._cursor], backend="mock", model=model)
        else:
            usage = Usage(
                input_tokens=_MOCK_INPUT_TOKENS,
                output_tokens=_MOCK_OUTPUT_TOKENS,
                cost_usd=0.0,
                backend="mock",
                model=model,
            )
        self._cursor += 1
        return reply, usage


class OpenRouterBackend:
    """One `litellm.completion()` call against OpenRouter, returned as `(Reply, Usage)`.

    The API key is only read when `complete()` runs, so tests can import and build
    this without `OPENROUTER_API_KEY`."""

    name = "openrouter"

    # Per-request timeout in seconds. LiteLLM's default is 6000s, so a provider that
    # stalls after connecting could hold up a cell for over an hour. Override with
    # OPENROUTER_TIMEOUT_S.
    TIMEOUT_S: float = float(os.environ.get("OPENROUTER_TIMEOUT_S", "180"))

    # Limited retries for a provider 429. OpenRouter returns `engine_overloaded` in
    # bursts, and otherwise the whole cell would restart from step 0. litellm's own
    # `num_retries` is not used because its sync path retries with no wait, so all the
    # retries land inside the same burst. Override with OPENROUTER_MAX_RETRIES and
    # OPENROUTER_RETRY_BASE_S.
    MAX_RETRIES: int = int(os.environ.get("OPENROUTER_MAX_RETRIES", "3"))
    RETRY_BASE_S: float = float(os.environ.get("OPENROUTER_RETRY_BASE_S", "2"))

    def __init__(self) -> None:
        # load a local .env if there is one
        load_dotenv()

    def _api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or add it to a .env file "
                "before calling OpenRouterBackend.complete()."
            )
        return key

    @staticmethod
    def _routing_for(model: str) -> dict:
        """Routing settings for this model string from `config.MODEL_ROUTING`, or {}.

        A model string that is not a grid model (a probe or smoke test) is not pinned.
        """
        from grant_eval.config import MODELS, MODEL_ROUTING

        for label, slug in MODELS.items():
            if slug == model:
                return MODEL_ROUTING.get(label, {})
        return {}

    @staticmethod
    def _label_for(model: str) -> str | None:
        """The grid label for this model string, or `None` if it is not a grid model.

        `None` tells the caller to skip the provider pin check.
        """
        from grant_eval.config import MODELS

        for label, slug in MODELS.items():
            if slug == model:
                return label
        return None

    def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        tools: list[dict],
    ) -> tuple[Reply, Usage]:
        api_key = self._api_key()

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "api_key": api_key,
            "timeout": self.TIMEOUT_S,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Pin provider, quantisation and reasoning effort. OpenRouter otherwise spreads
        # open-weight models across many providers and precisions, which would mix the
        # provider into the model comparison.
        routing = self._routing_for(model)
        if routing:
            extra: dict = {}
            if "provider" in routing:
                extra["provider"] = routing["provider"]
            if "reasoning_effort" in routing:
                extra["reasoning"] = {"effort": routing["reasoning_effort"]}
            if extra:
                kwargs["extra_body"] = extra

        response = self._completion_with_retry(kwargs)

        reply = self._to_reply(response)
        usage = self._to_usage(response, model)

        # Check the provider pin on every completion, since the provider could change on
        # any request. A mismatch raises `ProviderPinMismatch`, which halts the grid.
        # Imported here to avoid a circular import through `agent.extract`.
        label = self._label_for(model)
        if label is not None:
            from grant_eval.grid.preflight import assert_provider_pin

            assert_provider_pin(label, usage.provider)

        return reply, usage

    @staticmethod
    def _sleep(seconds: float) -> None:
        """The backoff wait, as a seam so tests do not spend it."""
        time.sleep(seconds)

    @staticmethod
    def _log_request_size(kwargs: dict, attempt: int) -> None:
        """Log the size of the outgoing request at DEBUG before it is sent.

        A context-length error is raised before anything reaches the trace, so this is
        the only record of the request that caused it. `approx_chars` is a character
        count, not tokens. `api_key` is in `kwargs` and is deliberately not logged.
        """
        if not _LOG.isEnabledFor(logging.DEBUG):
            return
        messages = kwargs.get("messages") or []
        try:
            approx_chars = len(json.dumps(messages, default=str))
        except (TypeError, ValueError):
            approx_chars = -1
        _LOG.debug(
            "completion request attempt=%d model=%s messages=%d approx_chars=%d "
            "tools=%d (chars, not tokens: approximate)",
            attempt + 1,
            kwargs.get("model"),
            len(messages),
            approx_chars,
            len(kwargs.get("tools") or []),
        )

    def _completion_with_retry(self, kwargs: dict):
        """`litellm.completion(**kwargs)`, retrying a 429 up to `MAX_RETRIES` times with
        exponential backoff and jitter, then re-raising it.

        `grid.policy.classify` decides what counts as a rate limit, so the transport and
        the grid worker agree. Other errors, including 5xx, are raised straight away and
        handled by the worker.
        """
        from grant_eval.grid.policy import Disposition, classify

        for attempt in range(self.MAX_RETRIES + 1):
            self._log_request_size(kwargs, attempt)
            try:
                return litellm.completion(**kwargs)
            except Exception as exc:
                if attempt == self.MAX_RETRIES:
                    raise
                if classify(exc) is not Disposition.RATE_LIMIT:
                    raise
                # Half fixed, half random: full jitter can sleep almost 0s and retry
                # inside the same burst.
                window = self.RETRY_BASE_S * 2 ** attempt
                self._sleep(window / 2 + random.random() * window / 2)

    @staticmethod
    def _reasoning_of(message) -> str | None:
        """A reasoning model's thinking, wherever the provider put it.

        DeepSeek v4 returns ``content=None`` with its text in ``reasoning_content``, and
        OpenRouter also copies it to ``provider_specific_fields['reasoning']``.
        """
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        extra = getattr(message, "provider_specific_fields", None) or {}
        if isinstance(extra, dict):
            value = extra.get("reasoning")
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _to_reply(response) -> Reply:
        message = response.choices[0].message
        text = getattr(message, "content", None)
        reasoning = OpenRouterBackend._reasoning_of(message)

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for tc in raw_tool_calls:
            fn = tc.function
            raw_args = fn.arguments
            # Models sometimes send truncated or invalid JSON arguments. Keep the raw
            # string instead of raising, so submit validation and the repair turn can see
            # the arguments were malformed rather than empty.
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (TypeError, json.JSONDecodeError) as e:
                arguments = {"_raw": raw_args, "_parse_error": str(e)}
            tool_calls.append(ToolCall(id=tc.id, name=fn.name, arguments=arguments))

        return Reply(text=text, tool_calls=tool_calls, reasoning=reasoning)

    @staticmethod
    def _served_provider(response) -> str | None:
        """The provider that answered, or `None` if the response does not say.

        Usually a top-level field, sometimes under `provider_specific_fields`. `None`
        is treated as a pass by `assert_provider_pin`."""
        top_level = getattr(response, "provider", None)
        if isinstance(top_level, str) and top_level:
            return top_level
        extra = getattr(response, "provider_specific_fields", None) or {}
        if isinstance(extra, dict):
            nested = extra.get("provider")
            if isinstance(nested, str) and nested:
                return nested
        return None

    @staticmethod
    def _to_usage(response, model: str) -> Usage:
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        cost = _safe_cost(response)

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            backend="openrouter",
            model=model,
            provider=OpenRouterBackend._served_provider(response),
        )


def _safe_cost(response) -> float:
    """Best-effort cost in USD.

    LiteLLM's `completion_cost` can raise for models it has no prices for, so fall back
    to the provider-reported cost, then 0.0."""
    try:
        cost = litellm.completion_cost(response)
        if cost is not None:
            return float(cost)
    except Exception:
        pass
    hidden = getattr(response, "_hidden_params", None) or {}
    fallback = hidden.get("response_cost")
    if fallback is not None:
        return float(fallback)
    return 0.0
