"""End-to-end tests for `run_extraction`.

These use the real Toolset against an in-process serve.py on the Leverhulme snapshot.
Only the model is mocked, so the submitted payloads have to be schema-valid.

"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from grant_eval.agent.extract import RunConfig, run_extraction
from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage
from grant_eval.config import AGENT_CONFIGS, RALPH
from grant_eval.dataset.serve import build_lookup, make_handler
from grant_eval.schema import ExtractionResult, Split


SNAP = Path("data/snapshots/leverhulme/2026-08-09")
ENTRY_URL = "https://www.leverhulme.ac.uk/early-career-fellowships"
SCHEMES_URL = "https://www.leverhulme.ac.uk/schemes-at-a-glance"

FUNDER = "Leverhulme Trust"


@pytest.fixture(scope="module")
def base_url():
    """Start serve.py in-process against the real snapshot and yield its base URL."""
    manifest = json.loads((SNAP / "manifest.json").read_text())
    lookup = build_lookup(SNAP, manifest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lookup, None, Counter()))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _opp_dict(oid: str, url: str) -> dict:
    return {
        "funder": FUNDER,
        "opportunity_id": oid,
        "title": f"Title {oid}",
        "url": url,
        "status": "open",
    }


# Two stable opportunities, submitted the same way every iteration so Ralph converges
# at iteration 2.
_TWO_OPPS = [
    _opp_dict("early-career-fellowships", ENTRY_URL),
    _opp_dict("research-project-grants", SCHEMES_URL),
]

# Fixed usage per call so cost sums are exact. Each loop run makes two calls
# (fetch, then submit).
_FETCH_USAGE = Usage(input_tokens=100, output_tokens=20, cost_usd=0.001, backend="mock", model="x")
_SUBMIT_USAGE = Usage(input_tokens=200, output_tokens=40, cost_usd=0.002, backend="mock", model="x")


def _fetch_then_submit_replies(opp_dicts):
    return [
        Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": ENTRY_URL})]),
        Reply(tool_calls=[ToolCall("c2", "submit_opportunities", {"opportunities": opp_dicts})]),
    ]


def _two_opp_factory():
    """Factory returning a fresh MockBackend that fetches then submits the two stable opportunities."""
    def factory():
        return MockBackend(
            replies=_fetch_then_submit_replies(_TWO_OPPS),
            usages=[_FETCH_USAGE, _SUBMIT_USAGE],
        )
    return factory


def _abstain_factory():
    def factory():
        return MockBackend(
            replies=[Reply(tool_calls=[ToolCall("c1", "submit_opportunities", {"opportunities": []})])],
            usages=[_SUBMIT_USAGE],
        )
    return factory


def _run_config(**overrides) -> RunConfig:
    base = dict(
        funder=FUNDER,
        split=Split.TUNE,
        snapshot_date=date(2026, 8, 9),
        model="openrouter/anthropic/claude-haiku-4.5",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        replicate=1,
    )
    base.update(overrides)
    return RunConfig(**base)


def test_happy_path_valid_extraction_result(base_url):
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )

    assert isinstance(result, ExtractionResult)

    # round-trips through JSON
    reloaded = ExtractionResult.model_validate_json(result.model_dump_json())
    assert reloaded == result

    assert len(result.opportunities) == 2
    assert result.schema_valid is True
    assert result.abstained is False
    assert result.submit_failed is False
    # `run_extraction` defaults to `config.RALPH`, which runs one iteration.
    assert result.ralph_iterations == 1
    assert result.scored_iteration_policy == "final"

    # telemetry populated
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.cost_usd is not None
    assert result.latency_s is not None
    assert result.backend  # non-empty


def test_cost_sums_across_iterations(base_url):
    """Cost and usage add up over every iteration, not just the scored one.

    Uses a multi-iteration config, since the default runs only one iteration.

    """
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
        ralph_config=dict(RALPH, max_iterations=5),
    )

    # Converges on the second pass: 2 iterations, each a fetch and a submit call.
    per_run_input = _FETCH_USAGE.input_tokens + _SUBMIT_USAGE.input_tokens
    per_run_output = _FETCH_USAGE.output_tokens + _SUBMIT_USAGE.output_tokens
    per_run_cost = _FETCH_USAGE.cost_usd + _SUBMIT_USAGE.cost_usd

    assert result.ralph_iterations == 2
    assert result.input_tokens == 2 * per_run_input
    assert result.output_tokens == 2 * per_run_output
    assert result.cost_usd == pytest.approx(2 * per_run_cost)


def test_config_hash_determinism(base_url):
    r1 = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    r2 = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    # same knobs -> same hash
    assert r1.config_hash == r2.config_hash

    # changing an experimental knob -> different hash
    r_temp = run_extraction(
        run_config=_run_config(temperature=0.7),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert r_temp.config_hash != r1.config_hash

    # The grid currently has one agent config, so a second one is supplied here. The
    # config label and its knobs should reach the hash however many configs the grid uses.
    r_cfg = run_extraction(
        run_config=_run_config(agent_config="wide"),
        agent_configs={**AGENT_CONFIGS, "wide": {"fetch_depth": 3, "step_budget": 32}},
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert r_cfg.config_hash != r1.config_hash

    # changing per-run identity -> same hash (identity, not config)
    r_rep = run_extraction(
        run_config=_run_config(replicate=4),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert r_rep.config_hash == r1.config_hash

    r_funder = run_extraction(
        run_config=_run_config(funder="Some Other Funder"),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert r_funder.config_hash == r1.config_hash


def test_out_path_writes_reloadable_json(base_url, tmp_path):
    out = tmp_path / "nested" / "dir" / "1.json"
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
        out_path=out,
    )
    assert out.exists()
    reloaded = ExtractionResult.model_validate_json(out.read_text())
    assert reloaded == result


def test_abstention_run(base_url):
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_abstain_factory(),
        entry_url=ENTRY_URL,
    )
    assert isinstance(result, ExtractionResult)
    assert result.abstained is True
    assert result.schema_valid is True
    assert result.opportunities == []
    assert result.ralph_iterations == 1  # abstention converges immediately

    # telemetry still filled in
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.cost_usd is not None
    assert result.latency_s is not None
    assert result.backend


def test_extraction_result_has_new_fields():
    fields = ExtractionResult.model_fields
    assert "schema_valid" in fields
    assert "submit_failed" in fields


# The provider that served the run is recorded on the result.
_FETCH_USAGE_WITH_PROVIDER = Usage(
    input_tokens=100, output_tokens=20, cost_usd=0.001, backend="mock", model="x",
    provider="ProviderA",
)
_SUBMIT_USAGE_WITH_PROVIDER = Usage(
    input_tokens=200, output_tokens=40, cost_usd=0.002, backend="mock", model="x",
    provider="ProviderB",
)


def _two_opp_factory_with_provider():
    def factory():
        return MockBackend(
            replies=_fetch_then_submit_replies(_TWO_OPPS),
            usages=[_FETCH_USAGE_WITH_PROVIDER, _SUBMIT_USAGE_WITH_PROVIDER],
        )
    return factory


def test_provider_is_recorded_on_the_result(base_url):
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory_with_provider(),
        entry_url=ENTRY_URL,
    )
    # The last completion reported ProviderB, so that is what gets recorded, matching how
    # `result.backend` and `result.model` work.
    assert result.provider == "ProviderB"


def test_provider_is_none_when_no_completion_reports_one(base_url):
    """With no provider in any usage, the field reads back as None, not an empty string."""
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert result.provider is None


# `cost_usd` is the spend of one attempt. A rate-limited cell is re-run from scratch
# and the failed attempt raised, so its usage never reached `run_extraction`.
# `attempts` makes that visible.
def test_a_re_run_records_the_attempt_count_so_the_cost_is_not_read_as_the_total(
    base_url, tmp_path
):
    """A second run into the same trace dir records `attempts=2` and only its own cost.

    Without the count, a reader (and `--max-spend-usd`) would take that cost as the
    cell's total bill.

    """
    def once():
        return run_extraction(
            run_config=_run_config(),
            base_url=base_url,
            backend_factory=_two_opp_factory(),
            entry_url=ENTRY_URL,
            trace_dir=str(tmp_path),
        )

    first, second = once(), once()

    assert first.attempts == 1
    assert second.attempts == 2
    assert second.cost_usd == pytest.approx(first.cost_usd), (
        "the second record must carry its OWN spend, not an accumulated one"
    )


def test_an_untraced_run_still_reports_one_attempt(base_url):
    """`attempts` comes from the trace, and a run without one counts as a first attempt."""
    result = run_extraction(
        run_config=_run_config(),
        base_url=base_url,
        backend_factory=_two_opp_factory(),
        entry_url=ENTRY_URL,
    )
    assert result.attempts == 1


# config_hash covers the loop parameters, so runs that differ in them never share
# an output path.
def _hash_args(**overrides):
    args = dict(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        agent_cfg={"fetch_depth": 1, "step_budget": 8},
        ralph_config={"max_iterations": 1, "termination": "t",
                      "scored_iteration_policy": "final"},
        loop_config={"budget_notice_at": 3, "carry_reasoning": True},
    )
    args.update(overrides)
    return args


def test_config_hash_changes_with_budget_notice_at():
    """budget_notice_at changes the hash, since it can decide whether a cell submits anything."""
    from grant_eval.agent.extract import config_hash

    a = config_hash(**_hash_args())
    b = config_hash(**_hash_args(
        loop_config={"budget_notice_at": 2, "carry_reasoning": True}))
    assert a != b


def test_config_hash_changes_with_carry_reasoning():
    from grant_eval.agent.extract import config_hash

    a = config_hash(**_hash_args())
    b = config_hash(**_hash_args(
        loop_config={"budget_notice_at": 3, "carry_reasoning": False}))
    assert a != b


def test_config_hash_is_order_independent():
    from grant_eval.agent.extract import config_hash

    a = config_hash(**_hash_args())
    b = config_hash(**_hash_args(
        loop_config={"carry_reasoning": True, "budget_notice_at": 3}))
    assert a == b
