"""End-to-end test of ``scoring/skeleton.py`` against the real Leverhulme snapshot, offline.

Only the model is mocked; the server, tools, submit validation and Ralph loop are real.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pytest

from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage
from grant_eval.schema import ExtractionResult
from grant_eval.scoring.skeleton import run_walking_skeleton, serve_snapshot


SNAP = Path("data/snapshots/leverhulme/2026-08-09")
FUNDER = "The Leverhulme Trust"
ENTRY_URL = "https://www.leverhulme.ac.uk/"
ECF_URL = "https://www.leverhulme.ac.uk/early-career-fellowships"
RPG_URL = "https://www.leverhulme.ac.uk/research-project-grants"

# One Ralph iteration keeps the test fast.
_ONE_ITER = {
    "max_iterations": 1,
    "termination": "schema_valid_and_no_new_opportunities",
    "scored_iteration_policy": "final",
}


def _opp(oid: str, url: str) -> dict:
    return {
        "funder": FUNDER,
        "opportunity_id": oid,
        "title": f"Title {oid}",
        "url": url,
        "status": "open",
    }


def _mock_factory():
    """Return a factory for a MockBackend that fetches the entry page then submits two opportunities."""
    replies = [
        Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": ENTRY_URL})]),
        Reply(
            tool_calls=[
                ToolCall(
                    "c2",
                    "submit_opportunities",
                    {"opportunities": [_opp("early-career-fellowships", ECF_URL),
                                       _opp("research-project-grants", RPG_URL)]},
                )
            ]
        ),
    ]
    usages = [
        Usage(100, 20, 0.001, "mock", "x"),
        Usage(200, 40, 0.002, "mock", "x"),
    ]
    return lambda: MockBackend(replies=list(replies), usages=list(usages))


def test_serve_snapshot_lifecycle_serves_then_closes():
    with serve_snapshot(SNAP) as base_url:
        assert base_url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(f"{base_url}/", timeout=15) as resp:
            assert resp.status == 200
            body = resp.read()
        assert b"Leverhulme" in body

    # the server should be closed once the context exits
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"{base_url}/", timeout=2)


def test_run_walking_skeleton_wires_end_to_end(tmp_path):
    out_path = tmp_path / "result.json"
    trace_dir = tmp_path / "traces"

    result = run_walking_skeleton(
        snapshot_dir=SNAP,
        funder=FUNDER,
        snapshot_date=date(2026, 8, 9),
        entry_url=ENTRY_URL,
        model="openrouter/anthropic/claude-haiku-4.5",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        replicate=1,
        ralph_config=_ONE_ITER,
        out_path=str(out_path),
        trace_dir=str(trace_dir),
        backend_factory=_mock_factory(),
    )

    assert isinstance(result, ExtractionResult)
    assert len(result.opportunities) == 2
    assert {o.opportunity_id for o in result.opportunities} == {
        "early-career-fellowships",
        "research-project-grants",
    }
    assert result.schema_valid is True
    assert result.abstained is False
    assert result.submit_failed is False
    assert result.step_budget_exhausted is False
    assert result.ralph_iterations == 1

    # summed over the two calls: 100/20 + 200/40
    assert result.input_tokens == 300
    assert result.output_tokens == 60
    assert result.cost_usd == 0.003
    assert result.latency_s is not None and result.latency_s >= 0
    assert result.backend == "mock"
    assert result.config_hash

    assert out_path.exists()
    reloaded = ExtractionResult.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert reloaded == result

    trace_file = trace_dir / "iter1.jsonl"
    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3  # at least: prompt, completion(s), result
    events = {json.loads(ln)["event"] for ln in lines}
    assert "prompt" in events
    assert "result" in events


def test_run_walking_skeleton_abstention(tmp_path):
    """An empty submit gives a valid ExtractionResult marked as abstained."""
    out_path = tmp_path / "abstain.json"
    trace_dir = tmp_path / "traces"

    def factory():
        return MockBackend(
            replies=[Reply(tool_calls=[ToolCall("c1", "submit_opportunities", {"opportunities": []})])],
            usages=[Usage(50, 10, 0.0005, "mock", "x")],
        )

    result = run_walking_skeleton(
        snapshot_dir=SNAP,
        funder=FUNDER,
        snapshot_date=date(2026, 8, 9),
        entry_url=ENTRY_URL,
        model="openrouter/anthropic/claude-haiku-4.5",
        ralph_config=_ONE_ITER,
        out_path=str(out_path),
        trace_dir=str(trace_dir),
        backend_factory=factory,
    )

    assert isinstance(result, ExtractionResult)
    assert result.opportunities == []
    assert result.abstained is True
    assert result.schema_valid is True
    assert out_path.exists()
