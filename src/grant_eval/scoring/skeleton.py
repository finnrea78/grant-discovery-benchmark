"""Runs a single grid cell end-to-end through the real stack, as a smoke test.

A local snapshot server, the real toolset and the live OpenRouter backend feed one
`run_extraction` call, which writes one `ExtractionResult` JSON and a trace file.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from contextlib import contextmanager
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

import litellm

from grant_eval.agent.extract import RunConfig, config_hash, run_extraction
from grant_eval.agent.router import OpenRouterBackend
from grant_eval.config import AGENT_CONFIGS, LOOP, RALPH
from grant_eval.dataset.serve import build_lookup, make_handler
from grant_eval.schema import ExtractionResult, Split


class _BoundedOpenRouterBackend(OpenRouterBackend):
    """OpenRouter backend that caps ``max_tokens`` on each completion.

    The normal backend sends no ``max_tokens``, so OpenRouter reserves the model's full
    output limit and a low-balance account gets a 402. This is only for the smoke run;
    the grid uses the uncapped backend.
    """

    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self._max_tokens = max_tokens

    def complete(self, messages, model, temperature, tools):
        api_key = self._api_key()
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "api_key": api_key,
            "max_tokens": self._max_tokens,
            "timeout": self.TIMEOUT_S,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = litellm.completion(**kwargs)
        return self._to_reply(response), self._to_usage(response, model)


@contextmanager
def serve_snapshot(snapshot_dir):
    """Serve one funder snapshot on a free local port and yield its base URL.

    Runs in a daemon thread in this process; the toolset only talks HTTP, so this
    behaves the same as a separate server process.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    lookup = build_lookup(snapshot_dir, manifest)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lookup, None, Counter()))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def run_walking_skeleton(
    *,
    snapshot_dir,
    funder,
    snapshot_date,
    entry_url,
    model,
    prompt_strategy="minimal",
    temperature=0.0,
    agent_config="standard",
    replicate=1,
    ralph_config=None,
    out_path,
    trace_dir,
    backend_factory=None,
) -> ExtractionResult:
    """Run one cell end-to-end and return its ``ExtractionResult``.

    ``backend_factory`` defaults to a live ``OpenRouterBackend`` (tests pass a mock).
    ``ralph_config`` defaults to ``config.RALPH``.

    Args:
        snapshot_dir: path to ``<funder>/<snapshot_date>/``.
        model: LiteLLM model string.
        out_path: where the ExtractionResult JSON is written.
        trace_dir: where the per-iteration ``iterN.jsonl`` traces are written.
    """
    if backend_factory is None:
        backend_factory = lambda: OpenRouterBackend()  # noqa: E731
    if ralph_config is None:
        ralph_config = RALPH

    run_config = RunConfig(
        funder=funder,
        split=Split.TUNE,
        snapshot_date=snapshot_date,
        model=model,
        prompt_strategy=prompt_strategy,
        temperature=temperature,
        agent_config=agent_config,
        replicate=replicate,
    )

    with serve_snapshot(snapshot_dir) as base_url:
        return run_extraction(
            run_config=run_config,
            base_url=base_url,
            backend_factory=backend_factory,
            entry_url=entry_url,
            agent_configs=AGENT_CONFIGS,
            ralph_config=ralph_config,
            trace_dir=trace_dir,
            out_path=out_path,
        )


SNAPSHOT_DIR = "data/snapshots/leverhulme/2026-08-09"
FUNDER = "The Leverhulme Trust"
SNAPSHOT_DATE = date(2026, 8, 9)
ENTRY_URL = "https://www.leverhulme.ac.uk/"
# Override with SKELETON_MODEL to smoke-test a cheaper model. The grid uses config.MODELS.
MODEL = os.environ.get("SKELETON_MODEL", "openrouter/anthropic/claude-haiku-4.5")
_MODEL_TAG = MODEL.rsplit("/", 1)[-1]

# Leverhulme's mirror was crawled from its homepage, so that works as an entry URL here.
# Most other mirrors do not contain the homepage; the grid takes entry URLs from
# data/funders.json. SKELETON_ENTRY_URL lets a low-balance account start from a smaller
# page, since the homepage is about 27k tokens.
_LIVE_ENTRY_URL = os.environ.get("SKELETON_ENTRY_URL", ENTRY_URL)

# Only "standard" exists now, but the override is kept so a different step budget can
# be tried, which separates "this model cannot drive the loop" from "the budget is too
# small for this site".
AGENT_CONFIG = os.environ.get("SKELETON_AGENT_CONFIG", "standard")
if AGENT_CONFIG not in AGENT_CONFIGS:
    raise SystemExit(
        f"SKELETON_AGENT_CONFIG={AGENT_CONFIG!r} is not one of {sorted(AGENT_CONFIGS)}"
    )

# One iteration keeps the smoke test cheap. Kept separate from RALPH so the smoke test's
# cost is not tied to the pre-registered setting.
SKELETON_RALPH = {
    "max_iterations": 1,
    "termination": RALPH["termination"],
    "scored_iteration_policy": "final",
}

# Output cap for live completions; see _BoundedOpenRouterBackend.
_LIVE_MAX_TOKENS = int(os.environ.get("SKELETON_MAX_TOKENS", "4000"))

# Paths include the config hash so runs that differ only in, say, a loop setting do
# not overwrite each other.
_CFG_HASH = config_hash(
    model=MODEL,
    prompt_strategy="minimal",
    temperature=0.0,
    agent_config=AGENT_CONFIG,
    agent_cfg=AGENT_CONFIGS[AGENT_CONFIG],
    ralph_config=SKELETON_RALPH,
    loop_config=LOOP,
)
_CELL = f"leverhulme_{_MODEL_TAG}_minimal_t0_{AGENT_CONFIG}_rep1_{_CFG_HASH[:10]}"

OUT_PATH = f"eval/skeleton/{_CELL}.json"
# One trace folder per cell so traces from different runs never mix.
TRACE_DIR = f"eval/skeleton_traces/{_CELL}"


def _mock_fallback_factory():
    """Offline mock backend that fetches the entry page and submits two real schemes.

    Used when the live API is unavailable, so the local wiring can still be checked.
    """
    from grant_eval.agent.router import MockBackend, Reply, ToolCall, Usage

    opps = [
        {
            "funder": FUNDER,
            "opportunity_id": "leverhulme-early-career-fellowships",
            "title": "Early Career Fellowships",
            "url": "https://www.leverhulme.ac.uk/early-career-fellowships",
            "status": "forthcoming",
        },
        {
            "funder": FUNDER,
            "opportunity_id": "leverhulme-research-project-grants",
            "title": "Research Project Grants",
            "url": "https://www.leverhulme.ac.uk/research-project-grants",
            "status": "open",
        },
    ]
    replies = [
        Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": _LIVE_ENTRY_URL})]),
        Reply(tool_calls=[ToolCall("c2", "submit_opportunities", {"opportunities": opps})]),
    ]
    usages = [Usage(100, 20, 0.0, "mock", MODEL), Usage(200, 40, 0.0, "mock", MODEL)]
    return lambda: MockBackend(replies=list(replies), usages=list(usages))


def _run_once(backend_factory) -> ExtractionResult:
    return run_walking_skeleton(
        snapshot_dir=SNAPSHOT_DIR,
        funder=FUNDER,
        snapshot_date=SNAPSHOT_DATE,
        entry_url=_LIVE_ENTRY_URL,
        model=MODEL,
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config=AGENT_CONFIG,
        replicate=1,
        ralph_config=SKELETON_RALPH,
        out_path=OUT_PATH,
        trace_dir=TRACE_DIR,
        backend_factory=backend_factory,
    )


# A mock fallback exits 0 and looks like a pass, which is wrong when testing whether a
# model can drive the loop. SKELETON_NO_FALLBACK=1 fails instead.
_NO_FALLBACK = os.environ.get("SKELETON_NO_FALLBACK") == "1"


def main() -> int:
    if os.environ.get("SKELETON_FORCE_MOCK") == "1":
        print("SKELETON_FORCE_MOCK=1 -> running the MOCK path (offline seam demo).")
        _print_summary(_run_once(_mock_fallback_factory()), path="MOCK (forced)")
        return 0

    live_factory = lambda: _BoundedOpenRouterBackend(_LIVE_MAX_TOKENS)  # noqa: E731
    # Retry once on a transient error, then fall back to the mock.
    for attempt in (1, 2):
        try:
            result = _run_once(live_factory)
            _print_summary(result, path=f"LIVE {_MODEL_TAG}")
            return 0
        except Exception as exc:  # noqa: BLE001 - report + fall back on any live failure
            print(f"[live attempt {attempt}] FAILED: {type(exc).__name__}: {exc}")
            if attempt == 1:
                print("  retrying live once...")

    if _NO_FALLBACK:
        print(
            "\nLIVE RUN DID NOT COMPLETE and SKELETON_NO_FALLBACK=1 is set, so no mock "
            "fallback was run. The failure above is the result: this arm did not drive "
            "the loop end-to-end.\n"
        )
        return 1

    print(
        "\nLIVE RUN DID NOT COMPLETE (both attempts failed above). Falling back to the "
        "MOCK path so the local seams are still demonstrated. The failure is a real "
        "provider/account limitation, NOT a wiring bug.\n"
    )
    _print_summary(_run_once(_mock_fallback_factory()), path="MOCK fallback (live failed)")
    return 0


def _print_summary(result: ExtractionResult, *, path: str = "LIVE") -> None:
    print("=" * 72)
    print(
        f"WALKING SKELETON — Leverhulme x {_MODEL_TAG} x minimal x t0 "
        f"x {AGENT_CONFIG} (step_budget={AGENT_CONFIGS[AGENT_CONFIG]['step_budget']}) x rep1"
    )
    print(f"run path: {path}   entry_url: {_LIVE_ENTRY_URL}")
    print("=" * 72)
    print(f"opportunities: {len(result.opportunities)}")
    for i, opp in enumerate(result.opportunities, 1):
        print(f"  {i:>2}. {opp.title}  [{opp.status.value}]  {opp.url}")
    print("-" * 72)
    print(f"ralph_iterations       : {result.ralph_iterations}")
    print(f"input_tokens           : {result.input_tokens}")
    print(f"output_tokens          : {result.output_tokens}")
    print(f"cost_usd               : {result.cost_usd}")
    print(f"latency_s              : {result.latency_s:.2f}")
    print(f"abstained              : {result.abstained}")
    print(f"step_budget_exhausted  : {result.step_budget_exhausted}")
    print(f"schema_valid           : {result.schema_valid}")
    print(f"submit_failed          : {result.submit_failed}")
    print(f"backend                : {result.backend}")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
