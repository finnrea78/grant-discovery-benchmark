"""Tests for the Toolset, run against an in-process serve.py on the real Leverhulme
snapshot. serve.py is reused rather than copied, since the Toolset should only see a
base URL and 200/404 responses, not how the mirror is laid out.

"""
from __future__ import annotations

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from grant_eval import config
from grant_eval.dataset.serve import build_lookup, make_handler
from grant_eval.schema import ExtractedOpportunity, Status
from grant_eval.agent.router import ToolCall
from grant_eval.agent.tools import Toolset, ToolResult, TRUNCATION_MARKER, CONTENT_CAP


SNAP = Path("data/snapshots/leverhulme/2026-08-09")

ECF_URL = "https://www.leverhulme.ac.uk/early-career-fellowships"
SCHEMES_URL = "https://www.leverhulme.ac.uk/schemes-at-a-glance"
OFF_MIRROR_URL = "https://www.leverhulme.ac.uk/this-page-does-not-exist"
PDF_URL = "https://www.leverhulme.ac.uk/sites/default/files/2017AR.pdf"


@pytest.fixture(scope="module")
def base_url():
    """Start serve.py in-process against the real snapshot and yield its base URL."""
    manifest = json.loads((SNAP / "manifest.json").read_text())
    lookup = build_lookup(SNAP, manifest)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(lookup, None, Counter())
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def toolset(base_url):
    return Toolset(base_url, {"fetch_depth": 1, "step_budget": 8})


def test_fetch_page_preserves_table_deadline(toolset):
    """The early-career-fellowships deadline sits in a table cell and survives cleaning."""
    text = toolset.fetch_page(ECF_URL)
    assert "MISS" not in text
    assert "18 February 2027" in text


def test_fetch_page_schemes_table_survives(toolset):
    """The schemes-at-a-glance page is a big table; its content survives cleaning."""
    text = toolset.fetch_page(SCHEMES_URL)
    assert text.strip()
    # table cells come from the per-scheme rows, e.g. their Duration / Value columns
    assert "Duration" in text
    assert "Value" in text
    assert "£" in text


def test_fetch_page_off_mirror_returns_miss(toolset):
    """Off-mirror pages 404 and come back as a MISS string, not an exception."""
    text = toolset.fetch_page(OFF_MIRROR_URL)
    assert "MISS" in text
    assert "not in the mirror" in text
    assert OFF_MIRROR_URL in text


# Transport errors and non-404 statuses must not be reported as an off-mirror MISS.
def _free_closed_port() -> int:
    """Bind an ephemeral port then close it, returning a port nothing is listening on."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_fetch_page_transport_error_is_not_miss():
    """A connection failure gives a FETCH ERROR observation, not a MISS, and does not raise."""
    port = _free_closed_port()
    ts = Toolset(f"http://127.0.0.1:{port}", {"fetch_depth": 1, "step_budget": 8})
    text = ts.fetch_page("https://www.leverhulme.ac.uk/whatever")
    assert "ERROR" in text
    assert "unreachable" in text
    assert "MISS" not in text


def test_fetch_page_non_404_status_is_not_miss():
    """A non-404 status (500 or 403) is a transport error, not an off-mirror MISS."""

    class _Always500(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")

        def log_message(self, *args):  # silence
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Always500)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ts = Toolset(f"http://127.0.0.1:{port}", {"fetch_depth": 1, "step_budget": 8})
        text = ts.fetch_page("https://www.leverhulme.ac.uk/whatever")
        assert "ERROR" in text
        assert "500" in text
        assert "MISS" not in text
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_page_raw_html_fallback_is_tagged():
    """When trafilatura can't extract the main content, the raw-HTML fallback is tagged in the output."""

    # trafilatura finds no main content in this HTML, so fetch_page falls back to raw
    # HTML and tags it.
    tiny_html = b"<html><head><title>t</title></head><body></body></html>"

    class _RawHtml(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(tiny_html)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RawHtml)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ts = Toolset(f"http://127.0.0.1:{port}", {"fetch_depth": 1, "step_budget": 8})
        text = ts.fetch_page("https://www.leverhulme.ac.uk/whatever")
        assert text.startswith("[raw-html fallback]")
    finally:
        server.shutdown()
        server.server_close()


def test_list_links_returns_onsite_links(toolset):
    text = toolset.list_links(ECF_URL)
    assert "MISS" not in text
    # at least a handful of on-site links with anchor text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) > 0
    # links are same-host absolute URLs
    assert "leverhulme.ac.uk" in text


def test_list_links_off_mirror_returns_miss(toolset):
    text = toolset.list_links(OFF_MIRROR_URL)
    assert "MISS" in text


def test_read_pdf_returns_text(toolset):
    text = toolset.read_pdf(PDF_URL)
    assert "MISS" not in text
    assert text.strip()
    assert "Leverhulme" in text


def test_read_pdf_off_mirror_returns_miss(toolset):
    text = toolset.read_pdf("https://www.leverhulme.ac.uk/sites/default/files/nope.pdf")
    assert "MISS" in text


def test_truncation_cap_helper(toolset):
    long = "x" * (CONTENT_CAP + 5000)
    capped = toolset._cap(long)
    assert len(capped) <= CONTENT_CAP + len(TRUNCATION_MARKER) + 2
    assert capped.endswith(TRUNCATION_MARKER)
    # head is kept (the tail is truncated)
    assert capped.startswith("x")


def test_truncation_cap_noop_when_short(toolset):
    short = "hello world"
    assert toolset._cap(short) == short


# Worst case seen in practice: 480,000 chars of tool output came to about 334k-338k
# tokens. Link-heavy output tokenises at well under the usual ~4 chars/token.
WORST_CASE_CHARS_PER_TOKEN = 1.42
# nova-lite-v1 has the 300k window that overflowed. gpt-oss-120b's window is smaller
# (131,072) but an overflow there is now scored as a result rather than halting the run.
NOVA_LITE_CONTEXT_TOKENS = 300_000


def test_content_cap_is_pinned_at_twelve_thousand_and_cap_applies_it(toolset):
    """CONTENT_CAP is 12,000 and step_budget * CONTENT_CAP still fits nova-lite's window.

    The cap helper test passes at any value, so the number is checked here together
    with the step budget. Raising `step_budget` without redoing this sum fails here.

    """
    assert CONTENT_CAP == 12_000

    worst_case_chars = config.AGENT_CONFIGS["standard"]["step_budget"] * CONTENT_CAP
    assert worst_case_chars == 288_000
    assert worst_case_chars / WORST_CASE_CHARS_PER_TOKEN < NOVA_LITE_CONTEXT_TOKENS

    capped = toolset._cap("x" * (CONTENT_CAP * 4))
    assert capped == "x" * 12_000 + "\n" + TRUNCATION_MARKER


def _valid_opp() -> dict:
    return {
        "funder": "Leverhulme Trust",
        "opportunity_id": "early-career-fellowships",
        "title": "Early Career Fellowships",
        "url": ECF_URL,
        "status": "open",
    }


def test_submit_valid_one_opportunity(toolset):
    result = toolset.submit_opportunities([_valid_opp()])
    assert isinstance(result, ToolResult)
    assert result.terminate is True
    assert result.abstained is False
    assert result.submission is not None
    assert len(result.submission) == 1
    assert isinstance(result.submission[0], ExtractedOpportunity)


def test_submit_empty_is_abstention(toolset):
    result = toolset.submit_opportunities([])
    assert result.terminate is True
    assert result.abstained is True
    assert result.submission == []


def test_submit_invalid_not_terminal(toolset):
    # missing required 'title' and 'url'
    bad = {"funder": "Leverhulme Trust", "opportunity_id": "x"}
    result = toolset.submit_opportunities([bad])
    assert result.terminate is False
    assert result.submission is None
    # the pydantic validation error is surfaced as observation text
    assert "valid" in result.content.lower() or "field required" in result.content.lower()


def test_tool_schemas_shape(toolset):
    schemas = toolset.tool_schemas()
    assert len(schemas) == 4
    names = {s["function"]["name"] for s in schemas}
    assert names == {"fetch_page", "list_links", "read_pdf", "submit_opportunities"}
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str)
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_submit_schema_items_are_not_bare_object(toolset):
    """The submit `opportunities` array uses the real ExtractedOpportunity schema.

    A bare {"type":"object"} would leave the model to learn the fields from validation errors.

    """
    schemas = toolset.tool_schemas()
    submit = next(s for s in schemas if s["function"]["name"] == "submit_opportunities")
    params = submit["function"]["parameters"]
    items = params["properties"]["opportunities"]["items"]

    # Not a bare object schema.
    assert items != {"type": "object"}

    props = items.get("properties", {})
    for key in ("status", "url", "deadlines"):
        assert key in props, f"missing {key} in submit item properties"

    blob = json.dumps(params)
    assert '"$ref"' not in blob
    assert '"$defs"' not in blob
    assert '"anyOf"' not in blob

    for key in ("deadlines", "next_open_date"):
        node = props[key]
        node = node.get("items", node)
        assert node.get("type") == "object", f"{key} must inline the PartialDate object"
        assert node.get("required") == ["raw"], f"{key} must keep PartialDate's required raw"
        assert set(node["properties"]) == {"raw", "resolved", "precision"}

    assert props["status"]["enum"] == [s.value for s in Status]


def test_dispatch_fetch_page(toolset):
    call = ToolCall(id="c1", name="fetch_page", arguments={"url": ECF_URL})
    result = toolset.dispatch(call)
    assert isinstance(result, ToolResult)
    assert "18 February 2027" in result.content
    assert result.terminate is False


def test_dispatch_submit_terminates(toolset):
    call = ToolCall(
        id="c2", name="submit_opportunities", arguments={"opportunities": []}
    )
    result = toolset.dispatch(call)
    assert result.terminate is True
    assert result.abstained is True


def test_dispatch_unknown_tool_is_error_not_crash(toolset):
    call = ToolCall(id="c3", name="not_a_tool", arguments={})
    result = toolset.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.terminate is False
    assert "unknown tool" in result.content.lower()


def test_dispatch_missing_arg_is_error_not_crash(toolset):
    call = ToolCall(id="c4", name="fetch_page", arguments={})
    result = toolset.dispatch(call)
    assert isinstance(result, ToolResult)
    assert result.terminate is False
    assert "error" in result.content.lower() or "url" in result.content.lower()
