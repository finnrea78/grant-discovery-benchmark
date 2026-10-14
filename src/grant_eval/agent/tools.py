"""The agent's tools, tied to one frozen mirror of a funder's website.

Every tool is an HTTP client for a single ``base_url``: a running
``grant_eval.dataset.serve`` instance for one funder snapshot. There is no code path to
the live web. URL resolution lives in ``serve.py``; this module only sends the path and
query, and the server answers 200 (mirrored) or 404 (not mirrored, a MISS). It uses
stdlib ``urllib`` so it needs no extra HTTP library.

The four tools:

* ``fetch_page``: cleaned main-content text with tables kept, capped at ``CONTENT_CAP``.
* ``list_links``: on-site links with their anchor text, deduplicated.
* ``read_pdf``: PDF text via pymupdf, falling back to pdfplumber, with the same cap.
* ``submit_opportunities``: validate and end the run; ``[]`` is a deliberate abstention.

A fetch outside the mirror returns a MISS message to the model rather than raising. An
invalid submit returns the validation error without ending the run, so the loop can
give the agent its one repair turn.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

import trafilatura
from pydantic import ValidationError

from grant_eval.schema import ExtractedOpportunity


# The longest possible transcript is roughly `step_budget` x CONTENT_CAP. At 20,000
# chars and 24 steps, nova-lite (300,000-token context) overflowed four times, because
# link lists are mostly URLs and tokenise at about 1.4 chars per token. 24 x 12,000 =
# 288,000 chars, about 203,000 tokens at that rate, which fits. `_cap` keeps the start
# of a page, where deadlines and the main details usually are.
CONTENT_CAP = 12_000
TRUNCATION_MARKER = "[truncated]"
_HTTP_TIMEOUT = 15  # seconds; short so a hung server fails fast


@dataclass
class ToolResult:
    """The outcome of one tool call.

    * ``content``: the text the model sees as the tool result. Always set.
    * ``terminate``: True only for a valid ``submit_opportunities`` call.
    * ``submission``: the parsed opportunities on a valid submit (maybe empty), else
      ``None``.
    * ``abstained``: True when a valid submit had an empty list.
    """

    content: str
    terminate: bool = False
    submission: list[ExtractedOpportunity] | None = None
    abstained: bool = False


class _LinkParser(HTMLParser):
    """Collect (href, anchor-text) pairs from ``<a>`` tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href = href
                self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            text = " ".join("".join(self._text).split())
            self.links.append((self._href, text))
            self._href = None
            self._text = []


class Toolset:
    """The agent's tools, tied to one snapshot through ``base_url``.

    The server does not need to be running when this is created; only tool calls make
    HTTP requests. ``agent_config`` is stored for the loop to read.
    """

    def __init__(self, base_url: str, agent_config: dict) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_config = dict(agent_config)

    def _server_url(self, url: str) -> str:
        """Map a requested URL to the same path and query on the mirror server.

        The host is dropped. A path from another site simply 404s, so no separate host
        check is needed.
        """
        parsed = urlparse(url)
        return f"{self.base_url}{urlunparse(('', '', parsed.path or '/', parsed.params, parsed.query, ''))}"

    def _get(self, url: str) -> tuple[int, bytes, str | None]:
        """GET a path and query from the mirror. Returns ``(status, body, error)``.

        An HTTP response gives ``(status, body, None)``; non-200 codes come back with an
        empty body. A connection failure or timeout gives ``(0, b'', reason)``, which
        the caller reports as a FETCH ERROR rather than a MISS.
        """
        server_url = self._server_url(url)
        try:
            with urllib.request.urlopen(server_url, timeout=_HTTP_TIMEOUT) as resp:
                return resp.status, resp.read(), None
        except urllib.error.HTTPError as e:
            # The server answered with an error code; keep it so 404 can be told apart.
            return e.code, b"", None
        except urllib.error.URLError as e:
            # No response at all: refused, unreachable or timed out.
            return 0, b"", str(getattr(e, "reason", e))

    def _fetch(self, url: str) -> tuple[str | None, bytes]:
        """Fetch a URL and return ``(observation, body)``.

        On 200 this is ``(None, body)``. Otherwise ``observation`` is a MISS message for
        a 404 or a FETCH ERROR message for anything else, worded differently so traces
        never count a connection failure as a MISS.
        """
        status, body, error = self._get(url)
        if status == 200:
            return None, body
        if error is not None:
            return f"FETCH ERROR: {url} — unreachable: {error}", b""
        if status == 404:
            return self._miss(url), b""
        return f"FETCH ERROR: {url} — HTTP {status}", b""

    @staticmethod
    def _miss(url: str) -> str:
        return f"MISS: {url} is not in the mirror"

    @staticmethod
    def _cap(text: str) -> str:
        """Cut to CONTENT_CAP chars, keeping the start and adding a marker if cut."""
        if len(text) <= CONTENT_CAP:
            return text
        return text[:CONTENT_CAP].rstrip() + "\n" + TRUNCATION_MARKER

    def fetch_page(self, url: str) -> str:
        """Fetch a page and return its cleaned main text, keeping tables.

        Tables are kept because deadlines are often listed in them.
        """
        observation, body = self._fetch(url)
        if observation is not None:
            return observation
        html = body.decode("utf-8", "replace")
        cleaned = trafilatura.extract(
            html, include_tables=True, favor_recall=True
        )
        if not cleaned:
            # Extraction can return None on sparse pages, so fall back to the raw HTML,
            # tagged so it can be spotted in traces.
            return self._cap("[raw-html fallback]\n" + html)
        return self._cap(cleaned)

    def list_links(self, url: str) -> str:
        """List the page's same-host links as absolute URLs with their anchor text.

        Returns every on-site link on the page; how far the agent goes is limited
        by the loop's step budget, not here.
        """
        observation, body = self._fetch(url)
        if observation is not None:
            return observation
        html = body.decode("utf-8", "replace")

        host = urlparse(url).netloc
        parser = _LinkParser()
        parser.feed(html)

        seen: set[str] = set()
        lines: list[str] = []
        for href, text in parser.links:
            absolute = urljoin(url, href)
            p = urlparse(absolute)
            if p.scheme not in ("http", "https"):
                continue
            if p.netloc != host:  # on-site only
                continue
            # drop fragments before deduplicating
            key = urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))
            if key in seen:
                continue
            seen.add(key)
            label = text or "(no anchor text)"
            lines.append(f"{key}  —  {label}")

        if not lines:
            return f"(no on-site links found on {url})"
        return self._cap("\n".join(lines))

    def read_pdf(self, url: str) -> str:
        """Fetch a PDF and extract its text with pymupdf, or pdfplumber if that fails."""
        observation, body = self._fetch(url)
        if observation is not None:
            return observation

        text = _extract_pdf_pymupdf(body)
        if not text.strip():
            text = _extract_pdf_pdfplumber(body)
        if not text.strip():
            return f"(no extractable text in PDF at {url})"
        return self._cap(text)

    def submit_opportunities(self, opportunities: list) -> ToolResult:
        """Validate ``opportunities`` against ``ExtractedOpportunity`` and end the run.

        An empty list is a valid abstention. An invalid payload returns the validation
        error without ending the run, so the loop can offer a repair turn.
        """
        if not isinstance(opportunities, list):
            return ToolResult(
                content=(
                    "submit_opportunities: `opportunities` must be a list "
                    f"(got {type(opportunities).__name__}). Nothing was submitted."
                ),
                terminate=False,
            )

        if len(opportunities) == 0:
            return ToolResult(
                content="Submitted 0 opportunities (deliberate abstention). Run complete.",
                terminate=True,
                submission=[],
                abstained=True,
            )

        parsed: list[ExtractedOpportunity] = []
        errors: list[str] = []
        for i, item in enumerate(opportunities):
            try:
                parsed.append(ExtractedOpportunity.model_validate(item))
            except ValidationError as e:
                errors.append(f"opportunity[{i}]: {e}")

        if errors:
            body = "\n\n".join(errors)
            return ToolResult(
                content=(
                    "submit_opportunities: payload is not valid — fix and resubmit.\n"
                    f"{body}"
                ),
                terminate=False,
            )

        return ToolResult(
            content=f"Submitted {len(parsed)} opportunity(ies). Run complete.",
            terminate=True,
            submission=parsed,
            abstained=False,
        )

    def tool_schemas(self) -> list[dict]:
        """OpenAI/LiteLLM function schemas for the four tools."""
        # Built from the Pydantic model so the tool schema always matches validation.
        # The `$ref`s are inlined (see `_inline_refs`).
        opp_schema = ExtractedOpportunity.model_json_schema()
        opp_defs = opp_schema.pop("$defs", None) or {}
        opp_item = _inline_refs(opp_schema, opp_defs)

        return [
            _schema(
                "fetch_page",
                "Fetch a page from the mirror and return its cleaned main-content "
                "text (tables preserved). Off-mirror URLs return a MISS observation.",
                {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL of the page to fetch.",
                    }
                },
                ["url"],
            ),
            _schema(
                "list_links",
                "List the on-site (same-host) hyperlinks on a page, as absolute URLs "
                "with their anchor text. Use this to discover pages to navigate to.",
                {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL of the page whose links to list.",
                    }
                },
                ["url"],
            ),
            _schema(
                "read_pdf",
                "Fetch a PDF from the mirror and return its extracted text. Some "
                "opportunities live only inside a linked PDF.",
                {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL of the PDF to read.",
                    }
                },
                ["url"],
            ),
            _schema(
                "submit_opportunities",
                "Submit the final list of funding opportunities you found and END the "
                "run. Submit an empty list to deliberately abstain (you are confident "
                "there are no in-scope opportunities).",
                {
                    "opportunities": {
                        "type": "array",
                        "description": (
                            "The opportunities you extracted. Each item must match the "
                            "opportunity schema (funder, opportunity_id, title, url, "
                            "status, deadlines, ...). Empty list = abstention."
                        ),
                        "items": opp_item,
                    }
                },
                ["opportunities"],
            ),
        ]

    def dispatch(self, call) -> ToolResult:
        """Run a ``router.ToolCall`` and return a ``ToolResult``. Unknown tools and bad
        arguments return an error message instead of raising."""
        name = call.name
        args = call.arguments if isinstance(call.arguments, dict) else {}

        if name == "submit_opportunities":
            return self.submit_opportunities(args.get("opportunities", []))

        if name in ("fetch_page", "list_links", "read_pdf"):
            url = args.get("url")
            if not isinstance(url, str) or not url.strip():
                return ToolResult(
                    content=f"Error: `{name}` requires a non-empty string `url` argument.",
                    terminate=False,
                )
            method = getattr(self, name)
            return ToolResult(content=method(url), terminate=False)

        return ToolResult(
            content=(
                f"Unknown tool {name!r}. Available tools: fetch_page, list_links, "
                "read_pdf, submit_opportunities."
            ),
            terminate=False,
        )


def _inline_refs(node, defs: dict, _seen: frozenset = frozenset()):
    """Replace ``$ref`` pointers with their definitions from ``defs`` and collapse
    ``anyOf: [X, {"type": "null"}]`` to ``X``.

    With ``$ref`` in the schema, gpt-oss-120b sent plain strings for ``deadlines`` and
    every submit was rejected; with the schema inlined it sent proper ``PartialDate``
    objects. deepseek-v4-flash handled both, so the ``$ref`` form favoured one model.
    The fields and validation are unchanged.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if name in _seen or name not in defs:
                return {k: v for k, v in node.items() if k != "$ref"}
            return _inline_refs(defs[name], defs, _seen | {name})

        options = node.get("anyOf")
        if isinstance(options, list):
            concrete = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
            if len(concrete) == 1 and len(concrete) < len(options):
                merged = dict(node)
                merged.pop("anyOf")
                resolved = _inline_refs(concrete[0], defs, _seen)
                if isinstance(resolved, dict):
                    merged = {**resolved, **merged}
                return {k: _inline_refs(v, defs, _seen) if k != "type" else v
                        for k, v in merged.items()}

        return {k: (v if k in ("title", "description", "type", "enum", "const", "default")
                    else _inline_refs(v, defs, _seen))
                for k, v in node.items() if k != "$defs"}

    if isinstance(node, list):
        return [_inline_refs(v, defs, _seen) for v in node]

    return node


def _schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
) -> dict:
    parameters: dict = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _extract_pdf_pymupdf(data: bytes) -> str:
    """Extract text with pymupdf (fitz). Returns "" on any error/empty."""
    try:
        import fitz  # PyMuPDF

        parts: list[str] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                parts.append(page.get_text())
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_pdf_pdfplumber(data: bytes) -> str:
    """Extract text with pdfplumber. Returns "" on any error/empty."""
    try:
        import io

        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""
