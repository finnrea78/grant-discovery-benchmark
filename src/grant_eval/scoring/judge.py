"""LLM-as-judge for the two free-text fields, `summary` and `eligibility_summary`.

`score.py` scores detection and the structured fields exactly, but a correct paraphrase
is not an exact match, so these fields get a secondary judge pass. F1 stays the only
pre-registered primary endpoint; nothing here feeds `ConfigAggregate`.

The module covers the single judge call (`run_judge`), the per-run scoring of the two
populations (`score_run_judge`), saving verdicts, a verdict cache, a threaded batch
runner, the corpus roll-up (`build_rollup`) and the CLI. The human spot-check lives in
`judge_agreement.py`; the CLI only forwards its flags there. Groundedness and
completeness are separate verdicts and are never merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import trafilatura
from pydantic import BaseModel, Field, ValidationError, field_validator

from grant_eval.agent.tools import _extract_pdf_pdfplumber, _extract_pdf_pymupdf
from grant_eval.dataset import serve
from grant_eval.dataset.registry import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SNAPSHOT_ROOT,
    FunderRecord,
    load_registry,
)
from grant_eval.schema import ExtractionResult
from grant_eval.scoring import score as scoremod
from grant_eval.scoring.groundtruth import DEFAULT_GROUND_TRUTH_DIR, GroundTruth, load_ground_truth

#: Shared by both dimensions and both fields, so claims are comparable across them.
CLAIM_TYPES: tuple[str, ...] = (
    "eligibility",
    "deadline",
    "amount",
    "purpose",
    "process",
    "other",
)

#: See `build_groundedness_prompt` / `build_completeness_prompt` for what each sees.
DIMENSIONS: tuple[str, ...] = ("completeness", "groundedness")

JUDGED_FIELDS: tuple[str, ...] = ("summary", "eligibility_summary")

#: Matches `config.MODEL_ROUTING["gemini-judge"]`. Not a grid model, so it must not be
#: added to `config.MODELS`.
JUDGE_MODEL: str = "openrouter/google/gemini-2.5-flash-lite"

#: Part of the CLI's `JudgeConfig`, so `JudgeConfig.config_hash` changes if the prompt
#: or rubric is ever revised.
PROMPT_VERSION: str = "v1"
RUBRIC_VERSION: str = "v1"

#: Where judge output is saved: one verdict file per run (`verdict_path`), a `cache/`
#: folder (`VerdictCache`) and one `rollup-<judge_config_hash>.json` per pass.
JUDGE_RESULTS_ROOT: Path = Path("eval/judge_results")

#: Longer justifications fail validation and count as a `judge_parse_failure`.
_JUSTIFICATION_MAX_LEN = 300


class Claim(BaseModel):
    """One checkable factual claim, the unit both dimensions measure.

    Scoring claims rather than whole records avoids partly measuring verbosity. What
    `supported` means depends on which prompt builder was used.
    """

    text: str
    claim_type: str
    supported: bool

    @field_validator("claim_type")
    @classmethod
    def _claim_type_in_vocabulary(cls, v: str) -> str:
        if v not in CLAIM_TYPES:
            valid = ", ".join(CLAIM_TYPES)
            raise ValueError(
                f"claim_type {v!r} is not in the shared vocabulary (expected one of {valid})"
            )
        return v


class _RawJudgeResponse(BaseModel):
    """The shape the judge is asked for: claims plus a justification."""

    claims: list[Claim]
    justification: str = Field(max_length=_JUSTIFICATION_MAX_LEN)


class JudgeVerdict(BaseModel):
    """One dimension's verdict on one record's field.

    `record_supported` is computed by `run_judge` as `all(c.supported for c in claims)`
    rather than taken from the model, which could contradict its own claims. Never built
    with empty `claims` (that case is `no_claims_found`).
    """

    field: str
    dimension: str
    claims: list[Claim]
    record_supported: bool
    justification: str = Field(max_length=_JUSTIFICATION_MAX_LEN)

    @field_validator("field")
    @classmethod
    def _field_is_judged(cls, v: str) -> str:
        if v not in JUDGED_FIELDS:
            valid = ", ".join(JUDGED_FIELDS)
            raise ValueError(f"field {v!r} is not judged (expected one of {valid})")
        return v

    @field_validator("dimension")
    @classmethod
    def _dimension_is_known(cls, v: str) -> str:
        if v not in DIMENSIONS:
            valid = ", ".join(DIMENSIONS)
            raise ValueError(f"dimension {v!r} is not one of {valid}")
        return v


@dataclass(frozen=True)
class JudgeConfig:
    """Identifies one judge pass. Verdict paths use both the run's and this config's hash."""

    model: str
    temperature: float
    prompt_version: str
    rubric_version: str

    @property
    def config_hash(self) -> str:
        """SHA-256 of the four fields as sorted JSON, like `agent.extract.config_hash`."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "rubric_version": self.rubric_version,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class JudgePrompt:
    """One rendered judge call.

    The prompt never names the model, prompt strategy, temperature, replicate or
    funder. `scaffold` is the prompt with the evidence blanked, so tests can check that
    only the evidence (for example a page mentioning the funder) contains such cues.
    """

    system: str
    user: str
    scaffold: str
    evidence: str


def _iso(snapshot_date) -> str:
    """`date(2026, 8, 9)` or `'2026-08-09'` -> `'2026-08-09'` (copy of `agent.prompts._iso`)."""
    iso = getattr(snapshot_date, "isoformat", None)
    return iso() if callable(iso) else str(snapshot_date).strip()


def _snapshot_instruction(snapshot_date) -> str:
    return (
        f"The snapshot date is {_iso(snapshot_date)}. All source text below was "
        "captured on this date from a frozen site mirror. Judge every date and status "
        "against this snapshot date, not against today; a claim can be true of the "
        "page at that date even though the page has since changed or the date has "
        "since passed."
    )


_PARAPHRASE_INSTRUCTION = (
    "Judge substance, not wording: a claim is supported if its meaning is present, "
    "even where the phrasing, ordering or level of detail differs from the source "
    "text. Do not mark a claim unsupported merely because it is not a verbatim match."
)


def _enumerate_instruction(claim_source: str) -> str:
    return (
        f"First enumerate every discrete, checkable factual claim in the {claim_source}, "
        f"each tagged with one claim type from the vocabulary above. A sentence with no "
        "checkable content (for example a bare restatement that this is a funding "
        "opportunity) yields no claim — do not invent one to pad the list. Only once the "
        "whole list is enumerated, judge each claim independently."
    )


_JSON_FORMAT_INSTRUCTION = (
    "Respond with a single JSON object and nothing else: no markdown code fences, no "
    "commentary before or after it. Its shape is exactly:\n"
    '{"claims": [{"text": "<the claim, in your own words>", '
    '"claim_type": "<one of the claim types above>", '
    '"supported": <true or false>}, ...], "justification": "<one sentence>"}\n'
    "Do not include a record-level verdict field of your own; it is computed from the "
    "claims you list."
)

_CLAIM_TYPES_LINE = f"Claim types: {', '.join(CLAIM_TYPES)}."


def _groundedness_system(snapshot_date) -> str:
    return "\n\n".join(
        [
            "You are checking whether a short piece of text extracted about a funding "
            "opportunity is GROUNDED in a source web page — whether everything it "
            "claims is actually supported by that page.",
            _snapshot_instruction(snapshot_date),
            _CLAIM_TYPES_LINE,
            _enumerate_instruction("AGENT'S TEXT"),
            _PARAPHRASE_INSTRUCTION,
            "Enumerate claims from the AGENT'S TEXT only. A claim is supported if the "
            "PAGE TEXT states it or clearly implies it; unsupported if the page "
            "contradicts it or never mentions it. The page text is your only source of "
            "truth here — you have no other evidence to draw on.",
            _JSON_FORMAT_INSTRUCTION,
        ]
    )


def _completeness_system(snapshot_date) -> str:
    return "\n\n".join(
        [
            "You are checking whether a short piece of text produced by an extraction "
            "agent CAPTURES what a reference description of a funding opportunity "
            "records — whether the agent's text is complete relative to the reference.",
            _snapshot_instruction(snapshot_date),
            _CLAIM_TYPES_LINE,
            _enumerate_instruction("REFERENCE TEXT"),
            _PARAPHRASE_INSTRUCTION,
            "Enumerate claims from the REFERENCE TEXT only. A claim is supported if the "
            "AGENT'S TEXT states it or clearly implies it; unsupported if the agent's "
            "text omits it or contradicts it. The agent's text is your only evidence of "
            "what was captured — you are not checking it against any source page here.",
            _JSON_FORMAT_INSTRUCTION,
        ]
    )


_GROUNDEDNESS_USER_TEMPLATE = (
    "FIELD: {field}\n\n"
    "AGENT'S TEXT (enumerate claims from this):\n{extracted_text}\n\n"
    "PAGE TEXT (the only evidence; judge support against this page alone):\n{page_text}"
)

_COMPLETENESS_USER_TEMPLATE = (
    "FIELD: {field}\n\n"
    "REFERENCE TEXT (enumerate claims from this):\n{gold_text}\n\n"
    "AGENT'S TEXT (the only evidence of what was captured):\n{extracted_text}"
)


def build_groundedness_prompt(
    *, field: str, extracted_text: str, page_text: str, snapshot_date
) -> JudgePrompt:
    """Groundedness: is every claim in the agent's text supported by the source page?

    Claims come from the agent's text and are checked against the page only. The judge
    never sees the gold text. This catches invented restrictions or amounts, the most
    harmful kind of error for a grant tool.
    """
    system = _groundedness_system(snapshot_date)
    user = _GROUNDEDNESS_USER_TEMPLATE.format(
        field=field, extracted_text=extracted_text, page_text=page_text
    )
    scaffold = system + "\n\n" + _GROUNDEDNESS_USER_TEMPLATE.format(
        field=field, extracted_text="", page_text=""
    )
    evidence = extracted_text + "\n\n" + page_text
    return JudgePrompt(system=system, user=user, scaffold=scaffold, evidence=evidence)


def build_completeness_prompt(
    *, field: str, extracted_text: str, gold_text: str, snapshot_date
) -> JudgePrompt:
    """Completeness: does the agent's text capture what the labeller recorded?

    Claims come from the gold text and are checked against the agent's text only; the
    judge never sees the page. This catches a missed eligibility restriction.
    """
    system = _completeness_system(snapshot_date)
    user = _COMPLETENESS_USER_TEMPLATE.format(
        field=field, extracted_text=extracted_text, gold_text=gold_text
    )
    scaffold = system + "\n\n" + _COMPLETENESS_USER_TEMPLATE.format(
        field=field, extracted_text="", gold_text=""
    )
    evidence = gold_text + "\n\n" + extracted_text
    return JudgePrompt(system=system, user=user, scaffold=scaffold, evidence=evidence)


class JudgeBackend(Protocol):
    """Returns the raw reply text; `run_judge` does the parsing.

    `total_cost_usd` is required: the batch runner's spend cap reads it and refuses to
    run against a backend without it.
    """

    name: str
    total_cost_usd: float

    def complete(self, *, system: str, user: str, model: str, temperature: float) -> str: ...


class FakeJudgeBackend:
    """Offline test backend that returns scripted replies (or raises scripted errors).

    Optional `costs` set the spend per call; a call that raises costs nothing. It is
    thread-safe. `delay` sleeps outside the lock so tests can force threads to overlap.
    """

    name = "fake"

    def __init__(
        self, replies: list[str | Exception], costs: list[float] | None = None,
        delay: float = 0.0,
    ) -> None:
        self._replies = list(replies)
        self._costs = list(costs) if costs is not None else None
        if self._costs is not None and len(self._costs) != len(self._replies):
            raise ValueError(
                "FakeJudgeBackend: `costs` must be the same length as `replies` "
                f"({len(self._costs)} vs {len(self._replies)})"
            )
        self._delay = delay
        self._cursor = 0
        #: One dict per call, for assertions on exactly what was sent.
        self.calls: list[dict] = []
        self.total_cost_usd = 0.0
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, model: str, temperature: float) -> str:
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.calls.append(
                {"system": system, "user": user, "model": model, "temperature": temperature}
            )
            if self._cursor >= len(self._replies):
                raise RuntimeError(
                    f"FakeJudgeBackend script exhausted after {self._cursor} call(s): no "
                    f"scripted reply for call #{self._cursor + 1}."
                )
            item = self._replies[self._cursor]
            cost = self._costs[self._cursor] if self._costs is not None else 0.0
            self._cursor += 1
            if not isinstance(item, Exception):
                self.total_cost_usd += cost
        if isinstance(item, Exception):
            raise item
        return item


def _judge_router_backend_class():
    """Build an `OpenRouterBackend` subclass routed by `config.MODEL_ROUTING["gemini-judge"]`.

    The judge is not a grid model, so it has no `MODELS` entry for the normal routing
    lookup. Built lazily so importing this module does not need `litellm`.
    """
    from grant_eval.agent.router import OpenRouterBackend

    class _GoogleAIStudioJudgeBackend(OpenRouterBackend):
        # Fixed judge timeout (the agent default is 180s). A timeout becomes a
        # `judge_call_failure` in `run_judge`.
        TIMEOUT_S: float = 60.0

        @staticmethod
        def _routing_for(model: str) -> dict:
            # Refuse to apply the judge's routing to any other model.
            if model != JUDGE_MODEL:
                raise ValueError(
                    f"_judge_router_backend_class is pinned to {JUDGE_MODEL!r}; refusing "
                    f"to apply its routing to {model!r}."
                )
            from grant_eval.config import MODEL_ROUTING

            return MODEL_ROUTING.get("gemini-judge", {})

    return _GoogleAIStudioJudgeBackend


class OpenRouterJudgeBackend:
    """The live judge backend: one single-turn completion with no tools.

    The API key is only read in `complete()`. `total_cost_usd` adds up the cost LiteLLM
    reports for each call, which the spend cap reads. Thread-safe.
    """

    name = "openrouter-judge"

    def __init__(self) -> None:
        self._router = _judge_router_backend_class()()
        self.total_cost_usd = 0.0
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, model: str, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        reply, usage = self._router.complete(
            messages=messages, model=model, temperature=temperature, tools=[]
        )
        with self._lock:
            self.total_cost_usd += usage.cost_usd
        return reply.text or ""


class JudgeOutcome(str, Enum):
    """What `run_judge` produced.

    Parse failures (a judge problem) and call failures (an infrastructure problem) are
    kept apart so their causes can be told apart.
    """

    OK = "ok"
    NO_CLAIMS_FOUND = "no_claims_found"
    PARSE_FAILURE = "judge_parse_failure"
    CALL_FAILURE = "judge_call_failure"


@dataclass(frozen=True)
class JudgeResult:
    """Result of one `run_judge` call. `verdict` is set only for `OK`; `error` holds the
    failure text for the two failure outcomes."""

    outcome: JudgeOutcome
    verdict: JudgeVerdict | None = None
    error: str | None = None


def _strip_markdown_fence(raw_text: str) -> str:
    """Remove a markdown code fence if it wraps the whole reply.

    The judge model wraps its JSON in a ```json fence even when asked not to, which
    made every reply in the first live run fail to parse. Nothing else is relaxed:
    invalid JSON inside the fence is still a `judge_parse_failure`.
    """
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline == -1:
        return text  # only a bare fence line, nothing inside
    body = text[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def run_judge(
    *, backend: JudgeBackend, judge_config: JudgeConfig, prompt: JudgePrompt,
    field: str, dimension: str,
) -> JudgeResult:
    """One judge call: complete, parse strictly, derive `record_supported`.

    Never raises for a bad reply or failed call; both come back as a `JudgeResult` so
    a batch keeps going.
    """
    try:
        raw_text = backend.complete(
            system=prompt.system, user=prompt.user,
            model=judge_config.model, temperature=judge_config.temperature,
        )
    except Exception as exc:  # 429/timeout/5xx surviving the backend's own retries
        return JudgeResult(outcome=JudgeOutcome.CALL_FAILURE, error=str(exc))

    try:
        data = json.loads(_strip_markdown_fence(raw_text))
        raw = _RawJudgeResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return JudgeResult(outcome=JudgeOutcome.PARSE_FAILURE, error=str(exc))

    if not raw.claims:
        # No verdict here, otherwise all([]) would count as fully supported.
        return JudgeResult(outcome=JudgeOutcome.NO_CLAIMS_FOUND)

    record_supported = all(c.supported for c in raw.claims)
    verdict = JudgeVerdict(
        field=field, dimension=dimension, claims=raw.claims,
        record_supported=record_supported, justification=raw.justification,
    )
    return JudgeResult(outcome=JudgeOutcome.OK, verdict=verdict)


#: Reasons a record gets no verdict: two decided before the call, three from `JudgeOutcome`.
FIELD_EMPTY = "field_empty"
UNRESOLVABLE_SOURCE = "unresolvable_source"
EXCLUSION_REASONS: tuple[str, ...] = (
    FIELD_EMPTY,
    UNRESOLVABLE_SOURCE,
    JudgeOutcome.NO_CLAIMS_FOUND.value,
    JudgeOutcome.PARSE_FAILURE.value,
    JudgeOutcome.CALL_FAILURE.value,
)


#: Cap on `RecordJudgement.error` so a huge reply cannot bloat the verdict file.
_ERROR_MAX_LEN = 500


@dataclass(frozen=True)
class RecordJudgement:
    """One record's judgement for one field and dimension.

    Either a verdict (`outcome == "ok"`) or one of `EXCLUSION_REASONS`. `error` holds
    the truncated failure text for parse and call failures, so they can be diagnosed
    from the verdict file.
    """

    opportunity_id: str
    field: str
    dimension: str
    outcome: str
    verdict: JudgeVerdict | None = None
    error: str | None = None


@dataclass
class FieldJudgement:
    """Rates and counts for one (field, dimension) cell of one run."""

    field: str
    dimension: str
    n_claims: int = 0
    n_claims_supported: int = 0
    n_records_judged: int = 0
    n_records_supported: int = 0
    exclusions: dict[str, int] = dc_field(default_factory=lambda: {r: 0 for r in EXCLUSION_REASONS})
    #: How often the judge got raw HTML instead of cleaned text (see `MirrorPage`).
    #: Not an exclusion. Always 0 for completeness, which does not use the mirror.
    n_raw_html_fallback: int = 0
    records: list[RecordJudgement] = dc_field(default_factory=list)

    @property
    def claim_rate_undefined(self) -> bool:
        return self.n_claims == 0

    @property
    def claim_rate(self) -> float:
        """Micro claim rate: all judged claims pooled, so longer records weigh more."""
        return 0.0 if self.claim_rate_undefined else self.n_claims_supported / self.n_claims

    @property
    def claim_rate_macro_undefined(self) -> bool:
        return self.n_records_judged == 0

    @property
    def claim_rate_macro(self) -> float:
        """Macro claim rate: the mean of each record's own claim rate.

        Every record counts equally, so this is not affected by verbosity.
        """
        if self.claim_rate_macro_undefined:
            return 0.0
        per_record = [
            sum(1 for c in r.verdict.claims if c.supported) / len(r.verdict.claims)
            for r in self.records
            if r.verdict is not None
        ]
        return sum(per_record) / len(per_record)

    @property
    def record_rate_undefined(self) -> bool:
        return self.n_records_judged == 0

    @property
    def record_rate(self) -> float:
        """Share of judged records where every claim is supported (stricter than claim rates)."""
        return 0.0 if self.record_rate_undefined else self.n_records_supported / self.n_records_judged


def funder_slug(display_name: str, registry: dict[str, FunderRecord]) -> str | None:
    """Map a funder display name ("The Leverhulme Trust") to its slug ("leverhulme").

    Returns `None` for an unknown name; callers treat that as `unresolvable_source`.
    """
    for record in registry.values():
        if record.display_name == display_name:
            return record.slug
    return None


@dataclass(frozen=True)
class MirrorPage:
    """One resolved mirror page. `raw_html_fallback` is True when `text` is raw HTML
    because cleaning returned nothing."""

    text: str
    raw_html_fallback: bool


def build_mirror_lookup(
    funder: str, snapshot_date, *, registry: dict[str, FunderRecord],
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict | None:
    """Build the URL lookup for one funder snapshot, once per run.

    Returns `None` for an unregistered funder or a snapshot with no manifest.
    """
    slug = funder_slug(funder, registry)
    if slug is None:
        return None
    snapshot_dir = Path(snapshot_root) / slug / _iso(snapshot_date)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return serve.build_lookup(snapshot_dir, manifest)


def resolve_mirror_path(lookup: dict | None, url: str) -> Path | None:
    """Resolve a URL to the mirrored file on disk, or `None` if it was not captured.

    Used by both `resolve_mirror_page` and the spot-check export, so there is one rule.
    """
    if lookup is None:
        return None
    parsed = urlparse(url)
    target = serve.resolve(lookup, parsed.path, parsed.query)
    if target is None or not target.is_file():
        return None
    return target


def resolve_mirror_page(lookup: dict | None, url: str) -> MirrorPage | None:
    """Resolve a URL to its cleaned page text, or `None` if it cannot be found."""
    target = resolve_mirror_path(lookup, url)
    if target is None:
        return None
    return _extract_mirror_page_text(target)


def resolve_mirror_text(
    funder: str, snapshot_date, url: str, *,
    registry: dict[str, FunderRecord], snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> str | None:
    """One-off resolution of a single URL (builds a lookup each time)."""
    lookup = build_mirror_lookup(funder, snapshot_date, registry=registry, snapshot_root=snapshot_root)
    page = resolve_mirror_page(lookup, url)
    return None if page is None else page.text


def _extract_mirror_page_text(path: Path) -> MirrorPage:
    """Clean text for one mirrored file, the same way the agent's `fetch_page` does.

    HTML uses the same trafilatura call and falls back to raw HTML when cleaning
    returns nothing, but without the trace tag or the length cap. PDFs use the agent's
    own PDF extractors.
    """
    if path.suffix.lower() == ".pdf":
        data = path.read_bytes()
        text = _extract_pdf_pymupdf(data)
        if not text.strip():
            text = _extract_pdf_pdfplumber(data)
        return MirrorPage(text=text, raw_html_fallback=False)
    html = path.read_text(encoding="utf-8", errors="replace")
    cleaned = trafilatura.extract(html, include_tables=True, favor_recall=True)
    if not cleaned:
        return MirrorPage(text=html, raw_html_fallback=True)
    return MirrorPage(text=cleaned, raw_html_fallback=False)


@dataclass
class RunJudgeScore:
    """Everything judged for one run, the free-text counterpart of `score.RunScore`.

    `fields` always has all four (field, dimension) keys, even when empty.
    """

    result: ExtractionResult
    schema_valid: bool | None
    abstained: bool
    judge_config_hash: str
    fields: dict[tuple[str, str], FieldJudgement]


def _is_blank(text: str | None) -> bool:
    return not (text or "").strip()


def _exclude(fj: FieldJudgement, opportunity_id: str, reason: str, error: str | None = None) -> None:
    fj.exclusions[reason] = fj.exclusions.get(reason, 0) + 1
    fj.records.append(
        RecordJudgement(
            opportunity_id=opportunity_id, field=fj.field, dimension=fj.dimension,
            outcome=reason, error=None if error is None else error[:_ERROR_MAX_LEN],
        )
    )


def _accumulate(fj: FieldJudgement, opportunity_id: str, judged: JudgeResult) -> None:
    """Add one `run_judge` result to its `FieldJudgement`.

    `OK` updates the counts; anything else is recorded as a named exclusion, never as
    an unsupported verdict.
    """
    if judged.outcome is JudgeOutcome.OK:
        v = judged.verdict
        fj.n_claims += len(v.claims)
        fj.n_claims_supported += sum(1 for c in v.claims if c.supported)
        fj.n_records_judged += 1
        if v.record_supported:
            fj.n_records_supported += 1
        fj.records.append(
            RecordJudgement(opportunity_id=opportunity_id, field=fj.field, dimension=fj.dimension, outcome="ok", verdict=v)
        )
    else:
        _exclude(fj, opportunity_id, judged.outcome.value, error=judged.error)


def _run_judge_cached(
    *, backend: JudgeBackend, judge_config: JudgeConfig, prompt: JudgePrompt,
    field: str, dimension: str, extracted_text: str, comparison_text: str,
    cache: VerdictCache | None,
) -> JudgeResult:
    """`run_judge` with an optional cache. Only `OK` results are cached."""
    key = None
    if cache is not None:
        key = verdict_cache_key(
            judge_config_hash=judge_config.config_hash, field=field, dimension=dimension,
            extracted_text=extracted_text, comparison_text=comparison_text,
        )
        cached = cache.get(key)
        if cached is not None:
            return JudgeResult(outcome=JudgeOutcome.OK, verdict=cached)

    result = run_judge(
        backend=backend, judge_config=judge_config, prompt=prompt, field=field, dimension=dimension,
    )
    if cache is not None and result.outcome is JudgeOutcome.OK:
        cache.put(key, result.verdict)
    return result


def score_run_judge(
    *, result: ExtractionResult, run_score: scoremod.RunScore, backend: JudgeBackend,
    judge_config: JudgeConfig, registry: dict[str, FunderRecord],
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT, cache: VerdictCache | None = None,
) -> RunJudgeScore:
    """Judge one run's `summary` and `eligibility_summary` on both dimensions.

    `run_score` must be `score.score_run` for the same `result`. Completeness covers
    true positives only, since an unmatched record has no gold text. Groundedness
    covers every record whose URL is in the mirror, false positives included. The two
    populations differ, so their rates are kept separate.
    """
    fields = {(f, d): FieldJudgement(field=f, dimension=d) for f in JUDGED_FIELDS for d in DIMENSIONS}

    for tp in run_score.detection.true_positives:
        for f in JUDGED_FIELDS:
            fj = fields[(f, "completeness")]
            extracted_text = getattr(tp.extracted, f)
            if _is_blank(extracted_text):
                # The agent left this field blank: an abstention, not an error.
                _exclude(fj, tp.extracted.opportunity_id, FIELD_EMPTY)
                continue
            gold_text = getattr(tp.gold, f) or ""
            prompt = build_completeness_prompt(
                field=f, extracted_text=extracted_text, gold_text=gold_text,
                snapshot_date=result.snapshot_date,
            )
            judged = _run_judge_cached(
                backend=backend, judge_config=judge_config, prompt=prompt,
                field=f, dimension="completeness",
                extracted_text=extracted_text, comparison_text=gold_text, cache=cache,
            )
            _accumulate(fj, tp.extracted.opportunity_id, judged)

    # Built once per run, since only the URL changes between opportunities.
    lookup = build_mirror_lookup(result.funder, result.snapshot_date, registry=registry, snapshot_root=snapshot_root)
    for opp in result.opportunities:
        # Resolved once per record; both fields use the same page.
        page = resolve_mirror_page(lookup, opp.url)
        for f in JUDGED_FIELDS:
            fj = fields[(f, "groundedness")]
            if page is None:
                _exclude(fj, opp.opportunity_id, UNRESOLVABLE_SOURCE)
                continue
            extracted_text = getattr(opp, f)
            if _is_blank(extracted_text):
                _exclude(fj, opp.opportunity_id, FIELD_EMPTY)
                continue
            if page.raw_html_fallback:
                fj.n_raw_html_fallback += 1
            prompt = build_groundedness_prompt(
                field=f, extracted_text=extracted_text, page_text=page.text,
                snapshot_date=result.snapshot_date,
            )
            judged = _run_judge_cached(
                backend=backend, judge_config=judge_config, prompt=prompt,
                field=f, dimension="groundedness",
                extracted_text=extracted_text, comparison_text=page.text, cache=cache,
            )
            _accumulate(fj, opp.opportunity_id, judged)

    return RunJudgeScore(
        result=result, schema_valid=result.schema_valid, abstained=result.abstained,
        judge_config_hash=judge_config.config_hash, fields=fields,
    )


def _field_judgement_to_dict(fj: FieldJudgement) -> dict:
    """JSON view of one cell, records included. Undefined rates become `null`."""
    return {
        "field": fj.field,
        "dimension": fj.dimension,
        "n_claims": fj.n_claims,
        "n_claims_supported": fj.n_claims_supported,
        "n_records_judged": fj.n_records_judged,
        "n_records_supported": fj.n_records_supported,
        "claim_rate": None if fj.claim_rate_undefined else fj.claim_rate,
        "claim_rate_undefined": fj.claim_rate_undefined,
        "claim_rate_macro": None if fj.claim_rate_macro_undefined else fj.claim_rate_macro,
        "claim_rate_macro_undefined": fj.claim_rate_macro_undefined,
        "record_rate": None if fj.record_rate_undefined else fj.record_rate,
        "record_rate_undefined": fj.record_rate_undefined,
        "n_raw_html_fallback": fj.n_raw_html_fallback,
        "exclusions": dict(fj.exclusions),
        "records": [
            {
                "opportunity_id": rec.opportunity_id,
                "outcome": rec.outcome,
                "record_supported": None if rec.verdict is None else rec.verdict.record_supported,
                "justification": None if rec.verdict is None else rec.verdict.justification,
                "claims": None if rec.verdict is None else [
                    {"text": c.text, "claim_type": c.claim_type, "supported": c.supported}
                    for c in rec.verdict.claims
                ],
                "error": rec.error,
            }
            for rec in fj.records
        ],
    }


def verdict_cache_key(judge_config_hash: str, field: str, dimension: str, extracted_text: str, comparison_text: str) -> str:
    """One SHA-256 over the judge config, field, dimension and hashes of both texts."""

    def _sha(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    payload = f"{judge_config_hash}|{field}|{dimension}|{_sha(extracted_text)}|{_sha(comparison_text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class VerdictCache:
    """Disk cache of verdicts at `root/cache/<key>.json`.

    Only `OK` verdicts are stored, so a temporary failure is never made permanent.
    """

    root: Path = JUDGE_RESULTS_ROOT

    def _path(self, key: str) -> Path:
        return Path(self.root) / "cache" / f"{key}.json"

    def get(self, key: str) -> JudgeVerdict | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return JudgeVerdict.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            # Treat a corrupt entry as a miss so one bad file cannot stop a batch; the
            # caller re-judges and overwrites it.
            return None

    def put(self, key: str, verdict: JudgeVerdict) -> None:
        write_verdict_file(self._path(key), json.loads(verdict.model_dump_json()))


#: Gemini 2.5 Flash Lite list prices, used only for the `--dry-run` estimate. Real spend
#: comes from the LiteLLM response.
_DRY_RUN_INPUT_USD_PER_MTOK = 0.10
_DRY_RUN_OUTPUT_USD_PER_MTOK = 0.40
#: Rough characters per token, to avoid adding a tokeniser dependency.
_DRY_RUN_CHARS_PER_TOKEN = 4.0
#: Generous fixed output size per reply, since a dry run has no real output to measure.
_DRY_RUN_OUTPUT_TOKENS = 300


def _estimate_call_cost_usd(system: str, user: str) -> float:
    input_tokens = (len(system) + len(user)) / _DRY_RUN_CHARS_PER_TOKEN
    return (
        input_tokens / 1e6 * _DRY_RUN_INPUT_USD_PER_MTOK
        + _DRY_RUN_OUTPUT_TOKENS / 1e6 * _DRY_RUN_OUTPUT_USD_PER_MTOK
    )


class _DryRunNoCall(Exception):
    """Raised by `_DryRunBackend` after recording a call; the outcome is never used."""


class _DryRunBackend:
    """Records the calls `score_run_judge` would make, without making them.

    Running the real `score_run_judge` logic keeps the dry-run count in line with a
    real pass.
    """

    name = "dry-run"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, model: str, temperature: float) -> str:
        with self._lock:
            self.calls.append({"system": system, "user": user, "model": model, "temperature": temperature})
        raise _DryRunNoCall()


def dry_run_batch(paths, *, ground_truth: GroundTruth, registry: dict[str, FunderRecord],
                   judge_config: JudgeConfig, snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT) -> dict:
    """Count the calls a real batch would make and estimate their cost, without calling anything."""
    n_calls = 0
    cost = 0.0
    for path in paths:
        result = scoremod.load_result(path)
        try:
            run_score = scoremod.score_run(result, ground_truth, source_path=Path(path))
        except KeyError:
            continue  # unlabelled funder, skipped by a real pass too
        dry_backend = _DryRunBackend()
        score_run_judge(
            result=result, run_score=run_score, backend=dry_backend, judge_config=judge_config,
            registry=registry, snapshot_root=snapshot_root,
        )
        n_calls += len(dry_backend.calls)
        cost += sum(_estimate_call_cost_usd(c["system"], c["user"]) for c in dry_backend.calls)
    return {"n_calls": n_calls, "estimated_cost_usd": cost}


@dataclass(frozen=True)
class JudgeRunOutcome:
    """One run's result within a batch.

    `cost_usd` is always 0.0: with several threads sharing one backend, spend cannot
    be split per run. Use `JudgeBatchResult.total_spend_usd` instead.
    """

    source_path: Path
    run_config_hash: str
    judge_config_hash: str
    resumed: bool
    to_dict: dict
    cost_usd: float


@dataclass
class JudgeBatchResult:
    """Outcome of a batch, including partial progress.

    Skipped, failed and cancelled runs are listed rather than raised, so the runs that
    did finish are never lost.
    """

    outcomes: list[JudgeRunOutcome]
    skipped_for_spend_cap: list[Path]
    errors: list[tuple[Path, str]]
    total_spend_usd: float
    cap_hit: bool
    #: Runs cancelled after a worker raised a non-`Exception` `BaseException`
    #: (e.g. `SystemExit`). Kept apart from the spend-cap list.
    cancelled_after_fatal: list[Path] = dc_field(default_factory=list)
    #: Runs cancelled because the main thread was interrupted (usually Ctrl-C).
    cancelled_after_interrupt: list[Path] = dc_field(default_factory=list)
    #: Name of the exception that interrupted the loop, or `None` if it finished.
    interrupted_by: str | None = None


def _judge_one_run(
    path: Path, *, ground_truth: GroundTruth, registry: dict[str, FunderRecord],
    snapshot_root: Path, backend: JudgeBackend, judge_config: JudgeConfig,
    cache: VerdictCache | None, results_root: Path, resume: bool,
) -> JudgeRunOutcome:
    """Judge one run, or reuse its saved verdict when resuming.

    Raises `KeyError` for an unlabelled funder and `ValueError` for a funder missing
    from the registry; the batch runner catches both per run.
    """
    result = scoremod.load_result(path)
    run_score = scoremod.score_run(result, ground_truth, source_path=Path(path))
    slug = funder_slug(result.funder, registry)
    if slug is None:
        raise ValueError(
            f"{result.funder!r} is not in the registry -- cannot build a stable "
            "verdict path for it (verdict_path needs a funder slug, not the display name)"
        )
    out_path = verdict_path(
        result.config_hash, judge_config.config_hash,
        funder_slug=slug, replicate=result.replicate, root=results_root,
    )

    if resume and out_path.is_file():
        return JudgeRunOutcome(
            source_path=Path(path), run_config_hash=result.config_hash,
            judge_config_hash=judge_config.config_hash, resumed=True,
            to_dict=read_verdict_file(out_path), cost_usd=0.0,
        )

    run_judge_score = score_run_judge(
        result=result, run_score=run_score, backend=backend, judge_config=judge_config,
        registry=registry, snapshot_root=snapshot_root, cache=cache,
    )
    d = to_dict(run_judge_score)
    write_verdict_file(out_path, d)
    return JudgeRunOutcome(
        source_path=Path(path), run_config_hash=result.config_hash,
        judge_config_hash=judge_config.config_hash, resumed=False,
        to_dict=d, cost_usd=0.0,
    )


def run_judge_batch(paths, *, ground_truth: GroundTruth, registry: dict[str, FunderRecord],
                     backend: JudgeBackend, judge_config: JudgeConfig,
                     snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT, workers: int = 8,
                     max_spend_usd: float = 25.0, cache: VerdictCache | None = None,
                     results_root: Path = JUDGE_RESULTS_ROOT, resume: bool = True) -> JudgeBatchResult:
    """Judge many runs in a thread pool with a spend cap.

    Spend is read from `backend.total_cost_usd` after each finished run. Once it
    reaches `max_spend_usd`, runs that have not started are cancelled and listed in
    `skipped_for_spend_cap`; runs already in progress finish, so spend can overshoot
    by up to `workers` runs. Refuses to start if the backend has no
    `total_cost_usd`.

    Errors inside a job are recorded in `errors` rather than raised. Otherwise the
    pool's `shutdown(wait=True)` would keep running (and paying for) every queued job
    with the cap no longer checked, then throw the results away.

    On Ctrl-C, queued runs are cancelled and listed, and the partial result is
    returned with `interrupted_by` set. Runs already in flight still finish and are
    saved to disk but may be missing from `outcomes`; `main` reports both counts. A
    second Ctrl-C while the pool drains is not caught.
    """
    if not hasattr(backend, "total_cost_usd"):
        raise AttributeError(
            f"{backend!r} has no `total_cost_usd` attribute, so the spend cap "
            "cannot be enforced against it (see JudgeBackend.total_cost_usd). "
            "Refusing to submit any work rather than running uncapped."
        )
    outcomes: list[JudgeRunOutcome] = []
    skipped: list[Path] = []
    errors: list[tuple[Path, str]] = []
    cancelled_after_fatal: list[Path] = []
    cancelled_after_interrupt: list[Path] = []
    interrupted_by: str | None = None
    total_spend = 0.0
    cap_hit = False

    def _job(path):
        try:
            return ("ok", _judge_one_run(
                path, ground_truth=ground_truth, registry=registry, snapshot_root=snapshot_root,
                backend=backend, judge_config=judge_config, cache=cache,
                results_root=results_root, resume=resume,
            ))
        except Exception as exc:
            return ("error", path, str(exc))
        except BaseException as exc:
            # SystemExit and friends would otherwise escape and leave the pool running
            # unchecked. Marked "fatal" so no more work is started. The type name is
            # included because `str(SystemExit())` is empty.
            return ("fatal", path, f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, p): p for p in paths}
        cancelled_futures = set()

        def _cancel_pending(current, into: list) -> None:
            """Cancel every future that has not started and record each one once in `into`.

            `cancel()` also returns True for an already-cancelled future, hence the set.
            """
            for other_fut, other_path in futures.items():
                if other_fut is current or other_fut in cancelled_futures:
                    continue
                if other_fut.cancel():
                    cancelled_futures.add(other_fut)
                    into.append(other_path)

        try:
            for fut in as_completed(futures):
                if fut.cancelled():
                    # Already recorded when it was cancelled.
                    continue
                outcome_kind, *rest = fut.result()
                # Check spend for errors too: a run can spend money and then fail.
                total_spend = backend.total_cost_usd
                if not cap_hit and total_spend >= max_spend_usd:
                    cap_hit = True
                    _cancel_pending(fut, skipped)
                if outcome_kind == "fatal":
                    # Stop starting new work, but keep collecting finished runs.
                    err_path, msg = rest
                    errors.append((err_path, msg))
                    _cancel_pending(fut, cancelled_after_fatal)
                    continue
                if outcome_kind == "error":
                    err_path, msg = rest
                    errors.append((err_path, msg))
                    continue
                (outcome,) = rest
                outcomes.append(outcome)
        except BaseException as exc:
            # Ctrl-C: cancel queued work and return the partial result (see docstring).
            interrupted_by = type(exc).__name__
            _cancel_pending(None, cancelled_after_interrupt)
            ex.shutdown(wait=False, cancel_futures=True)

    if interrupted_by is not None:
        # Re-read after the pool drains, since in-flight runs kept spending.
        total_spend = backend.total_cost_usd

    return JudgeBatchResult(
        outcomes=outcomes, skipped_for_spend_cap=skipped, errors=errors,
        total_spend_usd=total_spend, cap_hit=cap_hit,
        cancelled_after_fatal=cancelled_after_fatal,
        cancelled_after_interrupt=cancelled_after_interrupt,
        interrupted_by=interrupted_by,
    )


def rollup_path(judge_config_hash: str, *, root: Path = JUDGE_RESULTS_ROOT) -> Path:
    """Roll-up file for one judge config, so a new config never overwrites an old pass."""
    return Path(root) / f"rollup-{judge_config_hash}.json"


def verdict_path(
    run_config_hash: str, judge_config_hash: str, *, funder_slug: str, replicate: int,
    root: Path = JUDGE_RESULTS_ROOT,
) -> Path:
    """Path of one run's judged output.

    `run_config_hash` covers a whole grid arm, so funder slug and replicate are needed
    to identify a run. A different judge config gets its own file, so earlier passes
    are kept.
    """
    return Path(root) / run_config_hash / funder_slug / str(replicate) / f"{judge_config_hash}.json"


def write_verdict_file(path: Path, data: dict) -> None:
    """Write JSON atomically, creating parent folders.

    Writes to a unique temp file in the same folder and then renames it, so
    concurrent writers cannot corrupt each other. The temp file is removed on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, default=str))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_verdict_file(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_corpus_verdicts(
    judge_config_hash: str, *, root: Path = JUDGE_RESULTS_ROOT,
) -> tuple[list[dict], list[dict]]:
    """Every saved verdict under `root` for one judge config. Returns `(verdicts, unreadable)`.

    The roll-up is built from what is on disk rather than from one invocation, so
    judging a single funder cannot shrink the corpus figure, and resumed runs count.
    The glob matches `<run_hash>/<funder_slug>/<replicate>/<judge_hash>.json` (the same
    one `notebooks/judge.ipynb` uses) and never reaches the cache folder.

    Files that cannot be used are listed in `unreadable` instead of raising, since
    this runs at the end of a long, paid pass. Each file is checked by running
    `build_rollup` on it, so the check cannot drift from the real consumer.
    """
    verdicts: list[dict] = []
    unreadable: list[dict] = []
    for p in sorted(Path(root).glob(f"*/*/*/{judge_config_hash}.json")):
        try:
            data = read_verdict_file(p)
            build_rollup([data])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            unreadable.append({"path": str(p), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        verdicts.append(data)
    return verdicts, unreadable


def _rate(numerator: int, denominator: int) -> tuple[float | None, bool]:
    """`numerator / denominator`, or `None` (with the undefined flag) when it is zero."""
    undefined = denominator == 0
    return (None if undefined else numerator / denominator, undefined)


def _new_rollup_accumulator(field: str, dimension: str) -> dict:
    return {
        "field": field, "dimension": dimension,
        "n_claims": 0, "n_claims_supported": 0,
        "n_records_judged": 0, "n_records_supported": 0, "n_raw_html_fallback": 0,
        "exclusions": {r: 0 for r in EXCLUSION_REASONS},
        #: One claim rate per judged record across all runs. The corpus macro rate is
        #: their mean, not the mean of per-run macros (which would overweight small runs).
        "_per_record_claim_rates": [],
    }


def _fold_cell_into(acc: dict, cell: dict) -> None:
    acc["n_claims"] += cell["n_claims"]
    acc["n_claims_supported"] += cell["n_claims_supported"]
    acc["n_records_judged"] += cell["n_records_judged"]
    acc["n_records_supported"] += cell["n_records_supported"]
    acc["n_raw_html_fallback"] += cell["n_raw_html_fallback"]
    for reason, n in cell["exclusions"].items():
        acc["exclusions"][reason] = acc["exclusions"].get(reason, 0) + n
    for rec in cell["records"]:
        if rec["claims"]:
            supported = sum(1 for c in rec["claims"] if c["supported"])
            acc["_per_record_claim_rates"].append(supported / len(rec["claims"]))


def _finalise_rollup_cell(acc: dict) -> dict:
    claim_rate, claim_rate_undefined = _rate(acc["n_claims_supported"], acc["n_claims"])
    record_rate, record_rate_undefined = _rate(acc["n_records_supported"], acc["n_records_judged"])
    per_record = acc["_per_record_claim_rates"]
    claim_rate_macro_undefined = not per_record
    claim_rate_macro = None if claim_rate_macro_undefined else sum(per_record) / len(per_record)
    return {
        "field": acc["field"], "dimension": acc["dimension"],
        "n_claims": acc["n_claims"], "n_claims_supported": acc["n_claims_supported"],
        "n_records_judged": acc["n_records_judged"], "n_records_supported": acc["n_records_supported"],
        "n_raw_html_fallback": acc["n_raw_html_fallback"],
        "claim_rate": claim_rate, "claim_rate_undefined": claim_rate_undefined,
        "claim_rate_macro": claim_rate_macro, "claim_rate_macro_undefined": claim_rate_macro_undefined,
        "record_rate": record_rate, "record_rate_undefined": record_rate_undefined,
        "exclusions": acc["exclusions"],
    }


def build_rollup(run_dicts: list[dict]) -> dict:
    """Combine many runs' `to_dict()` output into corpus totals.

    Gives counts and micro/macro rates per (field, dimension), the same broken down
    `by_funder`, and an `exclusion_ledger` listing every excluded record and why.
    """
    totals: dict[str, dict] = {}
    by_funder: dict[str, dict[str, dict]] = {}
    ledger: list[dict] = []

    for run in run_dicts:
        funder = run["funder"]
        for key, cell in run["fields"].items():
            field, dimension = cell["field"], cell["dimension"]

            acc = totals.setdefault(key, _new_rollup_accumulator(field, dimension))
            _fold_cell_into(acc, cell)

            funder_cells = by_funder.setdefault(funder, {})
            funder_acc = funder_cells.setdefault(key, _new_rollup_accumulator(field, dimension))
            _fold_cell_into(funder_acc, cell)

            for rec in cell["records"]:
                if rec["outcome"] != "ok":
                    ledger.append({
                        "run_config_hash": run["config_hash"], "funder": funder,
                        "field": field, "dimension": dimension,
                        "opportunity_id": rec["opportunity_id"], "reason": rec["outcome"],
                        #: `.get` because older verdict files have no "error" key.
                        "error": rec.get("error"),
                    })

    return {
        "n_runs": len(run_dicts),
        "fields": {key: _finalise_rollup_cell(acc) for key, acc in totals.items()},
        "by_funder": {
            funder: {key: _finalise_rollup_cell(acc) for key, acc in cells.items()}
            for funder, cells in by_funder.items()
        },
        "exclusion_ledger": ledger,
    }


def to_dict(run_judge_score: RunJudgeScore) -> dict:
    """JSON view of one run's judged output, with `fields` keyed `"<field>:<dimension>"`."""
    r = run_judge_score.result
    return {
        "funder": r.funder,
        "model": r.model,
        "prompt_strategy": r.prompt_strategy,
        "temperature": r.temperature,
        "agent_config": r.agent_config,
        "replicate": r.replicate,
        "config_hash": r.config_hash,
        "schema_valid": run_judge_score.schema_valid,
        "abstained": run_judge_score.abstained,
        "judge_config_hash": run_judge_score.judge_config_hash,
        "fields": {
            f"{field}:{dimension}": _field_judgement_to_dict(fj)
            for (field, dimension), fj in run_judge_score.fields.items()
        },
    }


def expand_result_paths(paths) -> list[Path]:
    """Like `score._expand` but searches folders recursively, sorted.

    The grid is laid out as `grid/<funder>/*.json`, so the judge needs to recurse.
    Kept separate so `score` and `aggregate` do not start picking up nested files.
    """
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        out.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="judge",
        description=(
            "LLM-as-judge over the free-form fields (summary, eligibility_summary): "
            "completeness against the gold, groundedness against the mirror page. "
            "Secondary endpoint only -- see scoring/judge.py's module docstring."
        ),
    )
    p.add_argument(
        "results", nargs="*", default=[],
        help=(
            "ExtractionResult JSON file(s), or directories of them. Required for a "
            "normal pass, --dry-run and --export-spotcheck; not read at all for "
            "--agreement."
        ),
    )
    p.add_argument(
        "--ground-truth", default=str(DEFAULT_GROUND_TRUTH_DIR),
        help=f"ground-truth CSV directory (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    p.add_argument(
        "--snapshots", default=str(DEFAULT_SNAPSHOT_ROOT),
        help=f"snapshot mirror root (default: {DEFAULT_SNAPSHOT_ROOT})",
    )
    p.add_argument(
        "--results-root", default=str(JUDGE_RESULTS_ROOT),
        help=(
            "where verdicts, the verdict cache and the roll-up are written/read "
            f"(default: {JUDGE_RESULTS_ROOT}). Every path this CLI touches is keyed off "
            "this root, so a pass, its resume and its roll-up all stay together under it"
        ),
    )
    p.add_argument("--workers", type=int, default=8, help="thread pool size (default: 8)")
    p.add_argument(
        "--max-spend-usd", type=float, default=25.0,
        help="hard spend cap in USD, read off the API response (default: 25.0)",
    )
    p.add_argument("--no-cache", action="store_true", help="bypass the content-addressed verdict cache")
    p.add_argument(
        "--dry-run", action="store_true",
        help="build every prompt and report the call count and an ESTIMATED cost; call nothing",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of the report")
    # Spot-check and agreement flags, handled in judge_agreement.py.
    p.add_argument(
        "--export-spotcheck", type=int, default=None, metavar="N",
        help="write a stratified CSV of N judged claims for human coding (requires --out); see judge_agreement.py",
    )
    p.add_argument("--out", default=None, help="output CSV path for --export-spotcheck")
    p.add_argument(
        "--agreement", default=None, metavar="CODED.csv",
        help="join a human-coded spot-check CSV back to the stored verdicts and report judge-human agreement",
    )
    p.add_argument(
        "--test-retest", type=int, default=None, metavar="N",
        help=(
            "re-judge up to N records fresh (cache bypassed) and report judge self-agreement, "
            "drawn disjointly from the spot-check sample; makes a LIVE call per record "
            "-- see judge_agreement.py"
        ),
    )
    p.add_argument(
        "--spotcheck-csv", default=None, metavar="EXPORTED.csv",
        help=(
            "the spot-check CSV --export-spotcheck actually wrote; --test-retest excludes "
            "exactly the items in it. Without this the exclusion set is the DEFAULT "
            "selection regenerated at cell_target=25, which is only the right set if the "
            "export ran at --export-spotcheck 200 over the same results"
        ),
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Read before the --agreement branch, which also reads verdicts from this root.
    results_root = Path(args.results_root)

    # --agreement only needs the coded CSV and the saved verdicts, so it runs first.
    if args.agreement:
        from grant_eval.scoring import judge_agreement

        return judge_agreement.cli_agreement(
            Path(args.agreement), results_root=results_root, json_output=args.json,
        )

    gt = load_ground_truth(args.ground_truth)
    paths = expand_result_paths(args.results)
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1

    registry = load_registry(DEFAULT_REGISTRY_PATH)
    snapshot_root = Path(args.snapshots)
    # The fixed judge settings for every pass.
    judge_config = JudgeConfig(
        model=JUDGE_MODEL, temperature=0.0,
        prompt_version=PROMPT_VERSION, rubric_version=RUBRIC_VERSION,
    )

    # `is not None` so `--export-spotcheck 0` never falls through to a live paid pass.
    if args.export_spotcheck is not None:
        from grant_eval.scoring import judge_agreement

        return judge_agreement.cli_export_spotcheck(
            paths, ground_truth=gt, registry=registry, snapshot_root=snapshot_root,
            results_root=results_root, judge_config_hash=judge_config.config_hash,
            n_total=args.export_spotcheck, out_path=Path(args.out) if args.out else None,
            json_output=args.json,
        )

    # --dry-run must be checked before --test-retest, which makes live paid calls.
    # --export-spotcheck comes earlier because it only reads saved verdicts.
    if args.dry_run:
        if args.test_retest is not None:
            print(
                "--dry-run: --test-retest suppressed (it makes a LIVE call per record). "
                "The figures below are for a full judging pass over these paths, NOT for "
                "the test-retest sample.",
                file=sys.stderr,
            )
        report = dry_run_batch(
            paths, ground_truth=gt, registry=registry, snapshot_root=snapshot_root,
            judge_config=judge_config,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"dry run: {report['n_calls']} judge call(s) would be made")
            print(
                f"estimated cost: ${report['estimated_cost_usd']:.4f}  "
                "(ESTIMATE -- character-based, not tokenised; a real pass reads spend "
                "off the API response instead, see run_judge_batch)"
            )
        return 0

    if args.test_retest is not None:
        from grant_eval.scoring import judge_agreement

        backend = OpenRouterJudgeBackend()  # live, paid calls
        return judge_agreement.cli_test_retest(
            paths, ground_truth=gt, registry=registry, snapshot_root=snapshot_root,
            results_root=results_root, judge_config_hash=judge_config.config_hash,
            backend=backend, judge_config=judge_config,
            n=args.test_retest, max_spend_usd=args.max_spend_usd, json_output=args.json,
            spotcheck_csv=Path(args.spotcheck_csv) if args.spotcheck_csv else None,
        )

    backend = OpenRouterJudgeBackend()
    cache = None if args.no_cache else VerdictCache(root=results_root)
    batch = run_judge_batch(
        paths, ground_truth=gt, registry=registry, snapshot_root=snapshot_root,
        backend=backend, judge_config=judge_config, workers=args.workers,
        max_spend_usd=args.max_spend_usd, cache=cache, results_root=results_root,
    )
    for skipped_path, msg in batch.errors:
        print(f"skipped {skipped_path}: {msg}", file=sys.stderr)

    # The roll-up and the decision to write it both depend on what is on disk, not on
    # this invocation's outcomes, which can miss runs after an interrupt.
    corpus_verdicts, unreadable_verdicts = load_corpus_verdicts(
        judge_config.config_hash, root=results_root
    )
    for bad in unreadable_verdicts:
        print(f"skipped unreadable verdict file {bad['path']}: {bad['reason']}", file=sys.stderr)
    rollup = build_rollup(corpus_verdicts)
    rollup_out_path = rollup_path(judge_config.config_hash, root=results_root)
    rollup_written = bool(corpus_verdicts)
    if rollup_written:
        write_verdict_file(rollup_out_path, rollup)

    n_resumed = sum(1 for o in batch.outcomes if o.resumed)
    if args.json:
        print(json.dumps(
            {
                "rollup": rollup, "n_judged": len(batch.outcomes), "n_resumed": n_resumed,
                "n_skipped_spend_cap": len(batch.skipped_for_spend_cap),
                #: Cancelled runs are not on disk; the two causes are kept apart.
                "n_cancelled_after_fatal": len(batch.cancelled_after_fatal),
                "cancelled_after_fatal": [str(p) for p in batch.cancelled_after_fatal],
                "interrupted_by": batch.interrupted_by,
                "n_cancelled_after_interrupt": len(batch.cancelled_after_interrupt),
                "cancelled_after_interrupt": [str(p) for p in batch.cancelled_after_interrupt],
                "total_spend_usd": batch.total_spend_usd,
                #: Unreadable files are missing from the roll-up, so list them.
                "n_unreadable_verdict_files": len(unreadable_verdicts),
                "unreadable_verdict_files": unreadable_verdicts,
                #: Whether this pass rewrote the roll-up file.
                "rollup_written": rollup_written,
                "rollup_path": str(rollup_out_path),
                "n_runs_in_rollup": rollup["n_runs"],
            },
            indent=2,
        ))
    else:
        print(f"judged {len(batch.outcomes)} run(s) ({n_resumed} resumed from disk)")
        if batch.cancelled_after_fatal:
            print(
                f"fatal worker error: {len(batch.cancelled_after_fatal)} run(s) cancelled "
                "and never judged --",
                file=sys.stderr,
            )
            for cancelled_path in batch.cancelled_after_fatal:
                print(f"  cancelled {cancelled_path}", file=sys.stderr)
        if batch.interrupted_by:
            print(
                f"interrupted ({batch.interrupted_by}): "
                f"{len(batch.cancelled_after_interrupt)} run(s) cancelled and never judged. "
                "What was judged before the interrupt is on disk and in the roll-up below; "
                "re-running resumes from there.",
                file=sys.stderr,
            )
            for cancelled_path in batch.cancelled_after_interrupt:
                print(f"  cancelled {cancelled_path}", file=sys.stderr)
        if batch.skipped_for_spend_cap:
            print(
                f"spend cap ${args.max_spend_usd:.2f} reached: "
                f"{len(batch.skipped_for_spend_cap)} run(s) not judged"
            )
        print(f"total spend: ${batch.total_spend_usd:.4f}")
        if rollup_written:
            print(
                f"roll-up written to {rollup_out_path} "
                f"({rollup['n_runs']} run(s) on disk for this judge config)"
            )
        else:
            print(
                f"no verdict on disk for this judge config: no roll-up written to "
                f"{rollup_out_path}",
                file=sys.stderr,
            )
        # Explain when the in-memory count and the on-disk count disagree.
        if rollup_written and not batch.outcomes:
            print(
                f"note: this invocation recorded 0 outcome(s) in memory but found "
                f"{rollup['n_runs']} verdict file(s) on disk for this judge config. An "
                "interrupt or a fatal error can land after a run is written but before "
                "it is recorded; the roll-up is built from disk, so those runs ARE "
                "included in it.",
                file=sys.stderr,
            )
    # Exit 1 if the pass was cut short (fatal error or interrupt) or if every run
    # failed. Some failed runs alongside successes still exit 0, like score.py.
    if batch.cancelled_after_fatal or batch.interrupted_by:
        return 1
    if batch.errors and not batch.outcomes:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
