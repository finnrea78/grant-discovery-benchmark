"""Tests for `scoring/judge.py`, the LLM judge for completeness and groundedness.

Everything runs against `FakeJudgeBackend` or pure functions. The one test that builds
`OpenRouterJudgeBackend` mocks `litellm.completion`, so nothing touches the network.
"""
from __future__ import annotations

import csv
import random
from datetime import date

import pydantic
import pytest

import grant_eval.scoring.judge_agreement as ja
from grant_eval import config
from grant_eval.scoring.judge import (
    JUDGE_MODEL,
    Claim,
    FakeJudgeBackend,
    JudgeConfig,
    JudgeOutcome,
    JudgeVerdict,
    OpenRouterJudgeBackend,
    _judge_router_backend_class,
    build_completeness_prompt,
    build_groundedness_prompt,
    run_judge,
)

SNAPSHOT_DATE = date(2026, 8, 9)

#: A made-up funder name that can only reach a prompt through the evidence text below,
#: not through anything the prompt builders know.
FUNDER_SENTINEL = "The Zzyxquavian Endowment for Applied Cartography"


def _judge_config(**overrides) -> JudgeConfig:
    base = dict(
        model="openrouter/google/gemini-2.5-flash-lite",
        temperature=0.0,
        prompt_version="v1",
        rubric_version="v1",
    )
    base.update(overrides)
    return JudgeConfig(**base)


# Claim / JudgeVerdict schema
def test_claim_rejects_a_claim_type_outside_the_shared_vocabulary():
    with pytest.raises(pydantic.ValidationError):
        Claim(text="funds early-career researchers", claim_type="funding_scope", supported=True)


def test_judge_verdict_rejects_an_unjudged_field():
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdict(
            field="discipline_tags",
            dimension="groundedness",
            claims=[],
            record_supported=True,
            justification="x",
        )


def test_judge_verdict_rejects_an_unknown_dimension():
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdict(
            field="summary",
            dimension="plausibility",
            claims=[],
            record_supported=True,
            justification="x",
        )


def test_judge_verdict_rejects_an_overlong_justification():
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdict(
            field="summary",
            dimension="groundedness",
            claims=[],
            record_supported=True,
            justification="x" * 301,
        )


# JudgeConfig.config_hash
def test_judge_config_hash_moves_when_prompt_version_changes():
    a = _judge_config(prompt_version="v1")
    b = _judge_config(prompt_version="v2")
    assert a.config_hash != b.config_hash


def test_judge_config_hash_is_stable_for_identical_config():
    a = _judge_config()
    b = _judge_config()
    assert a.config_hash == b.config_hash


# Prompt builders
def test_groundedness_prompt_carries_page_text_and_not_gold_text():
    prompt = build_groundedness_prompt(
        field="summary",
        extracted_text="Funds early-career researchers.",
        page_text="This page states the PAGE-TEXT-SENTINEL award details.",
        snapshot_date=SNAPSHOT_DATE,
    )
    assert "PAGE-TEXT-SENTINEL" in prompt.user
    # there is no gold_text parameter, but check the negative directly anyway
    assert "GOLD-TEXT-SENTINEL-NEVER-PASSED" not in prompt.user


def test_completeness_prompt_carries_gold_text_and_not_page_text():
    prompt = build_completeness_prompt(
        field="summary",
        extracted_text="Funds early-career researchers.",
        gold_text="The gold record states the GOLD-TEXT-SENTINEL award details.",
        snapshot_date=SNAPSHOT_DATE,
    )
    assert "GOLD-TEXT-SENTINEL" in prompt.user
    assert "PAGE-TEXT-SENTINEL-NEVER-PASSED" not in prompt.user


@pytest.mark.parametrize("builder", [build_groundedness_prompt, build_completeness_prompt])
def test_prompts_carry_the_snapshot_date(builder):
    kwargs = dict(field="summary", extracted_text="x", snapshot_date=SNAPSHOT_DATE)
    kwargs["page_text" if builder is build_groundedness_prompt else "gold_text"] = "y"
    prompt = builder(**kwargs)
    assert "2026-08-09" in prompt.system


@pytest.mark.parametrize("builder", [build_groundedness_prompt, build_completeness_prompt])
def test_prompts_carry_the_paraphrase_tolerance_instruction(builder):
    kwargs = dict(field="summary", extracted_text="x", snapshot_date=SNAPSHOT_DATE)
    kwargs["page_text" if builder is build_groundedness_prompt else "gold_text"] = "y"
    prompt = builder(**kwargs)
    assert "wording" in prompt.system.lower()
    assert "substance" in prompt.system.lower()


@pytest.mark.parametrize("builder", [build_groundedness_prompt, build_completeness_prompt])
def test_prompts_never_reveal_identity_cues_outside_the_evidence(builder):
    """No prompt names the model, strategy, temperature or replicate.

    The funder name can legitimately appear in the evidence, so it is only checked to be
    absent from the prompt's own scaffold."""
    kwargs = dict(
        field="summary",
        extracted_text=f"Funded by {FUNDER_SENTINEL}, this scheme supports researchers.",
        snapshot_date=SNAPSHOT_DATE,
    )
    evidence_text = f"{FUNDER_SENTINEL} awards grants to early-career researchers."
    kwargs["page_text" if builder is build_groundedness_prompt else "gold_text"] = evidence_text
    prompt = builder(**kwargs)

    full_text = prompt.system + "\n" + prompt.user
    for banned in (
        "openrouter/deepseek/deepseek-v4-flash-0731",
        "openrouter/openai/gpt-oss-120b",
        "openrouter/amazon/nova-lite-v1",
        "chain_of_thought",
        "few_shot",
        "structured",
        "0.7",  # a TEMPERATURES value
        "replicate",
    ):
        assert banned not in full_text, f"identity cue {banned!r} leaked into the prompt"

    assert FUNDER_SENTINEL in prompt.user
    assert FUNDER_SENTINEL not in prompt.scaffold


# run_judge outcome classification, all against FakeJudgeBackend
WELL_FORMED_PROMPT = build_groundedness_prompt(
    field="summary", extracted_text="Funds early-career researchers by 2027-06-30.",
    page_text="Awards go to early-career researchers; deadline 30 June 2027.",
    snapshot_date=SNAPSHOT_DATE,
)


def _call(backend, prompt=WELL_FORMED_PROMPT, field="summary", dimension="groundedness"):
    return run_judge(
        backend=backend,
        judge_config=_judge_config(),
        prompt=prompt,
        field=field,
        dimension=dimension,
    )


def test_well_formed_response_parses_and_derives_record_supported():
    backend = FakeJudgeBackend(
        [
            '{"claims": ['
            '{"text": "funds early-career researchers", "claim_type": "eligibility", "supported": true}, '
            '{"text": "deadline 2027-06-30", "claim_type": "deadline", "supported": true}'
            '], "justification": "Both claims are supported by the page."}'
        ]
    )
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK
    assert result.verdict is not None
    assert result.verdict.record_supported is True
    assert len(result.verdict.claims) == 2


def test_model_supplied_record_supported_is_ignored_in_favour_of_the_derived_value():
    """A model-supplied record_supported is ignored; the value is derived from the claims."""
    backend = FakeJudgeBackend(
        [
            '{"claims": ['
            '{"text": "funds early-career researchers", "claim_type": "eligibility", "supported": true}, '
            '{"text": "deadline 2027-06-30", "claim_type": "deadline", "supported": false}'
            '], "justification": "One claim is unsupported.", "record_supported": true}'
        ]
    )
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK
    assert result.verdict.record_supported is False


def test_malformed_response_yields_judge_parse_failure_not_a_false_verdict():
    backend = FakeJudgeBackend(["this is not JSON at all"])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.PARSE_FAILURE
    assert result.verdict is None


def test_valid_json_with_the_wrong_shape_also_yields_judge_parse_failure():
    """Valid JSON with the wrong shape (no `claims`) is also a parse failure."""
    backend = FakeJudgeBackend(['{"justification": "no claims key at all"}'])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.PARSE_FAILURE
    assert result.verdict is None


def test_call_failure_after_retries_yields_judge_call_failure():
    backend = FakeJudgeBackend([TimeoutError("upstream timed out after retries")])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.CALL_FAILURE
    assert result.verdict is None


def test_call_failure_and_parse_failure_are_never_the_same_outcome():
    parse_result = _call(FakeJudgeBackend(["not JSON"]))
    call_result = _call(FakeJudgeBackend([RuntimeError("429")]))
    assert parse_result.outcome != call_result.outcome


# run_judge with markdown-fenced replies. Gemini 2.5 Flash Lite wraps its JSON in a
# ```json fence even when asked for bare JSON, so fenced replies have to parse, but
# malformed content inside a fence must still fail.
_OK_FENCE_BODY = (
    '{"claims": ['
    '{"text": "funds early-career researchers", "claim_type": "eligibility", "supported": true}'
    '], "justification": "The claim is supported by the page."}'
)


def _fenced(body: str, lang: str = "json") -> str:
    return f"```{lang}\n{body}\n```"


def test_fenced_reply_with_json_language_tag_still_parses():
    backend = FakeJudgeBackend([_fenced(_OK_FENCE_BODY, lang="json")])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK
    assert len(result.verdict.claims) == 1


def test_fenced_reply_with_bare_fence_still_parses():
    backend = FakeJudgeBackend([_fenced(_OK_FENCE_BODY, lang="")])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK


def test_fenced_reply_with_surrounding_whitespace_still_parses():
    backend = FakeJudgeBackend([f"  \n{_fenced(_OK_FENCE_BODY)}\n  "])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK


def test_unfenced_reply_still_parses_alongside_the_fence_fix():
    # Also covered above, repeated here next to the fence tests.
    backend = FakeJudgeBackend([_OK_FENCE_BODY])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.OK


def test_non_json_content_inside_a_fence_still_yields_parse_failure():
    """A fenced reply that isn't valid JSON is still a parse failure."""
    backend = FakeJudgeBackend([_fenced("this is not JSON at all")])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.PARSE_FAILURE
    assert result.verdict is None


def test_fenced_valid_json_failing_schema_still_yields_parse_failure():
    """A fenced reply with the wrong shape (no `claims`) is still a parse failure."""
    backend = FakeJudgeBackend([_fenced('{"justification": "no claims key at all"}')])
    result = _call(backend)
    assert result.outcome is JudgeOutcome.PARSE_FAILURE
    assert result.verdict is None


# _strip_markdown_fence on its own
from grant_eval.scoring.judge import _strip_markdown_fence  # noqa: E402


def test_strip_markdown_fence_removes_a_json_tagged_fence():
    assert _strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_markdown_fence_removes_a_bare_fence():
    assert _strip_markdown_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_markdown_fence_leaves_unfenced_text_unchanged():
    assert _strip_markdown_fence('{"a": 1}') == '{"a": 1}'


def test_strip_markdown_fence_strips_outer_whitespace_around_the_fence():
    assert _strip_markdown_fence('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'


def test_strip_markdown_fence_tolerates_a_missing_closing_fence():
    # No closing fence: only the opening line is dropped, and run_judge decides
    # whether the rest parses.
    assert _strip_markdown_fence('```json\n{"a": 1}') == '{"a": 1}'


def test_zero_claims_on_a_nonempty_field_yields_no_claims_found_not_a_100_percent_rate():
    backend = FakeJudgeBackend(
        ['{"claims": [], "justification": "The text asserts nothing checkable."}']
    )
    result = _call(backend)
    assert result.outcome is JudgeOutcome.NO_CLAIMS_FOUND
    assert result.verdict is None


def test_no_network_call_is_ever_attempted(monkeypatch):
    """A `run_judge` call against `FakeJudgeBackend` must never reach `litellm`."""
    import grant_eval.agent.router as router

    def _forbidden(**kwargs):
        raise AssertionError("network call attempted via litellm.completion")

    monkeypatch.setattr(router.litellm, "completion", _forbidden)

    backend = FakeJudgeBackend(
        ['{"claims": [], "justification": "The text asserts nothing checkable."}']
    )
    result = _call(backend)
    assert result.outcome is JudgeOutcome.NO_CLAIMS_FOUND


# OpenRouterJudgeBackend routing. litellm.completion is monkeypatched (as in
# tests/test_router.py) so the outgoing kwargs can be checked without a request.
class _FakeJudgeMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


class _FakeJudgeChoice:
    def __init__(self, message):
        self.message = message


class _FakeJudgeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeJudgeResponse:
    def __init__(self, content):
        self.choices = [_FakeJudgeChoice(_FakeJudgeMessage(content))]
        self.usage = _FakeJudgeUsage()
        self._hidden_params = {}


def test_openrouter_judge_backend_pins_the_google_ai_studio_routing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeJudgeResponse('{"claims": [], "justification": "x"}')

    monkeypatch.setattr(router.litellm, "completion", fake_completion)
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0)

    backend = OpenRouterJudgeBackend()
    text = backend.complete(system="s", user="u", model=JUDGE_MODEL, temperature=0.0)
    assert text == '{"claims": [], "justification": "x"}'
    assert captured["extra_body"]["provider"] == config.MODEL_ROUTING["gemini-judge"]["provider"]


def test_judge_router_backend_refuses_a_model_it_is_not_pinned_to():
    """`_routing_for` returns the pin for the model it is given, not always the judge's pin."""
    router_cls = _judge_router_backend_class()
    with pytest.raises(ValueError, match="gemini-2.5-flash-lite"):
        router_cls._routing_for("openrouter/deepseek/deepseek-v4-flash-0731")


# MODEL_ROUTING entry for the judge, and its effect on config_hash
def test_model_routing_pins_the_judge_to_google_ai_studio():
    """The judge model gets a `provider.only` pin with `allow_fallbacks: False`, like nova-lite."""
    routing = config.MODEL_ROUTING["gemini-judge"]
    assert routing["provider"]["only"] == ["google-ai-studio"]
    assert routing["provider"]["allow_fallbacks"] is False


def test_model_routing_addition_does_not_move_config_hash(monkeypatch):
    """Adding the judge's routing entry doesn't change `extract.config_hash`.

    Otherwise every existing run output path would move."""
    from grant_eval.agent import extract

    kwargs = dict(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        agent_cfg=config.AGENT_CONFIGS["standard"],
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    with_entry = extract.config_hash(**kwargs)

    without = dict(config.MODEL_ROUTING)
    del without["gemini-judge"]
    monkeypatch.setattr(config, "MODEL_ROUTING", without)
    without_entry = extract.config_hash(**kwargs)

    assert with_entry == without_entry


# Populations, rates and exclusions
from grant_eval.scoring.judge import (  # noqa: E402
    EXCLUSION_REASONS,
    FIELD_EMPTY,
    UNRESOLVABLE_SOURCE,
    FieldJudgement,
    RecordJudgement,
)


def _verdict(field="summary", dimension="groundedness", supported_pattern=(True,)) -> JudgeVerdict:
    claims = [
        Claim(text=f"claim {i}", claim_type="purpose", supported=s)
        for i, s in enumerate(supported_pattern)
    ]
    return JudgeVerdict(
        field=field, dimension=dimension, claims=claims,
        record_supported=all(supported_pattern), justification="j",
    )


def _record(oid: str, verdict: JudgeVerdict | None, outcome: str = "ok") -> RecordJudgement:
    return RecordJudgement(
        opportunity_id=oid, field="summary", dimension="groundedness",
        outcome=outcome, verdict=verdict,
    )


# FieldJudgement: the exclusion ledger always names all five reasons
def test_field_judgement_starts_with_every_exclusion_reason_named_at_zero():
    fj = FieldJudgement(field="summary", dimension="groundedness")
    assert fj.exclusions == {r: 0 for r in EXCLUSION_REASONS}
    assert set(EXCLUSION_REASONS) == {
        FIELD_EMPTY, UNRESOLVABLE_SOURCE, "no_claims_found",
        "judge_parse_failure", "judge_call_failure",
    }


# claim_rate (micro) and record_rate: independent, and undefined rather than 0
def test_claim_rate_and_record_rate_are_undefined_with_no_records_judged():
    fj = FieldJudgement(field="summary", dimension="groundedness")
    assert fj.claim_rate_undefined is True
    assert fj.claim_rate == 0.0
    assert fj.record_rate_undefined is True
    assert fj.record_rate == 0.0
    assert fj.claim_rate_macro_undefined is True
    assert fj.claim_rate_macro == 0.0


def test_one_record_with_four_of_five_claims_supported_gives_08_claim_rate_00_record_rate():
    """One unsupported claim out of five makes the record unsupported, but the claim rate is 4/5."""
    v = _verdict(supported_pattern=(True, True, True, True, False))
    fj = FieldJudgement(
        field="summary", dimension="groundedness",
        n_claims=5, n_claims_supported=4, n_records_judged=1, n_records_supported=0,
        records=[_record("r1", v)],
    )
    assert fj.claim_rate_undefined is False
    assert fj.claim_rate == 0.8
    assert fj.record_rate_undefined is False
    assert fj.record_rate == 0.0


# Micro vs macro claim rate
def test_micro_and_macro_claim_rates_differ_on_unequal_claim_counts():
    """Micro pools all claims (2/10) while macro averages per-record rates ((1.0 + 1/9) / 2)."""
    a = _verdict(supported_pattern=(True,))
    b = _verdict(supported_pattern=(True,) + (False,) * 8)
    fj = FieldJudgement(
        field="summary", dimension="groundedness",
        n_claims=10, n_claims_supported=2, n_records_judged=2, n_records_supported=0,
        records=[_record("a", a), _record("b", b)],
    )
    assert fj.claim_rate == pytest.approx(0.2)
    assert fj.claim_rate_macro == pytest.approx((1.0 + 1 / 9) / 2)
    assert fj.claim_rate != fj.claim_rate_macro


# Mirror resolution: funder display name vs slug, HTML cleaning, PDFs
import json  # noqa: E402

from grant_eval.dataset.registry import FunderRecord  # noqa: E402
from grant_eval.schema import Split  # noqa: E402
from grant_eval.scoring.judge import funder_slug, resolve_mirror_text  # noqa: E402

DISPLAY_NAME = "The Leverhulme Trust"
SLUG = "leverhulme"
SNAP_DATE = date(2026, 8, 9)


def _registry() -> dict[str, FunderRecord]:
    return {
        SLUG: FunderRecord(
            slug=SLUG, display_name=DISPLAY_NAME, split=Split.TUNE,
            snapshot_date=SNAP_DATE, entry_url="https://example.org/",
            content_hash="deadbeef",
        )
    }


def _build_snapshot(tmp_path, *, slug=SLUG, snapshot_date=SNAP_DATE, page_html=None, pdf_bytes=None):
    """Write a minimal snapshot directory: a manifest plus an optional HTML page and/or PDF."""
    snap_dir = tmp_path / "snapshots" / slug / snapshot_date.isoformat()
    files = []
    if page_html is not None:
        rel = "site/example.org/scheme.html"
        (snap_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap_dir / rel).write_text(page_html, encoding="utf-8")
        files.append({"url": "https://example.org/scheme", "path": rel, "kind": "page"})
    if pdf_bytes is not None:
        rel = "site/example.org/guidelines.pdf"
        (snap_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap_dir / rel).write_bytes(pdf_bytes)
        files.append({"url": "https://example.org/guidelines.pdf", "path": rel, "kind": "page"})
    (snap_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (snap_dir / "manifest.json").write_text(json.dumps({"content_hash": "x", "files": files}))
    return tmp_path / "snapshots"


def test_funder_slug_resolves_the_display_name_to_the_registry_slug():
    assert funder_slug(DISPLAY_NAME, _registry()) == SLUG


def test_funder_slug_returns_none_for_a_name_the_registry_does_not_carry():
    assert funder_slug("Not A Real Funder", _registry()) is None


def test_resolve_mirror_text_honours_the_display_name_not_the_slug(tmp_path):
    """Mirror lookup takes the funder's display name; a slug passed as the name finds nothing."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers with a doctorate.</p></body></html>")
    text = resolve_mirror_text(
        DISPLAY_NAME, SNAP_DATE, "https://example.org/scheme",
        registry=_registry(), snapshot_root=root,
    )
    assert text is not None
    assert "early career researchers" in text

    assert resolve_mirror_text(
        SLUG, SNAP_DATE, "https://example.org/scheme",
        registry=_registry(), snapshot_root=root,
    ) is None


def test_resolve_mirror_text_returns_none_for_a_url_never_captured(tmp_path):
    root = _build_snapshot(tmp_path, page_html="<html><body><p>x</p></body></html>")
    assert resolve_mirror_text(
        DISPLAY_NAME, SNAP_DATE, "https://example.org/never-crawled",
        registry=_registry(), snapshot_root=root,
    ) is None


def test_resolve_mirror_text_reads_a_pdf_via_the_pdf_path(tmp_path):
    """A PDF-backed URL is read through the PDF path rather than skipped."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Applications close on 30 June 2027 for early-career grants.")
    pdf_bytes = doc.tobytes()
    doc.close()

    root = _build_snapshot(tmp_path, pdf_bytes=pdf_bytes)
    text = resolve_mirror_text(
        DISPLAY_NAME, SNAP_DATE, "https://example.org/guidelines.pdf",
        registry=_registry(), snapshot_root=root,
    )
    assert text is not None
    assert "30 June 2027" in text


def test_resolve_mirror_text_falls_back_to_raw_html_on_a_sparse_page(tmp_path):
    """An empty trafilatura extraction falls back to the raw HTML, as `fetch_page` does.

    Otherwise the judge would be shown an empty page and mark well-grounded records as
    unsupported."""
    sparse_html = (
        "<html><head><title>Deadline info</title></head>"
        "<body><!-- Applications close 30 June 2027 --></body></html>"
    )
    import trafilatura as _trafilatura

    assert _trafilatura.extract(sparse_html, include_tables=True, favor_recall=True) is None

    root = _build_snapshot(tmp_path, page_html=sparse_html)
    text = resolve_mirror_text(
        DISPLAY_NAME, SNAP_DATE, "https://example.org/scheme",
        registry=_registry(), snapshot_root=root,
    )
    assert text is not None
    assert text != ""
    assert "Applications close 30 June 2027" in text


# score_run_judge: the two populations, exclusions, identity carry-through
from pathlib import Path  # noqa: E402

from grant_eval.schema import ExtractedOpportunity, ExtractionResult, GroundTruthOpportunity  # noqa: E402
from grant_eval.scoring import score as scoremod  # noqa: E402
from grant_eval.scoring.groundtruth import FunderGroundTruth, GroundTruth  # noqa: E402
from grant_eval.scoring.identity import build_key_index, normalise_funder  # noqa: E402
from grant_eval.scoring.judge import score_run_judge, to_dict  # noqa: E402

LEV = "The Leverhulme Trust"


def _ground_truth(funder: str, rows: list[GroundTruthOpportunity]) -> GroundTruth:
    by_key, shared = build_key_index(rows, funder)
    fgt = FunderGroundTruth(funder=funder, rows=rows, by_key=by_key, shared_urls=shared)
    return GroundTruth(funders={normalise_funder(funder): fgt})


def _gt_row(funder, oid, title, url, **kw) -> GroundTruthOpportunity:
    return GroundTruthOpportunity(funder=funder, opportunity_id=oid, title=title, url=url, **kw)


def _judge_result(funder, opportunities, **kw) -> ExtractionResult:
    base = dict(
        snapshot_date=SNAP_DATE, funder=funder, split=Split.TUNE,
        model="test/model", prompt_strategy="minimal", temperature=0.0,
        agent_config="standard", replicate=1, config_hash="test",
        opportunities=opportunities,
    )
    base.update(kw)
    return ExtractionResult(**base)


def _judge_extracted(funder, oid, title, url, summary=None, eligibility_summary=None) -> ExtractedOpportunity:
    return ExtractedOpportunity(
        funder=funder, opportunity_id=oid, title=title, url=url,
        summary=summary, eligibility_summary=eligibility_summary,
    )


def _ok_reply(supported: bool = True) -> str:
    return json.dumps(
        {"claims": [{"text": "c", "claim_type": "purpose", "supported": supported}], "justification": "j"}
    )


def test_completeness_skips_unmatched_records_groundedness_includes_them(monkeypatch):
    """Completeness only sees true positives; groundedness sees every record with a resolvable
    URL, false positives included."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))

    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    fp_opp = _judge_extracted(LEV, "g2", "Invented", "https://example.org/invented", summary="Funds y")
    result = _judge_result(LEV, [tp_opp, fp_opp])
    run_score = scoremod.score_run(result, gt)
    assert (run_score.detection.tp, run_score.detection.fp) == (1, 1)  # sanity check on the fixture

    backend = FakeJudgeBackend([_ok_reply()] * 10)
    out = score_run_judge(
        result=result, run_score=run_score, backend=backend, judge_config=_judge_config(),
        registry=_registry(),
    )
    completeness = out.fields[("summary", "completeness")]
    groundedness = out.fields[("summary", "groundedness")]
    assert completeness.n_records_judged == 1
    assert {r.opportunity_id for r in completeness.records if r.outcome == "ok"} == {"g1"}
    assert groundedness.n_records_judged == 2
    assert {r.opportunity_id for r in groundedness.records if r.outcome == "ok"} == {"g1", "g2"}


def test_unresolvable_url_leaves_both_denominators_and_is_counted_once(monkeypatch):
    """`unresolvable_source` only appears in groundedness's ledger, never completeness's."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: None)

    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    assert run_score.detection.tp == 1

    backend = FakeJudgeBackend([_ok_reply()])
    out = score_run_judge(
        result=result, run_score=run_score, backend=backend, judge_config=_judge_config(),
        registry=_registry(),
    )
    completeness = out.fields[("summary", "completeness")]
    groundedness = out.fields[("summary", "groundedness")]
    assert completeness.n_records_judged == 1
    assert completeness.exclusions[UNRESOLVABLE_SOURCE] == 0
    assert groundedness.n_records_judged == 0
    assert groundedness.exclusions[UNRESOLVABLE_SOURCE] == 1


def test_empty_field_is_field_empty_not_an_unsupported_verdict(monkeypatch):
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))

    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary=None)
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)

    backend = FakeJudgeBackend([])  # abstentions make no calls
    out = score_run_judge(
        result=result, run_score=run_score, backend=backend, judge_config=_judge_config(),
        registry=_registry(),
    )
    completeness = out.fields[("summary", "completeness")]
    groundedness = out.fields[("summary", "groundedness")]
    assert completeness.exclusions[FIELD_EMPTY] == 1
    assert completeness.n_records_judged == 0
    assert groundedness.exclusions[FIELD_EMPTY] == 1
    assert groundedness.n_records_judged == 0
    assert backend.calls == []


def test_parse_and_call_failures_are_counted_separately_and_leave_the_denominator():
    """With no snapshot root, groundedness resolves nothing, leaving just the completeness call."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    nowhere = Path("/nonexistent-snapshot-root-for-test")

    parse_failed = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend(["not json"]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    assert parse_failed.exclusions["judge_parse_failure"] == 1
    assert parse_failed.exclusions["judge_call_failure"] == 0
    assert parse_failed.n_records_judged == 0

    call_failed = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([RuntimeError("boom")]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    assert call_failed.exclusions["judge_call_failure"] == 1
    assert call_failed.exclusions["judge_parse_failure"] == 0
    assert call_failed.n_records_judged == 0


def test_parse_and_call_failure_records_carry_the_error_message():
    """A parse or call failure keeps its exception text on the record."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    nowhere = Path("/nonexistent-snapshot-root-for-test")

    parse_failed = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend(["not json"]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    parse_rec = parse_failed.records[0]
    assert parse_rec.outcome == "judge_parse_failure"
    assert parse_rec.error  # the JSONDecodeError text

    call_failed = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([RuntimeError("boom")]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    call_rec = call_failed.records[0]
    assert call_rec.outcome == "judge_call_failure"
    assert call_rec.error == "boom"


def test_ok_and_no_claims_found_records_carry_no_error():
    """Only the two failure outcomes carry an `error`."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    nowhere = Path("/nonexistent-snapshot-root-for-test")

    ok = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([_ok_reply()]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    assert ok.records[0].outcome == "ok"
    assert ok.records[0].error is None

    no_claims = score_run_judge(
        result=result, run_score=run_score,
        backend=FakeJudgeBackend(['{"claims": [], "justification": "nothing checkable"}']),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    assert no_claims.records[0].outcome == "no_claims_found"
    assert no_claims.records[0].error is None


def test_persisted_error_is_truncated_to_500_characters():
    """A huge error message is capped, keeping a real prefix."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    nowhere = Path("/nonexistent-snapshot-root-for-test")
    huge_message = "x" * 5000

    call_failed = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([RuntimeError(huge_message)]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    ).fields[("summary", "completeness")]
    error = call_failed.records[0].error
    assert error is not None
    assert len(error) <= 500
    assert huge_message.startswith(error)  # a prefix of the original


def test_rates_are_never_pooled_across_fields_or_dimensions(monkeypatch):
    """Each field and dimension gets its own FieldJudgement, so verdicts don't leak between them.

    Call order is completeness (summary, eligibility_summary) then groundedness, matching
    `score_run_judge`."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))

    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(
        LEV, "g1", "Scheme One", "https://example.org/scheme-one",
        summary="Funds x", eligibility_summary="Only UK residents",
    )
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)

    replies = [_ok_reply(True), _ok_reply(False), _ok_reply(True), _ok_reply(False)]
    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend(replies),
        judge_config=_judge_config(), registry=_registry(),
    )
    assert out.fields[("summary", "completeness")].claim_rate == 1.0
    assert out.fields[("eligibility_summary", "completeness")].claim_rate == 0.0
    assert out.fields[("summary", "groundedness")].claim_rate == 1.0
    assert out.fields[("eligibility_summary", "groundedness")].claim_rate == 0.0


def test_run_judge_score_carries_schema_valid_and_abstained_flags():
    gt = _ground_truth(LEV, [])
    result = _judge_result(LEV, [], schema_valid=False, abstained=True)
    run_score = scoremod.score_run(result, gt)
    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([]),
        judge_config=_judge_config(), registry=_registry(),
    )
    assert out.schema_valid is False
    assert out.abstained is True


def test_mirror_lookup_is_built_once_per_run_not_once_per_opportunity(monkeypatch, tmp_path):
    """`build_mirror_lookup` is called once per run, not once per opportunity."""
    import grant_eval.scoring.judge as judgemod

    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds researchers.</p></body></html>")
    calls = []
    real_build = judgemod.build_mirror_lookup

    def counting_build(*a, **kw):
        calls.append(1)
        return real_build(*a, **kw)

    monkeypatch.setattr(judgemod, "build_mirror_lookup", counting_build)

    gt = _ground_truth(LEV, [])
    opp1 = _judge_extracted(LEV, "g1", "One", "https://example.org/scheme", summary="x")
    opp2 = _judge_extracted(LEV, "g2", "Two", "https://example.org/scheme", summary="y")
    result = _judge_result(LEV, [opp1, opp2])
    run_score = scoremod.score_run(result, gt)

    score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([_ok_reply()] * 10),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=root,
    )
    assert len(calls) == 1


def test_raw_html_fallback_is_counted_not_silently_substituted(tmp_path):
    """A sparse page reaches the judge as raw HTML and is counted in `n_raw_html_fallback`.

    This uses the real filesystem with no mocks."""
    sparse_html = (
        "<html><head><title>x</title></head>"
        "<body><!-- Applications close 30 June 2027 --></body></html>"
    )
    root = _build_snapshot(tmp_path, page_html=sparse_html)

    gt = _ground_truth(LEV, [])
    opp = _judge_extracted(LEV, "g1", "Scheme", "https://example.org/scheme", summary="Funds x")
    result = _judge_result(LEV, [opp])
    run_score = scoremod.score_run(result, gt)

    backend = FakeJudgeBackend([_ok_reply()])
    out = score_run_judge(
        result=result, run_score=run_score, backend=backend, judge_config=_judge_config(),
        registry=_registry(), snapshot_root=root,
    )
    groundedness = out.fields[("summary", "groundedness")]
    assert groundedness.n_raw_html_fallback == 1
    assert groundedness.n_records_judged == 1  # the record was judged, not excluded
    assert "Applications close 30 June 2027" in backend.calls[0]["user"]


# to_dict: undefined rates serialise as null, not 0.0
def test_to_dict_serialises_an_undefined_rate_as_null_not_zero():
    """An empty cell serialises its rates as JSON null and keeps the `*_undefined` flags."""
    gt = _ground_truth(LEV, [])
    result = _judge_result(LEV, [])  # no opportunities, so every cell is empty
    run_score = scoremod.score_run(result, gt)
    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([]),
        judge_config=_judge_config(), registry=_registry(),
    )
    d = to_dict(out)
    cell = d["fields"]["summary:groundedness"]
    assert cell["claim_rate"] is None
    assert cell["claim_rate_undefined"] is True
    assert cell["claim_rate_macro"] is None
    assert cell["claim_rate_macro_undefined"] is True
    assert cell["record_rate"] is None
    assert cell["record_rate_undefined"] is True
    # Round-trip through json rather than just checking `is None`.
    reloaded = json.loads(json.dumps(d))
    assert reloaded["fields"]["summary:groundedness"]["claim_rate"] is None


def test_to_dict_serialises_a_defined_rate_as_a_float_not_null(monkeypatch):
    """A judged cell does not come back as null."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(
        judgemod, "resolve_mirror_page",
        lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False),
    )
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([_ok_reply()] * 4),
        judge_config=_judge_config(), registry=_registry(),
    )
    d = to_dict(out)
    cell = d["fields"]["summary:groundedness"]
    assert cell["claim_rate"] == 1.0
    assert cell["claim_rate_undefined"] is False
    assert isinstance(cell["claim_rate"], float)


def test_to_dict_carries_run_identity_exclusions_and_fallback_count():
    gt = _ground_truth(LEV, [])
    result = _judge_result(LEV, [], schema_valid=True, abstained=False)
    run_score = scoremod.score_run(result, gt)
    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([]),
        judge_config=_judge_config(), registry=_registry(),
    )
    d = to_dict(out)
    assert d["funder"] == LEV
    assert d["schema_valid"] is True
    assert d["abstained"] is False
    assert d["judge_config_hash"] == out.judge_config_hash
    cell = d["fields"]["summary:groundedness"]
    assert cell["exclusions"] == {r: 0 for r in EXCLUSION_REASONS}
    assert cell["n_raw_html_fallback"] == 0


def test_to_dict_carries_the_error_for_a_failure_record_and_null_for_an_ok_one():
    """`error` is written to the persisted record, not just kept in memory."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    nowhere = Path("/nonexistent-snapshot-root-for-test")

    out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend(["not json"]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    )
    d = to_dict(out)
    rec = d["fields"]["summary:completeness"]["records"][0]
    assert rec["outcome"] == "judge_parse_failure"
    assert rec["error"]

    ok_out = score_run_judge(
        result=result, run_score=run_score, backend=FakeJudgeBackend([_ok_reply()]),
        judge_config=_judge_config(), registry=_registry(), snapshot_root=nowhere,
    )
    ok_rec = to_dict(ok_out)["fields"]["summary:completeness"]["records"][0]
    assert ok_rec["outcome"] == "ok"
    assert ok_rec["error"] is None


# Persistence, concurrency, roll-up and CLI

# Persistence: verdict file paths
from grant_eval.scoring.judge import read_verdict_file, verdict_path, write_verdict_file  # noqa: E402


def test_verdict_path_is_keyed_by_both_hashes(tmp_path):
    p = verdict_path("runhash123", "judgehash456", funder_slug="leverhulme", replicate=1, root=tmp_path)
    assert p == tmp_path / "runhash123" / "leverhulme" / "1" / "judgehash456.json"


def test_verdict_path_disambiguates_funder_and_replicate_sharing_a_run_config_hash(tmp_path):
    """`verdict_path` keeps funders and replicates apart, since `run_config_hash` excludes them."""
    p_a = verdict_path("shared-hash", "judgehash", funder_slug="leverhulme", replicate=1, root=tmp_path)
    p_b = verdict_path("shared-hash", "judgehash", funder_slug="henry-moore", replicate=1, root=tmp_path)
    p_c = verdict_path("shared-hash", "judgehash", funder_slug="leverhulme", replicate=2, root=tmp_path)
    assert len({p_a, p_b, p_c}) == 3


def test_rollup_path_is_keyed_by_judge_config_hash(tmp_path):
    """The roll-up file name includes the judge config hash so a new config can't overwrite it."""
    p1 = judgemod.rollup_path("hash-a", root=tmp_path)
    p2 = judgemod.rollup_path("hash-b", root=tmp_path)
    assert p1 != p2
    assert p1 == tmp_path / "rollup-hash-a.json"


def test_write_then_read_verdict_file_round_trips(tmp_path):
    path = verdict_path("runhash", "judgehash", funder_slug="leverhulme", replicate=1, root=tmp_path)
    write_verdict_file(path, {"a": 1, "b": [1, 2, 3]})
    assert read_verdict_file(path) == {"a": 1, "b": [1, 2, 3]}


def test_a_changed_judge_config_hash_writes_a_new_file_and_leaves_the_old_one_intact(tmp_path):
    old_path = verdict_path("runhash", "judgehash-a", funder_slug="leverhulme", replicate=1, root=tmp_path)
    new_path = verdict_path("runhash", "judgehash-b", funder_slug="leverhulme", replicate=1, root=tmp_path)
    write_verdict_file(old_path, {"judge_config_hash": "judgehash-a"})
    write_verdict_file(new_path, {"judge_config_hash": "judgehash-b"})
    assert read_verdict_file(old_path) == {"judge_config_hash": "judgehash-a"}
    assert read_verdict_file(new_path) == {"judge_config_hash": "judgehash-b"}
    assert old_path.is_file() and new_path.is_file()


# VerdictCache: content-addressed on (judge_config_hash, field, dimension, texts)
from grant_eval.scoring.judge import VerdictCache, verdict_cache_key  # noqa: E402


def test_verdict_cache_key_is_stable_and_sensitive_to_every_component():
    base = dict(judge_config_hash="h", field="summary", dimension="groundedness", extracted_text="e", comparison_text="c")
    assert verdict_cache_key(**base) == verdict_cache_key(**base)
    for changed in (
        dict(judge_config_hash="h2"), dict(field="eligibility_summary"),
        dict(dimension="completeness"), dict(extracted_text="e2"), dict(comparison_text="c2"),
    ):
        varied = dict(base)
        varied.update(changed)
        assert verdict_cache_key(**varied) != verdict_cache_key(**base)


def test_verdict_cache_put_then_get_round_trips_the_verdict(tmp_path):
    cache = VerdictCache(root=tmp_path)
    key = verdict_cache_key(judge_config_hash="h", field="summary", dimension="groundedness", extracted_text="e", comparison_text="c")
    assert cache.get(key) is None
    v = _verdict(field="summary", dimension="groundedness", supported_pattern=(True, False))
    cache.put(key, v)
    got = cache.get(key)
    assert got is not None
    assert got.record_supported == v.record_supported
    assert [c.text for c in got.claims] == [c.text for c in v.claims]


def test_verdict_cache_get_treats_a_corrupt_file_as_a_miss_not_a_crash(tmp_path):
    """A corrupt cache entry is treated as a miss instead of raising."""
    cache = VerdictCache(root=tmp_path)
    key = verdict_cache_key(judge_config_hash="h", field="summary", dimension="groundedness", extracted_text="e", comparison_text="c")
    path = tmp_path / "cache" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"field": "summary", "dimension": "groundedness", "claims": [{"text"', encoding="utf-8")  # truncated JSON
    assert cache.get(key) is None


def test_write_verdict_file_gives_concurrent_writers_to_the_same_path_distinct_temp_files(tmp_path):
    """Concurrent writers to the same target each use their own temp file.

    With a shared `<path>.tmp`, two threads could write into the same file at once. The
    temp paths are captured by monkeypatching `os.replace`."""
    import os
    import threading as _threading

    import grant_eval.scoring.judge as judgemod

    target = tmp_path / "shared.json"
    seen_temp_paths: list[Path] = []
    lock = _threading.Lock()
    real_replace = os.replace

    def _capture_replace(src, dst):
        with lock:
            seen_temp_paths.append(Path(src))
        return real_replace(src, dst)

    barrier = _threading.Barrier(4)

    def _writer(i):
        barrier.wait()  # make all 4 writers hit the temp-file step together
        write_verdict_file(target, {"writer": i})

    monkeypatch_target = judgemod.os
    import unittest.mock as mock
    with mock.patch.object(monkeypatch_target, "replace", _capture_replace):
        threads = [_threading.Thread(target=_writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(seen_temp_paths) == 4
    assert len(set(seen_temp_paths)) == 4  # every writer used a different temp file
    # the final file is valid JSON from exactly one writer
    final = json.loads(target.read_text(encoding="utf-8"))
    assert final["writer"] in range(4)


def test_write_verdict_file_leaves_no_orphaned_temp_file_on_failure(tmp_path, monkeypatch):
    """A failed write removes its temp file."""
    import grant_eval.scoring.judge as judgemod

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(judgemod.os, "replace", _boom)
    path = tmp_path / "run" / "verdict.json"
    with pytest.raises(OSError):
        write_verdict_file(path, {"a": 1})
    leftover = list(path.parent.glob("*")) if path.parent.exists() else []
    assert leftover == []


def test_score_run_judge_reuses_a_cached_verdict_for_identical_inputs(tmp_path, monkeypatch):
    """A second call with the same cache makes no backend calls.

    The cache, not temperature 0, is what makes verdicts repeatable."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)
    cache = VerdictCache(root=tmp_path)

    backend1 = FakeJudgeBackend([_ok_reply()] * 2)
    out1 = score_run_judge(
        result=result, run_score=run_score, backend=backend1, judge_config=_judge_config(),
        registry=_registry(), cache=cache,
    )
    assert len(backend1.calls) == 2  # completeness + groundedness, this field

    backend2 = FakeJudgeBackend([])  # must NOT be called at all: everything is cached
    out2 = score_run_judge(
        result=result, run_score=run_score, backend=backend2, judge_config=_judge_config(),
        registry=_registry(), cache=cache,
    )
    assert backend2.calls == []
    c1 = out1.fields[("summary", "groundedness")].records[0].verdict
    c2 = out2.fields[("summary", "groundedness")].records[0].verdict
    assert c1.record_supported == c2.record_supported


def test_no_cache_bypasses_the_cache_and_calls_the_backend_again(tmp_path, monkeypatch):
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    run_score = scoremod.score_run(result, gt)

    backend1 = FakeJudgeBackend([_ok_reply()] * 2)
    score_run_judge(
        result=result, run_score=run_score, backend=backend1, judge_config=_judge_config(),
        registry=_registry(), cache=None,
    )
    backend2 = FakeJudgeBackend([_ok_reply()] * 2)
    score_run_judge(
        result=result, run_score=run_score, backend=backend2, judge_config=_judge_config(),
        registry=_registry(), cache=None,
    )
    assert len(backend2.calls) == 2  # no cache in play at all -> called again in full


# dry_run_batch: builds every prompt, calls nothing, estimates the cost
from grant_eval.scoring.judge import dry_run_batch  # noqa: E402


def test_dry_run_makes_zero_calls_on_the_real_backend_and_counts_calls(tmp_path, monkeypatch):
    """`--dry-run` counts the calls it would make without touching the backend."""
    import grant_eval.scoring.judge as judgemod

    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x", eligibility_summary="UK only")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    real_backend = FakeJudgeBackend([])  # empty script, so any call would raise
    report = dry_run_batch([path], ground_truth=gt, registry=_registry(), judge_config=_judge_config())
    assert real_backend.calls == []
    # 2 fields x 2 dimensions, one TP record with both fields populated
    assert report["n_calls"] == 4
    assert report["estimated_cost_usd"] > 0.0


def test_dry_run_writes_no_files(tmp_path, monkeypatch):
    """A dry run writes nothing: the results root is never created."""
    import grant_eval.scoring.judge as judgemod

    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "resolve_mirror_page", lambda *a, **kw: judgemod.MirrorPage(text="PAGE TEXT", raw_html_fallback=False))
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    path = fixtures_dir / "result.json"
    path.write_text(result.model_dump_json())

    dry_run_batch([path], ground_truth=gt, registry=_registry(), judge_config=_judge_config())
    assert not results_root.exists()


# run_judge_batch: resume, spend cap, concurrency
from grant_eval.scoring.judge import run_judge_batch  # noqa: E402

NOWHERE = Path("/nonexistent-snapshot-root-for-test")


def _write_run(tmp_path, name: str, config_hash: str) -> Path:
    """Write one result file that triggers exactly one judge call.

    One TP record with only `summary` filled gives one completeness call, and groundedness
    points at a missing mirror (`NOWHERE`) so it is excluded as `unresolvable_source`. This
    keeps the spend-cap arithmetic exact."""
    opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [opp], config_hash=config_hash)
    path = tmp_path / f"{name}.json"
    path.write_text(result.model_dump_json())
    return path


def _one_row_gt():
    return _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])


def test_run_judge_batch_persists_a_file_per_run(tmp_path):
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, "a", "run-a")]
    backend = FakeJudgeBackend([_ok_reply()])
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
    )
    assert len(batch.outcomes) == 1
    assert batch.outcomes[0].resumed is False
    out_path = verdict_path("run-a", _judge_config().config_hash, funder_slug=SLUG, replicate=1, root=results_root)
    assert out_path.is_file()
    assert read_verdict_file(out_path)["funder"] == LEV


def test_resume_skips_a_run_already_on_disk(tmp_path):
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, "a", "run-a")]
    first = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=FakeJudgeBackend([_ok_reply()]), judge_config=_judge_config(), results_root=results_root,
    )
    assert first.outcomes[0].resumed is False

    starved_backend = FakeJudgeBackend([])  # would raise "script exhausted" if ever called
    second = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=starved_backend, judge_config=_judge_config(), results_root=results_root,
    )
    assert second.outcomes[0].resumed is True
    assert starved_backend.calls == []
    assert not second.errors


def test_total_spend_usd_is_accurate_under_concurrency_not_inflated_by_racing_deltas(tmp_path):
    """Reported spend equals the backend's own total even with concurrent workers.

    `delay=` makes all 8 workers overlap, so a per-run before/after delta on the shared
    total would be inflated."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcdefgh"]
    costs = [7.0, 4.0, 2.0, 3.0, 1.0, 8.0, 6.0, 5.0]
    backend = FakeJudgeBackend([_ok_reply()] * 8, costs=costs, delay=0.05)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=8, max_spend_usd=1000.0,  # cap not relevant here
    )
    assert len(batch.outcomes) == 8
    assert batch.total_spend_usd == pytest.approx(sum(costs))
    assert batch.total_spend_usd == pytest.approx(backend.total_cost_usd)


def test_spend_cap_hard_stops_and_reports_partial_progress(tmp_path):
    """Once spend passes the cap the batch stops submitting work and reports partial progress.

    Whether run-c is skipped or already running is a race at `workers=1`, so it isn't
    checked. What is checked is that reported spend matches the backend total and no path
    is counted twice or dropped."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in ("a", "b", "c")]
    backend = FakeJudgeBackend([_ok_reply()] * 3, costs=[15.0, 15.0, 15.0])
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1, max_spend_usd=25.0,
    )
    assert batch.cap_hit is True
    assert len(batch.outcomes) in (2, 3)
    assert len(batch.outcomes) + len(batch.skipped_for_spend_cap) == 3
    assert batch.total_spend_usd == pytest.approx(backend.total_cost_usd)
    assert batch.total_spend_usd in (pytest.approx(30.0), pytest.approx(45.0))
    outcome_paths = {o.source_path for o in batch.outcomes}
    assert not (outcome_paths & set(batch.skipped_for_spend_cap))  # never both
    assert len(batch.skipped_for_spend_cap) == len(set(batch.skipped_for_spend_cap))  # no dupes


def test_spend_cap_never_double_counts_a_cancelled_run_as_skipped(tmp_path):
    """A cancelled future is not added to `skipped_for_spend_cap` twice.

    `as_completed` yields a future again once it is cancelled. With `workers=1` at most one
    of run-d/run-e can be running when the cap trips, but both may already have finished,
    so each appears 0 or 1 times. The `all_seen` check is what catches a double append."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcde"]
    backend = FakeJudgeBackend([_ok_reply()] * 5, costs=[15.0, 15.0, 1.0, 1.0, 1.0])
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1, max_spend_usd=25.0,
    )
    assert batch.cap_hit is True
    assert batch.skipped_for_spend_cap.count(paths[3]) in (0, 1)
    assert batch.skipped_for_spend_cap.count(paths[4]) in (0, 1)
    # No path anywhere is double-counted, whichever bucket it landed in.
    all_seen = (
        [o.source_path for o in batch.outcomes]
        + [p for p, _ in batch.errors]
        + batch.skipped_for_spend_cap
    )
    assert len(all_seen) == len(set(all_seen)) == len(paths)


def test_spend_cap_never_discards_a_run_that_was_already_paid_for_and_persisted(tmp_path):
    """Runs that finished before the cap tripped are reported as outcomes, not skipped.

    `future.cancel()` returns False for a finished future, so a run that was charged and
    written to disk has to show up in `batch.outcomes`. `delay=` makes several runs finish
    close together."""
    results_root = tmp_path / "results"
    names = "abcde"
    config_hashes = {n: f"run-{n}" for n in names}
    paths = [_write_run(tmp_path, n, config_hashes[n]) for n in names]
    # `_write_run` names the file by stem ("a") but the config hash is "run-a", so
    # compare config hashes throughout rather than path stems.
    path_to_hash = dict(zip(paths, (config_hashes[n] for n in names)))
    backend = FakeJudgeBackend([_ok_reply()] * 5, costs=[10.0] * 5, delay=0.05)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=5, max_spend_usd=25.0,
    )
    assert batch.cap_hit is True  # the cap fired (5 x $10 > $25)
    # verdict_path is four levels deep (<run_hash>/<funder_slug>/<replicate>/<judge_hash>.json)
    # and the cache only two (<root>/cache/<key>.json), so this glob can't match the cache.
    persisted_run_hashes = {
        p.relative_to(results_root).parts[0] for p in results_root.glob("*/*/*/*.json")
    }
    # The set checks below pass trivially on an empty set, so check the glob found
    # something first.
    assert len(persisted_run_hashes) >= 3
    outcome_hashes = {o.run_config_hash for o in batch.outcomes}
    skipped_hashes = {path_to_hash[p] for p in batch.skipped_for_spend_cap}
    # Every run with a file on disk was judged and charged, so it must be an outcome
    # and never also in skipped_for_spend_cap.
    assert persisted_run_hashes <= outcome_hashes
    assert not (persisted_run_hashes & skipped_hashes)


def test_run_judge_batch_processes_every_run_under_concurrency(tmp_path):
    """A pool wider than the run count judges every run exactly once."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in ("a", "b", "c", "d", "e")]
    backend = FakeJudgeBackend([_ok_reply()] * 5)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=4,
    )
    assert len(batch.outcomes) == 5
    assert {o.run_config_hash for o in batch.outcomes} == {f"run-{n}" for n in ("a", "b", "c", "d", "e")}
    assert not batch.errors


def test_an_unexpected_exception_is_recorded_as_an_error_and_other_runs_still_complete(tmp_path):
    """One malformed result file doesn't stop the batch.

    The other runs are judged and the bad one is listed in `errors`."""
    results_root = tmp_path / "results"
    good = [_write_run(tmp_path, n, f"run-{n}") for n in ("a", "b", "d", "e")]
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not valid json at all")
    paths = good[:2] + [bad_path] + good[2:]

    backend = FakeJudgeBackend([_ok_reply()] * 4)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1,
    )
    assert len(batch.outcomes) == 4
    assert len(batch.errors) == 1
    assert batch.errors[0][0] == bad_path
    assert batch.cap_hit is False


class _FatalWorkerError(BaseException):
    """A BaseException that isn't an Exception, like SystemExit or KeyboardInterrupt."""


def test_a_baseexception_in_a_worker_does_not_abort_the_pool_or_discard_the_batch(tmp_path, monkeypatch):
    """A BaseException in a worker is caught: finished runs are kept and queued ones cancelled.

    Otherwise `shutdown(wait=True)` would let every submitted job run (and spend money)
    with the cap no longer checked, and the whole result would be lost."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcde"]
    real_judge_one_run = judgemod._judge_one_run

    def fatal_on_c(path, **kw):
        if Path(path).name == "c.json":
            raise _FatalWorkerError("simulated interpreter-level failure")
        return real_judge_one_run(path, **kw)

    monkeypatch.setattr(judgemod, "_judge_one_run", fatal_on_c)
    # `delay=` keeps the worker busy on run-d while run-c's fatal is handled, so run-e
    # is still queued and `.cancel()` succeeds for it.
    backend = FakeJudgeBackend([_ok_reply()] * 5, delay=0.05)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1,
    )
    assert [p for p, _ in batch.errors] == [paths[2]]
    assert "_FatalWorkerError" in batch.errors[0][1]
    # The two runs that finished before the fatal are kept, not thrown away.
    assert {paths[0], paths[1]} <= {o.source_path for o in batch.outcomes}
    # Everything cut short is named and counted in its own bucket, not as a spend-cap skip.
    assert batch.cancelled_after_fatal
    assert set(batch.cancelled_after_fatal) <= {paths[3], paths[4]}
    assert not batch.skipped_for_spend_cap
    all_seen = (
        [o.source_path for o in batch.outcomes]
        + [p for p, _ in batch.errors]
        + batch.skipped_for_spend_cap
        + batch.cancelled_after_fatal
    )
    assert len(all_seen) == len(set(all_seen)) == len(paths)


def test_cli_reports_and_fails_when_a_fatal_worker_error_cut_the_pass_short(tmp_path, monkeypatch, capsys):
    """A pass cut short by a fatal error exits non-zero and names the cancelled runs."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()] * 5, delay=0.05))

    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcde"]
    real_judge_one_run = judgemod._judge_one_run

    def fatal_on_c(path, **kw):
        if Path(path).name == "c.json":
            raise _FatalWorkerError("simulated interpreter-level failure")
        return real_judge_one_run(path, **kw)

    monkeypatch.setattr(judgemod, "_judge_one_run", fatal_on_c)
    rc = judgemod.main([str(p) for p in paths] + ["--snapshots", str(NOWHERE), "--workers", "1"])
    assert rc != 0
    assert "cancelled" in capsys.readouterr().err


class _InterruptOnSpendRead:
    """Wrap a FakeJudgeBackend, raising KeyboardInterrupt on the Nth `total_cost_usd` read.

    `run_judge_batch` reads it on the main thread once per completion, which is where a
    real Ctrl-C would land. Read 1 is the `hasattr` check and read k+1 is the kth
    completion, so `interrupt_on_read=4` always fires after two completions whatever the
    scheduling."""

    name = "interrupting-fake"

    def __init__(self, inner, *, interrupt_on_read: int) -> None:
        self._inner = inner
        self._interrupt_on_read = interrupt_on_read
        self._reads = 0
        self._fired = False

    @property
    def calls(self):
        return self._inner.calls

    def complete(self, **kw):
        return self._inner.complete(**kw)

    @property
    def total_cost_usd(self):
        if not self._fired:
            self._reads += 1
            if self._reads >= self._interrupt_on_read:
                self._fired = True
                raise KeyboardInterrupt("simulated Ctrl-C during an unattended pass")
        return self._inner.total_cost_usd


def test_a_keyboard_interrupt_cancels_pending_work_instead_of_draining_the_pool(tmp_path):
    """Ctrl-C mid-batch cancels queued work and still returns what was already judged.

    The judging pass runs unattended for hours, so an interrupt is the likeliest early exit.
    Without handling it, `shutdown(wait=True)` would run every queued job with no cap check
    and the result would be lost."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcdefgh"]
    inner = FakeJudgeBackend([_ok_reply()] * 8, costs=[1.0] * 8, delay=0.05)
    backend = _InterruptOnSpendRead(inner, interrupt_on_read=4)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1, max_spend_usd=1e9,
    )
    assert batch.interrupted_by == "KeyboardInterrupt"

    # (a) the queue was cancelled, not drained.
    assert batch.cancelled_after_interrupt
    assert len(inner.calls) < len(paths)
    assert not batch.cap_hit  # the cap is not why these did not run

    # (b) the two runs recorded before the interrupt survive it, with their spend.
    assert len(batch.outcomes) == 2
    assert batch.total_spend_usd == pytest.approx(inner.total_cost_usd)
    assert batch.total_spend_usd >= 2.0
    # Never in two buckets at once.
    assert not (set(batch.cancelled_after_interrupt) & {o.source_path for o in batch.outcomes})


def test_cli_reports_and_fails_after_an_interrupt_but_still_writes_what_was_judged(tmp_path, monkeypatch, capsys):
    """An interrupted CLI pass still writes the roll-up, names cancelled runs and exits non-zero."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    inner = FakeJudgeBackend([_ok_reply()] * 8, costs=[1.0] * 8, delay=0.05)
    monkeypatch.setattr(
        judgemod, "OpenRouterJudgeBackend",
        lambda: _InterruptOnSpendRead(inner, interrupt_on_read=4),
    )

    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcdefgh"]
    # `--no-cache`: the fixtures all share the same texts, so with the cache on the
    # eight runs would collapse into one backend call.
    rc = judgemod.main(
        [str(p) for p in paths] + ["--snapshots", str(NOWHERE), "--workers", "1", "--no-cache"]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "interrupted" in err.lower()
    assert "cancelled" in err.lower()

    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    rollup = read_verdict_file(judgemod.rollup_path(pinned, root=results_root))
    assert rollup["n_runs"] == len(inner.calls)  # every run that was really judged


def _plant_unreadable_verdicts(results_root, judge_config_hash):
    """Write one corrupt verdict file and one that parses but has the wrong schema."""
    corrupt = results_root / "run-corrupt" / SLUG / "1" / f"{judge_config_hash}.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{ not valid json at all", encoding="utf-8")
    foreign = results_root / "run-foreign" / SLUG / "1" / f"{judge_config_hash}.json"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text(json.dumps({"not": "a verdict file"}), encoding="utf-8")
    return corrupt, foreign


def test_load_corpus_verdicts_names_a_bad_file_instead_of_aborting_the_whole_roll_up(tmp_path):
    """Unreadable verdict files are skipped and named instead of crashing the loader.

    This runs after the paid batch, so a crash here would lose the spend total and the
    roll-up."""
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, "a", "run-a")]
    judge_config = _judge_config()
    run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=FakeJudgeBackend([_ok_reply()]), judge_config=judge_config,
        results_root=results_root,
    )
    corrupt, foreign = _plant_unreadable_verdicts(results_root, judge_config.config_hash)

    verdicts, unreadable = judgemod.load_corpus_verdicts(judge_config.config_hash, root=results_root)

    assert len(verdicts) == 1  # the good run still loads
    assert {u["path"] for u in unreadable} == {str(corrupt), str(foreign)}
    # Named AND diagnosed: the path is in the record, not only in a lost traceback.
    assert all(u["reason"] for u in unreadable)
    # ... and what survives really does roll up, rather than blowing up one call later.
    assert judgemod.build_rollup(verdicts)["n_runs"] == 1


def test_a_bad_verdict_file_does_not_abort_a_pass_that_has_already_spent_the_money(
    tmp_path, monkeypatch, capsys
):
    """With bad verdict files the CLI still reports spend, writes the roll-up and exits 0."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    corrupt, foreign = _plant_unreadable_verdicts(results_root, pinned)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--snapshots", str(NOWHERE)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "total spend" in captured.out
    # Both bad files are named on stderr so they can be cleaned up later.
    assert str(corrupt) in captured.err
    assert str(foreign) in captured.err
    rollup_file = judgemod.rollup_path(pinned, root=results_root)
    assert rollup_file.is_file()
    assert read_verdict_file(rollup_file)["n_runs"] == 1


def test_an_interrupt_before_the_first_completion_still_writes_a_truthful_summary(
    tmp_path, monkeypatch, capsys
):
    """An interrupt on the first completion still writes the roll-up from the verdicts on disk.

    `batch.outcomes` is empty here even though runs were judged and written, so the
    roll-up can't depend on it. The summary also has to reconcile the in-memory count with
    the on-disk count instead of saying nothing was judged."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    inner = FakeJudgeBackend([_ok_reply()] * 8, costs=[1.0] * 8, delay=0.05)
    monkeypatch.setattr(
        judgemod, "OpenRouterJudgeBackend",
        lambda: _InterruptOnSpendRead(inner, interrupt_on_read=2),
    )

    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcdefgh"]
    # `--no-cache` for the same reason as the other interrupt CLI test.
    rc = judgemod.main(
        [str(p) for p in paths] + ["--snapshots", str(NOWHERE), "--workers", "4", "--no-cache"]
    )
    assert rc != 0
    captured = capsys.readouterr()

    # Runs really were judged, charged for and written.
    assert inner.calls
    assert inner.total_cost_usd > 0

    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    rollup_file = judgemod.rollup_path(pinned, root=results_root)
    assert rollup_file.is_file()
    assert read_verdict_file(rollup_file)["n_runs"] == len(inner.calls)

    # The summary must not claim nothing happened while the money was being spent.
    assert "no run judged in this pass" not in captured.err
    assert "left untouched" not in captured.err
    # ...and it must reconcile "0 recorded in memory" against "N on disk".
    assert f"{len(inner.calls)} run(s) on disk" in captured.out
    assert "recorded 0 outcome(s) in memory" in captured.err


def _raise_on_write(path, data):
    raise OSError("disk full")


def test_reported_spend_includes_calls_from_runs_that_later_failed(tmp_path, monkeypatch):
    """A run that is charged and then fails (here a failing write) still counts towards spend.

    The cap is set high so this only tests spend and error accounting; the next test
    covers the cap."""
    monkeypatch.setattr(judgemod, "write_verdict_file", _raise_on_write)
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in ("a", "b", "c")]
    backend = FakeJudgeBackend([_ok_reply()] * 3, costs=[5.0, 5.0, 5.0])
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1, max_spend_usd=1000.0,
    )
    assert len(batch.errors) == 3
    assert len(batch.outcomes) == 0
    assert batch.total_spend_usd == pytest.approx(15.0)
    assert batch.total_spend_usd == pytest.approx(backend.total_cost_usd)


def test_spend_cap_still_fires_when_every_run_fails_after_spending(tmp_path, monkeypatch):
    """With every write failing, the cap still trips and reported spend matches the backend."""
    monkeypatch.setattr(judgemod, "write_verdict_file", _raise_on_write)
    results_root = tmp_path / "results"
    paths = [_write_run(tmp_path, n, f"run-{n}") for n in "abcdef"]
    backend = FakeJudgeBackend([_ok_reply()] * 6, costs=[15.0] * 6)
    batch = run_judge_batch(
        paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        backend=backend, judge_config=_judge_config(), results_root=results_root,
        workers=1, max_spend_usd=25.0,
    )
    assert batch.cap_hit is True
    assert batch.total_spend_usd == pytest.approx(backend.total_cost_usd)
    assert len(batch.errors) + len(batch.outcomes) + len(batch.skipped_for_spend_cap) == 6


class _CostlessBackend:
    """A JudgeBackend with no spend tracking."""

    name = "costless"

    def complete(self, *, system, user, model, temperature):
        return _ok_reply()


def test_run_judge_batch_refuses_a_backend_with_no_cost_tracking(tmp_path):
    """A backend without `total_cost_usd` is refused before any work starts.

    Otherwise it would silently switch off the spend cap."""
    paths = [_write_run(tmp_path, "a", "run-a")]
    backend = _CostlessBackend()
    results_root = tmp_path / "results"
    with pytest.raises(AttributeError, match="total_cost_usd"):
        run_judge_batch(
            paths, ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
            backend=backend, judge_config=_judge_config(), results_root=results_root,
        )
    assert not results_root.exists()


# build_rollup: corpus totals, per-funder counts, exclusion ledger
from grant_eval.scoring.judge import build_rollup  # noqa: E402

FUNDER_A = "The Leverhulme Trust"
FUNDER_B = "The Wolfson Foundation"


def _run_dict_for_rollup(funder: str, config_hash: str, *, supported_pattern, excluded: dict | None = None) -> dict:
    """A minimal `to_dict()`-shaped run with only `summary:groundedness` filled in."""
    excluded = excluded or {}
    records = [
        {
            "opportunity_id": f"r{i}",
            "outcome": "ok",
            "record_supported": all(pattern),
            "justification": "j",
            "claims": [{"text": "c", "claim_type": "purpose", "supported": s} for s in pattern],
        }
        for i, pattern in enumerate(supported_pattern)
    ]
    n_claims = sum(len(p) for p in supported_pattern)
    n_claims_supported = sum(sum(p) for p in supported_pattern)
    n_records_supported = sum(1 for p in supported_pattern if all(p))
    exclusions = {r: 0 for r in EXCLUSION_REASONS}
    for reason, oids in excluded.items():
        exclusions[reason] = len(oids)
        for oid in oids:
            records.append({"opportunity_id": oid, "outcome": reason, "record_supported": None, "justification": None, "claims": None})
    cell = {
        "field": "summary", "dimension": "groundedness",
        "n_claims": n_claims, "n_claims_supported": n_claims_supported,
        "n_records_judged": len(supported_pattern), "n_records_supported": n_records_supported,
        "claim_rate": None, "claim_rate_undefined": n_claims == 0,
        "claim_rate_macro": None, "claim_rate_macro_undefined": len(supported_pattern) == 0,
        "record_rate": None, "record_rate_undefined": len(supported_pattern) == 0,
        "n_raw_html_fallback": 0, "exclusions": exclusions, "records": records,
    }
    empty_cell = {
        "field": "x", "dimension": "y", "n_claims": 0, "n_claims_supported": 0,
        "n_records_judged": 0, "n_records_supported": 0, "claim_rate": None,
        "claim_rate_undefined": True, "claim_rate_macro": None, "claim_rate_macro_undefined": True,
        "record_rate": None, "record_rate_undefined": True, "n_raw_html_fallback": 0,
        "exclusions": {r: 0 for r in EXCLUSION_REASONS}, "records": [],
    }
    return {
        "funder": funder, "model": "m", "prompt_strategy": "minimal", "temperature": 0.0,
        "agent_config": "standard", "replicate": 1, "config_hash": config_hash,
        "schema_valid": True, "abstained": False, "judge_config_hash": "jch",
        "fields": {
            "summary:groundedness": cell,
            "summary:completeness": dict(empty_cell, field="summary", dimension="completeness"),
            "eligibility_summary:groundedness": dict(empty_cell, field="eligibility_summary", dimension="groundedness"),
            "eligibility_summary:completeness": dict(empty_cell, field="eligibility_summary", dimension="completeness"),
        },
    }


def test_build_rollup_sums_claims_and_records_across_runs():
    run1 = _run_dict_for_rollup(FUNDER_A, "run-1", supported_pattern=[(True,), (True, False)])
    run2 = _run_dict_for_rollup(FUNDER_A, "run-2", supported_pattern=[(True, True, True)])
    rollup = build_rollup([run1, run2])
    cell = rollup["fields"]["summary:groundedness"]
    assert cell["n_claims"] == 6  # 1 + 2 + 3
    assert cell["n_claims_supported"] == 5  # 1 + 1 + 3
    assert cell["n_records_judged"] == 3
    assert cell["n_records_supported"] == 2  # the (True,) record and the (True,True,True) record
    assert cell["claim_rate"] == pytest.approx(5 / 6)
    assert cell["claim_rate_undefined"] is False


def test_build_rollup_reports_per_funder_counts():
    run_a = _run_dict_for_rollup(FUNDER_A, "run-a", supported_pattern=[(True,)])
    run_b = _run_dict_for_rollup(FUNDER_B, "run-b", supported_pattern=[(True, False)])
    rollup = build_rollup([run_a, run_b])
    by_funder = rollup["by_funder"]
    assert by_funder[FUNDER_A]["summary:groundedness"]["n_claims"] == 1
    assert by_funder[FUNDER_B]["summary:groundedness"]["n_claims"] == 2
    # a funder's total must not leak into another's
    assert by_funder[FUNDER_A]["summary:groundedness"]["n_records_judged"] == 1


def test_build_rollup_exclusion_ledger_names_every_excluded_record():
    run = _run_dict_for_rollup(
        FUNDER_A, "run-1", supported_pattern=[(True,)],
        excluded={FIELD_EMPTY: ["r-empty"], "judge_parse_failure": ["r-parse"]},
    )
    rollup = build_rollup([run])
    ledger = rollup["exclusion_ledger"]
    reasons = {(row["opportunity_id"], row["reason"]) for row in ledger}
    assert ("r-empty", FIELD_EMPTY) in reasons
    assert ("r-parse", "judge_parse_failure") in reasons
    assert all(row["funder"] == FUNDER_A for row in ledger)
    assert all(row["field"] == "summary" and row["dimension"] == "groundedness" for row in ledger)


def test_build_rollup_exclusion_ledger_carries_the_persisted_error():
    """The roll-up's exclusion ledger includes each record's stored `error`."""
    empty_cell = {
        "field": "x", "dimension": "y", "n_claims": 0, "n_claims_supported": 0,
        "n_records_judged": 0, "n_records_supported": 0, "claim_rate": None,
        "claim_rate_undefined": True, "claim_rate_macro": None, "claim_rate_macro_undefined": True,
        "record_rate": None, "record_rate_undefined": True, "n_raw_html_fallback": 0,
        "exclusions": {r: 0 for r in EXCLUSION_REASONS}, "records": [],
    }
    cell = dict(empty_cell, field="summary", dimension="completeness")
    cell["exclusions"] = {**cell["exclusions"], "judge_parse_failure": 1}
    cell["records"] = [{
        "opportunity_id": "g1", "outcome": "judge_parse_failure",
        "record_supported": None, "justification": None, "claims": None,
        "error": "Expecting value: line 1 column 1 (char 0)",
    }]
    run = {
        "funder": FUNDER_A, "model": "m", "prompt_strategy": "minimal", "temperature": 0.0,
        "agent_config": "standard", "replicate": 1, "config_hash": "run-1",
        "schema_valid": True, "abstained": False, "judge_config_hash": "jch",
        "fields": {
            "summary:completeness": cell,
            "summary:groundedness": dict(empty_cell, field="summary", dimension="groundedness"),
            "eligibility_summary:completeness": dict(empty_cell, field="eligibility_summary", dimension="completeness"),
            "eligibility_summary:groundedness": dict(empty_cell, field="eligibility_summary", dimension="groundedness"),
        },
    }
    rollup = build_rollup([run])
    ledger_row = rollup["exclusion_ledger"][0]
    assert ledger_row["opportunity_id"] == "g1"
    assert ledger_row["error"] == "Expecting value: line 1 column 1 (char 0)"


def test_build_rollup_undefined_cell_has_null_rates_not_zero():
    run = _run_dict_for_rollup(FUNDER_A, "run-1", supported_pattern=[])
    rollup = build_rollup([run])
    cell = rollup["fields"]["summary:completeness"]  # the empty_cell in the fixture
    assert cell["claim_rate"] is None
    assert cell["claim_rate_undefined"] is True


def test_build_rollup_macro_rate_pools_per_record_across_runs():
    """Corpus macro is a mean over records, not runs, so every record counts equally."""
    run1 = _run_dict_for_rollup(FUNDER_A, "run-1", supported_pattern=[(True,) + (False,) * 8])
    run2 = _run_dict_for_rollup(FUNDER_A, "run-2", supported_pattern=[(True,)])
    rollup = build_rollup([run1, run2])
    cell = rollup["fields"]["summary:groundedness"]
    assert cell["claim_rate_macro"] == pytest.approx((1 / 9 + 1.0) / 2)
    assert cell["claim_rate"] == pytest.approx(2 / 10)  # micro: pooled claims differ from macro


# CLI: build_parser, main
import grant_eval.scoring.judge as judgemod  # noqa: E402


def test_build_parser_defaults():
    args = judgemod.build_parser().parse_args(["result.json"])
    assert args.results == ["result.json"]
    assert args.workers == 8
    assert args.max_spend_usd == 25.0
    assert args.no_cache is False
    assert args.dry_run is False
    assert args.json is False


def test_cli_pins_the_judge_model_and_temperature(tmp_path, monkeypatch):
    """`main` builds the judge config with the pinned model and temperature.

    Checked against literal values rather than the constants."""
    captured = {}

    def fake_dry_run_batch(paths, *, ground_truth, registry, snapshot_root, judge_config):
        captured["judge_config"] = judge_config
        return {"n_calls": 0, "estimated_cost_usd": 0.0}

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "dry_run_batch", fake_dry_run_batch)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main(["--dry-run", str(path)])
    assert rc == 0
    cfg = captured["judge_config"]
    assert cfg.model == "openrouter/google/gemini-2.5-flash-lite"
    assert cfg.temperature == 0.0


def test_cli_dry_run_prints_call_count_and_labels_the_cost_as_an_estimate(tmp_path, monkeypatch, capsys):
    """The dry run prints the exact call count and labels the cost as an estimate.

    The fixture has one populated field on one TP record and an unresolvable mirror, so
    exactly one completeness call fires."""
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "DEFAULT_SNAPSHOT_ROOT", NOWHERE)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main(["--dry-run", "--snapshots", str(NOWHERE), str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run: 1 judge call(s) would be made" in out
    assert "estimate" in out.lower()


def test_cli_end_to_end_persists_a_file_and_writes_the_rollup(tmp_path, monkeypatch, capsys):
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--snapshots", str(NOWHERE)])
    assert rc == 0
    pinned_judge_config_hash = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    out_path = verdict_path("run-a", pinned_judge_config_hash, funder_slug=SLUG, replicate=1, root=results_root)
    assert out_path.is_file()
    # The roll-up is keyed by judge_config_hash so a changed judge config can't overwrite it.
    assert judgemod.rollup_path(pinned_judge_config_hash, root=results_root).is_file()
    out = capsys.readouterr().out
    assert "judged 1 run" in out


#: A second funder, so judging one funder's directory can be told apart from a
#: corpus-wide pass.
WELL = "Wellcome Trust"
WELL_SLUG = "wellcome"


def _two_funder_registry() -> dict[str, FunderRecord]:
    reg = _registry()
    reg[WELL_SLUG] = FunderRecord(
        slug=WELL_SLUG, display_name=WELL, split=Split.TUNE, snapshot_date=SNAP_DATE,
        entry_url="https://example.net/", content_hash="cafebabe",
    )
    return reg


def _two_funder_gt():
    lev = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    well = _ground_truth(WELL, [_gt_row(WELL, "g1", "Scheme One", "https://example.net/scheme-one")])
    return GroundTruth(funders={**lev.funders, **well.funders})


def _write_run_for_funder(tmp_path, name: str, config_hash: str, funder: str, url: str) -> Path:
    """Same as `_write_run`, with a configurable funder."""
    opp = _judge_extracted(funder, "g1", "Scheme One", url, summary="Funds x")
    result = _judge_result(funder, [opp], config_hash=config_hash)
    path = tmp_path / f"{name}.json"
    path.write_text(result.model_dump_json())
    return path


def test_the_rollup_covers_the_whole_corpus_on_disk_not_just_this_invocation(tmp_path, monkeypatch):
    """Judging one funder's directory doesn't narrow the corpus roll-up.

    The roll-up is built from every verdict on disk for this judge config, not just the
    paths given on the command line, since the notebook reads it as the corpus figure."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _two_funder_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _two_funder_registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()] * 4))

    lev_path = _write_run_for_funder(tmp_path, "lev", "run-lev", LEV, "https://example.org/scheme-one")
    well_path = _write_run_for_funder(tmp_path, "well", "run-well", WELL, "https://example.net/scheme-one")

    assert judgemod.main([str(lev_path), "--snapshots", str(NOWHERE)]) == 0
    assert judgemod.main([str(well_path), "--snapshots", str(NOWHERE)]) == 0

    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    rollup = read_verdict_file(judgemod.rollup_path(pinned, root=results_root))
    assert set(rollup["by_funder"]) == {LEV, WELL}
    assert rollup["n_runs"] == 2
    assert rollup["fields"]["summary:completeness"]["n_claims"] == 2


def test_an_all_error_pass_cannot_narrow_or_empty_the_existing_rollup(tmp_path, monkeypatch, capsys):
    """A pass where every run errors doesn't empty an existing roll-up.

    The roll-up is built from the corpus on disk, so a pass that judged nothing produces
    the same content. The CLI still reports the in-memory and on-disk counts."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    good_path = _write_run(tmp_path, "a", "run-a")
    assert judgemod.main([str(good_path), "--snapshots", str(NOWHERE)]) == 0
    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    good_rollup = read_verdict_file(judgemod.rollup_path(pinned, root=results_root))
    assert good_rollup["n_runs"] == 1

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not valid json at all")
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([]))
    assert judgemod.main([str(bad_path), "--snapshots", str(NOWHERE)]) != 0

    assert read_verdict_file(judgemod.rollup_path(pinned, root=results_root)) == good_rollup
    assert "recorded 0 outcome(s) in memory" in capsys.readouterr().err


def test_cli_succeeds_with_only_tune_results_present(tmp_path, monkeypatch):
    """The CLI works without validation results (`_write_run` uses `Split.TUNE`)."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    path = _write_run(tmp_path, "a", "run-a")
    result = ExtractionResult.model_validate_json(path.read_text())
    assert result.split is Split.TUNE
    assert judgemod.main([str(path), "--snapshots", str(NOWHERE)]) == 0


def test_cli_returns_nonzero_when_every_run_errors(tmp_path, monkeypatch):
    """A pass where every result file fails exits non-zero."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([]))

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not valid json at all")
    rc = judgemod.main([str(bad_path), "--snapshots", str(NOWHERE)])
    assert rc != 0


def test_cli_returns_zero_when_some_runs_error_but_others_succeed(tmp_path, monkeypatch):
    """A partial failure still exits 0, like `score.py`'s CLI does for a skipped path."""
    results_root = tmp_path / "results"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", results_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    good_path = _write_run(tmp_path, "a", "run-a")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ not valid json at all")
    rc = judgemod.main([str(good_path), str(bad_path), "--snapshots", str(NOWHERE)])
    assert rc == 0


def test_fake_judge_backend_delay_param_sleeps_before_returning():
    """FakeJudgeBackend with a per-call delay so tests can force thread overlap."""
    import time

    backend = FakeJudgeBackend(["ok"], delay=0.05)
    t0 = time.monotonic()
    backend.complete(system="s", user="u", model="m", temperature=0.0)
    assert time.monotonic() - t0 >= 0.04


def test_fake_judge_backend_accumulates_cost_only_on_success():
    """`total_cost_usd` sums the scripted costs; a call that raised costs nothing."""
    backend = FakeJudgeBackend(
        ["ok1", RuntimeError("boom"), "ok2"],
        costs=[0.01, 0.5, 0.02],
    )
    for _ in range(3):
        try:
            backend.complete(system="s", user="u", model="m", temperature=0.0)
        except RuntimeError:
            pass
    assert backend.total_cost_usd == pytest.approx(0.03)


def test_judge_router_backend_pins_a_60s_timeout():
    """The judge's 60s timeout is passed through to litellm.

    A timeout then becomes `judge_call_failure` like any other exception, which
    `test_call_failure_after_retries_yields_judge_call_failure` already covers."""
    router_cls = _judge_router_backend_class()
    assert router_cls.TIMEOUT_S == 60.0


def test_openrouter_judge_backend_accumulates_cost_from_the_litellm_response(monkeypatch):
    """Spend comes from `litellm.completion_cost` on the response, not from token counts."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.router as router

    monkeypatch.setattr(router.litellm, "completion", lambda **kw: _FakeJudgeResponse('{"claims": [], "justification": "x"}'))
    monkeypatch.setattr(router.litellm, "completion_cost", lambda response: 0.0042)

    backend = OpenRouterJudgeBackend()
    backend.complete(system="s", user="u", model=JUDGE_MODEL, temperature=0.0)
    assert backend.total_cost_usd == pytest.approx(0.0042)


# CLI flags: --export-spotcheck / --out / --agreement
def test_build_parser_accepts_export_spotcheck_out_and_agreement():
    args = judgemod.build_parser().parse_args(
        ["a.json", "--export-spotcheck", "200", "--out", "spot.csv"]
    )
    assert args.export_spotcheck == 200
    assert args.out == "spot.csv"
    assert args.agreement is None

    args2 = judgemod.build_parser().parse_args(["--agreement", "coded.csv"])
    assert args2.agreement == "coded.csv"
    assert args2.results == []


def test_build_parser_accepts_test_retest():
    """The parser accepts `--test-retest`."""
    args = judgemod.build_parser().parse_args(["a.json", "--test-retest", "200"])
    assert args.test_retest == 200
    assert judgemod.build_parser().parse_args(["a.json"]).test_retest is None


def test_main_dispatches_agreement_without_touching_ground_truth_or_registry(tmp_path, monkeypatch):
    """`--agreement` needs no results, ground truth or registry."""
    import grant_eval.scoring.judge_agreement as ja

    def _boom(*a, **kw):
        raise AssertionError("must not be called for --agreement")

    monkeypatch.setattr(judgemod, "load_ground_truth", _boom)
    monkeypatch.setattr(judgemod, "load_registry", _boom)

    captured = {}

    def fake_cli_agreement(coded_csv_path, *, results_root, json_output):
        captured["coded_csv_path"] = coded_csv_path
        captured["results_root"] = results_root
        captured["json_output"] = json_output
        return 0

    monkeypatch.setattr(ja, "cli_agreement", fake_cli_agreement)

    coded = tmp_path / "coded.csv"
    coded.write_text("item_id\n")
    rc = judgemod.main(["--agreement", str(coded)])
    assert rc == 0
    assert captured["coded_csv_path"] == Path(coded)


def test_main_dispatches_export_spotcheck_with_the_pinned_judge_config_hash(tmp_path, monkeypatch):
    import grant_eval.scoring.judge_agreement as ja

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())

    captured = {}

    def fake_cli_export_spotcheck(paths, *, ground_truth, registry, snapshot_root, results_root,
                                   judge_config_hash, n_total, out_path, json_output):
        captured["n_total"] = n_total
        captured["out_path"] = out_path
        captured["judge_config_hash"] = judge_config_hash
        return 0

    monkeypatch.setattr(ja, "cli_export_spotcheck", fake_cli_export_spotcheck)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--export-spotcheck", "200", "--out", "spot.csv"])
    assert rc == 0
    assert captured["n_total"] == 200
    assert captured["out_path"] == Path("spot.csv")
    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    assert captured["judge_config_hash"] == pinned


def test_main_dispatches_test_retest_with_a_real_backend(tmp_path, monkeypatch):
    """`--test-retest` makes live calls, so it builds `OpenRouterJudgeBackend` (faked here)."""
    import grant_eval.scoring.judge_agreement as ja

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    fake_backend = FakeJudgeBackend([])
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: fake_backend)

    captured = {}

    def fake_cli_test_retest(paths, *, ground_truth, registry, snapshot_root, results_root,
                              judge_config_hash, backend, judge_config, n, max_spend_usd,
                              json_output, spotcheck_csv):
        captured["n"] = n
        captured["backend"] = backend
        captured["max_spend_usd"] = max_spend_usd
        return 0

    monkeypatch.setattr(ja, "cli_test_retest", fake_cli_test_retest)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--test-retest", "50"])
    assert rc == 0
    assert captured["n"] == 50
    assert captured["backend"] is fake_backend


def test_export_spotcheck_zero_does_not_fall_through_to_a_live_paid_pass(tmp_path, monkeypatch):
    """`--export-spotcheck 0` doesn't fall through to a live paid pass (0 is falsy)."""
    import grant_eval.scoring.judge_agreement as ja

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())

    def _boom():
        raise AssertionError("OpenRouterJudgeBackend must never be constructed for --export-spotcheck 0")

    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", _boom)

    captured = {}
    monkeypatch.setattr(
        ja, "cli_export_spotcheck",
        lambda paths, **kw: (captured.update(n_total=kw["n_total"]), 0)[1],
    )

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--export-spotcheck", "0", "--out", "spot.csv"])
    assert rc == 0
    assert captured["n_total"] == 0


def test_dry_run_never_falls_through_to_a_live_paid_test_retest_pass(tmp_path, monkeypatch, capsys):
    """`--dry-run` with `--test-retest` makes no live calls and says test-retest was skipped."""
    import grant_eval.scoring.judge_agreement as ja

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())

    def _boom():
        raise AssertionError("OpenRouterJudgeBackend must never be constructed under --dry-run")

    def _never_live(*_a, **_kw):
        raise AssertionError("cli_test_retest makes LIVE calls and must never run under --dry-run")

    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", _boom)
    monkeypatch.setattr(ja, "cli_test_retest", _never_live)

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--snapshots", str(NOWHERE), "--dry-run", "--test-retest", "200"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "dry run: 1 judge call(s) would be made" in captured.out
    assert "--test-retest" in captured.err


def test_dry_run_does_not_displace_the_free_export_spotcheck_branch(tmp_path, monkeypatch):
    """`--export-spotcheck` takes precedence over `--dry-run`, since it makes no calls."""
    import grant_eval.scoring.judge_agreement as ja

    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())

    captured = {}
    monkeypatch.setattr(
        ja, "cli_export_spotcheck",
        lambda paths, **kw: (captured.update(n_total=kw["n_total"]), 0)[1],
    )

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--dry-run", "--export-spotcheck", "200", "--out", "spot.csv"])
    assert rc == 0
    assert captured["n_total"] == 200


# judge_agreement.py: Krippendorff's alpha, binary nominal case
def test_krippendorff_alpha_binary_matches_krippendorff_2011_worked_example():
    """Pinned against Krippendorff, K. (2011, updated 2013). "Computing
    Krippendorff's Alpha-Reliability." Annenberg School for Communication,
    University of Pennsylvania, Section A ("Binary or dichotomous data, two
    observers, no missing data"), p.2:
    https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff's%20Alpha-Reliability.pdf

    The paper's own 2-observer, 10-unit binary example (Meg/Owen):
        Items:  1 2 3 4 5 6 7 8 9 10
        Meg:    0 1 0 0 0 0 0 0 1 0
        Owen:   1 1 1 0 0 1 0 0 0 0
    gives coincidence-matrix marginals n0=14, n1=6, n=2N=20, o01=4 mismatches (the
    paper works this out cell by cell: units 1,3,6,9 are the four 0-1/1-0 pairs), and
    the paper's own closed form for this case:
        alpha = 1 - (n-1) * o01 / (n0 * n1) = 1 - 19*4/(14*6) = 1 - 76/84 = 0.095
    This is the same closed form `krippendorff_alpha_binary` derives from its general
    weighted coincidence matrix, so `weights=None` here is exercising the identical
    arithmetic the paper's simplified formula shortcuts to."""
    meg = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    owen = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    pairs = list(zip(meg, owen))
    alpha, undefined = ja.krippendorff_alpha_binary(pairs)
    assert undefined is False
    assert alpha == pytest.approx(1 - 19 * 4 / (14 * 6))
    assert alpha == pytest.approx(2 / 21)
    assert alpha == pytest.approx(0.095, abs=0.001)  # the paper's own rounded figure


def test_krippendorff_alpha_binary_undefined_with_fewer_than_two_items():
    """A single item (two ratings) is too few to estimate alpha, even if the raters disagree."""
    assert ja.krippendorff_alpha_binary([]) == (None, True)
    assert ja.krippendorff_alpha_binary([(True, False)]) == (None, True)


def test_krippendorff_alpha_binary_undefined_when_only_one_value_ever_appears():
    """If every rating is the same value, alpha is undefined: `(None, True)`, not 0.0 or 1.0."""
    pairs = [(True, True)] * 5
    assert ja.krippendorff_alpha_binary(pairs) == (None, True)


def test_krippendorff_alpha_binary_rejects_more_than_two_distinct_values():
    """Matches the error message, since a 3-value input could raise ValueError for another reason."""
    with pytest.raises(ValueError, match="expects exactly 2 distinct values"):
        ja.krippendorff_alpha_binary([("a", "b"), ("b", "c")])


def test_krippendorff_alpha_binary_weights_of_one_reproduce_the_unweighted_formula():
    """With every weight equal to 1 the weighted version reduces to the plain formula."""
    meg = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    owen = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    pairs = list(zip(meg, owen))
    unweighted, _ = ja.krippendorff_alpha_binary(pairs)
    weighted, _ = ja.krippendorff_alpha_binary(pairs, weights=[1.0] * len(pairs))
    assert weighted == pytest.approx(unweighted)


def test_krippendorff_alpha_binary_is_invariant_to_a_uniform_weight_rescaling():
    """Scaling every weight by the same factor doesn't change alpha.

    Weights are normalised to sum to `len(pairs)`, so the `(n-1)` correction uses the real
    observation count."""
    meg = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    owen = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    pairs = list(zip(meg, owen))
    unweighted, _ = ja.krippendorff_alpha_binary(pairs)
    scaled_2, _ = ja.krippendorff_alpha_binary(pairs, weights=[2.0] * len(pairs))
    scaled_200, _ = ja.krippendorff_alpha_binary(pairs, weights=[200.0] * len(pairs))
    assert scaled_2 == pytest.approx(unweighted)
    assert scaled_200 == pytest.approx(unweighted)


def test_raw_agreement_and_alpha_diverge_under_skewed_prevalence():
    """The kappa paradox (Feinstein & Cicchetti 1990): raw agreement 0.95 but alpha exactly 0.

    The one disagreement is on the only False item, so n=40, o_mismatch=1 and
    alpha = 1 - 39*1/39 = 0."""
    judge = [True] * 19 + [False]
    human = [True] * 20  # agrees on the first 19, disagrees on the lone False
    pairs = list(zip(human, judge))
    raw, raw_undef = ja._raw_agreement(pairs)
    alpha, alpha_undef = ja.krippendorff_alpha_binary(pairs)
    assert raw_undef is False and alpha_undef is False
    assert raw == pytest.approx(19 / 20)
    assert alpha == pytest.approx(0.0)
    assert raw - alpha > 0.5  # a divergence too large to attribute to rounding


def test_raw_agreement_undefined_for_empty_input():
    assert ja._raw_agreement([]) == (None, True)


def test_raw_agreement_is_weighted_when_weights_given():
    pairs = [(True, True), (True, False)]
    raw, undefined = ja._raw_agreement(pairs, weights=[3.0, 1.0])
    assert undefined is False
    assert raw == pytest.approx(3.0 / 4.0)


# Stratified sampling: allocate_targets, select_spotcheck_sample
def test_select_spotcheck_sample_oversamples_unsupported_from_a_skewed_corpus():
    """Equal allocation oversamples a rare class.

    `summary:completeness` has 1000 supported and 40 unsupported (~3.8%), but the sample
    should still be close to 50/50."""
    claims = []
    for i in range(1000):
        claims.append(_claim(field="summary", dimension="completeness", verdict=True, item_id=f"sup-{i}"))
    for i in range(40):
        claims.append(_claim(field="summary", dimension="completeness", verdict=False, item_id=f"unsup-{i}"))
    # Give every other cell plenty of population so none is short and none reallocates
    # into the cell under test.
    for cell in ja.ALL_CELLS:
        if cell == ("summary", "completeness", True) or cell == ("summary", "completeness", False):
            continue
        f, d, v = cell
        claims.extend(_claim(field=f, dimension=d, verdict=v, item_id=f"{f}-{d}-{v}-{i}") for i in range(30))
    selection = ja.select_spotcheck_sample(claims, cell_target=25, rng=random.Random(1))
    target_cell_claims = [
        c for c in selection.claims if c.field == "summary" and c.dimension == "completeness"
    ]
    n_supported = sum(1 for c in target_cell_claims if c.judge_verdict is True)
    n_unsupported = sum(1 for c in target_cell_claims if c.judge_verdict is False)
    assert n_supported == 25
    assert n_unsupported == 25  # 40 available >= the 25 target, no shortfall
    # The unsupported class is ~3.8% of the population but ~50% of the sample.
    population_share_unsupported = 40 / 1040
    sample_share_unsupported = n_unsupported / (n_supported + n_unsupported)
    assert sample_share_unsupported > population_share_unsupported * 5


def test_allocate_targets_gives_25_per_cell_when_population_is_ample():
    population = {cell: 100 for cell in ja.ALL_CELLS}
    targets = ja.allocate_targets(population, cell_target=25)
    assert targets == {cell: 25 for cell in ja.ALL_CELLS}


def test_allocate_targets_shortfall_reallocates_to_the_same_fields_other_dimension():
    """A short cell takes all it has and the rest goes to the same field's other dimension."""
    population = {cell: 100 for cell in ja.ALL_CELLS}
    population[("summary", "completeness", False)] = 10
    targets = ja.allocate_targets(population, cell_target=25)
    assert targets[("summary", "completeness", False)] == 10
    assert targets[("summary", "groundedness", False)] == 25 + 15
    # unaffected cells: same field/other verdict, and the other field entirely
    assert targets[("summary", "completeness", True)] == 25
    assert targets[("summary", "groundedness", True)] == 25
    assert targets[("eligibility_summary", "completeness", False)] == 25
    assert targets[("eligibility_summary", "groundedness", False)] == 25


def test_allocate_targets_double_shortfall_does_not_cascade_further():
    """When both dimensions are short the total comes in under 50 rather than being padded."""
    population = {cell: 100 for cell in ja.ALL_CELLS}
    population[("summary", "completeness", False)] = 5
    population[("summary", "groundedness", False)] = 8
    targets = ja.allocate_targets(population, cell_target=25)
    assert targets[("summary", "completeness", False)] == 5
    assert targets[("summary", "groundedness", False)] == 8


def test_allocate_targets_missing_cell_reads_as_zero_population():
    population = {}  # no cell present at all
    targets = ja.allocate_targets(population, cell_target=25)
    assert targets[("summary", "completeness", False)] == 0


def _claim(field="summary", dimension="completeness", verdict=True, item_id=None) -> "ja.JudgedClaim":
    return ja.JudgedClaim(
        item_id=item_id or f"id-{field}-{dimension}-{verdict}-{id(object())}",
        run_config_hash="run-x", judge_config_hash="jcfg-x", funder="Funder X",
        funder_slug="funder-x", replicate=1,
        opportunity_id="opp-1", field=field, dimension=dimension, claim_index=0,
        claim_text="text", claim_type="purpose", judge_verdict=verdict,
        extracted_text="extracted", comparison_display="ref",
    )


def test_select_spotcheck_sample_achieved_is_25_per_cell_with_ample_population():
    """With 30 available per cell and a target of 25, every cell achieves exactly 25."""
    claims = []
    for cell in ja.ALL_CELLS:
        f, d, v = cell
        claims.extend(_claim(field=f, dimension=d, verdict=v, item_id=f"{f}-{d}-{v}-{i}") for i in range(30))
    selection = ja.select_spotcheck_sample(claims, cell_target=25, rng=random.Random(1))
    assert selection.achieved == {cell: 25 for cell in ja.ALL_CELLS}
    assert len(selection.claims) == 25 * len(ja.ALL_CELLS) == 200


def test_select_spotcheck_sample_records_per_item_probability():
    """Each item's recorded probability is achieved / population for its cell."""
    claims = [_claim(item_id=f"c{i}") for i in range(40)]  # summary/completeness/True, pop=40
    for cell in ja.ALL_CELLS:
        if cell == ("summary", "completeness", True):
            continue
        f, d, v = cell
        claims.extend(_claim(field=f, dimension=d, verdict=v, item_id=f"{f}-{d}-{v}-{i}") for i in range(30))
    selection = ja.select_spotcheck_sample(claims, cell_target=25, rng=random.Random(1))
    target_cell_claims = [c for c in selection.claims if (c.field, c.dimension, c.judge_verdict) == ("summary", "completeness", True)]
    assert len(target_cell_claims) == 25
    for c in target_cell_claims:
        assert selection.probabilities[c.item_id] == pytest.approx(25 / 40)


def test_select_spotcheck_sample_row_order_is_randomised():
    """Different seeds give the same items in a different order."""
    claims = []
    for cell in ja.ALL_CELLS:
        f, d, v = cell
        claims.extend(_claim(field=f, dimension=d, verdict=v, item_id=f"{f}-{d}-{v}-{i}") for i in range(3))
    order_a = [c.item_id for c in ja.select_spotcheck_sample(claims, cell_target=3, rng=random.Random(1)).claims]
    order_b = [c.item_id for c in ja.select_spotcheck_sample(claims, cell_target=3, rng=random.Random(2)).claims]
    assert sorted(order_a) == sorted(order_b)  # same membership
    assert order_a != order_b  # different order


# write_spotcheck_csv: hides verdict/justification, item_id
def test_compute_item_id_is_stable_and_opaque():
    id1 = ja.compute_item_id("run-a", "slug-a", 1, "opp-1", "summary", "completeness", 0)
    id2 = ja.compute_item_id("run-a", "slug-a", 1, "opp-1", "summary", "completeness", 0)
    id3 = ja.compute_item_id("run-a", "slug-a", 1, "opp-1", "summary", "completeness", 1)
    assert id1 == id2  # stable
    assert id1 != id3  # a different claim gets a different id
    assert "opp-1" not in id1 and "summary" not in id1  # not a readable concatenation


def test_compute_item_id_distinguishes_funder_and_replicate():
    """Two runs sharing a `run_config_hash` and `opportunity_id` get different `item_id`s."""
    id_funder_a = ja.compute_item_id("run-a", "leverhulme", 1, "opp-1", "summary", "completeness", 0)
    id_funder_b = ja.compute_item_id("run-a", "henry-moore", 1, "opp-1", "summary", "completeness", 0)
    id_replicate_2 = ja.compute_item_id("run-a", "leverhulme", 2, "opp-1", "summary", "completeness", 0)
    assert len({id_funder_a, id_funder_b, id_replicate_2}) == 3


def test_write_spotcheck_csv_withholds_judge_verdict_and_justification(tmp_path):
    claims = [_claim(item_id="c1"), _claim(item_id="c2")]
    selection = ja.select_spotcheck_sample(claims, cell_target=25, rng=random.Random(1))
    out = tmp_path / "spot.csv"
    ja.write_spotcheck_csv(selection, out)
    with open(out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert "judge_verdict" not in fieldnames
    assert "justification" not in fieldnames
    assert len(rows) == 2
    for row in rows:
        assert row["human_code"] == ""
        assert row["human_notes"] == ""


def test_write_spotcheck_csv_records_the_sampling_probability(tmp_path):
    """The CSV records each item's sampling probability."""
    claims = [_claim(item_id=f"c{i}") for i in range(40)]  # summary/completeness/True
    for cell in ja.ALL_CELLS:
        if cell == ("summary", "completeness", True):
            continue
        f, d, v = cell
        claims.extend(_claim(field=f, dimension=d, verdict=v, item_id=f"{f}-{d}-{v}-{i}") for i in range(30))
    selection = ja.select_spotcheck_sample(claims, cell_target=25, rng=random.Random(1))
    out = tmp_path / "spot.csv"
    ja.write_spotcheck_csv(selection, out)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    target_rows = [r for r in rows if r["field"] == "summary" and r["dimension"] == "completeness" and r["item_id"].startswith("c")]
    assert len(target_rows) == 25
    for row in target_rows:
        assert float(row["sampling_probability"]) == pytest.approx(25 / 40)


# join_coded_rows: joins the coded CSV back to stored verdicts
def _write_stored_verdict(tmp_path, *, run_config_hash, judge_config_hash, field, dimension,
                           opportunity_id, claims, funder_slug="funder-x", replicate=1):
    """Write a minimal verdict file where `join_coded_rows` looks for it."""
    from grant_eval.scoring.judge import verdict_path, write_verdict_file

    other_field = "eligibility_summary" if field == "summary" else "summary"
    other_dimension = "groundedness" if dimension == "completeness" else "completeness"
    empty_cell = {"field": field, "dimension": dimension, "records": []}
    data = {
        "funder": "Funder X", "replicate": replicate,
        "config_hash": run_config_hash, "judge_config_hash": judge_config_hash,
        "fields": {
            f"{field}:{dimension}": {
                "field": field, "dimension": dimension,
                "records": [{
                    "opportunity_id": opportunity_id, "outcome": "ok",
                    "record_supported": all(c["supported"] for c in claims),
                    "justification": "j", "claims": claims,
                }],
            },
            f"{field}:{other_dimension}": {**empty_cell, "dimension": other_dimension},
            f"{other_field}:{dimension}": {**empty_cell, "field": other_field},
            f"{other_field}:{other_dimension}": {"field": other_field, "dimension": other_dimension, "records": []},
        },
    }
    path = verdict_path(run_config_hash, judge_config_hash, funder_slug=funder_slug, replicate=replicate, root=tmp_path)
    write_verdict_file(path, data)
    return tmp_path


def _coded_row(*, run_config_hash="run-a", judge_config_hash="jcfg-a", funder_slug="funder-x", replicate=1,
                opportunity_id="opp-1", field="summary", dimension="completeness", claim_index=0,
                human_code="supported", sampling_probability=1.0, item_id=None):
    row_item_id = item_id or ja.compute_item_id(
        run_config_hash, funder_slug, replicate, opportunity_id, field, dimension, claim_index,
    )
    return {
        "item_id": row_item_id, "run_config_hash": run_config_hash, "judge_config_hash": judge_config_hash,
        "funder": "Funder X", "funder_slug": funder_slug, "replicate": str(replicate),
        "opportunity_id": opportunity_id, "field": field, "dimension": dimension,
        "claim_index": str(claim_index), "claim_text": "text", "claim_type": "purpose",
        "extracted_text": "extracted", "comparison_reference": "ref",
        "sampling_probability": str(sampling_probability),
        "human_code": human_code, "human_notes": "",
    }


def test_join_coded_rows_reads_the_judge_verdict_off_the_stored_claim(tmp_path):
    claims = [{"text": "c0", "claim_type": "purpose", "supported": False}]
    root = _write_stored_verdict(
        tmp_path, run_config_hash="run-a", judge_config_hash="jcfg-a",
        field="summary", dimension="completeness", opportunity_id="opp-1", claims=claims,
    )
    row = _coded_row(human_code="unsupported")
    joined, unjoinable, n_uncoded = ja.join_coded_rows([row], results_root=root)
    assert unjoinable == []
    assert n_uncoded == 0
    assert len(joined) == 1
    assert joined[0].judge_verdict is False  # from the stored claim, not the coded row
    assert joined[0].human_code == "unsupported"


def test_join_coded_rows_reports_a_row_with_no_stored_verdict_file_as_unjoinable(tmp_path):
    row = _coded_row(run_config_hash="run-missing")
    joined, unjoinable, n_uncoded = ja.join_coded_rows([row], results_root=tmp_path)
    assert joined == []
    assert len(unjoinable) == 1
    assert unjoinable[0]["item_id"] == row["item_id"]


def test_join_coded_rows_reports_a_claim_index_the_decomposition_no_longer_has(tmp_path):
    """The stored claims no longer have claim_index 2 (e.g. re-judged since the export)."""
    claims = [{"text": "c0", "claim_type": "purpose", "supported": True}]
    root = _write_stored_verdict(
        tmp_path, run_config_hash="run-a", judge_config_hash="jcfg-a",
        field="summary", dimension="completeness", opportunity_id="opp-1", claims=claims,
    )
    row = _coded_row(
        claim_index=2,
        item_id=ja.compute_item_id("run-a", "funder-x", 1, "opp-1", "summary", "completeness", 2),
    )
    joined, unjoinable, n_uncoded = ja.join_coded_rows([row], results_root=root)
    assert joined == []
    assert len(unjoinable) == 1


def test_join_coded_rows_treats_a_blank_human_code_as_uncoded_not_unjoinable(tmp_path):
    claims = [{"text": "c0", "claim_type": "purpose", "supported": True}]
    root = _write_stored_verdict(
        tmp_path, run_config_hash="run-a", judge_config_hash="jcfg-a",
        field="summary", dimension="completeness", opportunity_id="opp-1", claims=claims,
    )
    row = _coded_row(human_code="")
    joined, unjoinable, n_uncoded = ja.join_coded_rows([row], results_root=root)
    assert joined == []
    assert unjoinable == []
    assert n_uncoded == 1


def test_join_coded_rows_reports_a_tampered_item_id_as_unjoinable(tmp_path):
    """A row whose item_id doesn't match its own key fields is not joined."""
    claims = [{"text": "c0", "claim_type": "purpose", "supported": True}]
    root = _write_stored_verdict(
        tmp_path, run_config_hash="run-a", judge_config_hash="jcfg-a",
        field="summary", dimension="completeness", opportunity_id="opp-1", claims=claims,
    )
    row = _coded_row(item_id="not-the-real-hash")
    joined, unjoinable, n_uncoded = ja.join_coded_rows([row], results_root=root)
    assert joined == []
    assert len(unjoinable) == 1


# compute_agreement: per stratum, reweighted overall, confusion counts
def _joined(field, dimension, human_code, judge_verdict, sampling_probability, item_id) -> "ja.JoinedRow":
    return ja.JoinedRow(
        item_id=item_id, field=field, dimension=dimension, human_code=human_code,
        judge_verdict=judge_verdict, sampling_probability=sampling_probability,
    )


def _hand_computed_agreement_rows():
    """Two strata with different sampling probabilities plus one `not_a_claim` row."""
    rows = [
        # stratum "summary:completeness:supported" (judge_verdict=True), p=0.5, n=4
        _joined("summary", "completeness", "supported", True, 0.5, "a1"),
        _joined("summary", "completeness", "supported", True, 0.5, "a2"),
        _joined("summary", "completeness", "supported", True, 0.5, "a3"),
        _joined("summary", "completeness", "unsupported", True, 0.5, "a4"),  # the one disagreement
        # stratum "summary:completeness:unsupported" (judge_verdict=False), p=0.2, n=2
        _joined("summary", "completeness", "unsupported", False, 0.2, "b1"),
        _joined("summary", "completeness", "unsupported", False, 0.2, "b2"),
        # not_a_claim elsewhere, excluded from every binary computation
        _joined("eligibility_summary", "completeness", "not_a_claim", True, 0.5, "c1"),
    ]
    return rows


def test_compute_agreement_matches_hand_computed_values():
    """Every figure here is worked out by hand from `_hand_computed_agreement_rows`.

    stratum "summary:completeness:supported" (n=4): 3 agree, 1 disagrees, so
      raw = 3/4 = 0.75. Coincidence pairs: 3x(True,True), 1x(False,True); o_mismatch=1,
      n_true=7 (6 from the 3 matches + 1 from the mismatch's judge side), n_false=1
      (the mismatch's human side); n=8; alpha = 1 - 7*1/(7*1) = 0.

    stratum "summary:completeness:unsupported" (n=2): both agree (human=unsupported,
      judge=False), so raw = 1.0, but every rating is False. With zero variance alpha is
      undefined (None), not 1.0, which is why raw agreement and alpha are kept separate.

    reweighted overall (weight = 1/p, normalised to sum to 6): raw weights are 2 (1/0.5)
      for stratum A's 4 rows and 5 (1/0.2) for stratum B's 2 rows, summing to 18.
      Normalising multiplies every weight by 6/18 = 1/3, giving 2/3 for A's rows and 5/3
      for B's. raw_overall doesn't change (a common scale cancels): weighted agree =
      (2/3)*3 + (5/3)*2 = 16/3; total weight = 6; raw_overall = (16/3)/6 = 8/9.
      Weighted alpha uses the true count for n: o_mismatch = 2/3; n_true = (2/3)*2*3
      + 2/3 = 14/3; n_false = 2/3 + (5/3)*2*2 = 22/3; n = 2*6 = 12;
      alpha = 1 - 11*(2/3)/((14/3)*(22/3)) = 1 - 3/14 = 11/14.

    by_field/by_dimension: all 6 binary rows are field=summary, dimension=completeness
      (5 of 6 agree, so 5/6); eligibility_summary and groundedness have no binary rows
      (the lone eligibility_summary row is the excluded not_a_claim), so undefined.

    confusion: tallied over all 7 rows (not_a_claim included) by (human_code,
      judge verdict as supported/unsupported)."""
    report = ja.compute_agreement(_hand_computed_agreement_rows())

    assert report.n_joined == 7
    assert report.not_a_claim_rate == pytest.approx(1 / 7)
    assert report.not_a_claim_rate_undefined is False

    supported_cell = report.by_stratum["summary:completeness:supported"]
    assert supported_cell["n"] == 4
    assert supported_cell["raw_agreement"] == pytest.approx(0.75)
    assert supported_cell["raw_agreement_undefined"] is False
    assert supported_cell["alpha"] == pytest.approx(0.0)
    assert supported_cell["alpha_undefined"] is False

    unsupported_cell = report.by_stratum["summary:completeness:unsupported"]
    assert unsupported_cell["n"] == 2
    assert unsupported_cell["raw_agreement"] == pytest.approx(1.0)
    assert unsupported_cell["alpha"] is None
    assert unsupported_cell["alpha_undefined"] is True

    assert report.raw_agreement_overall_reweighted == pytest.approx(8 / 9)
    assert report.raw_agreement_overall_reweighted_undefined is False
    assert report.alpha_overall_reweighted == pytest.approx(11 / 14)
    assert report.alpha_overall_reweighted_undefined is False

    assert report.by_field["summary"]["n"] == 6
    assert report.by_field["summary"]["raw_agreement"] == pytest.approx(5 / 6)
    assert report.by_field["eligibility_summary"]["n"] == 0
    assert report.by_field["eligibility_summary"]["raw_agreement_undefined"] is True

    assert report.by_dimension["completeness"]["raw_agreement"] == pytest.approx(5 / 6)
    assert report.by_dimension["groundedness"]["raw_agreement_undefined"] is True

    assert report.confusion["supported"] == {"supported": 3, "unsupported": 0}
    assert report.confusion["unsupported"] == {"supported": 1, "unsupported": 2}
    assert report.confusion["not_a_claim"] == {"supported": 1, "unsupported": 0}


def _weighted_agreement_rows():
    """Four strata across both fields and dimensions at different sampling probabilities.

    This way every weighted figure differs from its unweighted twin and from the overall."""
    return [
        # summary:completeness:supported, p=0.5 (weight 2): 3 agree, 1 disagrees
        _joined("summary", "completeness", "supported", True, 0.5, "a1"),
        _joined("summary", "completeness", "supported", True, 0.5, "a2"),
        _joined("summary", "completeness", "supported", True, 0.5, "a3"),
        _joined("summary", "completeness", "unsupported", True, 0.5, "a4"),
        # summary:completeness:unsupported, p=0.2 (weight 5): both agree
        _joined("summary", "completeness", "unsupported", False, 0.2, "b1"),
        _joined("summary", "completeness", "unsupported", False, 0.2, "b2"),
        # eligibility_summary:groundedness:supported, p=0.5 (weight 2): 1 agrees, 1 does not
        _joined("eligibility_summary", "groundedness", "supported", True, 0.5, "d1"),
        _joined("eligibility_summary", "groundedness", "unsupported", True, 0.5, "d2"),
        # eligibility_summary:groundedness:unsupported, p=0.1 (weight 10): agrees
        _joined("eligibility_summary", "groundedness", "unsupported", False, 0.1, "e1"),
        # one not_a_claim, p=0.25 (weight 4), excluded from every binary computation
        _joined("summary", "groundedness", "not_a_claim", True, 0.25, "f1"),
    ]


def test_by_field_by_dimension_and_not_a_claim_are_reweighted_for_the_sampling_design():
    """by_field/by_dimension agreement and not_a_claim_rate are reweighted, with unweighted twins.

    The sample oversamples `unsupported`, so plain sample proportions don't estimate the
    corpus. Expected values from `_weighted_agreement_rows` (weight = 1/p):

    by_field summary: binary rows a1..a4 (w=2) and b1,b2 (w=5).
      weighted: agree = 3*2 + 2*5 = 16, total = 4*2 + 2*5 = 18, so 8/9.
      unweighted: 5 of 6 agree, so 5/6.
    by_field eligibility_summary: d1 (w=2, agrees), d2 (w=2, disagrees), e1 (w=10,
      agrees). weighted: 12/14 = 6/7. unweighted: 2/3.
    by_dimension completeness: the same rows as summary, so 8/9 weighted, 5/6 not.
    by_dimension groundedness: the same rows as eligibility_summary (f1 is
      not_a_claim and excluded), so 6/7 weighted, 2/3 not.
    not_a_claim rate: over all 10 rows, weighted numerator = 4 (f1's weight);
      total weight = 4*2 + 2*5 + 2*2 + 10 + 4 = 36, so 1/9. unweighted: 1/10."""
    report = ja.compute_agreement(_weighted_agreement_rows())

    assert report.by_field["summary"]["raw_agreement_reweighted"] == pytest.approx(8 / 9)
    assert report.by_field["summary"]["raw_agreement"] == pytest.approx(5 / 6)
    assert report.by_field["eligibility_summary"]["raw_agreement_reweighted"] == pytest.approx(6 / 7)
    assert report.by_field["eligibility_summary"]["raw_agreement"] == pytest.approx(2 / 3)

    assert report.by_dimension["completeness"]["raw_agreement_reweighted"] == pytest.approx(8 / 9)
    assert report.by_dimension["completeness"]["raw_agreement"] == pytest.approx(5 / 6)
    assert report.by_dimension["groundedness"]["raw_agreement_reweighted"] == pytest.approx(6 / 7)
    assert report.by_dimension["groundedness"]["raw_agreement"] == pytest.approx(2 / 3)

    assert report.not_a_claim_rate_reweighted == pytest.approx(1 / 9)
    assert report.not_a_claim_rate_reweighted_undefined is False
    assert report.not_a_claim_rate == pytest.approx(1 / 10)


def test_reweighted_by_field_is_undefined_not_zero_when_a_field_has_no_binary_rows():
    """A field with no binary rows gives None plus its flag for the reweighted figure, not 0.0."""
    report = ja.compute_agreement(_hand_computed_agreement_rows())
    cell = report.by_field["eligibility_summary"]
    assert cell["raw_agreement_reweighted"] is None
    assert cell["raw_agreement_reweighted_undefined"] is True


def test_format_agreement_report_never_prints_an_unweighted_figure_unlabelled():
    """The printed output shows reweighted and unweighted figures, both labelled."""
    report = ja.compute_agreement(_weighted_agreement_rows())
    text = ja.format_agreement_report(report, n_uncoded=0, unjoinable=[])
    assert ja.UNWEIGHTED_SAMPLE_CAVEAT in text
    assert "reweighted=0.889" in text          # by field: summary, 8/9
    assert "unweighted=0.833" in text          # by field: summary, 5/6, labelled
    assert "reweighted=0.857" in text          # by dimension: groundedness, 6/7
    assert "unweighted=0.667" in text          # by dimension: groundedness, 2/3


def test_agreement_report_to_dict_carries_both_the_reweighted_and_unweighted_figures():
    report = ja.compute_agreement(_weighted_agreement_rows())
    payload = ja.agreement_report_to_dict(report, n_uncoded=0, unjoinable=[])
    assert payload["by_field"]["summary"]["raw_agreement_reweighted"] == pytest.approx(8 / 9)
    assert payload["by_field"]["summary"]["raw_agreement"] == pytest.approx(5 / 6)
    assert payload["not_a_claim_rate_reweighted"] == pytest.approx(1 / 9)
    assert payload["not_a_claim_rate"] == pytest.approx(1 / 10)
    assert payload["unweighted_sample_caveat"] == ja.UNWEIGHTED_SAMPLE_CAVEAT


def test_agreement_report_to_dict_serialises_undefined_as_null_not_zero():
    report = ja.compute_agreement(_hand_computed_agreement_rows())
    payload = ja.agreement_report_to_dict(report, n_uncoded=2, unjoinable=[{"item_id": "x", "reason": "r"}])
    assert payload["by_field"]["eligibility_summary"]["raw_agreement"] is None
    assert payload["by_field"]["eligibility_summary"]["raw_agreement_undefined"] is True
    assert payload["n_uncoded"] == 2
    assert payload["n_unjoinable"] == 1
    assert payload["unjoinable"] == [{"item_id": "x", "reason": "r"}]
    # The reweighted alpha carries a caveat field in the JSON too.
    assert payload["alpha_overall_reweighted_is_published_estimator"] is False
    assert payload["alpha_overall_reweighted_caveat"] == ja.ALPHA_REWEIGHTED_CAVEAT
    assert payload["by_stratum_alpha_caveat"] == ja.PER_STRATUM_ALPHA_CAVEAT


def test_format_agreement_report_states_when_raw_and_alpha_diverge():
    """The divergence note is printed for the kappa-paradox fixture (raw=0.95, alpha=0.0)."""
    rows = []
    for i in range(19):
        rows.append(_joined("summary", "completeness", "supported", True, 1.0, f"m{i}"))
    rows.append(_joined("summary", "completeness", "supported", False, 1.0, "m19"))
    report = ja.compute_agreement(rows)
    text = ja.format_agreement_report(report, n_uncoded=0, unjoinable=[])
    assert "diverge" in text.lower()
    assert "Feinstein" in text


def test_format_agreement_report_lists_unjoinable_ids_not_just_the_count():
    """An unjoinable item_id shows up in the printed output, not just in `--json`."""
    report = ja.compute_agreement([])
    unjoinable = [{"item_id": "deadbeef1234", "reason": "no stored verdict file for run 'ghost-run'"}]
    text = ja.format_agreement_report(report, n_uncoded=0, unjoinable=unjoinable)
    assert "deadbeef1234" in text
    assert "no stored verdict file for run 'ghost-run'" in text


# load_judged_records / explode_claims: rebuilding export context from the original
# results plus the stored verdict
def test_load_judged_records_reconstructs_context_from_the_original_results(tmp_path):
    """Completeness compares against gold text; groundedness shows the mirror path but
    compares against the cleaned page text."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers with a doctorate.</p></body></html>")
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Gold summary text")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    batch = judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True), _ok_reply(True)]),
        judge_config=judge_config, results_root=results_root,
    )
    assert batch.errors == []

    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
    )
    assert loaded.skipped_unjudged == []
    assert loaded.skipped_unlabelled_funder == []
    assert loaded.mismatched_funder == []
    by_dimension = {r.dimension: r for r in loaded.records}
    assert set(by_dimension) == {"completeness", "groundedness"}
    assert by_dimension["completeness"].funder_slug == SLUG
    assert by_dimension["completeness"].replicate == 1

    completeness = by_dimension["completeness"]
    assert completeness.extracted_text == "Funds x"
    assert completeness.comparison_text == "Gold summary text"
    assert completeness.comparison_display == "Gold summary text"

    groundedness = by_dimension["groundedness"]
    assert groundedness.extracted_text == "Funds x"
    assert "early career researchers" in groundedness.comparison_text
    assert groundedness.comparison_display.endswith("scheme.html")
    assert groundedness.comparison_display != groundedness.comparison_text


def test_load_judged_records_names_and_counts_records_not_stored_ok(tmp_path):
    """Records with a non-OK stored outcome are named and counted, not silently skipped."""
    gt = _ground_truth(LEV, [
        _gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Gold 1"),
        _gt_row(LEV, "g2", "Scheme Two", "https://example.org/scheme-two", summary="Gold 2"),
    ])
    opp1 = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds 1")
    opp2 = _judge_extracted(LEV, "g2", "Scheme Two", "https://example.org/scheme-two", summary="Funds 2")
    result = _judge_result(LEV, [opp1, opp2])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    # opp1's completeness reply parses and opp2's doesn't. Both groundedness calls
    # point at NOWHERE and are excluded before reaching the backend.
    judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=NOWHERE,
        backend=FakeJudgeBackend([_ok_reply(True), "not json"]),
        judge_config=judge_config, results_root=results_root,
    )

    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=NOWHERE,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
    )
    completeness_records = [r for r in loaded.records if r.dimension == "completeness"]
    assert {r.opportunity_id for r in completeness_records} == {"g1"}  # g2 excluded, not merged in

    excluded_completeness = [e for e in loaded.excluded_records if e["dimension"] == "completeness"]
    assert {e["opportunity_id"] for e in excluded_completeness} == {"g2"}
    assert excluded_completeness[0]["reason"] == "judge_parse_failure"
    # The stored parse error comes along too, not just the reason.
    assert excluded_completeness[0]["error"]


def test_load_judged_records_skips_a_run_with_no_stored_verdict_and_reports_it(tmp_path):
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=NOWHERE,
        results_root=tmp_path / "no-such-results", judge_config_hash="whatever-hash",
    )
    assert loaded.records == []
    assert loaded.skipped_unjudged == [path]


def test_load_judged_records_reports_an_unregistered_funder_and_a_funder_mismatch(tmp_path):
    """An unlabelled funder and a stored verdict for the wrong funder are both named and counted."""
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme-one")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme-one", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    # No registry entry at all for LEV -> no slug -> skipped_unlabelled_funder.
    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry={}, snapshot_root=NOWHERE,
        results_root=tmp_path / "no-such-results", judge_config_hash="whatever-hash",
    )
    assert loaded.records == []
    assert loaded.skipped_unjudged == []
    assert loaded.skipped_unlabelled_funder == [path]

    # A verdict is at the right path but names a different funder, which the loader
    # must report.
    from grant_eval.scoring.judge import verdict_path as _verdict_path, write_verdict_file as _write_verdict_file

    results_root = tmp_path / "judge-results"
    vpath = _verdict_path("test", "whatever-hash", funder_slug=SLUG, replicate=1, root=results_root)
    _write_verdict_file(vpath, {"funder": "Someone Else Entirely", "replicate": 1, "fields": {}})
    loaded2 = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=NOWHERE,
        results_root=results_root, judge_config_hash="whatever-hash",
    )
    assert loaded2.records == []
    assert len(loaded2.mismatched_funder) == 1
    assert loaded2.mismatched_funder[0]["expected_funder"] == LEV
    assert loaded2.mismatched_funder[0]["stored_funder"] == "Someone Else Entirely"


def test_explode_claims_computes_one_item_id_per_claim_index():
    rec = ja.JudgedRecord(
        run_config_hash="run-a", judge_config_hash="jcfg-a", funder="Funder X",
        funder_slug="funder-x", replicate=1,
        opportunity_id="opp-1", field="summary", dimension="completeness",
        extracted_text="x", comparison_text="y", comparison_display="y",
        snapshot_date=SNAP_DATE,
        claims=[
            {"text": "c0", "claim_type": "purpose", "supported": True},
            {"text": "c1", "claim_type": "eligibility", "supported": False},
        ],
        record_supported=False,
    )
    claims = ja.explode_claims([rec])
    assert [c.claim_index for c in claims] == [0, 1]
    assert [c.judge_verdict for c in claims] == [True, False]
    assert claims[0].item_id == ja.compute_item_id("run-a", "funder-x", 1, "opp-1", "summary", "completeness", 0)
    assert claims[1].item_id == ja.compute_item_id("run-a", "funder-x", 1, "opp-1", "summary", "completeness", 1)


# Test-retest: judge self-agreement, drawn separately from the spot-check
def _judged_record(*, run_config_hash="run-a", funder_slug="funder-x", replicate=1, opportunity_id="opp-1",
                    field="summary", dimension="completeness", record_supported=True) -> "ja.JudgedRecord":
    return ja.JudgedRecord(
        run_config_hash=run_config_hash, judge_config_hash="jcfg-a", funder="Funder X",
        funder_slug=funder_slug, replicate=replicate,
        opportunity_id=opportunity_id, field=field, dimension=dimension,
        extracted_text="extracted", comparison_text="gold or page text", comparison_display="ref",
        snapshot_date=SNAP_DATE,
        claims=[{"text": "c0", "claim_type": "purpose", "supported": record_supported}],
        record_supported=record_supported,
    )


def test_select_test_retest_sample_excludes_the_spot_check_sample_records():
    """`exclude_keys` is honoured (the end-to-end test below checks a real export)."""
    records = [_judged_record(opportunity_id=f"opp-{i}") for i in range(5)]
    exclude = {ja.record_key(records[0]), ja.record_key(records[2])}
    sample = ja.select_test_retest_sample(records, exclude_keys=exclude, n=10, rng=random.Random(1))
    assert {ja.record_key(r) for r in sample} == {ja.record_key(records[1]), ja.record_key(records[3]), ja.record_key(records[4])}


def test_spotcheck_record_keys_and_test_retest_draw_disjoint_samples_end_to_end(tmp_path):
    """A test-retest sample is disjoint from a real spot-check export.

    `spotcheck_record_keys` uses the same seeded selection as `cli_export_spotcheck`;
    checked here by round-tripping an actual export CSV."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers.</p></body></html>")
    gt_rows = [_gt_row(LEV, f"g{i}", f"Scheme {i}", "https://example.org/scheme", summary=f"Gold {i}") for i in range(30)]
    gt = _ground_truth(LEV, gt_rows)
    opps = [_judge_extracted(LEV, f"g{i}", f"Scheme {i}", "https://example.org/scheme", summary=f"Funds {i}") for i in range(30)]
    result = _judge_result(LEV, opps)
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    # Only enough replies for the 30 completeness calls, so the groundedness calls get
    # judge_call_failure and are excluded. Only completeness records load, all supported.
    judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True)] * 30),
        judge_config=judge_config, results_root=results_root,
    )

    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
    )
    out_csv = tmp_path / "spot.csv"
    # n_total=200 so cell_target is 25 (200 // 8), the same default
    # `spotcheck_record_keys` uses; otherwise the two selections would differ.
    ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=200, out_path=out_csv, json_output=False,
    )
    with open(out_csv, newline="", encoding="utf-8") as fh:
        exported_record_keys = {
            (row["run_config_hash"], row["funder_slug"], int(row["replicate"]),
             row["opportunity_id"], row["field"], row["dimension"])
            for row in csv.DictReader(fh)
        }
    assert exported_record_keys  # the fixture exported something

    exclude_keys = ja.spotcheck_record_keys(loaded.records)
    assert exclude_keys == exported_record_keys  # matches the real export exactly

    sample = ja.select_test_retest_sample(loaded.records, exclude_keys=exclude_keys, n=100)
    sample_keys = {ja.record_key(r) for r in sample}
    assert sample_keys.isdisjoint(exported_record_keys)


def test_select_test_retest_sample_caps_at_n():
    records = [_judged_record(opportunity_id=f"opp-{i}") for i in range(10)]
    sample = ja.select_test_retest_sample(records, exclude_keys=set(), n=3, rng=random.Random(1))
    assert len(sample) == 3


def test_run_test_retest_reports_agreement_between_fresh_and_stored_verdicts():
    """Fresh replies are compared with each record's stored verdict; parse failures are excluded.

    Record 1 flips, record 2 matches and record 3 fails to parse."""
    records = [
        _judged_record(opportunity_id="opp-1", record_supported=True),
        _judged_record(opportunity_id="opp-2", record_supported=True),
        _judged_record(opportunity_id="opp-3", record_supported=True),
    ]
    backend = FakeJudgeBackend([_ok_reply(False), _ok_reply(True), "not json"])
    result = ja.run_test_retest(records, backend=backend, judge_config=_judge_config())
    assert result.n_compared == 2  # the parse failure is excluded
    assert result.n_agree == 1  # only opp-2's fresh reply matched
    assert result.n_fresh_failed == 1
    assert result.self_agreement == pytest.approx(0.5)
    assert result.self_agreement_undefined is False


def test_run_test_retest_self_agreement_undefined_when_every_fresh_call_fails():
    records = [_judged_record()]
    backend = FakeJudgeBackend([RuntimeError("boom")])
    result = ja.run_test_retest(records, backend=backend, judge_config=_judge_config())
    assert result.n_compared == 0
    assert result.self_agreement is None
    assert result.self_agreement_undefined is True


def test_run_test_retest_stops_at_the_spend_cap(tmp_path):
    """`run_test_retest` stops once spend reaches `max_spend_usd`.

    Five records at $0.01 each with a $0.025 cap must not all run."""
    records = [_judged_record(opportunity_id=f"opp-{i}") for i in range(5)]
    backend = FakeJudgeBackend(
        [_ok_reply(True)] * 5, costs=[0.01, 0.01, 0.01, 0.01, 0.01],
    )
    result = ja.run_test_retest(records, backend=backend, judge_config=_judge_config(), max_spend_usd=0.025)
    assert result.cap_hit is True
    assert result.n_compared < 5
    assert result.n_skipped_spend_cap == 5 - result.n_compared
    assert result.total_spend_usd == pytest.approx(backend.total_cost_usd)
    assert backend.total_cost_usd < 0.05  # didn't run all 5


def test_run_test_retest_refuses_a_backend_it_cannot_charge():
    """`run_test_retest` refuses a backend without `total_cost_usd`, like `run_judge_batch`."""
    class _NoSpendBackend:
        name = "no-spend"

        def complete(self, **kw):
            return _ok_reply(True)

    with pytest.raises(AttributeError):
        ja.run_test_retest([_judged_record()], backend=_NoSpendBackend(), judge_config=_judge_config())


# cli_export_spotcheck / cli_agreement end to end (fake backend, no live calls)
def test_cli_export_spotcheck_requires_out(capsys):
    rc = ja.cli_export_spotcheck(
        [], ground_truth=_one_row_gt(), registry=_registry(), snapshot_root=NOWHERE,
        results_root=Path("/nonexistent"), judge_config_hash="jcfg", n_total=200,
        out_path=None, json_output=False,
    )
    assert rc == 1
    assert "--out" in capsys.readouterr().err


def test_cli_export_spotcheck_writes_a_csv_from_a_real_judged_run(tmp_path, capsys):
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers with a doctorate.</p></body></html>")
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Gold summary text")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    batch = judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True), _ok_reply(False)]),
        judge_config=judge_config, results_root=results_root,
    )
    assert batch.errors == []

    out_csv = tmp_path / "spot.csv"
    rc = ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=8, out_path=out_csv, json_output=False,
    )
    assert rc == 0
    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # one completeness claim, one groundedness claim
    assert {r["dimension"] for r in rows} == {"completeness", "groundedness"}
    for r in rows:
        assert r["human_code"] == ""
    out = capsys.readouterr().out
    assert "wrote 2 claim(s)" in out
    # The funder spread is recorded in the export itself, via the `funder` column.
    assert {r["funder"] for r in rows} == {LEV}
    assert {r["funder_slug"] for r in rows} == {SLUG}
    assert {r["replicate"] for r in rows} == {"1"}


def test_cli_export_spotcheck_names_every_excluded_record_not_just_counting_them(tmp_path, capsys):
    """`--json` lists the excluded records in full, not just a count."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers.</p></body></html>")
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Gold summary text")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True), _ok_reply(True)]),
        judge_config=judge_config, results_root=results_root,
    )
    # Turn the stored completeness record into a failure so the loader excludes it.
    vpath = judgemod.verdict_path(
        "test", judge_config.config_hash, funder_slug=SLUG, replicate=1, root=results_root,
    )
    stored = judgemod.read_verdict_file(vpath)
    stored["fields"]["summary:completeness"]["records"][0]["outcome"] = "judge_parse_failure"
    judgemod.write_verdict_file(vpath, stored)

    rc = ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=8, out_path=tmp_path / "spot.csv", json_output=True,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_excluded_records"] == 1
    assert payload["excluded_records"] == [{
        "path": str(path), "opportunity_id": "g1", "field": "summary",
        "dimension": "completeness", "reason": "judge_parse_failure",
        # The record was edited by hand with no error text, so this is None.
        "error": None,
    }]


def _many_judged_records(tmp_path, n: int = 60):
    """Judge `n` completeness records for one funder through the real persistence path.

    Returns `(result_path, ground_truth, snapshot_root, results_root, judge_config, loaded)`."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers.</p></body></html>")
    gt_rows = [_gt_row(LEV, f"g{i}", f"Scheme {i}", "https://example.org/scheme", summary=f"Gold {i}") for i in range(n)]
    gt = _ground_truth(LEV, gt_rows)
    opps = [_judge_extracted(LEV, f"g{i}", f"Scheme {i}", "https://example.org/scheme", summary=f"Funds {i}") for i in range(n)]
    result = _judge_result(LEV, opps)
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True)] * n),
        judge_config=judge_config, results_root=results_root,
    )
    loaded = ja.load_judged_records(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
    )
    return path, gt, root, results_root, judge_config, loaded


def _csv_record_keys(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {
            (row["run_config_hash"], row["funder_slug"], int(row["replicate"]),
             row["opportunity_id"], row["field"], row["dimension"])
            for row in csv.DictReader(fh)
        }


def test_spotcheck_record_keys_from_csv_reads_the_export_that_actually_happened(tmp_path):
    """Test-retest exclusions come from the export being coded, not a regenerated default.

    Exported at n_total=80 (cell_target 10) so it can't match the 200/25 default by chance."""
    path, gt, root, results_root, judge_config, loaded = _many_judged_records(tmp_path)
    out_csv = tmp_path / "spot.csv"
    ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=80, out_path=out_csv, json_output=False,
    )
    exported = _csv_record_keys(out_csv)
    # cell_target=10 for summary:completeness:supported, plus the 10 that
    # summary:groundedness:supported can't fill (it's empty), so 20 in total.
    assert len(exported) == 20

    keys, malformed = ja.spotcheck_record_keys_from_csv(out_csv)
    assert keys == exported
    assert malformed == []
    # The default regeneration would have excluded a different set.
    assert keys != ja.spotcheck_record_keys(loaded.records)


def test_spotcheck_record_keys_from_csv_names_a_malformed_row_rather_than_dropping_it(tmp_path):
    """A CSV row without a usable record key (bad `replicate`) is reported, not skipped."""
    bad_csv = tmp_path / "bad.csv"
    with open(bad_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ja.SPOTCHECK_FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "item_id": "good", "run_config_hash": "rc", "funder_slug": SLUG,
            "replicate": "1", "opportunity_id": "g1", "field": "summary",
            "dimension": "completeness",
        })
        writer.writerow({
            "item_id": "bad", "run_config_hash": "rc", "funder_slug": SLUG,
            "replicate": "not-a-number", "opportunity_id": "g2", "field": "summary",
            "dimension": "completeness",
        })
    keys, malformed = ja.spotcheck_record_keys_from_csv(bad_csv)
    assert keys == {("rc", SLUG, 1, "g1", "summary", "completeness")}
    assert len(malformed) == 1
    assert malformed[0]["item_id"] == "bad"


def test_cli_test_retest_excludes_the_export_it_is_given_not_a_regenerated_default(tmp_path, capsys):
    """The CLI excludes what was exported: 60 candidates minus 20 exported leaves 40.

    Regenerating the default selection would exclude 50 and could overlap with the items
    being coded."""
    path, gt, root, results_root, judge_config, loaded = _many_judged_records(tmp_path)
    out_csv = tmp_path / "spot.csv"
    ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=80, out_path=out_csv, json_output=False,
    )
    capsys.readouterr()

    rc = ja.cli_test_retest(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        backend=FakeJudgeBackend([_ok_reply(True)] * 60), judge_config=judge_config,
        n=60, max_spend_usd=1e9, json_output=True, spotcheck_csv=out_csv,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exclusion_source"] == "exported_csv"
    assert payload["spotcheck_csv"] == str(out_csv)
    assert payload["n_excluded_as_spotcheck"] == 20
    assert payload["n_drawn"] == 40
    assert payload["n_malformed_spotcheck_rows"] == 0


def test_cli_test_retest_without_an_export_says_which_selection_it_regenerated(tmp_path, capsys):
    """Without an export, the regenerated default exclusion set is labelled with its `cell_target`."""
    path, gt, root, results_root, judge_config, _loaded = _many_judged_records(tmp_path)
    rc = ja.cli_test_retest(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        backend=FakeJudgeBackend([_ok_reply(True)] * 60), judge_config=judge_config,
        n=60, max_spend_usd=1e9, json_output=True,
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["exclusion_source"] == "regenerated_default_selection"
    assert payload["spotcheck_csv"] is None
    assert payload["spotcheck_cell_target"] == ja.SPOTCHECK_CELL_TARGET
    assert payload["n_excluded_as_spotcheck"] == 50
    assert payload["n_drawn"] == 10
    assert "--spotcheck-csv" in captured.err


def test_build_parser_accepts_spotcheck_csv_for_test_retest():
    args = judgemod.build_parser().parse_args(["r.json", "--test-retest", "200", "--spotcheck-csv", "spot.csv"])
    assert args.test_retest == 200
    assert args.spotcheck_csv == "spot.csv"


def test_main_threads_the_spotcheck_csv_through_to_cli_test_retest(tmp_path, monkeypatch):
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([]))

    captured = {}
    monkeypatch.setattr(
        ja, "cli_test_retest",
        lambda paths, **kw: (captured.update(spotcheck_csv=kw["spotcheck_csv"]), 0)[1],
    )
    path = _write_run(tmp_path, "a", "run-a")
    assert judgemod.main([str(path), "--test-retest", "50", "--spotcheck-csv", "spot.csv"]) == 0
    assert captured["spotcheck_csv"] == Path("spot.csv")


def test_cli_agreement_reports_raw_agreement_and_alpha(tmp_path, capsys):
    """Round-trip a real export through `--agreement` with hand-filled codes."""
    root = _build_snapshot(tmp_path, page_html="<html><body><p>Funds early career researchers with a doctorate.</p></body></html>")
    gt = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Gold summary text")])
    tp_opp = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x")
    result = _judge_result(LEV, [tp_opp])
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    judgemod.run_judge_batch(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        backend=FakeJudgeBackend([_ok_reply(True), _ok_reply(False)]),
        judge_config=judge_config, results_root=results_root,
    )
    out_csv = tmp_path / "spot.csv"
    ja.cli_export_spotcheck(
        [path], ground_truth=gt, registry=_registry(), snapshot_root=root,
        results_root=results_root, judge_config_hash=judge_config.config_hash,
        n_total=8, out_path=out_csv, json_output=False,
    )
    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["human_code"] = "supported"  # the human agrees with every judge verdict
    coded_csv = tmp_path / "coded.csv"
    with open(coded_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ja.SPOTCHECK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    rc = ja.cli_agreement(coded_csv, results_root=results_root, json_output=False)
    assert rc == 0
    out = capsys.readouterr().out
    # one row's judge verdict was `unsupported` and the human coded `supported`,
    # so one of the two joined rows disagrees.
    assert "raw agreement: 0.500" in out
    # The reweighted alpha is always labelled as not a published estimator, in both
    # the CLI text and the constant.
    assert ja.ALPHA_REWEIGHTED_CAVEAT in out
    assert ja.PER_STRATUM_ALPHA_CAVEAT in out


# Two funders sharing a config_hash must not collide on one verdict file.
# `agent.extract.config_hash` leaves out funder, replicate and snapshot_date, so
# run_config_hash alone doesn't identify a run.
def test_two_funders_sharing_a_config_hash_do_not_collide_on_one_verdict_file(tmp_path):
    """Two funders in the same grid arm are both judged, with no false resume hits.

    `workers=1` keeps the submission order deterministic."""
    from grant_eval.dataset.registry import FunderRecord as _FunderRecord
    from grant_eval.scoring.groundtruth import GroundTruth as _GroundTruth

    FUNDER_B = "Henry Moore Foundation"
    SLUG_B = "henry-moore"
    root = _build_snapshot(tmp_path, slug=SLUG, page_html="<html><body><p>Funds early career researchers.</p></body></html>")
    _build_snapshot(tmp_path, slug=SLUG_B, page_html="<html><body><p>Funds sculpture conservation.</p></body></html>")
    registry = dict(_registry())
    registry[SLUG_B] = _FunderRecord(
        slug=SLUG_B, display_name=FUNDER_B, split=Split.TUNE,
        snapshot_date=SNAP_DATE, entry_url="https://example.org/", content_hash="deadbeef",
    )

    gt_a = _ground_truth(LEV, [_gt_row(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Gold A")])
    gt_b = _ground_truth(FUNDER_B, [_gt_row(FUNDER_B, "g1", "Scheme One", "https://example.org/scheme", summary="Gold B")])
    gt = _GroundTruth(funders={**gt_a.funders, **gt_b.funders})

    SHARED_CONFIG_HASH = "shared-arm-hash"  # same grid arm: model/prompt/temperature identical
    opp_a = _judge_extracted(LEV, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x (A)")
    opp_b = _judge_extracted(FUNDER_B, "g1", "Scheme One", "https://example.org/scheme", summary="Funds x (B)")
    result_a = _judge_result(LEV, [opp_a], config_hash=SHARED_CONFIG_HASH)
    result_b = _judge_result(FUNDER_B, [opp_b], config_hash=SHARED_CONFIG_HASH)
    path_a = tmp_path / "a.json"
    path_a.write_text(result_a.model_dump_json())
    path_b = tmp_path / "b.json"
    path_b.write_text(result_b.model_dump_json())

    results_root = tmp_path / "judge-results"
    judge_config = _judge_config()
    backend = FakeJudgeBackend([_ok_reply(True)] * 4)  # up to 2 calls each (completeness/groundedness x summary)
    batch = judgemod.run_judge_batch(
        [path_a, path_b], ground_truth=gt, registry=registry, snapshot_root=root,
        backend=backend, judge_config=judge_config, results_root=results_root, workers=1,
    )
    assert batch.errors == []
    assert len(batch.outcomes) == 2
    outcomes_by_path = {o.source_path: o for o in batch.outcomes}

    # Neither run may be a false "resume" hit off the other's file.
    assert outcomes_by_path[path_a].resumed is False
    assert outcomes_by_path[path_b].resumed is False
    # Each run's own stored verdict must name its OWN funder, never the other's.
    assert outcomes_by_path[path_a].to_dict["funder"] == LEV
    assert outcomes_by_path[path_b].to_dict["funder"] == FUNDER_B


# judge_notebook.py helpers
import grant_eval.scoring.judge_notebook as jn  # noqa: E402


def _run_dict_with(funder, config_hash, *, model=None, prompt_strategy=None, temperature=None, **kw) -> dict:
    d = _run_dict_for_rollup(funder, config_hash, **kw)
    if model is not None:
        d["model"] = model
    if prompt_strategy is not None:
        d["prompt_strategy"] = prompt_strategy
    if temperature is not None:
        d["temperature"] = temperature
    return d


def test_group_rollup_splits_totals_by_model_without_leaking_across_groups():
    run_gpt = _run_dict_with(FUNDER_A, "run-gpt", model="gpt", supported_pattern=[(True,)])
    run_claude = _run_dict_with(FUNDER_A, "run-claude", model="claude", supported_pattern=[(True, False)])
    grouped = jn.group_rollup([run_gpt, run_claude], key="model")
    assert grouped["gpt"]["summary:groundedness"]["n_claims"] == 1
    assert grouped["claude"]["summary:groundedness"]["n_claims"] == 2
    # a group's total must not leak into another's
    assert grouped["gpt"]["summary:groundedness"]["n_claims_supported"] == 1


def test_group_rollup_pools_multiple_runs_into_the_same_group():
    run1 = _run_dict_with(FUNDER_A, "run-1", model="gpt", supported_pattern=[(True,)])
    run2 = _run_dict_with(FUNDER_B, "run-2", model="gpt", supported_pattern=[(True, True, False)])
    grouped = jn.group_rollup([run1, run2], key="model")
    assert grouped["gpt"]["summary:groundedness"]["n_claims"] == 4  # 1 + 3, pooled
    assert grouped["gpt"]["summary:groundedness"]["n_claims_supported"] == 3  # 1 + 2


def test_group_rollup_groups_by_temperature_as_a_string():
    run_cold = _run_dict_with(FUNDER_A, "run-cold", temperature=0.0, supported_pattern=[(True,)])
    run_warm = _run_dict_with(FUNDER_A, "run-warm", temperature=0.7, supported_pattern=[(False,)])
    grouped = jn.group_rollup([run_cold, run_warm], key="temperature")
    assert set(grouped) == {"0.0", "0.7"}
    assert grouped["0.0"]["summary:groundedness"]["n_claims_supported"] == 1
    assert grouped["0.7"]["summary:groundedness"]["n_claims_supported"] == 0


def test_exclusion_reason_counts_tallies_the_ledger():
    ledger = [
        {"reason": FIELD_EMPTY}, {"reason": FIELD_EMPTY}, {"reason": "judge_parse_failure"},
    ]
    assert jn.exclusion_reason_counts(ledger) == {FIELD_EMPTY: 2, "judge_parse_failure": 1}


def test_exclusion_reason_counts_of_an_empty_ledger_is_empty():
    assert jn.exclusion_reason_counts([]) == {}


def test_worst_unsupported_claims_excludes_supported_ones():
    claims = [_claim(verdict=True, item_id="s1"), _claim(verdict=False, item_id="u1")]
    worst = jn.worst_unsupported_claims(claims, n=10)
    assert [c.item_id for c in worst] == ["u1"]


def test_worst_unsupported_claims_ranks_groundedness_before_completeness():
    comp = _claim(dimension="completeness", verdict=False, item_id="comp-1")
    ground = _claim(dimension="groundedness", verdict=False, item_id="ground-1")
    worst = jn.worst_unsupported_claims([comp, ground], n=10)
    assert [c.item_id for c in worst] == ["ground-1", "comp-1"]


def test_worst_unsupported_claims_respects_n():
    claims = [_claim(dimension="groundedness", verdict=False, item_id=f"u{i}") for i in range(5)]
    assert len(jn.worst_unsupported_claims(claims, n=3)) == 3


def test_make_fake_reply_validates_against_the_real_judge_schema():
    rng = random.Random(1)
    for _ in range(50):
        reply = jn.make_fake_reply(rng)
        data = json.loads(reply)
        raw = judgemod._RawJudgeResponse.model_validate(data)  # raises on any schema violation
        assert 1 <= len(raw.claims) <= 4


def test_make_fake_reply_support_rate_shapes_the_supported_proportion():
    rng = random.Random(7)
    n_supported = 0
    n_total = 0
    for _ in range(500):
        raw = judgemod._RawJudgeResponse.model_validate(json.loads(jn.make_fake_reply(rng, support_rate=0.9)))
        n_total += len(raw.claims)
        n_supported += sum(1 for c in raw.claims if c.supported)
    assert n_supported / n_total == pytest.approx(0.9, abs=0.05)


def _selection_of(claims: list["ja.JudgedClaim"]) -> "ja.SpotcheckSelection":
    return ja.SpotcheckSelection(
        claims=claims, probabilities={c.item_id: 1.0 for c in claims},
        population={cell: 0 for cell in ja.ALL_CELLS}, achieved={cell: 0 for cell in ja.ALL_CELLS},
    )


def test_simulate_human_codes_covers_every_claim_with_a_valid_code():
    claims = [_claim(verdict=(i % 2 == 0), item_id=f"c{i}") for i in range(20)]
    codes = jn.simulate_human_codes(_selection_of(claims), rng=random.Random(3))
    assert set(codes) == {c.item_id for c in claims}
    assert all(v in ja._VALID_HUMAN_CODES for v in codes.values())


def test_simulate_human_codes_disagreement_rate_actually_flips_some_codes():
    """disagreement_rate=1.0 flips every binary code relative to the judge."""
    claims = [_claim(verdict=True, item_id=f"c{i}") for i in range(20)]
    codes = jn.simulate_human_codes(
        _selection_of(claims), rng=random.Random(3), disagreement_rate=1.0, not_a_claim_rate=0.0,
    )
    assert all(v == "unsupported" for v in codes.values())  # judge said True, human must disagree


def test_apply_simulated_human_codes_round_trips_through_the_real_csv(tmp_path):
    claims = [_claim(verdict=True, item_id="c0"), _claim(verdict=False, item_id="c1")]
    selection = ja.SpotcheckSelection(
        claims=claims, probabilities={"c0": 1.0, "c1": 1.0},
        population={cell: 0 for cell in ja.ALL_CELLS}, achieved={cell: 0 for cell in ja.ALL_CELLS},
    )
    out_path = tmp_path / "spot.csv"
    ja.write_spotcheck_csv(selection, out_path)
    jn.apply_simulated_human_codes(out_path, {"c0": "supported", "c1": "not_a_claim"})
    rows = {r["item_id"]: r for r in ja.load_coded_csv(out_path)}
    assert rows["c0"]["human_code"] == "supported"
    assert rows["c1"]["human_code"] == "not_a_claim"
    assert "SYNTHETIC" in rows["c0"]["human_notes"]
    assert rows["c0"]["claim_text"] == "text"  # untouched columns survive the round trip


def test_apply_simulated_human_codes_leaves_uncoded_rows_blank(tmp_path):
    claims = [_claim(verdict=True, item_id="c0"), _claim(verdict=False, item_id="c1")]
    selection = ja.SpotcheckSelection(
        claims=claims, probabilities={"c0": 1.0, "c1": 1.0},
        population={cell: 0 for cell in ja.ALL_CELLS}, achieved={cell: 0 for cell in ja.ALL_CELLS},
    )
    out_path = tmp_path / "spot.csv"
    ja.write_spotcheck_csv(selection, out_path)
    jn.apply_simulated_human_codes(out_path, {"c0": "supported"})  # c1 left uncoded
    rows = {r["item_id"]: r for r in ja.load_coded_csv(out_path)}
    assert rows["c1"]["human_code"] == ""


def test_apply_simulated_human_codes_rejects_an_invalid_code(tmp_path):
    claims = [_claim(verdict=True, item_id="c0")]
    selection = ja.SpotcheckSelection(
        claims=claims, probabilities={"c0": 1.0},
        population={cell: 0 for cell in ja.ALL_CELLS}, achieved={cell: 0 for cell in ja.ALL_CELLS},
    )
    out_path = tmp_path / "spot.csv"
    ja.write_spotcheck_csv(selection, out_path)
    with pytest.raises(ValueError, match="not a valid human_code"):
        jn.apply_simulated_human_codes(out_path, {"c0": "definitely_maybe"})


# expand_result_paths / --results-root
def test_expand_result_paths_walks_nested_funder_directories(tmp_path):
    """A grid root is searched recursively, since runs sit in `grid/<funder>/*.json`."""
    grid = tmp_path / "grid"
    (grid / "leverhulme").mkdir(parents=True)
    (grid / "wellcome").mkdir(parents=True)
    (grid / "leverhulme" / "a.json").write_text("{}")
    (grid / "leverhulme" / "b.json").write_text("{}")
    (grid / "wellcome" / "c.json").write_text("{}")
    (grid / "notes.txt").write_text("not a run")

    found = judgemod.expand_result_paths([grid])
    assert [p.name for p in found] == ["a.json", "b.json", "c.json"]
    # Sorted, so the pool's submission order is deterministic across invocations.
    assert found == sorted(found)


def test_expand_result_paths_still_takes_explicit_files(tmp_path):
    """A file argument is passed through unchanged."""
    f = tmp_path / "one.json"
    f.write_text("{}")
    assert judgemod.expand_result_paths([f]) == [f]


def test_expand_result_paths_does_not_change_score_expand(tmp_path):
    """The recursion is only in judge, so the `score` and `aggregate` CLIs are unaffected."""
    from grant_eval.scoring import score as scoremod

    nested = tmp_path / "cfg" / "deeper"
    nested.mkdir(parents=True)
    (nested / "deep.json").write_text("{}")
    (tmp_path / "cfg" / "shallow.json").write_text("{}")

    assert [p.name for p in scoremod._expand([tmp_path / "cfg"])] == ["shallow.json"]
    assert [p.name for p in judgemod.expand_result_paths([tmp_path / "cfg"])] == [
        "deep.json", "shallow.json",
    ]


def test_cli_results_root_redirects_verdicts_and_the_rollup(tmp_path, monkeypatch, capsys):
    """`--results-root` moves both the verdict file and the roll-up, leaving the default alone."""
    default_root = tmp_path / "default"
    chosen_root = tmp_path / "chosen"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", default_root)
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--snapshots", str(NOWHERE), "--results-root", str(chosen_root)])
    assert rc == 0

    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    assert verdict_path("run-a", pinned, funder_slug=SLUG, replicate=1, root=chosen_root).is_file()
    assert judgemod.rollup_path(pinned, root=chosen_root).is_file()
    # Nothing was written to the default root.
    assert not default_root.exists()


def test_cli_results_root_defaults_to_the_pinned_root(tmp_path, monkeypatch):
    """Without the flag, output goes to the usual default location."""
    default_root = tmp_path / "default"
    monkeypatch.setattr(judgemod, "JUDGE_RESULTS_ROOT", default_root)
    # The parser's default is captured from the constant at import time, so assert on
    # the value main() actually resolves rather than on the help string.
    monkeypatch.setattr(judgemod, "load_ground_truth", lambda *_a, **_kw: _one_row_gt())
    monkeypatch.setattr(judgemod, "load_registry", lambda *_a, **_kw: _registry())
    monkeypatch.setattr(judgemod, "OpenRouterJudgeBackend", lambda: FakeJudgeBackend([_ok_reply()]))

    path = _write_run(tmp_path, "a", "run-a")
    rc = judgemod.main([str(path), "--snapshots", str(NOWHERE)])
    assert rc == 0
    pinned = judgemod.JudgeConfig(
        model=judgemod.JUDGE_MODEL, temperature=0.0,
        prompt_version=judgemod.PROMPT_VERSION, rubric_version=judgemod.RUBRIC_VERSION,
    ).config_hash
    assert verdict_path("run-a", pinned, funder_slug=SLUG, replicate=1,
                        root=judgemod.JUDGE_RESULTS_ROOT).is_file()


# Judge leaderboard: whole configurations, not one factor at a time
def test_group_rollup_by_config_keeps_configs_differing_only_in_temperature_apart():
    """Configs that share model and prompt but differ in temperature stay separate rows."""
    hot = _run_dict_with(FUNDER_A, "run-hot", model="m", prompt_strategy="minimal",
                         temperature=0.7, supported_pattern=[(True,)])
    cold = _run_dict_with(FUNDER_A, "run-cold", model="m", prompt_strategy="minimal",
                          temperature=0.0, supported_pattern=[(True, False)])
    grouped = jn.group_rollup_by_config([hot, cold])
    assert len(grouped) == 2
    assert "m | minimal | t0.7" in grouped
    assert "m | minimal | t0.0" in grouped
    assert grouped["m | minimal | t0.7"]["summary:groundedness"]["n_claims"] == 1
    assert grouped["m | minimal | t0.0"]["summary:groundedness"]["n_claims"] == 2


def test_group_rollup_by_config_uses_the_same_labels_as_the_f1_leaderboard():
    """Leaderboard labels come from `analysis.dataset.config_label`, matching the F1 leaderboard."""
    from grant_eval.analysis.dataset import config_label

    run = _run_dict_with(FUNDER_A, "r", model="mdl", prompt_strategy="few_shot",
                         temperature=0.7, supported_pattern=[(True,)])
    grouped = jn.group_rollup_by_config([run])
    assert list(grouped) == [config_label("mdl", "few_shot", 0.7)]


def test_rank_configs_orders_best_first_and_reports_how_many_cells_contributed():
    """`combined_macro` averages only cells that have a rate, and `n_cells` says how many."""
    good = _run_dict_with(FUNDER_A, "good", model="a", prompt_strategy="minimal",
                          temperature=0.0, supported_pattern=[(True,)])
    poor = _run_dict_with(FUNDER_A, "poor", model="b", prompt_strategy="minimal",
                          temperature=0.0, supported_pattern=[(True, False)])
    rows = jn.rank_configs([good, poor])
    assert [r["config"] for r in rows] == ["a | minimal | t0.0", "b | minimal | t0.0"]
    assert rows[0]["combined_macro"] == 1.0
    assert rows[1]["combined_macro"] == 0.5
    assert all(r["n_cells"] == 1 for r in rows)
    # The three empty cells are None, never a 0.0 that would drag the mean down.
    assert rows[0]["rates"]["summary:completeness"] is None
    assert rows[0]["rates"]["summary:groundedness"] == 1.0


def test_rank_configs_sorts_a_wholly_undefined_config_last_not_as_a_zero():
    """A config with no judged claims has no rate and sorts below a genuine 0.0."""
    real = _run_dict_with(FUNDER_A, "real", model="a", prompt_strategy="minimal",
                          temperature=0.0, supported_pattern=[(False,)])
    empty = _run_dict_with(FUNDER_A, "empty", model="z", prompt_strategy="minimal",
                           temperature=0.0, supported_pattern=[])
    rows = jn.rank_configs([real, empty])
    assert rows[-1]["config"] == "z | minimal | t0.0"
    assert rows[-1]["combined_macro"] is None
    assert rows[-1]["combined_macro_undefined"] is True
    assert rows[-1]["n_cells"] == 0
    # ...and the genuine 0.0 is still reported as a real rate, above it.
    assert rows[0]["combined_macro"] == 0.0
    assert rows[0]["combined_macro_undefined"] is False


def test_best_config_per_cell_picks_each_cell_independently():
    """Each cell's best config is picked independently of the combined ranking."""
    a = _run_dict_with(FUNDER_A, "a", model="a", prompt_strategy="minimal",
                       temperature=0.0, supported_pattern=[(True,)])
    b = _run_dict_with(FUNDER_A, "b", model="b", prompt_strategy="minimal",
                       temperature=0.0, supported_pattern=[(True, False)])
    best = jn.best_config_per_cell([a, b])
    assert best["summary:groundedness"]["config"] == "a | minimal | t0.0"
    assert best["summary:groundedness"]["rate"] == 1.0
    # A cell no config has any claims in is undefined, never awarded to someone.
    assert best["summary:completeness"]["undefined"] is True
    assert best["summary:completeness"]["config"] is None


# claim_type_breakdown
def test_claim_type_breakdown_counts_by_type_and_dimension():
    """Every claim type and dimension is present, with zeros where there are no claims."""
    run = _run_dict_with(FUNDER_A, "r", supported_pattern=[(True, False, True)])
    out = jn.claim_type_breakdown([run])
    purpose = out["purpose"]["groundedness"]
    assert purpose["n_claims"] == 3
    assert purpose["n_supported"] == 2
    assert purpose["support_rate"] == pytest.approx(2 / 3)
    assert purpose["support_rate_undefined"] is False
    # Every claim type in the shared vocabulary is represented, both dimensions.
    assert set(out) == set(judgemod.CLAIM_TYPES)
    assert set(out["deadline"]) == set(judgemod.DIMENSIONS)
    assert out["deadline"]["groundedness"]["n_claims"] == 0
    assert out["deadline"]["groundedness"]["support_rate"] is None
    assert out["deadline"]["groundedness"]["support_rate_undefined"] is True


def test_claim_type_breakdown_ignores_excluded_records_which_carry_no_claims():
    """An excluded record with `claims: null` counts as zero claims instead of raising."""
    run = _run_dict_with(
        FUNDER_A, "r", supported_pattern=[(True,)],
        excluded={"judge_parse_failure": ["bad-1", "bad-2"]},
    )
    out = jn.claim_type_breakdown([run])
    assert out["purpose"]["groundedness"]["n_claims"] == 1


def test_rank_agreement_with_f1_joins_on_the_shared_config_label(tmp_path):
    """The judge and F1 tables join on the same config label, so nothing comes back None."""
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "configs.json").write_text(json.dumps([
        {"config": {"model": "a", "prompt_strategy": "minimal", "temperature": 0.0},
         "macro_f1": 0.4},
        {"config": {"model": "b", "prompt_strategy": "minimal", "temperature": 0.0},
         "macro_f1": 0.9},
    ]))
    run_a = _run_dict_with(FUNDER_A, "a", model="a", prompt_strategy="minimal",
                           temperature=0.0, supported_pattern=[(True,)])
    run_b = _run_dict_with(FUNDER_A, "b", model="b", prompt_strategy="minimal",
                           temperature=0.0, supported_pattern=[(True, False)])
    rows = jn.rank_agreement_with_f1([run_a, run_b], scored_dir=scored)

    # Ordered by F1 rank, so the F1 winner leads even though it is the judge's loser.
    assert [r["config"] for r in rows] == ["b | minimal | t0.0", "a | minimal | t0.0"]
    assert rows[0]["f1_rank"] == 1 and rows[0]["macro_f1"] == 0.9
    assert rows[0]["judge_rank"] == 2
    assert rows[1]["f1_rank"] == 2 and rows[1]["judge_rank"] == 1
    # The join worked: no None columns.
    assert all(r["judge_combined_macro"] is not None for r in rows)


def test_rank_agreement_with_f1_keeps_a_config_missing_from_one_side_visible(tmp_path):
    """An unmatched config is kept, with `None` for the missing side."""
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "configs.json").write_text(json.dumps([
        {"config": {"model": "a", "prompt_strategy": "minimal", "temperature": 0.0},
         "macro_f1": 0.4},
        {"config": {"model": "unjudged", "prompt_strategy": "minimal", "temperature": 0.0},
         "macro_f1": 0.5},
    ]))
    run_a = _run_dict_with(FUNDER_A, "a", model="a", prompt_strategy="minimal",
                           temperature=0.0, supported_pattern=[(True,)])
    rows = jn.rank_agreement_with_f1([run_a], scored_dir=scored)
    by_label = {r["config"]: r for r in rows}
    assert by_label["unjudged | minimal | t0.0"]["judge_rank"] is None
    assert by_label["unjudged | minimal | t0.0"]["macro_f1"] == 0.5
    assert by_label["a | minimal | t0.0"]["judge_rank"] == 1
