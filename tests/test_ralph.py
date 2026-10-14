"""Tests for the Ralph outer loop, run offline with a fresh MockBackend per
iteration and a fake Toolset. The pre-registered config runs one iteration, so the
convergence tests use an explicit multi-iteration config instead.
"""
from __future__ import annotations

from grant_eval.agent.loop import SUBMIT_TOOL_NAME
from grant_eval.agent.ralph import (
    RalphResult,
    canonical_url,
    identity_key,
    run_ralph,
    shared_urls_in,
)
from grant_eval.agent.router import MockBackend, Reply, ToolCall
from grant_eval.agent.tools import ToolResult
from grant_eval.config import RALPH
from grant_eval.schema import ExtractedOpportunity


class FakeToolset:
    """Scripted stand-in for Toolset.

    `plan` maps a tool name to one ToolResult (returned every time) or a list
    consumed in order. Ralph reuses one toolset across iterations.
    """

    def __init__(self, plan: dict):
        self._plan = {k: list(v) if isinstance(v, list) else v for k, v in plan.items()}
        self.dispatched: list[ToolCall] = []

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": SUBMIT_TOOL_NAME,
                    "description": "Submit the final opportunities and end the run.",
                    "parameters": {
                        "type": "object",
                        "properties": {"opportunities": {"type": "array"}},
                        "required": ["opportunities"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_page",
                    "description": "Fetch a page.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        self.dispatched.append(call)
        entry = self._plan[call.name]
        if isinstance(entry, list):
            return entry.pop(0)
        return entry


def _opp(oid: str, url: str | None = None) -> ExtractedOpportunity:
    return ExtractedOpportunity(
        funder="Leverhulme Trust",
        opportunity_id=oid,
        title=f"Title {oid}",
        url=url or f"https://example.org/{oid}",
    )


_COMMON = dict(
    model="fake/model",
    temperature=0.0,
    strategy="minimal",
    entry_url="https://example.org/",
    funder="Leverhulme Trust",
    step_budget=8,
    snapshot_date="2026-08-09",
)


# `config.RALPH` runs one iteration, so it cannot exercise convergence. These tests
# use their own multi-iteration config; pointing them at `RALPH` would drop the only
# coverage the convergence logic has.
_MULTI_ITER = 5
_MULTI = dict(RALPH, max_iterations=_MULTI_ITER)


def _fetch_then_submit_replies(oids):
    """A two-step script: fetch the entry page, then submit the given opportunity ids."""
    return [
        Reply(tool_calls=[ToolCall("c1", "fetch_page", {"url": "https://example.org/"})]),
        Reply(
            tool_calls=[
                ToolCall(
                    "c2",
                    SUBMIT_TOOL_NAME,
                    {"opportunities": [{"id": o} for o in oids]},
                )
            ]
        ),
    ]


def test_stable_nonempty_set_terminates_at_iteration_two():
    opps = [_opp("o1"), _opp("o2")]
    factory_calls = {"n": 0}
    backends: list[MockBackend] = []

    def backend_factory():
        factory_calls["n"] += 1
        b = MockBackend(replies=_fetch_then_submit_replies(["o1", "o2"]))
        backends.append(b)
        return b

    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: ToolResult(
                content="Submitted 2.", terminate=True, submission=opps, abstained=False
            ),
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=_MULTI, **_COMMON
    )

    assert isinstance(result, RalphResult)
    assert result.ralph_iterations == 2
    assert result.scored_iteration_policy == "final"
    assert {o.opportunity_id for o in result.final.opportunities} == {"o1", "o2"}
    assert len(result.iterations) == 2
    assert factory_calls["n"] == 2
    # a distinct backend instance per iteration
    assert len(set(id(b) for b in backends)) == 2


def test_abstention_converges_at_iteration_one():
    factory_calls = {"n": 0}

    def backend_factory():
        factory_calls["n"] += 1
        return MockBackend(
            replies=[Reply(tool_calls=[ToolCall("c1", SUBMIT_TOOL_NAME, {"opportunities": []})])]
        )

    toolset = FakeToolset(
        {
            SUBMIT_TOOL_NAME: ToolResult(
                content="abstain", terminate=True, submission=[], abstained=True
            )
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=_MULTI, **_COMMON
    )

    assert result.ralph_iterations == 1
    assert result.final.abstained is True
    assert result.final.schema_valid is True
    assert result.final.opportunities == []
    assert len(result.iterations) == 1
    assert factory_calls["n"] == 1


def test_never_converges_runs_full_budget():
    counter = {"i": 0}
    factory_calls = {"n": 0}

    def backend_factory():
        factory_calls["n"] += 1
        counter["i"] += 1
        oid = f"o{counter['i']}"  # a NEW opportunity id every iteration
        return MockBackend(replies=_fetch_then_submit_replies([oid]))

    # Ralph keys on the submission, so each iteration needs its own terminal ToolResult
    # matching the new id (fetch is always the same, submits are consumed in order).
    max_iter = _MULTI_ITER
    submit_results = [
        ToolResult(
            content=f"Submitted {i}.",
            terminate=True,
            submission=[_opp(f"o{i}")],
            abstained=False,
        )
        for i in range(1, max_iter + 1)
    ]
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: submit_results,
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=_MULTI, **_COMMON
    )

    assert result.ralph_iterations == max_iter == 5
    assert len(result.iterations) == 5
    assert factory_calls["n"] == 5
    # final is the last iteration's result
    assert result.final is result.iterations[-1]
    assert {o.opportunity_id for o in result.final.opportunities} == {"o5"}


def test_nothing_carries_fresh_backend_per_iteration():
    counter = {"i": 0}
    built: list[int] = []

    def backend_factory():
        counter["i"] += 1
        oid = f"o{counter['i']}"
        b = MockBackend(replies=_fetch_then_submit_replies([oid]))
        b._ralph_tag = counter["i"]  # tag to prove distinct instances
        built.append(id(b))
        return b

    submit_results = [
        ToolResult(
            content=f"Submitted {i}.",
            terminate=True,
            submission=[_opp(f"o{i}")],
            abstained=False,
        )
        for i in range(1, _MULTI_ITER + 1)
    ]
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: submit_results,
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=_MULTI, **_COMMON
    )

    # one distinct backend per iteration that actually ran
    assert len(built) == result.ralph_iterations
    assert len(set(built)) == result.ralph_iterations


def test_order_insensitive_no_new_terminates_at_two():
    opps = [_opp("o1"), _opp("o2")]
    opps_rev = [_opp("o2"), _opp("o1")]
    submits = [
        ToolResult(content="s1", terminate=True, submission=opps, abstained=False),
        ToolResult(content="s2", terminate=True, submission=opps_rev, abstained=False),
    ]
    order = {"i": 0}

    def backend_factory():
        # iteration 1 submits o1,o2 ; iteration 2 submits o2,o1 (reversed)
        oids = ["o1", "o2"] if order["i"] == 0 else ["o2", "o1"]
        order["i"] += 1
        return MockBackend(replies=_fetch_then_submit_replies(oids))

    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: submits,
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=_MULTI, **_COMMON
    )

    assert result.ralph_iterations == 2
    assert {o.opportunity_id for o in result.final.opportunities} == {"o1", "o2"}


def test_identity_key_canonicalisation():
    base = _opp("x", url="https://example.org/scheme")
    trailing_slash = _opp("x", url="https://example.org/scheme/")
    with_query = _opp("x", url="https://example.org/scheme?utm=foo")
    upper = _opp("x", url="HTTPS://EXAMPLE.ORG/scheme")

    k = identity_key(base, "Leverhulme Trust")
    assert identity_key(trailing_slash, "Leverhulme Trust") == k
    assert identity_key(with_query, "Leverhulme Trust") == k
    assert identity_key(upper, "Leverhulme Trust") == k
    # funder casing / surrounding whitespace folds too
    assert identity_key(base, "  leverhulme trust  ") == k

    # canonical_url helper directly
    assert canonical_url("https://example.org/scheme/") == "https://example.org/scheme"
    assert canonical_url("https://example.org/scheme?q=1") == "https://example.org/scheme"
    assert canonical_url("HTTPS://Example.org/Scheme") == "https://example.org/scheme"

    # a different url gives a different key
    assert identity_key(_opp("x", url="https://example.org/other"), "Leverhulme Trust") != k


def test_trace_dir_writes_per_iteration_files(tmp_path):
    counter = {"i": 0}

    def backend_factory():
        counter["i"] += 1
        return MockBackend(replies=_fetch_then_submit_replies([f"o{counter['i']}"]))

    submit_results = [
        ToolResult(
            content=f"Submitted {i}.",
            terminate=True,
            submission=[_opp(f"o{i}")],
            abstained=False,
        )
        for i in range(1, _MULTI_ITER + 1)
    ]
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: submit_results,
        }
    )

    trace_dir = tmp_path / "ralph_traces"
    result = run_ralph(
        backend_factory=backend_factory,
        toolset=toolset,
        ralph_config=_MULTI,
        trace_dir=str(trace_dir),
        **_COMMON,
    )

    # never converges -> 5 iterations -> iter1..iter5.jsonl all exist
    assert result.ralph_iterations == 5
    for i in range(1, 6):
        assert (trace_dir / f"iter{i}.jsonl").exists()
        assert result.iterations[i - 1].trace_path == str(trace_dir / f"iter{i}.jsonl")


# Where the URL is not shared the title is ignored, so paraphrasing a title on an
# unambiguous URL is not penalised.
def test_identity_key_ignores_title_when_url_is_not_shared():
    funder = "Society of Antiquaries of London"
    url = "https://www.sal.org.uk/what-we-do/grants/our-grant-programmes"
    a = ExtractedOpportunity(funder=funder, opportunity_id="a", title="Research Awards", url=url)
    b = ExtractedOpportunity(
        funder=funder, opportunity_id="b", title="Research Awards (2026 round)", url=url
    )
    # No shared-URL set: the URL alone decides, so the paraphrase does not matter.
    assert identity_key(a, funder) == identity_key(b, funder)
    # The key is a 3-tuple with an empty title slot in that branch.
    assert identity_key(a, funder)[2] == ""


# Where the URL is shared the title tells records apart, e.g. society-antiquaries
# has 8 schemes on one page.
def test_identity_key_disambiguates_on_title_when_url_is_shared():
    url = "https://www.sal.org.uk/what-we-do/grants/our-grant-programmes"
    funder = "Society of Antiquaries of London"
    titles = [
        "Research Awards",
        "Janet Arnold Awards",
        "Beatrice de Cardi Awards",
        "Margaret and Tom Jones Awards",
        "Tessa and Mortimer Wheeler Travel Awards",
        "William Lambarde Travel Awards",
        "William and Jane Morris Church Conservation Grants",
        "The Linda Parry Textile Research and Conservation Grants",
    ]
    opps = [
        ExtractedOpportunity(funder=funder, opportunity_id=f"sal-{i}", title=t, url=url)
        for i, t in enumerate(titles)
    ]
    shared = shared_urls_in(opps)
    assert shared == frozenset({url})
    assert len({identity_key(o, funder, shared) for o in opps}) == 8

    # Without a shared-URL set they do collapse, so a slip back to URL-only keying
    # fails here rather than silently in the grid.
    assert len({identity_key(o, funder) for o in opps}) == 1


def test_identity_key_title_normalisation_is_deterministic():
    url = "https://example.org/grants"
    funder = "F"
    shared = frozenset({url})

    def key(title):
        return identity_key(
            ExtractedOpportunity(funder=funder, opportunity_id="x", title=title, url=url),
            funder,
            shared,
        )

    # casefold, collapsed whitespace, punctuation, typographic dashes, leading "The"
    assert key("Research Awards") == key("  research   AWARDS  ")
    assert key("Conferences, Lectures and Publications") == key(
        "Conferences Lectures and Publications"
    )
    assert key("Grants – Outline") == key("Grants - Outline")
    assert key("The Linda Parry Grants") == key("Linda Parry Grants")
    # ...but it is not fuzzy: two different schemes stay different.
    assert key("Janet Arnold Awards") != key("Beatrice de Cardi Awards")
    assert key("Research Awards") != key("Research Award Scheme")


def test_identity_key_empty_url_fallbacks():
    # `url` is validated non-empty, but normalisation can still empty it.
    bare_slash = ExtractedOpportunity(
        funder="F", opportunity_id="oid-1", title="Some Scheme", url="/"
    )
    assert identity_key(bare_slash, "F") == ("f", "oid-1", "")

    no_id = ExtractedOpportunity(funder="F", opportunity_id="  ", title="Some Scheme", url="/")
    assert identity_key(no_id, "F") == ("f", "some scheme", "")


def _scripted_run(funder, per_iteration):
    """Run Ralph over a fixed list of per-iteration submissions."""
    submits = [
        ToolResult(content=f"s{i}", terminate=True, submission=opps, abstained=False)
        for i, opps in enumerate(per_iteration, 1)
    ]
    # Pad to max_iterations so a non-converging run cannot run off the end of the script.
    submits += [submits[-1]] * (_MULTI_ITER - len(submits))
    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: submits,
        }
    )
    common = dict(_COMMON)
    common["funder"] = funder
    return run_ralph(
        backend_factory=lambda: MockBackend(replies=_fetch_then_submit_replies(["x"])),
        toolset=toolset,
        ralph_config=_MULTI,
        **common,
    )


# A new scheme on an already-seen shared URL is a new opportunity, so Ralph must
# not converge. With a (funder, url) key, {A@u, B@u} then {A@u, C@u} would look
# like a subset and stop early.
def test_new_scheme_on_a_shared_url_does_not_converge_spuriously():
    url = "https://www.sal.org.uk/what-we-do/grants/our-grant-programmes"
    funder = "Society of Antiquaries of London"

    def sal(oid, title):
        return ExtractedOpportunity(funder=funder, opportunity_id=oid, title=title, url=url)

    result = _scripted_run(
        funder,
        [
            [sal("a", "Research Awards"), sal("b", "Janet Arnold Awards")],
            [sal("a", "Research Awards"), sal("c", "Beatrice de Cardi Awards")],
            [sal("a", "Research Awards"), sal("c", "Beatrice de Cardi Awards")],
        ],
    )
    # iteration 2 introduced "Beatrice de Cardi Awards" -> not a subset -> ran on;
    # iteration 3 repeated iteration 2 -> converged there.
    assert result.ralph_iterations == 3
    assert {o.opportunity_id for o in result.final.opportunities} == {"a", "c"}


def test_a_non_final_policy_is_not_actually_implemented_by_run_ralph():
    """run_ralph only echoes scored_iteration_policy and always scores the last iteration.

    This is why preflight only accepts "final". If another policy is ever
    implemented this test fails, and that list can be widened.
    """
    policy_config = dict(_MULTI, scored_iteration_policy="best_of")
    submits = [
        ToolResult(content=f"s{i}", terminate=True,
                   submission=[_opp(f"o{i}")], abstained=False)
        for i in range(1, _MULTI_ITER + 1)
    ]
    toolset = FakeToolset({
        "fetch_page": ToolResult(content="PAGE", terminate=False),
        SUBMIT_TOOL_NAME: submits,
    })
    counter = {"i": 0}

    def backend_factory():
        counter["i"] += 1
        return MockBackend(replies=_fetch_then_submit_replies([f"o{counter['i']}"]))

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset,
        ralph_config=policy_config, **_COMMON,
    )

    assert result.scored_iteration_policy == "best_of"  # the LABEL was taken...
    assert result.final is result.iterations[-1]        # ...and then ignored
    assert {o.opportunity_id for o in result.final.opportunities} == {"o5"}


# The shared-URL set only grows during a run, so a record's key cannot change
# meaning just because a URL stopped being shared.
def test_shared_url_set_is_monotone_across_iterations():
    url = "https://example.org/grants"
    funder = "F"

    def opp(oid, title):
        return ExtractedOpportunity(funder=funder, opportunity_id=oid, title=title, url=url)

    result = _scripted_run(
        funder,
        [
            [opp("a", "Alpha Grant")],                          # url not yet shared
            [opp("a", "Alpha Grant"), opp("b", "Beta Grant")],  # url becomes shared
            [opp("a", "Alpha Grant"), opp("b", "Beta Grant")],  # stable -> converge
        ],
    )
    assert result.ralph_iterations == 3


def test_pre_registered_max_iterations_is_one():
    """The pre-registered iteration count is one; nothing else pins this value."""
    assert RALPH["max_iterations"] == 1


def test_one_iteration_runs_the_loop_once_and_scores_it():
    """With one iteration the loop runs once and scores that single result."""
    factory_calls = {"n": 0}

    def backend_factory():
        factory_calls["n"] += 1
        return MockBackend(replies=_fetch_then_submit_replies(["o1", "o2"]))

    toolset = FakeToolset(
        {
            "fetch_page": ToolResult(content="PAGE", terminate=False),
            SUBMIT_TOOL_NAME: ToolResult(
                content="Submitted 2.",
                terminate=True,
                submission=[_opp("o1"), _opp("o2")],
                abstained=False,
            ),
        }
    )

    result = run_ralph(
        backend_factory=backend_factory, toolset=toolset, ralph_config=RALPH, **_COMMON
    )

    assert factory_calls["n"] == 1
    assert result.ralph_iterations == 1
    assert len(result.iterations) == 1
    assert result.final is result.iterations[0]
    assert result.scored_iteration_policy == "final"
    assert {o.opportunity_id for o in result.final.opportunities} == {"o1", "o2"}
