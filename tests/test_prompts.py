"""Byte-stability tests for grant_eval.agent.prompts.

The prompt is byte-identical across models for fixed inputs, so differences in
behaviour can be put down to the model. The four strategies are checked against
golden strings, along with the shared task core, fixed user prompt, exemplar
leakage and distinct fingerprints.

"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from grant_eval.agent.prompts import _TASK_CORE, render
from grant_eval.config import PROMPT_STRATEGIES

# Fixed inputs used for the golden strings below.
FUNDER = "The Leverhulme Trust"
ENTRY_URL = "https://www.leverhulme.ac.uk/"

# Expected bytes of render("minimal", ...).
SNAPSHOT_DATE = "2026-08-09"

GOLDEN_MINIMAL_SYSTEM = (
    "You are examining one UK research funder's website to find its currently-open "
    "or forthcoming funding opportunities.\n"
    "\n"
    "An open opportunity is a distinct call that, on the snapshot date given below, "
    "is accepting applications (or states a future open date) and is available to "
    "some UK-based researchers. Closed, expired and purely informational pages are "
    "out of scope. A call with no stated deadline counts only if the page says it is "
    "currently open. So currently-open calls and forthcoming calls with a stated "
    "future open date are in scope; closed, expired and informational pages are not.\n"
    "\n"
    "Judge every date against the snapshot date, not against today. The pages are a "
    "frozen capture, so a deadline that has passed relative to the snapshot date is "
    "closed and one that has not is still open.\n"
    "\n"
    "Use the provided tools to read pages and follow links, starting from the entry "
    "URL. Everything you can see is a frozen snapshot of that one site.\n"
    "\n"
    "For each opportunity in scope, record what the pages state: its title and URL; "
    "whether it is open, forthcoming, closed or rolling; its deadlines and next open "
    "date; the award amount; who is eligible to apply; a short summary of what it "
    "funds; and its discipline areas. Where the pages do not state something, omit "
    "that field entirely rather than inferring a value or sending an empty one.\n"
    "\n"
    "When you are done, call submit_opportunities with the structured records. "
    "Submitting an empty list is the correct, valid action when there are no in-scope "
    "opportunities."
)

GOLDEN_MINIMAL_USER = (
    "Funder: The Leverhulme Trust\n"
    "Snapshot date: 2026-08-09\n"
    "Start from this entry URL: https://www.leverhulme.ac.uk/"
)

# The three scaffolded strategies: the minimal system prompt, a blank line, and one
# scaffold each. Written as a sum so the shared prefix is visible here too.

GOLDEN_STRUCTURED_SCAFFOLD = (
    "Work through the task in this order.\n"
    "\n"
    "1. Survey the links reachable from the entry URL and build a list of candidate "
    "opportunity pages.\n"
    "2. Open each candidate page in turn.\n"
    "3. Test that page against the inclusion rule above and decide it in or out.\n"
    "4. For a page that is in, record its fields from what the page says.\n"
    "5. When every candidate has been decided, call submit_opportunities once with the "
    "records you kept.\n"
    "\n"
    "The fields of a record are: funder, opportunity_id, title, url, source_url, "
    "status, deadlines, next_open_date, amount, eligibility_summary, summary, "
    "discipline_tags, uncertain_fields."
)

GOLDEN_CHAIN_OF_THOUGHT_SCAFFOLD = (
    "Reason explicitly before you act, rather than acting first and explaining "
    "afterwards.\n"
    "\n"
    "Before each tool call, say what you are looking for and why that page or link is "
    "the next thing to look at.\n"
    "\n"
    "Before you record an opportunity, work through the inclusion rule above one "
    "clause at a time against what the page actually says, quoting the wording you are "
    "relying on, and reach a verdict before you write anything down. Do the same for a "
    "candidate you decide to leave out."
)

GOLDEN_FEW_SHOT_SCAFFOLD = (
    "Five worked examples, judged against a 2026-08-28 snapshot date.\n"
    "\n"
    "1. IN SCOPE — deadline still ahead of the snapshot date.\n"
    "Page: \"Dr Williams's Trust Library Bursaries for Mid-Career Researchers. Three "
    "awards of £3,500 each. Deadline: 31 March 2027.\"\n"
    "2027-03-31 is after the snapshot date, so the call is still accepting "
    "applications.\n"
    "Record: title \"Dr Williams's Trust Library Bursaries for Mid-Career "
    "Researchers\"; url https://www.history.ac.uk/fellowships-funding/awards-bursaries-prizes/dr-williamss-trust-library-bursaries-mid-career-researchers ; "
    "status open; deadlines [\"2027-03-31\"]; amount_raw \"Three awards of £3,500 "
    "each\".\n"
    "\n"
    "2. IN SCOPE — narrow eligibility.\n"
    "Page: \"Young Engineer of the Year. Five awards of £3,000 each year, for UK "
    "engineers early in their careers. Nominations open in September 2026.\"\n"
    "The rule asks whether a call is available to SOME UK-based researchers, so "
    "eligibility this narrow does not exclude it.\n"
    "Record: title \"Young Engineer of the Year\"; url "
    "https://raeng.org.uk/programmes-and-prizes/prizes/raeng-young-engineer-of-the-year/ ; "
    "status forthcoming; next_open_date \"2026-09\"; deadlines []; amount_raw \"five "
    "awards of £3,000 each year\"; uncertain_fields [\"next_open_date\"].\n"
    "\n"
    "3. IN SCOPE — staged call, several dates.\n"
    "Page: \"Green Future Fellowships, Round 3: opens September 2026, closes April "
    "2027. Two stages: expression of interest, then invited full application. Awards "
    "of up to £3 million per individual.\"\n"
    "Not open on the snapshot date, but a future opening date is stated. Record every "
    "date the page gives, in one record for the whole staged call; a month-only date "
    "keeps month precision — never invent a day — and goes in uncertain_fields.\n"
    "Record: title \"Green Future Fellowships\"; url "
    "https://raeng.org.uk/programmes-and-prizes/programmes/uk-grants-and-prizes/support-for-research/gff/green-future-fellowships/ ; "
    "status forthcoming; next_open_date \"2026-09\"; deadlines [\"2027-04\"]; "
    "amount_raw \"Awards of up to £3 million per individual\"; uncertain_fields "
    "[\"deadlines\", \"next_open_date\"].\n"
    "\n"
    "4. IN SCOPE — no deadline, page says it is open.\n"
    "Page: \"Green Future Fellowships Accelerated International Route (GFF AIR). There "
    "is no deadline for applications; this route remains open continuously for an "
    "initial two-year period.\"\n"
    "A call with no deadline counts when the page says it is currently open. Leave "
    "deadlines empty rather than guessing one.\n"
    "Record: title \"Green Future Fellowships Accelerated International Route (GFF "
    "AIR)\"; url "
    "https://raeng.org.uk/programmes-and-prizes/programmes/uk-grants-and-prizes/support-for-research/gff/green-future-fellowships-air/ ; "
    "status rolling; deadlines [].\n"
    "\n"
    "5. NOT IN SCOPE — expired call still on the site.\n"
    "Page: \"The Pollard Prize. Applications open: until Friday 17 July 2026. First "
    "prize: fast-track publication plus £500 of books.\"\n"
    "The words \"applications open\" are not the test — the date is. 17 July 2026 is "
    "before the snapshot date, the window has expired, and no future opening date is "
    "given. No record submitted."
)

GOLDEN_SYSTEM = {
    "minimal": GOLDEN_MINIMAL_SYSTEM,
    "structured": GOLDEN_MINIMAL_SYSTEM + "\n\n" + GOLDEN_STRUCTURED_SCAFFOLD,
    "chain_of_thought": (
        GOLDEN_MINIMAL_SYSTEM + "\n\n" + GOLDEN_CHAIN_OF_THOUGHT_SCAFFOLD
    ),
    "few_shot": GOLDEN_MINIMAL_SYSTEM + "\n\n" + GOLDEN_FEW_SHOT_SCAFFOLD,
}

ALL_STRATEGIES = ["minimal", "structured", "chain_of_thought", "few_shot"]


def test_minimal_is_byte_identical_to_golden():
    system, user = render("minimal", entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    assert system == GOLDEN_MINIMAL_SYSTEM
    assert user == GOLDEN_MINIMAL_USER


@pytest.mark.parametrize("strategy", ["structured", "chain_of_thought", "few_shot"])
def test_scaffolded_strategies_are_byte_identical_to_golden(strategy):
    system, user = render(
        strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
    )
    assert system == GOLDEN_SYSTEM[strategy]
    assert user == GOLDEN_MINIMAL_USER


def test_the_golden_set_covers_every_registered_strategy():
    """Every strategy has golden bytes, so a new one can't be added without them."""
    assert sorted(GOLDEN_SYSTEM) == sorted(PROMPT_STRATEGIES)


def test_task_core_is_a_byte_identical_prefix_of_every_strategy():
    """Scaffolded levels keep the minimal task text unchanged.

    Otherwise the contrast with `minimal` would mix a better task description with the scaffolding.

    """
    minimal_system, _ = render(
        "minimal", entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
    )
    assert minimal_system == _TASK_CORE, "`minimal` must be the task core and nothing else"

    for strategy in PROMPT_STRATEGIES:
        system, _ = render(
            strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
        )
        assert system.startswith(_TASK_CORE), (
            f"{strategy!r} does not start with the task core verbatim; the levels are "
            f"no longer differing only in scaffolding"
        )
        remainder = system[len(_TASK_CORE):]
        assert remainder == "" or remainder.startswith("\n\n"), (
            f"{strategy!r} joins its scaffold to the core with {remainder[:4]!r}, not a "
            f"blank line"
        )


def test_the_levels_are_orthogonal_not_nested():
    """Each level carries its own scaffold and no other level's, so contrasts don't mix manipulations."""
    tails = {}
    for strategy in ("structured", "chain_of_thought", "few_shot"):
        system, _ = render(
            strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
        )
        tails[strategy] = system[len(_TASK_CORE) + 2:]

    for strategy, tail in tails.items():
        for other, other_tail in tails.items():
            if other == strategy:
                continue
            assert other_tail not in tail, (
                f"{strategy!r} contains the whole {other!r} scaffold; the levels are "
                f"nested, not orthogonal"
            )


def test_user_prompt_is_invariant_across_strategies():
    """All scaffolding is in the system prompt; the user prompt only carries per-run identity."""
    users = {
        strategy: render(
            strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
        )[1]
        for strategy in PROMPT_STRATEGIES
    }
    assert set(users.values()) == {GOLDEN_MINIMAL_USER}


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_contains_inputs(strategy):
    _, user = render(strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    assert FUNDER in user
    assert ENTRY_URL in user
    assert SNAPSHOT_DATE in user


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_snapshot_date_accepts_a_date_object(strategy):
    """A `date` and its ISO string render the same bytes."""
    from datetime import date

    as_obj = render(strategy, entry_url=ENTRY_URL, funder=FUNDER,
                    snapshot_date=date(2026, 8, 9))
    as_str = render(strategy, entry_url=ENTRY_URL, funder=FUNDER,
                    snapshot_date="2026-08-09")
    assert as_obj == as_str


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_no_trailing_whitespace_surprises(strategy):
    system, user = render(strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    for text in (system, user):
        for line in text.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace on line: {line!r}"
    assert not system.endswith("\n")
    assert not user.endswith("\n")


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_determinism(strategy):
    """The `few_shot` exemplars are fixed constants, never sampled."""
    a = render(strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    b = render(strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    assert a == b


def test_unknown_strategy_raises_value_error():
    with pytest.raises(ValueError) as exc:
        render("bogus", entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
    assert "bogus" in str(exc.value)


def test_every_registered_strategy_is_handled():
    """Every name in PROMPT_STRATEGIES renders without hitting the unknown-strategy path."""
    for strategy in PROMPT_STRATEGIES:
        try:
            render(strategy, entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE)
        except ValueError:
            pytest.fail(f"{strategy!r} is registered but render() treats it as unknown")


# Leakage control
def test_few_shot_cites_no_split_funder():
    """Exemplars come only from the deferred `ihr` and `raeng` and name no scored funder.

    Reads `data/funders.json` rather than a hardcoded list; this check is what ruled out
    an exemplar whose page mentioned the Leverhulme Trust.

    """
    registry = json.loads(Path("data/funders.json").read_text(encoding="utf-8"))
    assert len(registry) == 24, "funders.json is no longer the 24-funder registry"

    system, _ = render(
        "few_shot", entry_url=ENTRY_URL, funder=FUNDER, snapshot_date=SNAPSHOT_DATE
    )
    haystack = system.lower()

    for funder in registry:
        slug = funder["slug"].lower()
        # Match on word boundaries, otherwise short slugs like `rsc` and `cara` match inside
        # other words.
        assert not re.search(rf"(?<![a-z0-9]){re.escape(slug)}(?![a-z0-9])", haystack), (
            f"few_shot prompt cites the slug of split funder {funder['display_name']!r}"
        )

        name = funder["display_name"].lower()
        for candidate in {name, name[4:] if name.startswith("the ") else name}:
            assert candidate not in haystack, (
                f"few_shot prompt names split funder {funder['display_name']!r}"
            )

        host = (urlparse(funder["entry_url"]).netloc or "").lower()
        for candidate in {host, host[4:] if host.startswith("www.") else host}:
            assert candidate and candidate not in haystack, (
                f"few_shot prompt cites the entry-URL domain of split funder "
                f"{funder['display_name']!r}"
            )


# Cell identity
def test_prompt_fingerprints_are_all_distinct():
    """Strategies have distinct fingerprints, since `config_hash` uses it to name the output path."""
    from grant_eval.agent.extract import prompt_fingerprint

    fingerprints = {s: prompt_fingerprint(s) for s in PROMPT_STRATEGIES}
    assert len(set(fingerprints.values())) == len(PROMPT_STRATEGIES), fingerprints
    for strategy, fp in fingerprints.items():
        assert not fp.startswith("unimplemented:"), f"{strategy!r} still unauthored"
