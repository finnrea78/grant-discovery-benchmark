"""Prompt strategies for the grant-discovery agent.

Plain string templates, kept apart from the tools and router so the prompts can be
rendered and compared on their own. For fixed inputs the output is the same for every
model, so differences in behaviour come from the model and not the prompt.

Four strategies, matching ``grant_eval.config.PROMPT_STRATEGIES``:

* ``minimal``: ``_TASK_CORE`` alone.
* ``structured``: core plus an ordered procedure and the list of record fields.
* ``chain_of_thought``: core plus an instruction to reason before acting.
* ``few_shot``: core plus five worked examples.

Each non-minimal strategy is the core plus one scaffold. A scaffold only adds
organisation or process and never changes the inclusion rule, otherwise the factor
would measure a better task description rather than scaffolding. The user message is
the same for all four. ``tests/test_prompts.py`` checks the exact text of each.
"""
from __future__ import annotations

from grant_eval.config import PROMPT_STRATEGIES

# The task definition, shared word for word by all four strategies as the start of
# every system prompt.

_TASK_CORE = (
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

# structured: an ordered procedure plus the field names from `schema.Opportunity`. The
# submit schema already has these fields, so this only changes organisation. Names only,
# since explaining a field would add task detail the core does not have.

_STRUCTURED_SCAFFOLD = (
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

# chain_of_thought: reason before acting. It does not say where to put the reasoning,
# because two of the three models already return a separate reasoning stream.

_CHAIN_OF_THOUGHT_SCAFFOLD = (
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

# few_shot: five fixed worked examples.
#
# To avoid leakage every example comes from `raeng` or `ihr`
# (`data/ground-truth/deferred-batch3/`), which are in neither split. Their `split`
# column reads `exemplar`, which `schema._parse_split` refuses, so they cannot end up in
# a scored split by accident. `tests/test_prompts.py::test_few_shot_cites_no_split_funder`
# checks that no example names a funder from `data/funders.json`.

_FEW_SHOT_SCAFFOLD = (
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

#: strategy name -> its scaffold. `minimal` has none.
_SCAFFOLDS: dict[str, str] = {
    "structured": _STRUCTURED_SCAFFOLD,
    "chain_of_thought": _CHAIN_OF_THOUGHT_SCAFFOLD,
    "few_shot": _FEW_SHOT_SCAFFOLD,
}

# The same for all four strategies: it only carries the run's funder, date and URL.
_USER = (
    "Funder: {funder}\n"
    "Snapshot date: {snapshot_date}\n"
    "Start from this entry URL: {entry_url}"
)


def _iso(snapshot_date) -> str:
    """``date(2026, 8, 9)`` or ``'2026-08-09'`` -> ``'2026-08-09'``, so the output does not
    depend on the caller's type."""
    iso = getattr(snapshot_date, "isoformat", None)
    return iso() if callable(iso) else str(snapshot_date).strip()


def _system_for(strategy: str) -> str:
    """``_TASK_CORE``, plus this strategy's single scaffold if it has one."""
    scaffold = _SCAFFOLDS.get(strategy)
    return _TASK_CORE if scaffold is None else f"{_TASK_CORE}\n\n{scaffold}"


def render(strategy: str, *, entry_url: str, funder: str, snapshot_date) -> tuple[str, str]:
    """Return (system_prompt, user_prompt), identical across models for fixed inputs.

    Args:
        strategy:  one of ``grant_eval.config.PROMPT_STRATEGIES``.
        entry_url: the funder's entry URL the agent starts from.
        funder:    the funder's display name.
        snapshot_date: the date the mirror was captured, as a ``date`` or ISO string.
            The inclusion rule depends on it. Without it a DeepSeek run guessed "early
            2026" for a 2026-08-09 snapshot and never submitted.

    Raises:
        ValueError: for an unknown strategy (names the bad value and the valid set).
    """
    if strategy != "minimal" and strategy not in _SCAFFOLDS:
        valid = ", ".join(PROMPT_STRATEGIES)
        raise ValueError(f"unknown prompt strategy {strategy!r}; valid strategies: {valid}")
    return _system_for(strategy), _USER.format(
        funder=funder, entry_url=entry_url, snapshot_date=_iso(snapshot_date)
    )
