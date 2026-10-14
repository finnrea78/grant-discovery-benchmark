"""Ground-truth and extraction schemas for the grant-discovery evaluation.

Hand-labelled ground-truth records and the agent's output share one `Opportunity`
shape so the scorer can match them at the (funder, opportunity) level.
"""
from __future__ import annotations

import csv
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Split(str, Enum):
    """Which dataset a funder belongs to. Fixed before any grid run.

    There is no `exemplar` member on purpose: the `few_shot` exemplar funders carry
    `split = exemplar`, so `_parse_split` refuses to load them into a scored split."""

    TUNE = "tune"
    VALIDATION = "validation"


class Status(str, Enum):
    """Lifecycle state of a call at snapshot time. The ground-truth CSV only exercises
    forthcoming/open/closed; rolling/unknown are kept for the agent's benefit."""

    FORTHCOMING = "forthcoming"  # not yet open, a future open date is stated
    OPEN = "open"                # currently accepting applications
    CLOSED = "closed"            # closed, no future opening stated (distractor)
    ROLLING = "rolling"          # perpetually open / no fixed deadline
    UNKNOWN = "unknown"          # state could not be determined


class EdgeCaseTag(str, Enum):
    """Edge-case tags (RQ4). A ground-truth opportunity can carry any number of these."""

    ROLLING = "rolling"                              # perpetually-open / no fixed deadline
    MULTI_DEADLINE = "multi_deadline"                # outline+full stages; multiple rounds
    PDF_ONLY = "pdf_only"                            # opportunity only inside a linked PDF
    JS_PORTAL = "js_portal"                          # client-side rendered / embedded portal
    AMBIGUOUS_ELIGIBILITY = "ambiguous_eligibility"  # eligibility split/hedged across pages
    RESTRICTED = "restricted"                        # embargoed / invite-only
    RECENTLY_CLOSED = "recently_closed"              # closed call still visible (distractor)


class DatePrecision(str, Enum):
    """How precisely a `PartialDate` is pinned down."""

    DAY = "day"      # full YYYY-MM-DD
    MONTH = "month"  # YYYY-MM
    YEAR = "year"    # YYYY
    NONE = "none"    # season / free text (e.g. "autumn 2026"), unparseable


class PartialDate(BaseModel):
    """A date that may be imprecise: a full date (`2026-07-01`), a month (`2026-09`), a
    year (`2027`) or free text (`autumn 2026`).

    `raw` holds the source string. `resolved` is set only when the value is a full day,
    so `2027` never becomes 2027-01-01. Precision follows the string's shape, not
    calendar validity: `2026-99` counts as MONTH but never resolves to a `date`."""

    raw: str
    resolved: date | None = None
    precision: DatePrecision = DatePrecision.NONE

    @classmethod
    def parse(cls, value: str) -> "PartialDate":
        """Build a PartialDate from a raw cell, inferring precision without coercion."""
        raw = value.strip()
        parts = raw.split("-")
        # Full YYYY-MM-DD
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            try:
                resolved = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                return cls(raw=raw, precision=DatePrecision.NONE)
            return cls(raw=raw, resolved=resolved, precision=DatePrecision.DAY)
        # YYYY-MM
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return cls(raw=raw, precision=DatePrecision.MONTH)
        # YYYY
        if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) == 4:
            return cls(raw=raw, precision=DatePrecision.YEAR)
        # season / free text
        return cls(raw=raw, precision=DatePrecision.NONE)


class Amount(BaseModel):
    """Funding amount as a range. Either bound may be missing, as many calls state only
    one or none. `raw` keeps the page text for amounts that are not a clean min/max."""

    min_gbp: float | None = None
    max_gbp: float | None = None
    raw: str | None = None


class Opportunity(BaseModel):
    """Fields the agent emits and the ground truth shares, matched at scoring time."""

    funder: str
    opportunity_id: str                   # slug, stable id for matching
    title: str
    url: str
    source_url: str | None = None         # page/mirror the record was extracted from
    status: Status = Status.UNKNOWN
    deadlines: list[PartialDate] = Field(default_factory=list)  # 0+ application deadlines
    next_open_date: PartialDate | None = None
    amount: Amount | None = None
    eligibility_summary: str | None = None
    summary: str | None = None
    discipline_tags: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)   # fields the labeller/agent flagged

    @field_validator("url")
    @classmethod
    def _url_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must be non-empty")
        return v.strip()


class ExtractedOpportunity(Opportunity):
    """One opportunity the agent claims to have found in a run. Same shape as the base;
    a distinct type so the scorer never confuses an extraction with a gold record."""


def _parse_in_scope(value: str | None, opportunity_id: str) -> bool:
    """Parse the ground-truth `in_scope` cell into a bool, raising on anything else.

    Only an explicit `"no"` excludes a row; an empty cell counts as in scope so a
    data-entry slip cannot drop a gold record."""
    cell = (value or "").strip().lower()
    if cell in ("", "yes"):
        return True
    if cell == "no":
        return False
    raise ValueError(
        f"{opportunity_id}: unrecognised in_scope value {value!r} (expected yes/no or empty)"
    )


def _parse_split(value: str | None, opportunity_id: str) -> Optional[Split]:
    """Parse the ground-truth `split` cell, raising on anything `Split` lacks.

    Blank stays `None`. The check exists for `exemplar`, used by the `few_shot` exemplar
    funders in `data/ground-truth/deferred-batch3/`: moving either CSV into
    `data/ground-truth/` should fail rather than add a funder quoted in the prompt to
    the scored tune split."""
    cell = (value or "").strip()
    if not cell:
        return None
    try:
        return Split(cell)
    except ValueError:
        valid = ", ".join(s.value for s in Split)
        raise ValueError(
            f"{opportunity_id}: unrecognised split {cell!r} (expected one of {valid}, or "
            f"empty). `exemplar` marks a funder used only as a few_shot prompt exemplar "
            f"(agent/prompts.py); such a funder must never enter a scored split."
        ) from None


class GroundTruthOpportunity(Opportunity):
    """A hand-verified record labelled from the mirror, which the grid is scored
    against. Also carries labelling fields the agent never sees."""

    split: Optional[Split] = None
    edge_case_tags: list[EdgeCaseTag] = Field(default_factory=list)
    in_scope: bool = True
    exclusion_reason: str | None = None
    verified_date: date | None = None
    snapshot_date: date | None = None
    labeller: str | None = None
    pass_id: str | None = None
    double_labelled: bool = False          # part of the >=20% kappa sample
    extraction_notes: str | None = None

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "GroundTruthOpportunity":
        """Map one `csv.DictReader` row of the frozen ground-truth CSV into a record.

        Handles `;`-separated multi-values, `yes/no` -> bool, partial dates, and empty
        cells -> None/[]."""

        def clean(key: str) -> str | None:
            v = (row.get(key) or "").strip()
            return v or None

        def multi(key: str) -> list[str]:
            v = (row.get(key) or "").strip()
            return [p.strip() for p in v.split(";") if p.strip()]

        def num(key: str) -> float | None:
            v = clean(key)
            return float(v) if v is not None else None

        split_raw = clean("split")
        next_open = clean("next_open_date")

        return cls(
            funder=row["funder"].strip(),
            opportunity_id=row["opportunity_id"].strip(),
            title=row["title"].strip(),
            url=row["url"].strip(),
            source_url=clean("source_page"),
            status=Status(row["status"].strip()) if clean("status") else Status.UNKNOWN,
            deadlines=[PartialDate.parse(d) for d in multi("deadlines")],
            next_open_date=PartialDate.parse(next_open) if next_open else None,
            amount=Amount(
                min_gbp=num("amount_min_gbp"),
                max_gbp=num("amount_max_gbp"),
                raw=clean("amount_raw"),
            ),
            eligibility_summary=clean("eligibility_summary"),
            summary=clean("summary"),
            discipline_tags=multi("discipline_tags"),
            uncertain_fields=multi("uncertain_fields"),
            split=_parse_split(split_raw, row["opportunity_id"].strip()),
            edge_case_tags=[EdgeCaseTag(t) for t in multi("edge_case_tags")],
            in_scope=_parse_in_scope(clean("in_scope"), row["opportunity_id"].strip()),
            exclusion_reason=clean("exclusion_reason"),
            verified_date=(date.fromisoformat(v) if (v := clean("verified_date")) else None),
            snapshot_date=(date.fromisoformat(v) if (v := clean("snapshot_date")) else None),
            labeller=clean("labeller"),
            pass_id=clean("pass_id"),
            extraction_notes=clean("notes"),
        )

    @classmethod
    def load_csv(cls, path: str | Path) -> list["GroundTruthOpportunity"]:
        """Load every data row of a ground-truth CSV into records."""
        with open(path, newline="", encoding="utf-8") as fh:
            return [cls.from_csv_row(row) for row in csv.DictReader(fh)]


class ExtractionResult(BaseModel):
    """One grid run's output for one funder: the opportunities found, the run identity,
    and cost and reliability telemetry.

    Written by `grid/store.py` to `<results-root>/grid/<funder>/<cell_id>.json`.
    `config_hash` ties each record to the exact config that produced it.
    """

    # run identity
    snapshot_date: date
    funder: str
    split: Split
    model: str                            # pinned dated snapshot id
    prompt_strategy: str
    temperature: float
    agent_config: str
    replicate: int
    config_hash: str

    # payload
    opportunities: list[ExtractedOpportunity] = Field(default_factory=list)

    # cost and reliability telemetry
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    ralph_iterations: int | None = None
    scored_iteration_policy: str | None = None   # "final" | "best_of"
    backend: str | None = None                   # transport / provider seam (e.g. litellm route)
    provider: str | None = None                  # OpenRouter provider that served the last
                                                 # completion; None for MockBackend or when
                                                 # the response does not say.
    step_budget_exhausted: bool = False          # agent hit its step cap before finishing
    abstained: bool = False                      # agent declined to emit any opportunity
    schema_valid: bool | None = None             # a valid submit happened (empty counts)
    submit_failed: bool | None = None            # the one repair turn was used up (2nd invalid submit)
    repair_turns_used: int | None = None         # schema repairs taken, outside step_budget,
                                                 # summed over iterations for a per-model repair rate.
    attempts: int | None = None                  # which attempt at this cell produced the record.
                                                 # >1 means a 429 restarted the cell from step 0, so
                                                 # `cost_usd` covers only this attempt; earlier ones
                                                 # are in the trace.
    timed_out: bool = False                      # hit the per-cell time cap. Scored on what it had
                                                 # and never retried, since retrying slow models
                                                 # would select on outcome.
    context_overflow: bool = False               # the provider refused an over-long transcript.
                                                 # Scored and never retried like `timed_out`, but
                                                 # counted by the circuit breaker, because a model
                                                 # that overflows on every cell is a real fault.
