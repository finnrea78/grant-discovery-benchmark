"""Tests that the Leverhulme ground-truth CSV loads into GroundTruthOpportunity.

Includes the awkward rows with imprecise dates and uncertain fields.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from grant_eval.schema import (
    DatePrecision,
    GroundTruthOpportunity,
    Split,
    Status,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "ground-truth" / "leverhulme_2026-08-09.csv"


@pytest.fixture(scope="module")
def records() -> list[GroundTruthOpportunity]:
    return GroundTruthOpportunity.load_csv(CSV_PATH)


def by_id(records, opp_id) -> GroundTruthOpportunity:
    (match,) = [r for r in records if r.opportunity_id == opp_id]
    return match


def test_all_twelve_rows_load(records):
    assert len(records) == 12
    assert all(isinstance(r, GroundTruthOpportunity) for r in records)


def test_split_loads_from_csv(records):
    # data/funders.md puts Leverhulme in the tune split.
    assert all(r.split is Split.TUNE for r in records)


def test_multi_deadline_row_yields_two_full_dates(records):
    rpg = by_id(records, "leverhulme-research-project-grants")
    assert rpg.status is Status.OPEN
    assert len(rpg.deadlines) == 2
    assert [d.resolved for d in rpg.deadlines] == [date(2026, 7, 1), date(2026, 9, 1)]
    assert all(d.precision is DatePrecision.DAY for d in rpg.deadlines)
    # The page says "autumn 2026"; the labelling protocol records the bare year and
    # flags the field as uncertain.
    assert rpg.next_open_date is not None
    assert rpg.next_open_date.raw == "2026"
    assert rpg.next_open_date.resolved is None  # NOT coerced to 2026-01-01
    assert rpg.next_open_date.precision is DatePrecision.YEAR
    assert "next_open_date" in rpg.uncertain_fields


def test_bare_year_next_open_preserves_year_precision(records):
    art = by_id(records, "leverhulme-art-scholarship-bursaries-undergraduates-and-postgraduates")
    assert art.status is Status.FORTHCOMING
    assert art.next_open_date is not None
    assert art.next_open_date.raw == "2027"
    assert art.next_open_date.resolved is None  # NOT coerced to 2027-01-01
    assert art.next_open_date.precision is DatePrecision.YEAR


def test_visiting_professorships_preserves_uncertain_fields(records):
    vp = by_id(records, "leverhulme-visiting-professorships")
    assert vp.uncertain_fields == ["status"]
    assert vp.status is Status.OPEN
    # single full-date deadline
    assert len(vp.deadlines) == 1
    assert vp.deadlines[0].resolved == date(2026, 9, 1)
    # amount range present
    assert vp.amount is not None
    assert vp.amount.min_gbp == 10000
    assert vp.amount.max_gbp == 150000


def test_month_only_dates_are_month_precision(records):
    centres = by_id(records, "leverhulme-leverhulme-research-centres")
    assert centres.next_open_date.raw == "2027-02"
    assert centres.next_open_date.precision is DatePrecision.MONTH
    assert centres.next_open_date.resolved is None
    assert len(centres.deadlines) == 1
    assert centres.deadlines[0].precision is DatePrecision.MONTH


def test_excluded_row_carries_reason(records):
    doc = by_id(records, "leverhulme-leverhulme-doctoral-scholarships")
    assert doc.in_scope is False
    assert doc.exclusion_reason
    assert doc.status is Status.CLOSED


def test_labeller_and_pass_id_and_notes(records):
    ecf = by_id(records, "leverhulme-early-career-fellowships")
    # every row is human-reviewed, so `labeller` is the reviewer
    assert ecf.labeller == "Finn Rea"
    assert ecf.pass_id == "A"
    assert ecf.extraction_notes
    assert ecf.double_labelled is False
    # full-date next_open + full-date deadline
    assert ecf.next_open_date.resolved == date(2027, 1, 1)
    assert ecf.deadlines[0].resolved == date(2027, 2, 18)


def test_optional_split_injection_still_valid():
    # Assigning split after load stays legal, whether or not the CSV carried one.
    records = GroundTruthOpportunity.load_csv(CSV_PATH)
    r = records[0]
    r.split = Split.TUNE
    assert r.split is Split.TUNE


def test_free_text_next_open_date_stays_raw():
    """A date given only in words keeps its raw text and does not resolve.

    The real CSVs no longer have such a cell, so this uses a synthetic row.
    """
    r = GroundTruthOpportunity.from_csv_row(
        _minimal_row(status="forthcoming", next_open_date="autumn 2026")
    )
    assert r.next_open_date is not None
    assert r.next_open_date.raw == "autumn 2026"
    assert r.next_open_date.resolved is None
    assert r.next_open_date.precision is DatePrecision.NONE


def _minimal_row(**overrides) -> dict[str, str]:
    row = {
        "funder": "leverhulme",
        "opportunity_id": "leverhulme-test",
        "title": "Test",
        "url": "https://example.org/test",
        "status": "open",
    }
    row.update(overrides)
    return row


def test_in_scope_no_yields_false():
    r = GroundTruthOpportunity.from_csv_row(_minimal_row(in_scope="no"))
    assert r.in_scope is False


def test_in_scope_empty_defaults_to_true():
    r = GroundTruthOpportunity.from_csv_row(_minimal_row(in_scope=""))
    assert r.in_scope is True


def test_in_scope_unrecognised_value_raises():
    with pytest.raises(ValueError):
        GroundTruthOpportunity.from_csv_row(_minimal_row(in_scope="maybe"))


def test_split_empty_stays_none():
    r = GroundTruthOpportunity.from_csv_row(_minimal_row(split=""))
    assert r.split is None


def test_split_unrecognised_value_raises():
    """An unrecognised cell surfaces rather than defaulting into a scored split."""
    with pytest.raises(ValueError) as exc:
        GroundTruthOpportunity.from_csv_row(_minimal_row(split="holdout"))
    assert "holdout" in str(exc.value)
    assert "leverhulme-test" in str(exc.value)


# The few_shot exemplar funders must not load into a scored split.
DEFERRED = Path(__file__).resolve().parents[1] / "data" / "ground-truth" / "deferred-batch3"


@pytest.mark.parametrize("name", ["raeng_2026-08-28.csv", "ihr_2026-08-28.csv"])
def test_exemplar_funders_refuse_to_load_as_ground_truth(name):
    """The `ihr` and `raeng` CSVs (used for few_shot examples) refuse to load.

    Their pages are quoted in the prompt, so they must never end up in a scored split.
    """
    with pytest.raises(ValueError) as exc:
        GroundTruthOpportunity.load_csv(DEFERRED / name)
    assert "exemplar" in str(exc.value)


def test_split_enum_has_no_exemplar_member():
    """`Split` has only tune and validation, so the exemplar refusal above still works."""
    assert [s.value for s in Split] == ["tune", "validation"]


def test_the_outcome_flags_default_false_so_older_records_stay_readable():
    """`timed_out` and `context_overflow` default to False so records without them still load."""
    from grant_eval.schema import ExtractionResult

    record = ExtractionResult(
        snapshot_date=date(2026, 8, 9),
        funder="The British Academy",
        split=Split.TUNE,
        model="openrouter/amazon/nova-lite-v1",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        replicate=1,
        config_hash="deadbeef",
    )
    assert record.timed_out is False
    assert record.context_overflow is False

    # a record saved without the field still loads
    older = record.model_dump_json(exclude={"context_overflow"})
    assert ExtractionResult.model_validate_json(older).context_overflow is False
