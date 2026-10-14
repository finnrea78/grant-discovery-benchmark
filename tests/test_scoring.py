"""Tests for the scorer.

Runs offline against the ground-truth CSVs and six saved skeleton results in
`tests/fixtures/skeleton/`, so new runs under `eval/` cannot break them.
"""
from __future__ import annotations

from datetime import date

import pytest

from grant_eval.schema import (
    Amount,
    DatePrecision,
    EdgeCaseTag,
    ExtractedOpportunity,
    ExtractionResult,
    GroundTruthOpportunity,
    PartialDate,
    Split,
    Status,
)
from grant_eval.scoring import groundtruth as gtmod
from grant_eval.scoring import mirror as mirrormod
from grant_eval.scoring import score as scoremod
from grant_eval.scoring.identity import (
    build_key_index,
    match_key,
    normalise_title,
    normalise_url,
    shared_urls,
)

RESULTS = "tests/fixtures/skeleton"
HAND_SCORED = f"{RESULTS}/leverhulme_deepseek-v4-flash-0731_minimal_t0_shallow_rep1_4ae8403f2e.json"
NOVA = f"{RESULTS}/leverhulme_nova-lite-v1_minimal_t0_standard_rep1_3b4b116bfb.json"
EMPTY = f"{RESULTS}/leverhulme_gemini-2.5-flash-lite_minimal_t0_standard_rep1_632e627c5b.json"

SAL = "Society of Antiquaries of London"
SAL_URL = "https://www.sal.org.uk/what-we-do/grants/our-grant-programmes"
LEV = "The Leverhulme Trust"


@pytest.fixture(scope="module")
def gt():
    return gtmod.load_ground_truth()


# Normalisation
def test_normalise_url_matches_the_protocol_rule():
    assert normalise_url("https://Example.org/Scheme/") == "https://example.org/scheme"
    assert normalise_url("https://example.org/scheme?utm=x") == "https://example.org/scheme"
    assert normalise_url("https://example.org/scheme#apply") == "https://example.org/scheme"
    # the fragment is dropped first, so a query inside a fragment goes too
    assert normalise_url("https://example.org/scheme#a?b=1") == "https://example.org/scheme"
    assert normalise_url("") == ""


def test_normalise_title_ladder():
    assert normalise_title("  Research   AWARDS ") == "research awards"
    assert normalise_title("Conferences, Lectures and Publications") == (
        "conferences lectures and publications"
    )
    assert normalise_title("Grants – Outline") == normalise_title("Grants - Outline")
    assert normalise_title("The Linda Parry Grants") == "linda parry grants"
    assert normalise_title("“Quoted” Award!") == "quoted award"
    # exact matching, not fuzzy
    assert normalise_title("Janet Arnold Awards") != normalise_title("Janet Arnold Award Fund")


def test_shared_urls_counts_rows_not_distinct_titles():
    assert shared_urls(["https://a/x", "https://a/x/", "https://a/y"]) == frozenset(
        {"https://a/x"}
    )
    assert shared_urls(["https://a/x"]) == frozenset()


# Ground-truth selection and loading
def test_corpus_totals(gt):
    # `wellbeing-of-women` was moved to `withdrawn/`. In-scope stays at 167 because
    # all 19 of its rows were out of scope.
    assert len(gt.files) == 24
    assert len(gt.rows) == 292
    assert len(gt.in_scope) == 167
    assert len(gt.funders) == 24


def test_badly_named_csvs_are_rejected_with_a_reason(tmp_path):
    """Files that fail the filename check are listed as rejected, each with a reason."""
    import shutil

    shutil.copy("data/ground-truth/leverhulme_2026-08-09.csv", tmp_path)
    for name in ("_TEMPLATE.csv", "ground-truth_all-funders (rsc).csv"):
        (tmp_path / name).write_text("funder,title\n", encoding="utf-8")
    gt = gtmod.load_ground_truth(tmp_path)
    rejected = {p.name: reason for p, reason in gt.rejected}
    assert set(rejected) == {"_TEMPLATE.csv", "ground-truth_all-funders (rsc).csv"}
    for reason in rejected.values():
        assert reason
    assert len(gt.files) == 1


def test_the_stale_bes_export_is_not_double_counted(gt):
    bes = gt.for_funder("British Ecological Society")
    assert len(bes.rows) == 12
    assert len(bes.in_scope) == 8
    ids = [r.opportunity_id for r in bes.rows]
    assert len(ids) == len(set(ids))


def test_non_utf8_file_is_decoded_not_crashed(tmp_path):
    """A cp1252 file falls through to the cp1252 decoder instead of crashing."""
    path = tmp_path / "cp1252.csv"
    path.write_bytes("funder,title\nBES,Caf\u00e9 grant\n".encode("cp1252"))
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding="utf-8")
    rows, encoding = gtmod.read_rows(path)
    assert encoding == "cp1252"
    assert rows == [{"funder": "BES", "title": "Caf\u00e9 grant"}]


def test_in_scope_strata_counts_match_d4_and_d5(gt):
    counts: dict[str, int] = {}
    for row in gt.in_scope:
        for tag in row.edge_case_tags:
            counts[tag.value] = counts.get(tag.value, 0) + 1
    assert counts["restricted"] == 78
    assert round(78 / len(gt.in_scope), 2) == 0.47
    assert counts["multi_deadline"] == 57
    assert counts["rolling"] == 19
    assert counts["ambiguous_eligibility"] == 10
    # Not tested, but pinned so a relabelling that makes them testable gets noticed.
    assert counts["pdf_only"] == 3
    assert counts["js_portal"] == 1


def test_shared_url_rows_are_the_d1_disambiguation_cases(gt):
    """15 in-scope rows share a URL with another row, across three funders."""
    per_funder = {}
    for key, fgt in gt.funders.items():
        n = sum(
            1
            for row in fgt.in_scope
            if normalise_url(row.url) in fgt.shared_urls
        )
        if n:
            per_funder[fgt.funder] = n
    assert per_funder == {
        SAL: 8,
        "Henry Moore Foundation": 5,
        "PRS Foundation": 2,
    }
    assert sum(per_funder.values()) == 15


def test_shared_urls_are_computed_over_distractors_too(gt):
    """Shared URLs are worked out over all rows, not just in-scope ones.

    Otherwise a distractor on a shared URL would get the same URL-only key as a real row.
    """
    mixed = 0
    for fgt in gt.funders.values():
        for url in fgt.shared_urls:
            group = [r for r in fgt.rows if normalise_url(r.url) == url]
            if len({r.in_scope for r in group}) > 1:
                mixed += 1
    assert mixed == 3


def test_key_index_raises_on_a_collision():
    rows = [
        GroundTruthOpportunity(
            funder="F", opportunity_id="a", title="Same Title", url="https://a/x"
        ),
        GroundTruthOpportunity(
            funder="F", opportunity_id="b", title="same  title!", url="https://a/x/"
        ),
    ]
    with pytest.raises(ValueError, match="collide on match key"):
        build_key_index(rows, "F")


# The hand-scored cell
def test_reproduces_the_hand_scored_leverhulme_cell(gt):
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    d = score.detection
    assert (d.tp, d.fp, d.fn) == (7, 1, 3)
    assert d.n_extracted == 8
    assert d.n_in_scope == 10
    assert round(d.precision, 3) == 0.875
    assert round(d.recall, 3) == 0.700
    assert round(d.f1, 3) == 0.778


def test_the_single_false_positive_is_the_d3_distractor(gt):
    """Major Research Fellowships has a closing date but no opening date, so it is out
    of scope and counts as a distractor false positive."""
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    (fp,) = score.detection.false_positives
    assert fp.reason == "distractor"
    assert fp.gold.opportunity_id == "leverhulme-major-research-fellowships"
    assert fp.gold.in_scope is False
    assert "no future opening date" in fp.gold.exclusion_reason


def test_records_not_just_counts(gt):
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    assert {fn.opportunity_id for fn in score.detection.false_negatives} == {
        "leverhulme-art-scholarship-bursaries-undergraduates-and-postgraduates",
        "leverhulme-art-scholarship-training-opportunities-under-18s",
        "leverhulme-philip-leverhulme-prizes",
    }
    tp = {t.opportunity_id for t in score.detection.true_positives}
    assert "leverhulme-research-project-grants" in tp
    # each TP keeps both records so mismatches can be checked without re-joining
    for t in score.detection.true_positives:
        assert isinstance(t.extracted, ExtractedOpportunity)
        assert isinstance(t.gold, GroundTruthOpportunity)


def test_paraphrased_title_on_a_unique_url_is_still_a_true_positive(gt):
    """A different title still matches when the URL belongs to only one row."""
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    by_id = {t.opportunity_id: t for t in score.detection.true_positives}
    rpg = by_id["leverhulme-research-project-grants"]
    assert rpg.gold.title != rpg.extracted.title
    assert rpg.key[2] == ""  # title slot empty -> URL alone decided


# The other real fixtures
def test_nova_lite_staged_call_produces_a_duplicate_false_positive(gt):
    """Outline and full stages of one call are a single row, so a second record for the
    same call is a `duplicate` false positive."""
    score = scoremod.score_run(scoremod.load_result(NOVA), gt)
    d = score.detection
    assert (d.tp, d.fp, d.fn) == (5, 1, 5)
    assert d.fp_by_reason() == {"duplicate": 1}
    (fp,) = d.false_positives
    assert fp.gold.opportunity_id == "leverhulme-research-project-grants"
    assert round(d.precision, 3) == 0.833
    assert round(d.recall, 3) == 0.500
    assert round(d.f1, 3) == 0.625


def test_empty_run_reports_precision_as_undefined(gt):
    score = scoremod.score_run(scoremod.load_result(EMPTY), gt)
    d = score.detection
    assert d.n_extracted == 0
    assert d.precision_undefined is True
    assert d.precision == 0.0
    assert d.recall == 0.0
    assert d.f1 == 0.0
    assert d.fn == 10


def test_every_committed_result_scores(gt):
    from pathlib import Path

    paths = sorted(Path(RESULTS).glob("*.json"))
    assert len(paths) == 6
    scores = scoremod.score_paths(paths, gt)
    assert len(scores) == 6
    # 0.333 and 0.778 come from the same funder, arm and temperature
    f1s = sorted(round(s.detection.f1, 3) for s in scores)
    assert f1s == [0.0, 0.0, 0.333, 0.333, 0.625, 0.778]


# The shared-URL side of the key, at scorer level
def _result(funder, opportunities) -> ExtractionResult:
    return ExtractionResult(
        snapshot_date=date(2026, 8, 21),
        funder=funder,
        split=Split.TUNE,
        model="test/model",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        replicate=1,
        config_hash="test",
        opportunities=opportunities,
    )


def _extracted(funder, oid, title, url, **kw) -> ExtractedOpportunity:
    return ExtractedOpportunity(funder=funder, opportunity_id=oid, title=title, url=url, **kw)


def test_shared_url_requires_the_title_to_match(gt):
    """On a URL shared by several rows, only a matching title gives a true positive."""
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
    good = _result(SAL, [_extracted(SAL, f"s{i}", t, SAL_URL) for i, t in enumerate(titles)])
    d = scoremod.score_run(good, gt).detection
    assert d.n_in_scope == 8
    assert (d.tp, d.fp, d.fn) == (8, 0, 0)
    assert d.f1 == 1.0

    bad = _result(
        SAL, [_extracted(SAL, f"s{i}", "Grants", SAL_URL) for i in range(8)]
    )
    d = scoremod.score_run(bad, gt).detection
    assert d.tp == 0
    assert d.fp_by_reason() == {"ambiguous_title": 8}


def test_unmatched_url_is_a_different_false_positive_from_a_distractor(gt):
    res = _result(
        LEV,
        [
            _extracted(LEV, "x", "Invented Scheme", "https://www.leverhulme.ac.uk/not-a-page"),
            _extracted(
                LEV,
                "y",
                "Major Research Fellowships",
                "https://www.leverhulme.ac.uk/major-research-fellowships",
            ),
        ],
    )
    d = scoremod.score_run(res, gt).detection
    assert d.fp_by_reason() == {"unmatched": 1, "distractor": 1}
    unmatched = next(f for f in d.false_positives if f.reason == "unmatched")
    assert unmatched.gold is None


def test_unattributable_false_positives_are_reported_outside_the_strata(gt):
    res = _result(
        LEV, [_extracted(LEV, "x", "Invented", "https://www.leverhulme.ac.uk/nope")]
    )
    score = scoremod.score_run(res, gt)
    assert score.fp_unattributable == 1
    assert sum(s.fp_attributable for s in score.strata.values()) == 0


def test_restricted_stratum_splits_the_positives(gt):
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    restricted = score.strata["restricted"]
    other = score.strata["not_restricted"]
    assert restricted.n_in_scope + other.n_in_scope == score.detection.n_in_scope
    assert restricted.tp + other.tp == score.detection.tp
    assert restricted.fn + other.fn == score.detection.fn
    assert restricted.n_in_scope == 6
    assert round(restricted.recall, 3) == 0.5


# Field-level accuracy
def _pair(gold_kw, ext_kw) -> scoremod.DetectionScore:
    gold = GroundTruthOpportunity(
        funder="F", opportunity_id="row-1", title="T", url="https://a/x", **gold_kw
    )
    ext = ExtractedOpportunity(
        funder="F", opportunity_id="row-1", title="T", url="https://a/x", **ext_kw
    )
    d = scoremod.DetectionScore(n_in_scope=1, n_extracted=1)
    d.true_positives.append(
        scoremod.TruePositive(key=("f", "https://a/x", ""), extracted=ext, gold=gold)
    )
    return d


def test_deadlines_require_the_full_set_to_match():
    both = [PartialDate.parse("2026-07-01"), PartialDate.parse("2026-09-01")]
    one = [PartialDate.parse("2026-09-01")]
    fields = scoremod.score_fields(_pair({"deadlines": both}, {"deadlines": one}))
    assert fields["deadlines"].correct == 0
    assert fields["deadlines"].incorrect == 1

    fields = scoremod.score_fields(_pair({"deadlines": both}, {"deadlines": list(reversed(both))}))
    assert fields["deadlines"].correct == 1  # a SET: order is irrelevant


def test_deadlines_compare_at_the_precision_stated_not_the_raw_string():
    """`2027-06` and a month-precision 'June 2027' compare equal, but a day does not."""
    gt_month = PartialDate.parse("2027-06")
    assert gt_month.precision is DatePrecision.MONTH and gt_month.resolved is None
    agent_month = PartialDate(
        raw="June 2027", resolved=date(2027, 6, 1), precision=DatePrecision.MONTH
    )
    assert scoremod.deadline_key(gt_month) == scoremod.deadline_key(agent_month) == "2027-06"

    # ...but a coarse value is never equal to a precise one
    agent_day = PartialDate(
        raw="15 June 2027", resolved=date(2027, 6, 15), precision=DatePrecision.DAY
    )
    assert scoremod.deadline_key(agent_day) != scoremod.deadline_key(gt_month)


def test_uncertain_field_is_excluded_not_scored():
    """A field marked `uncertain` in the ground truth is left out of the denominator."""
    fields = scoremod.score_fields(
        _pair(
            {
                "amount": Amount(min_gbp=1000.0),
                "uncertain_fields": ["amount_min_gbp"],
            },
            {"amount": Amount(min_gbp=9999.0)},
        )
    )
    acc = fields["amount_min_gbp"]
    assert acc.excluded_uncertain == 1
    assert acc.scored == 0
    assert acc.accuracy is None


def test_blank_is_a_real_target_and_an_asserted_value_is_wrong():
    """A blank ground-truth field means the page says nothing, so a value there is wrong."""
    fields = scoremod.score_fields(_pair({"amount": Amount()}, {"amount": Amount(min_gbp=500.0)}))
    acc = fields["amount_min_gbp"]
    assert (acc.correct, acc.incorrect) == (0, 1)
    assert acc.gt_blank_agent_asserted == 1
    assert acc.excluded_uncertain == 0

    # both blank counts as correct
    fields = scoremod.score_fields(_pair({"amount": Amount()}, {"amount": Amount()}))
    assert fields["amount_min_gbp"].correct == 1


def test_omitting_a_stated_value_is_tracked_separately_from_asserting_a_blank():
    fields = scoremod.score_fields(
        _pair({"amount": Amount(max_gbp=500.0)}, {"amount": Amount()})
    )
    acc = fields["amount_max_gbp"]
    assert acc.incorrect == 1
    assert acc.gt_value_agent_blank == 1
    assert acc.gt_blank_agent_asserted == 0


def test_blank_status_has_no_target_and_is_excluded():
    """A blank `status` is a labelling gap, so it is excluded under its own counter."""
    fields = scoremod.score_fields(
        _pair({"status": Status.UNKNOWN}, {"status": Status.OPEN})
    )
    acc = fields["status"]
    assert acc.excluded_no_target == 1
    assert acc.excluded_uncertain == 0
    assert acc.scored == 0


def test_status_is_compared_exactly():
    fields = scoremod.score_fields(
        _pair({"status": Status.FORTHCOMING}, {"status": Status.OPEN})
    )
    assert fields["status"].incorrect == 1
    fields = scoremod.score_fields(_pair({"status": Status.OPEN}, {"status": Status.OPEN}))
    assert fields["status"].correct == 1


def test_field_scoring_only_covers_true_positive_pairs(gt):
    score = scoremod.score_run(scoremod.load_result(HAND_SCORED), gt)
    for acc in score.fields.values():
        total = acc.scored + acc.excluded_uncertain + acc.excluded_no_target
        assert total == score.detection.tp


def test_edge_case_tag_enum_covers_the_stratum():
    assert scoremod.STRATUM_TAG == EdgeCaseTag.RESTRICTED.value


# CLI
def test_cli_scores_a_result_and_prints_a_report(capsys):
    rc = scoremod.main([HAND_SCORED, "--show-rejected"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "precision 0.875" in out
    assert "recall 0.700" in out
    assert "F1 0.778" in out
    assert "leverhulme-major-research-fellowships" in out


def test_cli_json_mode_round_trips(capsys):
    import json

    rc = scoremod.main([HAND_SCORED, "--json"])
    assert rc == 0
    (payload,) = json.loads(capsys.readouterr().out)
    assert payload["detection"]["tp"] == 7
    assert payload["detection"]["fp_by_reason"] == {"distractor": 1}
    assert len(payload["detection"]["false_negatives"]) == 3
    assert payload["strata"]["restricted"]["n_in_scope"] == 6


def test_cli_accepts_a_directory_and_prints_a_summary(capsys):
    rc = scoremod.main([RESULTS, "--quiet-records"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUMMARY" in out
    assert out.count("=" * 78) >= 6


def test_cli_skips_an_unlabelled_funder_rather_than_aborting(tmp_path, capsys):
    path = tmp_path / "unknown.json"
    path.write_text(_result("Not A Labelled Funder", []).model_dump_json(), encoding="utf-8")
    rc = scoremod.main([str(path)])
    assert rc == 1
    assert "no ground truth for funder" in capsys.readouterr().err


def test_match_key_is_a_three_tuple_in_both_branches():
    unshared = match_key(funder="F", url="https://a/x", title="T", shared=frozenset())
    sharedk = match_key(funder="F", url="https://a/x", title="T", shared={"https://a/x"})
    assert unshared == ("f", "https://a/x", "")
    assert sharedk == ("f", "https://a/x", "t")
    assert unshared != sharedk


# The hallucination split: a fabricated URL versus a page nobody labelled
LEV_UNLABELLED = (
    "https://www.leverhulme.ac.uk/art-scholarship-training-opportunities-under-18s-0"
)
LEV_FABRICATED = "https://www.leverhulme.ac.uk/art-scholarship-under-18s"


@pytest.fixture(scope="module")
def lev_mirror():
    """The real committed Leverhulme mirror. Offline: reads the snapshot manifest."""
    return mirrormod.load_mirror().for_funder(LEV)


def test_a_mirror_splits_unmatched_into_unlabelled_page_and_fabricated_url(gt, lev_mirror):
    """With a mirror, `unmatched` splits into unlabelled pages and fabricated URLs."""
    res = _result(
        LEV,
        [
            _extracted(LEV, "a", "Art Scholarships", LEV_UNLABELLED),
            _extracted(LEV, "b", "Art Scholarships Under 18s", LEV_FABRICATED),
        ],
    )
    d = scoremod.score_run(res, gt, mirror=lev_mirror).detection
    assert d.fp_by_reason() == {"unlabelled_page": 1, "fabricated_url": 1}
    assert d.mirror_resolved is True


def test_without_a_mirror_the_reason_stays_unmatched(gt):
    """Without a mirror the scorer keeps `unmatched` and marks the split unresolved."""
    res = _result(
        LEV,
        [
            _extracted(LEV, "a", "Art Scholarships", LEV_UNLABELLED),
            _extracted(LEV, "b", "Art Scholarships Under 18s", LEV_FABRICATED),
        ],
    )
    d = scoremod.score_run(res, gt).detection
    assert d.fp_by_reason() == {"unmatched": 2}
    assert d.mirror_resolved is False


def test_hallucination_rate_counts_only_fabricated_urls(gt, lev_mirror):
    """Hallucination rate is the share of extracted records whose URL is not in the mirror."""
    res = _result(
        LEV,
        [
            _extracted(LEV, "a", "Art Scholarships", LEV_UNLABELLED),
            _extracted(LEV, "b", "Art Scholarships Under 18s", LEV_FABRICATED),
            _extracted(
                LEV,
                "c",
                "Major Research Fellowships",
                "https://www.leverhulme.ac.uk/major-research-fellowships",
            ),
            _extracted(
                LEV,
                "d",
                "Research Project Grants",
                "https://www.leverhulme.ac.uk/research-project-grants",
            ),
        ],
    )
    d = scoremod.score_run(res, gt, mirror=lev_mirror).detection
    assert d.n_extracted == 4
    assert d.fp_by_reason()["fabricated_url"] == 1
    assert d.hallucination_rate == 0.25
    assert d.hallucination_rate_undefined is False


def test_hallucination_rate_is_undefined_not_zero_without_a_mirror(gt):
    """Without a mirror the rate is flagged undefined, since 0.0 would claim nothing was made up."""
    res = _result(LEV, [_extracted(LEV, "b", "Invented", LEV_FABRICATED)])
    d = scoremod.score_run(res, gt).detection
    assert d.hallucination_rate_undefined is True
    assert d.hallucination_rate == 0.0


def test_hallucination_rate_is_undefined_when_the_run_extracted_nothing(gt, lev_mirror):
    """An empty run has no denominator, the same as precision."""
    d = scoremod.score_run(_result(LEV, []), gt, mirror=lev_mirror).detection
    assert d.n_extracted == 0
    assert d.hallucination_rate_undefined is True


def test_to_dict_nulls_an_undefined_hallucination_rate(gt, lev_mirror):
    """An undefined hallucination rate is written as null in the JSON output."""
    unresolved = scoremod.to_dict(scoremod.score_run(_result(LEV, []), gt))
    assert unresolved["detection"]["hallucination_rate"] is None
    assert unresolved["detection"]["hallucination_rate_undefined"] is True
    assert unresolved["detection"]["mirror_resolved"] is False

    resolved = scoremod.to_dict(
        scoremod.score_run(
            _result(LEV, [_extracted(LEV, "b", "Invented", LEV_FABRICATED)]),
            gt,
            mirror=lev_mirror,
        )
    )
    assert resolved["detection"]["hallucination_rate"] == 1.0
    assert resolved["detection"]["mirror_resolved"] is True


def test_a_url_that_normalises_to_nothing_is_fabricated_not_unlabelled(gt, lev_mirror):
    """A URL that normalises to nothing counts as fabricated."""
    res = _result(LEV, [_extracted(LEV, "a", "No URL", "https://?a=1")])
    d = scoremod.score_run(res, gt, mirror=lev_mirror).detection
    assert d.fp_by_reason() == {"fabricated_url": 1}


def test_the_mirror_never_changes_which_records_match(gt, lev_mirror):
    """The mirror only labels false positives and never changes TP, FP or FN."""
    result = scoremod.load_result(HAND_SCORED)
    without = scoremod.score_run(result, gt).detection
    with_mirror = scoremod.score_run(result, gt, mirror=lev_mirror).detection
    assert (without.tp, without.fp, without.fn) == (with_mirror.tp, with_mirror.fp, with_mirror.fn)
    assert without.precision == with_mirror.precision
    assert without.recall == with_mirror.recall
    assert without.f1 == with_mirror.f1


def test_cli_reports_the_hallucination_rate_and_can_be_told_not_to(capsys):
    rc = scoremod.main([HAND_SCORED, "--quiet-records"])
    assert rc == 0
    assert "hallucination rate: 0.000" in capsys.readouterr().out

    rc = scoremod.main([HAND_SCORED, "--quiet-records", "--no-mirror"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hallucination rate: —" in out
    assert "the split was not resolved" in out


def test_cli_names_a_funder_whose_mirror_is_absent_rather_than_going_quiet(
    tmp_path, capsys
):
    registry = tmp_path / "funders.json"
    registry.write_text(
        '[{"slug": "gone", "display_name": "Gone", "split": "tune",'
        ' "snapshot_date": "2026-01-01", "entry_url": "https://x/", "content_hash": "h"}]'
    )
    rc = scoremod.main(
        [HAND_SCORED, "--quiet-records", "--registry", str(registry), "--snapshots", str(tmp_path)]
    )
    assert rc == 0
    assert "no mirror for gone" in capsys.readouterr().err


def test_to_dict_keeps_every_key_it_already_emitted(gt):
    """`to_dict` still has all its older keys, which `aggregate.py` relies on."""
    payload = scoremod.to_dict(scoremod.score_run(scoremod.load_result(HAND_SCORED), gt))
    assert set(payload) >= {
        "funder", "model", "prompt_strategy", "temperature", "agent_config",
        "replicate", "config_hash", "source_path", "detection", "strata",
        "fp_unattributable", "tag_counts", "fields",
    }
    assert payload["detection"]["tp"] == 7
    assert payload["detection"]["precision"] == pytest.approx(0.875)
    assert payload["strata"]["restricted"]["n_in_scope"] == 6


def test_to_dict_carries_the_runs_telemetry(gt):
    """`to_dict` includes the cost and failure flags under `telemetry`.

    The keys are listed by hand rather than read from `TELEMETRY_FIELDS`, so a key
    dropped from that constant would be caught.
    """
    payload = scoremod.to_dict(scoremod.score_run(scoremod.load_result(HAND_SCORED), gt))
    for key in (
        "split", "cost_usd", "latency_s", "input_tokens", "output_tokens",
        "timed_out", "context_overflow", "abstained", "schema_valid",
        "submit_failed", "step_budget_exhausted", "attempts", "provider",
    ):
        assert key in payload["telemetry"], key
        assert key in scoremod.TELEMETRY_FIELDS, key
    assert payload["telemetry"]["split"] == "tune"


def test_run_failed_is_a_timeout_or_a_context_overflow_only(gt):
    """`run_failed` is set only by a timeout or context overflow, not a submit failure."""
    result = scoremod.load_result(EMPTY)
    telemetry = scoremod.to_dict(scoremod.score_run(result, gt))["telemetry"]
    assert telemetry["run_failed"] == bool(
        telemetry["timed_out"] or telemetry["context_overflow"]
    )
    assert telemetry["abstained"] is False
    assert telemetry["submit_failed"] is True
    assert telemetry["run_failed"] is False


def test_run_failed_stays_false_for_abstention_alone(gt):
    """Abstaining is model behaviour, not a failed run, so `run_failed` stays False."""
    result = scoremod.load_result(HAND_SCORED).model_copy(update={"abstained": True})
    telemetry = scoremod.to_dict(scoremod.score_run(result, gt))["telemetry"]
    assert telemetry["timed_out"] is False
    assert telemetry["context_overflow"] is False
    assert telemetry["abstained"] is True
    assert telemetry["run_failed"] is False


def test_run_failed_is_true_for_context_overflow_alone(gt):
    """A context overflow on its own sets `run_failed`."""
    result = scoremod.load_result(HAND_SCORED).model_copy(update={"context_overflow": True})
    telemetry = scoremod.to_dict(scoremod.score_run(result, gt))["telemetry"]
    assert telemetry["timed_out"] is False
    assert telemetry["context_overflow"] is True
    assert telemetry["run_failed"] is True


def test_a_missing_cost_serialises_as_null_not_zero(gt):
    """A missing cost is written as null, because zero would mean the run was free."""
    import json
    result = scoremod.load_result(HAND_SCORED)
    result = result.model_copy(update={"cost_usd": None})
    payload = json.loads(json.dumps(scoremod.to_dict(scoremod.score_run(result, gt))))
    assert payload["telemetry"]["cost_usd"] is None
