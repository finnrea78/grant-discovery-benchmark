"""Tests for checking whether an extracted URL exists in the frozen snapshot (`scoring/mirror.py`).

This separates invented URLs from real pages the labeller did not list. Lookups use
each snapshot's manifest.json, so nothing touches the network.
"""
from __future__ import annotations

import json

import pytest

from grant_eval.scoring import mirror as mirrormod

LEV = "The Leverhulme Trust"
LEV_REAL = "https://www.leverhulme.ac.uk/art-scholarship-training-opportunities-under-18s-0"
LEV_FAKE = "https://www.leverhulme.ac.uk/art-scholarship-under-18s"


@pytest.fixture(scope="module")
def mirror():
    return mirrormod.load_mirror()


def _index(*urls):
    return mirrormod.MirrorIndex(
        funder="F",
        slug="f",
        snapshot_date="2026-01-01",
        content_hash="abc",
        urls=frozenset(mirrormod.normalise_url(u) for u in urls),
        n_files=len(urls),
    )


def test_membership_uses_the_identity_normalisation_ladder():
    index = _index("https://Example.org/Scheme/")
    assert index.contains("https://example.org/scheme")
    assert index.contains("https://EXAMPLE.org/Scheme/")
    assert index.contains("https://example.org/scheme?utm=x")
    assert index.contains("https://example.org/scheme#apply")
    assert not index.contains("https://example.org/other")


def test_a_url_that_normalises_to_nothing_is_never_in_the_mirror():
    # "?a=1" normalises to "", so "" is in `urls`; `contains` must still reject it.
    index = _index("https://example.org/scheme", "?a=1")
    assert "" in index.urls
    assert not index.contains("")
    assert not index.contains("?a=1")


def test_load_mirror_covers_every_tune_funder(mirror):
    tune = {f for f, m in mirror.by_funder.items() if m.split == "tune"}
    assert len(tune) == 16
    assert LEV in tune


def test_the_leverhulme_snapshot_separates_a_real_page_from_an_invented_one(mirror):
    # Both URLs came from real grid runs; the second is a made-up slug.
    index = mirror.for_funder(LEV)
    assert index is not None
    assert index.snapshot_date == "2026-08-09"
    assert index.contains(LEV_REAL)
    assert not index.contains(LEV_FAKE)


def test_the_index_records_the_snapshot_content_hash(mirror):
    index = mirror.for_funder(LEV)
    assert index.content_hash == (
        "1dc0c98146f7f90e4e0139ff304908dfc6c33dd32b852d024b0022f9a970999a"
    )


def test_normalisation_collapses_duplicate_manifest_urls(mirror):
    index = mirror.for_funder(LEV)
    assert index.n_files == 482
    assert len(index.urls) == 395


def test_an_unknown_funder_resolves_to_none_not_an_empty_mirror(mirror):
    assert mirror.for_funder("Not A Funder") is None


def test_a_funder_whose_snapshot_is_absent_is_named_in_missing(tmp_path):
    registry = tmp_path / "funders.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "slug": "present",
                    "display_name": "Present Funder",
                    "split": "tune",
                    "snapshot_date": "2026-01-01",
                    "entry_url": "https://present.example/",
                    "content_hash": "h",
                },
                {
                    "slug": "absent",
                    "display_name": "Absent Funder",
                    "split": "tune",
                    "snapshot_date": "2026-01-01",
                    "entry_url": "https://absent.example/",
                    "content_hash": "h",
                },
            ]
        )
    )
    snap = tmp_path / "snapshots" / "present" / "2026-01-01"
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(
        json.dumps(
            {
                "content_hash": "h",
                "files": [
                    {"path": "site/a.html", "url": "https://present.example/a", "kind": "page"}
                ],
            }
        )
    )

    loaded = mirrormod.load_mirror(registry=registry, snapshots_dir=tmp_path / "snapshots")
    assert loaded.for_funder("Present Funder") is not None
    assert loaded.for_funder("Absent Funder") is None
    assert [slug for slug, _ in loaded.missing] == ["absent"]
    assert loaded.missing[0][1]


def test_a_manifest_entry_without_a_url_is_counted_not_dropped_silently(tmp_path):
    registry = tmp_path / "funders.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "slug": "f",
                    "display_name": "F",
                    "split": "tune",
                    "snapshot_date": "2026-01-01",
                    "entry_url": "https://f.example/",
                    "content_hash": "h",
                }
            ]
        )
    )
    snap = tmp_path / "snapshots" / "f" / "2026-01-01"
    snap.mkdir(parents=True)
    (snap / "manifest.json").write_text(
        json.dumps(
            {
                "content_hash": "h",
                "files": [
                    {"path": "site/a.html", "url": "https://f.example/a", "kind": "page"},
                    {"path": "site/b.html", "kind": "page"},
                ],
            }
        )
    )

    index = mirrormod.load_mirror(
        registry=registry, snapshots_dir=tmp_path / "snapshots"
    ).for_funder("F")
    assert index.n_files == 2
    assert index.n_without_url == 1
    assert len(index.urls) == 1
