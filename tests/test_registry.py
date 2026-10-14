"""Tests for the funder registry in data/funders.json and its snapshots."""
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pytest

from grant_eval.dataset.registry import (
    FunderRecord,
    funders_for_split,
    load_registry,
)
from grant_eval.dataset.serve import build_lookup, resolve
from grant_eval.schema import Split
from grant_eval.scoring.groundtruth import load_ground_truth

REGISTRY = Path("data/funders.json")
FUNDERS_MD = Path("data/funders.md")

# Same format as `snapshot._LOG_LINE`, copied here so the test does not share code with
# what it checks.
WGET_LOG_LINE = re.compile(r'URL:(\S+)\s+\[[^\]]*\]\s+->\s+"([^"]+)"')


def test_registry_has_24_funders_split_16_8():
    reg = load_registry(REGISTRY)
    assert len(reg) == 24
    assert len(funders_for_split(reg, Split.TUNE)) == 16
    assert len(funders_for_split(reg, Split.VALIDATION)) == 8
    assert "wellbeing-of-women" not in reg


def test_leverhulme_record_is_exact():
    rec = load_registry(REGISTRY)["leverhulme"]
    assert rec.display_name == "The Leverhulme Trust"
    assert rec.split is Split.TUNE
    assert rec.snapshot_date == date(2026, 8, 9)
    assert rec.entry_url == "https://www.leverhulme.ac.uk/"


def test_every_snapshot_dir_exists_with_a_manifest():
    for rec in load_registry(REGISTRY).values():
        manifest = rec.snapshot_dir() / "manifest.json"
        assert manifest.is_file(), f"{rec.slug}: missing {manifest}"


def test_content_hash_matches_the_manifest():
    for rec in load_registry(REGISTRY).values():
        manifest = json.loads(
            (rec.snapshot_dir() / "manifest.json").read_text(encoding="utf-8")
        )
        assert rec.content_hash == manifest["content_hash"], rec.slug


def test_entry_url_matches_the_first_url_crawled():
    """`entry_url` matches the first URL in meta/wget.log.

    The manifest's `url` field can be rebuilt from the path, so wget's own log is used instead.
    """
    for rec in load_registry(REGISTRY).values():
        log_path = rec.snapshot_dir() / "meta" / "wget.log"
        assert log_path.is_file(), f"{rec.slug}: missing {log_path}"
        first = None
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = WGET_LOG_LINE.search(line)
            if m:
                first = m.group(1)
                break
        assert first is not None, f"{rec.slug}: no parseable wget.log entry"
        assert rec.entry_url.rstrip("/") == first.rstrip("/"), rec.slug


def test_entry_url_resolves_to_a_captured_page():
    """`entry_url` resolves to a captured page using the same resolver the agent's fetch tool uses."""
    for rec in load_registry(REGISTRY).values():
        snap_dir = rec.snapshot_dir()
        manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
        lookup = build_lookup(snap_dir, manifest)
        parsed = urlparse(rec.entry_url)
        hit = resolve(lookup, parsed.path or "/", parsed.query)
        assert hit is not None, f"{rec.slug}: entry_url {rec.entry_url} is a 404"


def test_split_membership_matches_funders_md():
    """The tune and validation slug sets match the lists in data/funders.md."""
    text = FUNDERS_MD.read_text(encoding="utf-8")

    def slugs_after(label: str) -> set[str]:
        m = re.search(rf"\*\*{label}[^:]*:\*\*(.*?)\.\n", text, re.S)
        assert m, f"could not find a {label!r} list in {FUNDERS_MD}"
        body = m.group(1).replace("\n", " ")
        return {s.strip(" `") for s in body.split(",") if s.strip(" `")}

    reg = load_registry(REGISTRY)
    tune_slugs = {r.slug for r in funders_for_split(reg, Split.TUNE)}
    val_slugs = {r.slug for r in funders_for_split(reg, Split.VALIDATION)}

    assert tune_slugs == slugs_after("Tune")
    assert val_slugs == slugs_after("Validation")


def test_funder_record_is_frozen():
    rec = load_registry(REGISTRY)["leverhulme"]
    with pytest.raises(Exception):
        rec.slug = "nope"


def test_load_registry_rejects_a_duplicate_slug(tmp_path):
    bad = tmp_path / "dupe.json"
    row = {
        "slug": "x", "display_name": "X", "split": "tune",
        "snapshot_date": "2026-08-09", "entry_url": "https://x/",
        "content_hash": "a" * 64,
    }
    bad.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(bad)


def test_every_registry_funder_has_at_least_one_in_scope_row():
    """Every funder has at least one in-scope ground-truth row.

    A funder with none has a recall denominator of 0 and so tells us nothing.
    """
    gt = load_ground_truth()
    barren = []
    for rec in load_registry(REGISTRY).values():
        try:
            rows = gt.for_funder(rec.display_name)
        except KeyError:
            barren.append(f"{rec.slug}: no ground-truth file at all")
            continue
        if not rows.in_scope:
            barren.append(f"{rec.slug}: {len(rows.rows)} labelled rows, 0 in scope")
    assert not barren, "registry funders with no in-scope rows: " + "; ".join(barren)
