"""Tests that the hub links each funder to a page its snapshot actually contains.

Some funders were crawled from a subpath, so their snapshot has no ``/`` page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grant_eval.dataset.serve_all import discover, hub_index, make_browse_handler


def build_snapshot(
    root: Path, funder: str, entry_url: str, urls: list[str], date: str = "2026-08-21"
) -> Path:
    """Write a small snapshot with a manifest, files on disk and meta/snapshot.json."""
    snap = root / funder / date
    files = []
    for url in urls:
        rel = url.split("//", 1)[1]  # host/path
        path = f"site/{rel.rstrip('/')}/index.html" if url.endswith("/") else f"site/{rel}.html"
        dest = snap / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"<html><body>{url}</body></html>")
        files.append({"url": url, "path": path, "kind": "page", "bytes": 32})
    (snap / "manifest.json").write_text(json.dumps({"files": files}))
    (snap / "meta").mkdir(parents=True, exist_ok=True)
    (snap / "meta" / "snapshot.json").write_text(json.dumps({"entry_url": entry_url}))
    return snap


@pytest.fixture
def sites(tmp_path: Path):
    snaps, gt = tmp_path / "snapshots", tmp_path / "gt"
    gt.mkdir()
    # Crawled from a subpath: '/' is genuinely not in the snapshot.
    build_snapshot(
        snaps,
        "subpath-funder",
        "https://ex.org/what-we-do/",
        ["https://ex.org/what-we-do/", "https://ex.org/what-we-do/fellowships"],
    )
    # Crawled from the site root.
    build_snapshot(
        snaps, "root-funder", "https://root.org/", ["https://root.org/", "https://root.org/grants"]
    )
    found = discover(snaps, gt)
    for i, s in enumerate(found, start=1):
        s["port"] = 8100 + i
        # Populates s['captured'] and the landing path the hub links to.
        make_browse_handler(s, 8100, {}, rewrite=True, strict=False)
        add_render_stats(s)
    return {s["funder"]: s for s in found}


def add_render_stats(s: dict) -> None:
    """Add the counts main() computes before passing a site to hub_index."""
    files = s["manifest"]["files"]
    s["n_files"] = len(files)
    s["n_pages"] = sum(1 for f in files if f.get("kind") == "page")
    s["n_docs"] = sum(1 for f in files if f.get("kind") == "document")
    s["host"] = "example.org"
    s["n_style"] = 0
    s["style_cell"] = "<span class='pill pend'>none</span>"


def test_subpath_snapshot_lands_on_its_entry_not_root(sites):
    """A snapshot crawled from a subpath lands on its entry page, not '/'."""
    s = sites["subpath-funder"]
    assert s["landing"] == "/what-we-do/"
    assert s["captured"](s["landing"], "") is True
    assert s["captured"]("/", "") is False


def test_root_snapshot_still_lands_on_root(sites):
    s = sites["root-funder"]
    assert s["landing"] == "/"
    assert s["captured"]("/", "") is True


def test_hub_links_to_the_landing_path(sites):
    html = hub_index(sorted(sites.values(), key=lambda s: s["funder"]), 8100).decode()
    sub = sites["subpath-funder"]
    assert f"http://localhost:{sub['port']}/what-we-do/" in html
    # the bare origin would 404 for this funder
    assert f"'http://localhost:{sub['port']}/'" not in html


def test_landing_falls_back_to_a_captured_page_when_entry_is_missing(tmp_path: Path):
    """Without meta/snapshot.json the landing page is still a captured page."""
    snaps, gt = tmp_path / "snapshots", tmp_path / "gt"
    gt.mkdir()
    snap = build_snapshot(
        snaps, "no-meta", "https://ex.org/grants/", ["https://ex.org/grants/deep/award"]
    )
    (snap / "meta" / "snapshot.json").unlink()
    found = discover(snaps, gt)
    s = found[0]
    s["port"] = 8101
    make_browse_handler(s, 8100, {}, rewrite=True, strict=False)
    assert s["captured"](s["landing"], "") is True
