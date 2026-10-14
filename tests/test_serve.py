"""Tests that resolve() finds captured files whose URLs contain percent-escapes.

wget names files after the decoded path but the manifest and the browser use the
encoded URL, so both forms need to resolve.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grant_eval.dataset.serve import build_lookup, resolve


@pytest.fixture
def lookup(tmp_path: Path):
    snap = tmp_path / "snap"
    files = [
        # encoded URL, file on disk named with the decoded space
        {
            "url": "https://ex.org/wp-content/plugins/divi_layout_injector%203.5/style.css",
            "path": "site/ex.org/wp-content/plugins/divi_layout_injector 3.5/style.css",
            "kind": "asset",
        },
        # ordinary path, no escapes
        {"url": "https://ex.org/plain.css", "path": "site/ex.org/plain.css", "kind": "asset"},
        # query string preserved in the filename, as wget writes it
        {
            "url": "https://ex.org/theme.css?ver=1.2",
            "path": "site/ex.org/theme.css?ver=1.2.css",
            "kind": "asset",
        },
    ]
    for f in files:
        p = snap / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("body{}")
    (snap / "manifest.json").write_text(json.dumps({"files": files}))
    return build_lookup(snap, {"files": files})


def test_percent_encoded_request_finds_the_file(lookup):
    """The encoded form, as the browser sends it, resolves."""
    assert resolve(lookup, "/wp-content/plugins/divi_layout_injector%203.5/style.css", "")


def test_decoded_request_finds_the_same_file(lookup):
    """The decoded form resolves too."""
    assert resolve(lookup, "/wp-content/plugins/divi_layout_injector 3.5/style.css", "")


def test_plain_paths_are_unaffected(lookup):
    assert resolve(lookup, "/plain.css", "")
    assert resolve(lookup, "/theme.css", "ver=1.2")


def test_still_returns_none_for_something_never_captured(lookup):
    assert resolve(lookup, "/nope.css", "") is None
    assert resolve(lookup, "/wp-content/plugins/other%20plugin/style.css", "") is None
