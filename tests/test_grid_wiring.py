"""Checks on console scripts, trace storage and the pre-registration file.

These are file-content checks, since pyproject entries and prose files cannot be run.
"""

import tomllib
from pathlib import Path


def test_console_scripts_are_registered():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["grid_run"] == "grant_eval.run:main"
    assert scripts["grid_status"] == "grant_eval.grid.status:main"


def test_console_scripts_no_longer_expose_the_generic_run_name():
    # `run` is too generic a name for a shared environment.
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert "run" not in scripts
    # `grid` (the cell-count entry point) should still be there.
    assert scripts["grid"] == "grant_eval.config:main"


def test_uncompressed_grid_traces_are_gitignored():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "eval/run-traces/grid/**/*.jsonl" in text


def test_preregistration_records_the_six_new_lines():
    text = Path("eval/preregistration.md").read_text(encoding="utf-8")
    for needle in [
        "GRID_SEED",
        "crawl entry URL",
        "timed_out",
        "5 replicates",
        "dry run",
        "scope-cut",
    ]:
        assert needle in text, needle


def test_preregistration_no_longer_claims_the_homepage_entry():
    text = Path("eval/preregistration.md").read_text(encoding="utf-8").lower()
    assert "starts from the funder homepage" not in text


def test_funders_md_cites_funders_json():
    text = Path("data/funders.md").read_text(encoding="utf-8")
    assert "funders.json" in text
