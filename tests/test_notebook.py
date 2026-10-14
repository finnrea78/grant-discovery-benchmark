"""Checks that notebooks/analysis.ipynb only presents numbers from the tested analysis code.

A statistic computed inside a cell would be an untested second copy that could disagree
with the module that owns it.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

NOTEBOOK = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "analysis.ipynb"


def _code_source() -> str:
    """Return the source of every code cell joined together (markdown cells are skipped)."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_the_notebook_computes_nothing_itself():
    """No code cell does its own statistical arithmetic."""
    source = _code_source()
    for banned in ("scipy", "np.percentile", "wilcoxon(", "default_rng", "/ len("):
        assert banned not in source, f"notebook computes its own statistics: {banned}"


def test_the_notebook_runs_end_to_end_and_leaves_no_errors():
    """The notebook executes cleanly with nbconvert."""
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--ExecutePreprocessor.kernel_name=python3",
         "--stdout", str(NOTEBOOK)],
        check=True, capture_output=True, timeout=900,
    )
