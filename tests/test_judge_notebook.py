"""Checks that notebooks/judge.ipynb only presents numbers from the tested judge code.

Kept separate from test_notebook.py so a failure is reported under the right notebook.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

NOTEBOOK = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "judge.ipynb"


def _code_source() -> str:
    """Return the source of every code cell joined together.

    Markdown is skipped because the prose mentions `1 / sampling probability`.
    """
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_the_notebook_computes_nothing_itself():
    """No code cell does its own statistical arithmetic."""
    source = _code_source()
    for banned in ("scipy", "np.percentile", "wilcoxon(", "default_rng", "/ len("):
        assert banned not in source, f"notebook computes its own statistics: {banned}"


def test_the_notebook_reads_the_real_judged_corpus():
    """The notebook reads the real judge output and the real human review."""
    source = _code_source()
    assert "eval\" / \"judge_results" in source or "judge_results" in source, (
        "notebook does not point at the real judge output root"
    )
    assert "judge_fake_demo" not in source, "notebook still points at the fake-demo root"
    assert "run_judge_batch" not in source, (
        "notebook still generates its own verdicts; it must read the committed pass"
    )
    assert "FakeJudgeBackend" not in source, "notebook still uses a fake judge backend"
    assert "simulate_human_codes" not in source, "notebook still simulates human codes"
    assert "judge_check.csv" in source, "notebook does not read the human review"


def test_the_notebook_verifies_provenance_before_presenting_numbers():
    """The notebook checks the judge config hash before showing any rates.

    The hash is recomputed from the module constants, so a changed judge model,
    temperature or prompt makes the notebook stop instead of plotting stale verdicts.
    """
    source = _code_source()
    assert "judgemod.rollup_path(judge_config.config_hash" in source
    assert "assert rollup_out_path.is_file()" in source, (
        "notebook presents rates without first asserting the roll-up matches the "
        "judge config it recomputed"
    )


def test_the_notebook_runs_end_to_end_and_leaves_no_errors():
    """The notebook executes cleanly with nbconvert.

    The long timeout is because Section 7 resolves every mirror page from disk.
    """
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--ExecutePreprocessor.kernel_name=python3",
         "--stdout", str(NOTEBOOK)],
        check=True, capture_output=True, timeout=1800,
    )
