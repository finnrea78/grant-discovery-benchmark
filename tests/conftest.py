"""Shared test fixtures."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def git_checkout(tmp_path_factory, monkeypatch):
    """Make `git rev-parse` and `git status` work when the code is not a git checkout.

    If the repo root has no `.git`, point git at a throwaway repo with one empty commit
    and the real files as its work tree. Nothing is tracked, so the tree reads as clean.
    """
    if (REPO_ROOT / ".git").exists():
        yield
        return
    git_dir = tmp_path_factory.mktemp("gitdir") / ".git"
    subprocess.run(["git", "init", "-q", str(git_dir.parent)], check=True)
    monkeypatch.setenv("GIT_DIR", str(git_dir))
    monkeypatch.setenv("GIT_WORK_TREE", str(REPO_ROOT))
    subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=Test",
         "commit", "-q", "--allow-empty", "-m", "init"],
        check=True,
    )
    yield
