"""Tests for `grid_status`, which reports progress, spend and ETA from the results directory alone."""
from pathlib import Path

from grant_eval.dataset.registry import load_registry
from grant_eval.grid import status, store
from grant_eval.grid.cells import enumerate_cells
from grant_eval.schema import ExtractionResult, Split

REG = load_registry(Path("data/funders.json"))


def _plan():
    return enumerate_cells(REG, Split.TUNE, funders=["leverhulme"],
                           models=["nova-lite"], prompts=["minimal"])


def _result(cell, *, cost, latency):
    return ExtractionResult(
        snapshot_date=cell.snapshot_date, funder=cell.display_name, split=cell.split,
        model=cell.model_slug, prompt_strategy=cell.prompt_strategy,
        temperature=cell.temperature, agent_config=cell.agent_config,
        replicate=cell.replicate, config_hash=cell.config_hash,
        cost_usd=cost, latency_s=latency,
    )


def test_summary_of_an_empty_directory(tmp_path):
    plan = _plan()
    rep = status.summarise(tmp_path, plan)
    assert rep.total == len(plan)
    assert rep.complete == 0
    assert rep.pending == len(plan)


def test_summary_counts_and_costs(tmp_path):
    """Counts come from the plan, not the directory, and costs add up correctly."""
    plan = _plan()
    store.write_result_atomic(plan[0].result_path(tmp_path),
                              _result(plan[0], cost=0.01, latency=20.0))
    store.write_result_atomic(plan[1].result_path(tmp_path),
                              _result(plan[1], cost=0.03, latency=40.0))
    store.write_failure(plan[2], tmp_path, exc=RuntimeError("x"), attempts=2)
    rep = status.summarise(tmp_path, plan)
    assert rep.complete == 2
    assert rep.failed == 1
    assert rep.pending == len(plan) - 3
    assert rep.total_cost_usd == 0.04
    assert rep.mean_latency_s == 30.0


def test_status_works_with_no_running_process(tmp_path):
    """Status works without the executor, a lock file or any running process."""
    plan = _plan()
    store.write_result_atomic(plan[0].result_path(tmp_path),
                              _result(plan[0], cost=0.01, latency=10.0))
    assert status.summarise(tmp_path, plan).complete == 1
