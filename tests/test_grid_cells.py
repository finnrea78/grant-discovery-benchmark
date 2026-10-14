"""Tests for cell enumeration and the seeded execution plan.

The execution plan is a seeded shuffle of all cells, so changes in provider behaviour over time
are not confounded with funder or config.
"""
from __future__ import annotations

from pathlib import Path

from grant_eval import config
from grant_eval.dataset.registry import load_registry
from grant_eval.grid.cells import Cell, enumerate_cells, ordered_plan
from grant_eval.schema import Split

REG = load_registry(Path("data/funders.json"))


def test_tune_grid_is_1920_cells():
    assert len(enumerate_cells(REG, Split.TUNE)) == 1920


def test_validation_grid_is_960_cells():
    """8 validation funders x 120 cells. The lock file picks the 40 cells actually run."""
    assert len(enumerate_cells(REG, Split.VALIDATION)) == 960


def test_every_cell_id_is_unique():
    ids = [c.cell_id for c in enumerate_cells(REG, Split.TUNE)]
    assert len(set(ids)) == len(ids)


def test_every_cell_id_is_unique_for_validation():
    """Cell IDs are unique on the validation split too."""
    ids = [c.cell_id for c in enumerate_cells(REG, Split.VALIDATION)]
    assert len(set(ids)) == len(ids)


def test_every_cell_id_is_unique_across_both_splits_combined():
    """No tune cell ID clashes with a validation one, since both share the same output root."""
    tune_ids = [c.cell_id for c in enumerate_cells(REG, Split.TUNE)]
    validation_ids = [c.cell_id for c in enumerate_cells(REG, Split.VALIDATION)]
    combined = tune_ids + validation_ids
    assert len(set(combined)) == len(combined)


def test_cell_id_shape():
    """Cell IDs have the expected shape.

    Cells are looked up by temperature because "t07" sorts before "t0_" in ASCII.
    """
    cells = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )
    assert len(cells) == 2  # two temperatures
    by_temp = {c.temperature: c.cell_id for c in cells}
    assert set(by_temp) == {0.0, 0.7}
    assert by_temp[0.0].startswith("leverhulme_nova-lite-v1_minimal_t0_standard_rep1_")
    assert by_temp[0.7].startswith("leverhulme_nova-lite-v1_minimal_t07_standard_rep1_")
    assert len(by_temp[0.0].rsplit("_", 1)[1]) == 10


def test_temperature_tag_renders_bare_digits_not_decimals():
    """Temperatures render as `0` and `07`, not `0.0` and `0.7`."""
    cells = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )
    tags = {c.temperature: c.cell_id.split("_t", 1)[1].split("_", 1)[0] for c in cells}
    assert tags[0.0] == "0"
    assert tags[0.7] == "07"


def test_paths_are_nested_by_funder():
    cell = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[0]
    p = cell.result_path(Path("eval/results/phase1"))
    assert p.parent == Path("eval/results/phase1/grid/leverhulme")
    assert cell.failed_path(Path("eval/results/phase1")).name.endswith(".failed.json")


def test_trace_dir_is_flat_and_keyed_by_cell_id():
    """`trace_dir` has no funder subdirectory, since the cell ID already starts with the funder."""
    cell = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[0]
    root = Path("eval/results/phase1")
    assert cell.trace_dir(root) == root / "grid" / cell.cell_id


def test_enumeration_is_deterministic():
    a = [c.cell_id for c in enumerate_cells(REG, Split.TUNE)]
    b = [c.cell_id for c in enumerate_cells(REG, Split.TUNE)]
    assert a == b


def test_plan_is_deterministic_given_the_seed_and_shuffles():
    cells = enumerate_cells(REG, Split.TUNE)
    p1 = [c.cell_id for c in ordered_plan(cells, config.GRID_SEED)]
    p2 = [c.cell_id for c in ordered_plan(cells, config.GRID_SEED)]
    p3 = [c.cell_id for c in ordered_plan(cells, config.GRID_SEED + 1)]
    assert p1 == p2
    assert p1 != p3
    assert sorted(p1) == sorted(p3)


def test_ordered_plan_does_not_mutate_its_input():
    """`ordered_plan` shuffles a copy and leaves its input alone."""
    cells = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme", "wellcome"], models=["nova-lite"],
        prompts=["minimal"],
    )
    before = list(cells)
    ordered_plan(cells, config.GRID_SEED)
    assert cells == before


def test_plan_interleaves_funders():
    """The first 32 cells cover many funders (about 14 expected, 13 with the real seed)."""
    plan = ordered_plan(enumerate_cells(REG, Split.TUNE), config.GRID_SEED)
    assert len({c.funder for c in plan[:32]}) >= 10


def test_plan_is_not_funder_contiguous():
    """Adjacent cells rarely share a funder (about 1 in 16 expected)."""
    plan = ordered_plan(enumerate_cells(REG, Split.TUNE), config.GRID_SEED)
    same_as_prev = sum(
        1 for a, b in zip(plan, plan[1:]) if a.funder == b.funder
    )
    assert same_as_prev / (len(plan) - 1) < 0.25


def test_cell_carries_the_registry_entry_url():
    cell = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"])[0]
    assert cell.entry_url == REG["leverhulme"].entry_url
    assert cell.display_name == "The Leverhulme Trust"


def test_config_hash_uses_the_resolved_model_slug_not_the_label():
    """The config hash uses the full model slug, not the short grid label."""
    from grant_eval.agent.extract import config_hash

    cell = enumerate_cells(
        REG, Split.TUNE, funders=["leverhulme"], models=["nova-lite"],
        prompts=["minimal"], replicates=[1],
    )[0]
    expected = config_hash(
        model=config.MODELS["nova-lite"],
        prompt_strategy="minimal",
        temperature=cell.temperature,
        agent_config="standard",
        agent_cfg=config.AGENT_CONFIGS["standard"],
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    assert cell.config_hash == expected
    # Check the label gives a different hash, otherwise the assert above proves nothing.
    mislabelled = config_hash(
        model="nova-lite",
        prompt_strategy="minimal",
        temperature=cell.temperature,
        agent_config="standard",
        agent_cfg=config.AGENT_CONFIGS["standard"],
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    assert cell.config_hash != mislabelled


def test_grid_covers_all_three_models_and_four_prompt_strategies():
    """Every model and prompt strategy appears in the grid."""
    cells = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"])
    assert {c.model for c in cells} == set(config.MODELS)
    assert {c.prompt_strategy for c in cells} == set(config.PROMPT_STRATEGIES)


def test_unfiltered_call_reaches_every_split_funder():
    """The tune grid includes every tune funder from the registry."""
    from grant_eval.dataset.registry import funders_for_split

    expected_slugs = {r.slug for r in funders_for_split(REG, Split.TUNE)}
    cells = enumerate_cells(REG, Split.TUNE)
    assert {c.funder for c in cells} == expected_slugs
