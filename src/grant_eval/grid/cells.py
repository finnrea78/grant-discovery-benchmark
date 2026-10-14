"""Enumerate the grid cells and fix their execution order.

Cells are sorted before shuffling so the grid does not depend on dict order. The
plan is one seeded shuffle over the whole list, so provider drift over time becomes
noise rather than being confounded with a funder or config.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from grant_eval import config
from grant_eval.agent.extract import config_hash as _config_hash
from grant_eval.dataset.registry import FunderRecord, funders_for_split
from grant_eval.schema import Split


def _temperature_tag(temperature: float) -> str:
    """Render `0.0` as `"0"` and `0.7` as `"07"` for use inside a `cell_id`."""
    return str(temperature).replace(".", "").rstrip("0") or "0"


@dataclass(frozen=True)
class Cell:
    """One fully resolved grid cell: everything needed to run it and name its outputs."""

    funder: str
    display_name: str
    split: Split
    snapshot_date: date
    entry_url: str
    model: str          # grid label, e.g. "nova-lite"
    model_slug: str      # LiteLLM/OpenRouter string, e.g. "openrouter/amazon/nova-lite-v1"
    prompt_strategy: str
    temperature: float
    agent_config: str
    replicate: int
    config_hash: str

    @property
    def cell_id(self) -> str:
        """A filesystem-safe id, unique per cell.

        Includes the first 10 hex characters of `config_hash`, so a config change can
        never overwrite an old result."""
        model_tag = self.model_slug.rsplit("/", 1)[-1]
        temp_tag = _temperature_tag(self.temperature)
        return (
            f"{self.funder}_{model_tag}_{self.prompt_strategy}_t{temp_tag}_"
            f"{self.agent_config}_rep{self.replicate}_{self.config_hash[:10]}"
        )

    def identity(self) -> dict:
        """This cell's identity as a JSON-ready dict, for logs and the run header."""
        return {
            "cell_id": self.cell_id,
            "funder": self.funder,
            "model": self.model,
            "model_slug": self.model_slug,
            "prompt_strategy": self.prompt_strategy,
            "temperature": self.temperature,
            "agent_config": self.agent_config,
            "replicate": self.replicate,
            "config_hash": self.config_hash,
        }

    def result_path(self, root: Path) -> Path:
        return root / "grid" / self.funder / f"{self.cell_id}.json"

    def failed_path(self, root: Path) -> Path:
        return root / "grid" / self.funder / f"{self.cell_id}.failed.json"

    def trace_dir(self, root: Path) -> Path:
        return root / "grid" / self.cell_id


def _resolved_config_hash(
    model_label: str, prompt_strategy: str, temperature: float, agent_config: str
) -> str:
    """`extract.config_hash` for one (model, prompt, temperature, agent_config)."""
    return _config_hash(
        model=config.MODELS[model_label],
        prompt_strategy=prompt_strategy,
        temperature=temperature,
        agent_config=agent_config,
        agent_cfg=config.AGENT_CONFIGS[agent_config],
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )


def enumerate_cells(
    registry: dict[str, FunderRecord],
    split: Split,
    *,
    funders: list[str] | None = None,
    models: list[str] | None = None,
    prompts: list[str] | None = None,
    replicates: list[int] | None = None,
) -> list[Cell]:
    """Every `Cell` for `split`, optionally narrowed by the filters.

    Returned in a fixed sorted order so `ordered_plan` always shuffles the same input.
    """
    funder_records = funders_for_split(registry, split)
    if funders is not None:
        wanted = set(funders)
        funder_records = [r for r in funder_records if r.slug in wanted]

    raw_cells = config.grid_cells()
    if models is not None:
        wanted_models = set(models)
        raw_cells = [c for c in raw_cells if c["model"] in wanted_models]
    if prompts is not None:
        wanted_prompts = set(prompts)
        raw_cells = [c for c in raw_cells if c["prompt_strategy"] in wanted_prompts]
    if replicates is not None:
        wanted_reps = set(replicates)
        raw_cells = [c for c in raw_cells if c["replicate"] in wanted_reps]

    # `config_hash` never depends on funder or replicate, so compute it once and reuse.
    hash_cache: dict[tuple[str, str, float, str], str] = {}

    cells: list[Cell] = []
    for record in funder_records:
        for raw in raw_cells:
            key = (raw["model"], raw["prompt_strategy"], raw["temperature"], raw["agent_config"])
            if key not in hash_cache:
                hash_cache[key] = _resolved_config_hash(*key)
            cells.append(
                Cell(
                    funder=record.slug,
                    display_name=record.display_name,
                    split=record.split,
                    snapshot_date=record.snapshot_date,
                    entry_url=record.entry_url,
                    model=raw["model"],
                    model_slug=config.MODELS[raw["model"]],
                    prompt_strategy=raw["prompt_strategy"],
                    temperature=raw["temperature"],
                    agent_config=raw["agent_config"],
                    replicate=raw["replicate"],
                    config_hash=hash_cache[key],
                )
            )

    cells.sort(
        key=lambda c: (
            c.funder,
            c.model,
            c.prompt_strategy,
            c.temperature,
            c.agent_config,
            c.replicate,
        )
    )
    return cells


def ordered_plan(cells: list[Cell], seed: int) -> list[Cell]:
    """Seeded shuffle of a copy of `cells`. The same seed always gives the same order,
    so a resumed run gets the same order."""
    plan = list(cells)
    random.Random(seed).shuffle(plan)
    return plan
