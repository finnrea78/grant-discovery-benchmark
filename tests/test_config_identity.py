"""Tests for `GRID_SEED` and how prompt text feeds into `config_hash`.

`GRID_SEED` only sets execution order. `config_hash` includes the prompt template
rendered with fixed sentinel values, not the per-run text.
"""
from __future__ import annotations

from datetime import date

import pytest

from grant_eval import config
from grant_eval.agent import extract
from grant_eval.agent.extract import SENTINEL_PROMPT_CONTEXT, config_hash, prompt_fingerprint

BASE = dict(
    model="openrouter/deepseek/deepseek-v4-flash-0731",
    prompt_strategy="minimal",
    temperature=0.0,
    agent_config="standard",
    agent_cfg={"step_budget": 16},
    ralph_config=config.RALPH,
    loop_config=config.LOOP,
)


def test_grid_seed_exists_and_is_an_int():
    assert isinstance(config.GRID_SEED, int)


def test_grid_seed_is_not_in_the_config_hash_payload(monkeypatch):
    """Changing GRID_SEED does not change the config hash.

    The `hasattr` check catches `extract.py` importing GRID_SEED by name, which would
    make the monkeypatch below ineffective.
    """
    assert not hasattr(extract, "GRID_SEED"), (
        "extract.py now imports GRID_SEED by name; config_hash must read it (if at "
        "all) via config.GRID_SEED, or this test's monkeypatch stops being effective"
    )
    before = config_hash(**BASE)
    monkeypatch.setattr(config, "GRID_SEED", config.GRID_SEED + 1)
    monkeypatch.setattr(extract, "GRID_SEED", config.GRID_SEED, raising=False)
    assert config_hash(**BASE) == before


def test_sentinel_prompt_context_values():
    """The sentinel context is fixed values, so nothing like `date.today()` can creep in."""
    assert SENTINEL_PROMPT_CONTEXT == {
        "entry_url": "https://example.invalid/",
        "funder": "SENTINEL FUNDER",
        "snapshot_date": date(2000, 1, 1),
    }


def test_sentinel_prompt_context_cannot_be_mutated():
    """The sentinel context is read-only, since `prompt_fingerprint` is cached on strategy alone."""
    with pytest.raises(TypeError):
        SENTINEL_PROMPT_CONTEXT["funder"] = "SOMETHING ELSE"


def test_prompt_fingerprint_is_stable_and_strategy_specific():
    assert prompt_fingerprint("minimal") == prompt_fingerprint("minimal")
    assert len(prompt_fingerprint("minimal")) == 64


def test_config_hash_moves_when_the_prompt_system_template_changes(monkeypatch):
    """Changing the system prompt template changes the config hash."""
    import grant_eval.agent.prompts as prompts

    before = config_hash(**BASE)
    real = prompts.render

    def tweaked(strategy, **kwargs):
        system, user = real(strategy, **kwargs)
        return system + " EXTRA SENTENCE.", user

    monkeypatch.setattr(prompts, "render", tweaked)
    prompt_fingerprint.cache_clear()
    try:
        assert config_hash(**BASE) != before
    finally:
        prompt_fingerprint.cache_clear()


def test_config_hash_moves_when_the_prompt_user_template_changes(monkeypatch):
    """Changing the user prompt template changes the config hash too."""
    import grant_eval.agent.prompts as prompts

    before = config_hash(**BASE)
    real = prompts.render

    def tweaked(strategy, **kwargs):
        system, user = real(strategy, **kwargs)
        return system, user + " EXTRA."

    monkeypatch.setattr(prompts, "render", tweaked)
    prompt_fingerprint.cache_clear()
    try:
        assert config_hash(**BASE) != before
    finally:
        prompt_fingerprint.cache_clear()


def test_every_registered_strategy_fingerprints_for_real():
    """Every registered strategy has a real SHA-256 fingerprint, not the sentinel."""
    for strategy in config.PROMPT_STRATEGIES:
        fp = prompt_fingerprint(strategy)
        assert len(fp) == 64, f"{strategy!r} is registered but not authored: {fp}"


def test_unimplemented_strategy_fingerprints_as_a_sentinel(monkeypatch):
    """A strategy whose prompt is not written yet gets an `unimplemented:` fingerprint.

    This lets the grid be planned before a prompt exists; preflight refuses to run such cells.
    """
    import grant_eval.agent.prompts as prompts

    def unwritten(strategy, **kwargs):
        raise NotImplementedError(strategy)

    monkeypatch.setattr(prompts, "render", unwritten)
    prompt_fingerprint.cache_clear()
    try:
        assert prompt_fingerprint("chain_of_thought") == "unimplemented:chain_of_thought"
    finally:
        prompt_fingerprint.cache_clear()


def test_prompt_fingerprint_differs_across_strategies():
    """Each strategy has a different fingerprint even though they share `_TASK_CORE`."""
    fingerprints = {s: prompt_fingerprint(s) for s in config.PROMPT_STRATEGIES}
    assert len(set(fingerprints.values())) == len(config.PROMPT_STRATEGIES), fingerprints


def test_config_hash_is_a_pure_function_of_its_arguments():
    """Changing only `prompt_strategy` changes the hash."""
    a = config_hash(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        agent_cfg={"step_budget": 16},
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    b = config_hash(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        prompt_strategy="structured",
        temperature=0.0,
        agent_config="standard",
        agent_cfg={"step_budget": 16},
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    assert a != b


def test_config_hash_canary_for_the_frozen_deepseek_minimal_cell():
    """The hash for one real grid cell (deepseek-flash, minimal, t=0.0, standard) has not changed.

    The other tests only compare hashes, so this catches the value moving by accident.
    If it fails, work out why before updating the expected value.
    """
    h = config_hash(
        model="openrouter/deepseek/deepseek-v4-flash-0731",
        prompt_strategy="minimal",
        temperature=0.0,
        agent_config="standard",
        agent_cfg=config.AGENT_CONFIGS["standard"],
        ralph_config=config.RALPH,
        loop_config=config.LOOP,
    )
    assert h == "6193098e749b172a7fa3120654fcb230dd99c9da9d666e75ad1ee266c5855472"
