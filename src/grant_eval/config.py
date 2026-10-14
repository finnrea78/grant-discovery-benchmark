"""Grid definition and pinned model snapshots for the evaluation.

Two cheap or open-weight models plus one low-cost commercial reference model.
"""
from __future__ import annotations

from itertools import product

# Models (IV). Values are LiteLLM/OpenRouter model strings, checked against the
# OpenRouter catalogue before the grid run.
MODELS: dict[str, str] = {
    # label             : pinned slug
    "deepseek-flash": "openrouter/deepseek/deepseek-v4-flash-0731",  # dated snapshot
    "gpt-oss-120b":   "openrouter/openai/gpt-oss-120b",              # open weights
    "nova-lite":      "openrouter/amazon/nova-lite-v1",              # commercial reference
}

# Models considered and not used:
#   openai/gpt-5.6-luna           - takes no `temperature`, which is one of the IVs.
#   google/gemini-2.0-flash       - delisted.
#   anthropic/claude-haiku-4.5    - dropped on cost ($1.00/$5.00), not capability.
#   google/gemini-2.5-flash-lite  - $0.10/$0.40 against nova-lite's $0.06/$0.24.
#   amazon/nova-micro-v1          - risk of floor effects.
#   openai/gpt-4o-mini-2024-07-18 - the commercial slot went to nova-lite, on cost.
#   qwen/qwen3.7-flash            - cheap with a 1M context, but rate-limited upstream.

# Per-model routing, held constant across the grid.
# Used by `router.OpenRouterBackend._routing_for`. The provider that served each
# response is read back and checked against `["provider"]["only"]`; a mismatch halts the
# grid. Quantisation cannot be checked because no response reports it, so
# `allow_fallbacks: False` plus the provider pin is the best available control.
MODEL_ROUTING: dict[str, dict] = {
    "deepseek-flash": {
        "provider": {"only": ["parasail"], "quantizations": ["fp8"],
                     "allow_fallbacks": False},
        "reasoning_effort": "medium",
    },
    "gpt-oss-120b": {
        "provider": {"only": ["deepinfra"], "quantizations": ["bf16"],
                     "allow_fallbacks": False},
        "reasoning_effort": "medium",
    },
    "nova-lite": {
        # Closed weights on Bedrock, so no quantisation to pin, and no reasoning mode.
        "provider": {"only": ["amazon-bedrock"], "allow_fallbacks": False},
    },
    "gemini-judge": {
        # The LLM-as-judge model used by `scoring/judge.py`, not a grid arm, so it must
        # never be added to `MODELS`. Pinned anyway so the judge cannot drift to another
        # provider mid-pass. `judge.py` reads this entry by label because the model
        # string has no `MODELS` entry. It is not part of `config_hash`, so adding it
        # moved no result paths.
        "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False},
    },
}

# Prompt caching is disabled: providers cache differently, so leaving it on would make
# the cost comparison partly a comparison of caching policies.
PROMPT_CACHING: str = "disabled"  # "disabled" | "enabled_reported_both"

# Prompt strategies (IV)
PROMPT_STRATEGIES: list[str] = ["minimal", "structured", "chain_of_thought", "few_shot"]

# Decoding temperature (IV)
TEMPERATURES: list[float] = [0.0, 0.7]

# Agent operational config, a single level.
# `step_budget` is 24 rather than 16: at 16 gpt-oss-120b ran out of steps on two prompts
# and scored F1 0.000 there, while deepseek-v4-flash needed only 6 to 10. A budget that
# cuts off one model mid-task would confound the model comparison.
AGENT_CONFIGS: dict[str, dict] = {
    "standard": {"step_budget": 24},
}

# Replicates, for the reliability analysis (RQ3).
REPLICATES: int = 5

# Seed for the shuffled run order of all 1,920 cells. Random order turns provider drift
# over a multi-day run into noise rather than bias against one funder or config. It
# cannot change a cell's result, so it is kept out of `config_hash` and `result_sha`.
GRID_SEED: int = 20260831

# Agent loop parameters, fixed in the pre-registration. Both change results, so both
# are in the config hash.
#
# `budget_notice_at`: with this many steps or fewer left, the loop says how many remain
#   and that submit_opportunities must be called. At 2 a DeepSeek run submitted nine
#   good records on its last step, failed validation on one field, and had no budget
#   for the repair. 3 leaves room for submit, rejection and resubmit. 0 disables it.
# `carry_reasoning`: use a reasoning model's `reasoning` as assistant content when
#   `content` is empty. Two of the three models return reasoning.
LOOP: dict = {
    "budget_notice_at": 3,
    "carry_reasoning": True,
}

# Ralph loop parameters, fixed in the pre-registration (`eval/preregistration.md`).
# One iteration: nothing carries between iterations, so repeated draws of a cell come
# from `REPLICATES` instead. `termination` and `scored_iteration_policy` have no effect
# at one iteration but stay so the `config_hash` payload keeps its shape.
RALPH: dict = {
    "max_iterations": 1,
    "termination": "schema_valid_and_no_new_opportunities",  # inert at 1 iteration
    "scored_iteration_policy": "final",   # "final" | "best_of"; inert at 1 iteration
}


def grid_cells() -> list[dict]:
    """Every (model, prompt, temperature, agent_config, replicate) cell, without funders.

    Funders are added per split at run time.
    """
    cells = []
    for model, prompt, temp, cfg, rep in product(
        MODELS, PROMPT_STRATEGIES, TEMPERATURES, AGENT_CONFIGS, range(1, REPLICATES + 1)
    ):
        cells.append(
            {
                "model": model,
                "prompt_strategy": prompt,
                "temperature": temp,
                "agent_config": cfg,
                "replicate": rep,
            }
        )
    return cells


def main() -> int:
    n = len(grid_cells())
    print(f"{n} cells/funder  "
          f"({len(MODELS)} models x {len(PROMPT_STRATEGIES)} prompts x "
          f"{len(TEMPERATURES)} temps x {len(AGENT_CONFIGS)} configs x {REPLICATES} reps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
