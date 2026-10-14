"""Helpers for `notebooks/judge.ipynb`, so the notebook only loads, calls and plots.

Includes corpus breakdowns and rankings built on `judge.build_rollup`, plus synthetic
fixtures (a fake judge reply and a simulated human coder) used to exercise the
pipeline before any real judging or human coding exists.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from grant_eval.scoring import judge as judgemod
from grant_eval.scoring.judge import CLAIM_TYPES
from grant_eval.scoring.judge_agreement import (
    SPOTCHECK_FIELDNAMES,
    JudgedClaim,
    SpotcheckSelection,
)
from grant_eval.scoring.judge_agreement import _VALID_HUMAN_CODES as _VALID_HUMAN_CODES


def group_rollup(run_dicts: list[dict], *, key: str) -> dict[str, dict[str, dict]]:
    """Like `build_rollup`'s `by_funder`, but grouped by any run attribute such as `model`.

    Uses `judge.py`'s own accumulator helpers so the rates match `build_rollup`.
    """
    groups: dict[str, dict[str, dict]] = {}
    for run in run_dicts:
        group_value = str(run[key])
        cells = groups.setdefault(group_value, {})
        for cell_key, cell in run["fields"].items():
            field, dimension = cell["field"], cell["dimension"]
            acc = cells.setdefault(cell_key, judgemod._new_rollup_accumulator(field, dimension))
            judgemod._fold_cell_into(acc, cell)
    return {
        group_value: {cell_key: judgemod._finalise_rollup_cell(acc) for cell_key, acc in cells.items()}
        for group_value, cells in groups.items()
    }


def exclusion_reason_counts(ledger: list[dict]) -> dict[str, int]:
    """Tally `build_rollup`'s `exclusion_ledger` by `reason`."""
    counts: dict[str, int] = {}
    for row in ledger:
        counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    return counts


#: Fixed cell order for per-config rows, so table columns and means use the same order.
CELL_ORDER: tuple[str, ...] = tuple(
    f"{f}:{d}" for f in judgemod.JUDGED_FIELDS for d in judgemod.DIMENSIONS
)


def group_rollup_by_config(run_dicts: list[dict]) -> dict[str, dict[str, dict]]:
    """`group_rollup` grouped by the whole configuration rather than one factor.

    Uses `analysis.dataset.config_label` so the labels match the F1 leaderboard.
    """
    from grant_eval.analysis.dataset import config_label

    groups: dict[str, dict[str, dict]] = {}
    for run in run_dicts:
        label = config_label(run["model"], run["prompt_strategy"], run["temperature"])
        cells = groups.setdefault(label, {})
        for cell_key, cell in run["fields"].items():
            acc = cells.setdefault(
                cell_key, judgemod._new_rollup_accumulator(cell["field"], cell["dimension"])
            )
            judgemod._fold_cell_into(acc, cell)
    return {
        label: {k: judgemod._finalise_rollup_cell(acc) for k, acc in cells.items()}
        for label, cells in groups.items()
    }


def rank_configs(run_dicts: list[dict], *, rate_key: str = "claim_rate_macro") -> list[dict]:
    """Every configuration, best `combined_macro` first (the judge leaderboard).

    `combined_macro` is the plain mean of the four (field, dimension) cell rates. It is
    for presentation only, not a pre-registered endpoint: completeness varies much more
    than groundedness so it drives the mean, and groundedness rewards extracting less.
    Undefined cells are skipped and `n_cells` says how many were used. `rate_key` picks
    the cell rate; macro is the default because micro favours configs that write more.
    """
    grouped = group_rollup_by_config(run_dicts)
    rows: list[dict] = []
    for label, cells in grouped.items():
        rates: dict[str, float | None] = {}
        for cell_key in CELL_ORDER:
            cell = cells.get(cell_key)
            if cell is None or cell[f"{rate_key}_undefined"]:
                rates[cell_key] = None
            else:
                rates[cell_key] = cell[rate_key]
        defined = [v for v in rates.values() if v is not None]
        rows.append({
            "config": label,
            "rates": rates,
            "combined_macro": (sum(defined) / len(defined)) if defined else None,
            "combined_macro_undefined": not defined,
            "n_cells": len(defined),
            "n_claims": sum(
                cells[k]["n_claims"] for k in CELL_ORDER if k in cells
            ),
        })
    # Undefined sorts last rather than as zero, as in `report.leaderboard`.
    rows.sort(key=lambda r: (r["combined_macro"] is None, -(r["combined_macro"] or 0.0)))
    return rows


def best_config_per_cell(run_dicts: list[dict], *, rate_key: str = "claim_rate_macro") -> dict[str, dict]:
    """The top configuration for each (field, dimension) cell on its own.

    Shown next to `rank_configs` because the per-cell winners can differ from the
    combined ranking (groundedness tends to go to the model that abstains most).
    """
    grouped = group_rollup_by_config(run_dicts)
    out: dict[str, dict] = {}
    for cell_key in CELL_ORDER:
        best_label, best_rate = None, None
        for label, cells in grouped.items():
            cell = cells.get(cell_key)
            if cell is None or cell[f"{rate_key}_undefined"]:
                continue
            if best_rate is None or cell[rate_key] > best_rate:
                best_label, best_rate = label, cell[rate_key]
        out[cell_key] = {
            "config": best_label, "rate": best_rate, "undefined": best_rate is None,
        }
    return out


def rank_agreement_with_f1(run_dicts: list[dict], *, scored_dir=None) -> list[dict]:
    """Each configuration's judge rank next to its macro-F1 rank, best F1 first.

    If two different measures agree on the ranking, the result is more convincing.
    Both sides use `analysis.dataset.config_label`, and the join lives here so it can
    be tested. `macro_f1` comes from the frozen `configs.json`. A config missing from
    one side gets `None` there instead of being dropped.
    """
    from grant_eval.analysis import report as reportmod

    configs = reportmod._load_configs() if scored_dir is None else reportmod._load_configs(scored_dir)
    from grant_eval.analysis.dataset import config_label

    f1_by_label = {
        config_label(c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"]):
            c["macro_f1"]
        for c in configs
    }
    f1_ordered = sorted(
        f1_by_label, key=lambda k: (f1_by_label[k] is None, -(f1_by_label[k] or 0.0))
    )
    f1_rank = {label: i for i, label in enumerate(f1_ordered, start=1)}

    judge_rows = rank_configs(run_dicts)
    judge_rank = {r["config"]: i for i, r in enumerate(judge_rows, start=1)}
    judge_combined = {r["config"]: r["combined_macro"] for r in judge_rows}

    rows = [
        {
            "config": label,
            "f1_rank": f1_rank.get(label),
            "judge_rank": judge_rank.get(label),
            "macro_f1": f1_by_label.get(label),
            "judge_combined_macro": judge_combined.get(label),
        }
        for label in sorted(set(f1_by_label) | set(judge_rank),
                            key=lambda k: (f1_rank.get(k) is None, f1_rank.get(k) or 0))
    ]
    return rows


def claim_type_breakdown(run_dicts: list[dict]) -> dict[str, dict[str, dict]]:
    """Supported/unsupported counts per (claim_type, dimension) over all stored claims.

    Shows which kind of fact (deadline, amount, eligibility, ...) is missed or
    unsupported. Counted from the saved `records[].claims[]`; only `ok` records have
    claims.
    """
    out: dict[str, dict[str, dict]] = {
        ct: {d: {"n_claims": 0, "n_supported": 0} for d in judgemod.DIMENSIONS}
        for ct in CLAIM_TYPES
    }
    for run in run_dicts:
        for cell in run["fields"].values():
            dimension = cell["dimension"]
            for rec in cell["records"]:
                for claim in (rec.get("claims") or []):
                    bucket = out[claim["claim_type"]][dimension]
                    bucket["n_claims"] += 1
                    if claim["supported"]:
                        bucket["n_supported"] += 1
    for by_dim in out.values():
        for bucket in by_dim.values():
            n = bucket["n_claims"]
            bucket["support_rate"] = (bucket["n_supported"] / n) if n else None
            bucket["support_rate_undefined"] = n == 0
    return out


#: An unsupported groundedness claim is worse than a completeness gap. Lower sorts first.
_DIMENSION_SEVERITY = {"groundedness": 0, "completeness": 1}


def worst_unsupported_claims(claims: list[JudgedClaim], *, n: int = 10) -> list[JudgedClaim]:
    """The first `n` unsupported claims as worked examples, groundedness first.

    Ties are broken by `(field, opportunity_id, claim_index)` so the order is stable.
    """
    unsupported = [c for c in claims if c.judge_verdict is False]
    unsupported.sort(
        key=lambda c: (_DIMENSION_SEVERITY[c.dimension], c.field, c.opportunity_id, c.claim_index)
    )
    return unsupported[:n]


def make_fake_reply(rng: random.Random, *, support_rate: float = 0.8, max_claims: int = 4) -> str:
    """One fake judge reply with 1..`max_claims` claims, each supported with `support_rate`."""
    n = rng.randint(1, max_claims)
    claims = [
        {
            "text": f"synthetic claim {i}",
            "claim_type": rng.choice(CLAIM_TYPES),
            "supported": rng.random() < support_rate,
        }
        for i in range(n)
    ]
    return json.dumps({"claims": claims, "justification": "synthetic fixture reply"})


def simulate_human_codes(
    selection: SpotcheckSelection, *, rng: random.Random,
    disagreement_rate: float = 0.12, not_a_claim_rate: float = 0.06,
) -> dict[str, str]:
    """Simulated `human_code` per claim.

    Each claim is `"not_a_claim"` with `not_a_claim_rate`, the opposite of the judge
    with `disagreement_rate`, and otherwise agrees with the judge.
    """
    codes: dict[str, str] = {}
    for c in selection.claims:
        u = rng.random()
        if u < not_a_claim_rate:
            codes[c.item_id] = "not_a_claim"
        elif u < not_a_claim_rate + disagreement_rate:
            codes[c.item_id] = "unsupported" if c.judge_verdict else "supported"
        else:
            codes[c.item_id] = "supported" if c.judge_verdict else "unsupported"
    return codes


_SYNTHETIC_NOTE = "SYNTHETIC -- simulated for the fake-backend demo, not a real human coder"


def apply_simulated_human_codes(csv_path: Path, codes: dict[str, str]) -> None:
    """Fill `human_code`/`human_notes` in an exported spot-check CSV from `codes`.

    Simulates a returned coding file. Rows missing from `codes` stay uncoded.
    """
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        code = codes.get(row["item_id"])
        if code is None:
            continue
        if code not in _VALID_HUMAN_CODES:
            raise ValueError(f"{code!r} is not a valid human_code (expected one of {_VALID_HUMAN_CODES})")
        row["human_code"] = code
        row["human_notes"] = _SYNTHETIC_NOTE
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPOTCHECK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
