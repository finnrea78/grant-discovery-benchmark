"""`uv run lock_config`: write `eval/locked-config.json` for the validation run.

`--split validation` refuses to run without this file. Configs are ranked on
`macro_f1` from the frozen tune `configs.json`, the same figure the leaderboard
prints. It refuses scored data that includes validation runs (choosing on validation
would be circular), refuses to overwrite an existing lock without `--force`, and
refuses labels that match no scored config. `--top-n 2` lets the validation run
check whether the ranking holds, not just the winner.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from grant_eval.analysis import dataset as ds
from grant_eval.dataset.registry import DEFAULT_REGISTRY_PATH, load_registry
from grant_eval.grid.cells import enumerate_cells
from grant_eval.schema import Split

#: Must match `run.LOCKED_CONFIG_PATH`; `tests/test_lock_config.py` checks this.
DEFAULT_LOCK_PATH = Path("eval/locked-config.json")


@dataclass(frozen=True)
class RankedConfig:
    """One scored configuration and its place on the tune leaderboard."""

    label: str
    model: str
    prompt_strategy: str
    temperature: float
    macro_f1: float | None
    macro_precision: float | None
    macro_recall: float | None
    n_funders: int

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "temperature": self.temperature,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "n_funders": self.n_funders,
        }


def _require_scored_dir(scored_dir: Path) -> Path:
    scored_dir = Path(scored_dir)
    if not (scored_dir / "configs.json").is_file():
        raise SystemExit(
            f"lock_config: no frozen scored artefact at {scored_dir} "
            f"(expected {scored_dir / 'configs.json'}). Produce one with "
            "`uv run freeze_scores <results-dir> --out-dir <scored-dir>`."
        )
    return scored_dir


def rank_configs(scored_dir: Path = ds.DEFAULT_SCORED_DIR) -> list[RankedConfig]:
    """Every scored configuration, best macro F1 first.

    Same order as `analysis/report.py:leaderboard`: an undefined `macro_f1` sorts last
    rather than being treated as 0.0.
    """
    scored_dir = _require_scored_dir(scored_dir)
    configs = json.loads((scored_dir / "configs.json").read_text(encoding="utf-8"))

    def sort_key(c: dict) -> tuple[bool, float]:
        f1 = c["macro_f1"]
        return (f1 is None, -(f1 if f1 is not None else 0.0))

    ranked = []
    for c in sorted(configs, key=sort_key):
        cfg = c["config"]
        ranked.append(
            RankedConfig(
                label=ds.config_label(cfg["model"], cfg["prompt_strategy"], cfg["temperature"]),
                model=cfg["model"],
                prompt_strategy=cfg["prompt_strategy"],
                temperature=cfg["temperature"],
                macro_f1=c["macro_f1"],
                macro_precision=c["macro_precision"],
                macro_recall=c["macro_recall"],
                n_funders=c["n_funders"],
            )
        )
    return ranked


def _assert_evidence_is_tune(scored_dir: Path) -> None:
    """Refuse a scored artefact containing anything but tune runs.

    Checks each run in `runs.jsonl`, since a tune artefact with validation runs added
    later would pass a manifest-level check.
    """
    runs_path = Path(scored_dir) / "runs.jsonl"
    if not runs_path.is_file():
        raise SystemExit(
            f"lock_config: no frozen scored artefact at {runs_path}. Produce one with "
            "`uv run freeze_scores <results-dir> --out-dir <scored-dir>`."
        )
    splits: set[str] = set()
    with runs_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            splits.add(json.loads(line).get("telemetry", {}).get("split", "unknown"))
    off_split = sorted(s for s in splits if s != Split.TUNE.value)
    if off_split:
        raise SystemExit(
            f"lock_config refuses to rank on {scored_dir}: it contains runs from "
            f"split(s) {', '.join(off_split)}, not tune alone. Choosing the locked "
            "configuration from validation runs and then spending the validation split "
            "on it is circular — the held-out split exists to prevent exactly that. "
            "Point --scored-dir at the frozen TUNE artefact."
        )


def _chosen(
    ranked: list[RankedConfig], *, top_n: int, labels: list[str] | None
) -> list[RankedConfig]:
    if labels is None:
        return ranked[:top_n]
    by_label = {c.label: c for c in ranked}
    chosen = []
    for label in labels:
        try:
            chosen.append(by_label[label])
        except KeyError:
            raise SystemExit(
                f"lock_config: no scored configuration labelled {label!r}. "
                f"Known labels:\n  " + "\n  ".join(sorted(by_label))
            ) from None
    return chosen


def build_lock(
    *,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
    registry: Path = DEFAULT_REGISTRY_PATH,
    top_n: int = 1,
    labels: list[str] | None = None,
    now: datetime | None = None,
) -> dict:
    """The lock file's contents: cells to run, when it was written, and the evidence.

    `run.py` needs `cells`, `locked_at` and `tune_evidence`; `configs` and
    `tune_source_hash` are kept so the choice can be traced back to the scorer.
    """
    scored_dir = _require_scored_dir(scored_dir)
    _assert_evidence_is_tune(scored_dir)

    chosen = _chosen(rank_configs(scored_dir), top_n=top_n, labels=labels)

    reg = load_registry(registry)
    validation_cells = enumerate_cells(reg, Split.VALIDATION)
    wanted = {(c.model, c.prompt_strategy, c.temperature) for c in chosen}
    cells = [
        cell.cell_id
        for cell in validation_cells
        if (cell.model_slug, cell.prompt_strategy, cell.temperature) in wanted
    ]
    if not cells:
        raise SystemExit(
            "lock_config: the chosen configuration(s) produced no validation cells. "
            "The grid definition has moved since the tune runs were scored, so the "
            "lock would name cells the dispatcher cannot enumerate."
        )

    manifest_path = scored_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    source_hash = manifest.get("scorer", {}).get("source_hash")

    stamp = (now or datetime.now(timezone.utc)).isoformat()
    summary = "; ".join(
        f"{c.label} (macro F1 {c.macro_f1:.3f} over {c.n_funders} funders)"
        if c.macro_f1 is not None
        else f"{c.label} (macro F1 undefined)"
        for c in chosen
    )
    evidence = (
        f"Chosen on the pre-registered primary endpoint (detection F1, preregistration "
        f"§1/§5) from the frozen tune artefact at {scored_dir.as_posix()} "
        f"(scorer source_hash {source_hash}). Ranked best first: {summary}."
    )

    return {
        "cells": cells,
        "locked_at": stamp,
        "tune_evidence": evidence,
        "configs": [c.as_dict() for c in chosen],
        "tune_scored_dir": scored_dir.as_posix(),
        "tune_source_hash": source_hash,
        "n_validation_funders": len({c.split("_", 1)[0] for c in cells}),
    }


def write_lock(lock: dict, path: Path = DEFAULT_LOCK_PATH, *, force: bool = False) -> Path:
    """Write `lock` to `path`, refusing to clobber an existing lock unless `force`."""
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(
            f"lock_config: {path} already exists and will not be overwritten. The "
            "validation split is single-shot: the pre-registration fixes that as 'no "
            "config iteration after seeing validation results', and a lock that can be "
            "rewritten after a disappointing run is how that stops being true. If this "
            "lock was written in error and NO validation run has been dispatched "
            "against it, re-run with --force."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lock_config",
        description="Emit eval/locked-config.json: the configuration(s) the "
        "single-shot validation run is locked to, chosen on tune F1.",
    )
    parser.add_argument("--scored-dir", type=Path, default=ds.DEFAULT_SCORED_DIR,
                        help="frozen TUNE scored artefact (default: eval/scored/grid)")
    parser.add_argument("--out", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--top-n", type=int, default=1,
                        help="lock the best N configs on macro F1 (default: 1). "
                             "2 makes H4's ranking claim testable out of sample")
    parser.add_argument("--config", action="append", default=None, dest="labels",
                        help="lock this exact config label instead of the top-N; "
                             "repeatable")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing lock (see the refusal text)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be locked; write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_n < 1:
        raise SystemExit(f"--top-n: {args.top_n} must be at least 1.")

    lock = build_lock(
        scored_dir=args.scored_dir,
        registry=args.registry,
        top_n=args.top_n,
        labels=args.labels,
    )

    for i, cfg in enumerate(lock["configs"], start=1):
        f1 = "undefined" if cfg["macro_f1"] is None else f"{cfg['macro_f1']:.3f}"
        print(f"  {i}. {cfg['label']}  macro F1 {f1}")
    print(
        f"{len(lock['cells'])} validation cells "
        f"across {lock['n_validation_funders']} funders"
    )

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    path = write_lock(lock, args.out, force=args.force)
    print(f"wrote {path} — commit it before running `grid_run --split validation`")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
