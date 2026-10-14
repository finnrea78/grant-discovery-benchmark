"""Tune-to-validation gap for the lock file's configs, and whether the ranking holds.

The tune and validation funder sets do not overlap, so there is nothing to pair on: the
gap is a plain difference of macro means and the ranking is reported as an ordering,
not a significance test. Undefined macro values stay `None` rather than becoming 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import dataset as ds

__all__ = [
    "DEFAULT_LOCK_PATH",
    "DEFAULT_VALIDATION_SCORED_DIR",
    "ConfigGap",
    "GeneralisationResult",
    "macro_by_label",
    "tune_validation_gap",
]

#: The lock file naming the configurations run on the validation split.
DEFAULT_LOCK_PATH = Path("eval/locked-config.json")

#: Where `freeze_scores eval/validation/grid/` writes the held-out artefact.
DEFAULT_VALIDATION_SCORED_DIR = Path("eval/scored/validation")

#: Tolerance for the lock's macro F1 against the tune artefact. The lock stores full
#: precision, so anything larger is a real change.
_LOCK_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ConfigGap:
    """Tune and validation macro figures for one config in the lock file, and the gap.

    `gap` is `validation - tune` (negative means worse on validation), or `None` if
    either side is undefined. Ranks are 1-based within the lock file's configs only.
    """

    label: str
    tune_macro_f1: float | None
    tune_macro_precision: float | None
    tune_macro_recall: float | None
    tune_n_funders: int
    validation_macro_f1: float | None
    validation_macro_precision: float | None
    validation_macro_recall: float | None
    validation_n_funders: int
    gap: float | None
    tune_rank: int | None
    validation_rank: int | None


@dataclass(frozen=True)
class GeneralisationResult:
    """The full tune-versus-validation comparison, with its source paths.

    `ranking_preserved` is `None` when fewer than two configs can be compared.
    `unlocked_validation_configs` should always be empty; it is kept so a problem shows.
    """

    gaps: tuple[ConfigGap, ...]
    ranking_preserved: bool | None
    tune_scored_dir: str
    validation_scored_dir: str
    lock_path: str
    locked_at: str
    source_hash: str
    unlocked_validation_configs: tuple[str, ...]

    @property
    def mean_gap(self) -> float | None:
        """Mean `gap` over the configurations that have one, or `None` if none do."""
        defined = [g.gap for g in self.gaps if g.gap is not None]
        if not defined:
            return None
        return sum(defined) / len(defined)


def _macro(config: dict, field: str) -> float | None:
    """One macro figure from a `configs.json` entry, using the `<field>_undefined` flag."""
    if config.get(f"{field}_undefined"):
        return None
    return config.get(field)


def macro_by_label(scored_dir: str | Path) -> dict[str, dict]:
    """`{config_label: configs.json entry}` for one scored directory.

    Uses `dataset.config_label`, the same key as the lock file.
    """
    path = Path(scored_dir) / "configs.json"
    configs = json.loads(path.read_text())
    return {
        ds.config_label(
            c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"]
        ): c
        for c in configs
    }


def _check_consistent_with_lock(lock: dict, tune_configs: dict[str, dict]) -> None:
    """Raise if the tune artefact no longer matches the macro F1 the lock recorded."""
    for entry in lock["configs"]:
        label = entry["label"]
        if label not in tune_configs:
            raise KeyError(
                f"locked config {label!r} is not in the tune artefact at "
                f"{lock.get('tune_scored_dir')!r}; the artefact the lock was written "
                "against has been replaced"
            )
        recorded = entry.get("macro_f1")
        current = _macro(tune_configs[label], "macro_f1")
        if recorded is None or current is None:
            continue
        if abs(recorded - current) > _LOCK_TOLERANCE:
            raise ValueError(
                f"locked config {label!r} recorded tune macro F1 {recorded!r} but the "
                f"tune artefact now reports {current!r}; the tune evidence moved under "
                "the lock and the out-of-sample claim no longer rests on it"
            )


def _source_hash(scored_dir: Path) -> str:
    """The scorer `source_hash` a frozen artefact was produced under."""
    return json.loads((scored_dir / "manifest.json").read_text())["scorer"]["source_hash"]


def _check_same_scorer(lock: dict, tune_dir: Path, validation_dir: Path) -> None:
    """Raise if tune and validation were scored with different scorer code.

    Otherwise the gap would mix a scorer change in with the split.
    """
    hashes = {
        str(tune_dir): _source_hash(tune_dir),
        str(validation_dir): _source_hash(validation_dir),
    }
    locked = lock.get("tune_source_hash")
    if locked:
        hashes["the lock"] = locked
    distinct = set(hashes.values())
    if len(distinct) > 1:
        detail = "; ".join(f"{where} -> {h[:12]}" for where, h in sorted(hashes.items()))
        raise ValueError(
            "tune and validation were not scored by the same scorer, so the "
            f"tune-to-validation gap would confound the split with a scorer change: {detail}"
        )


def _ranks(values: dict[str, float | None]) -> dict[str, int | None]:
    """1-based ranks, highest first. Ties share a rank; undefined values get `None`."""
    defined = sorted(
        ((label, v) for label, v in values.items() if v is not None),
        key=lambda pair: -pair[1],
    )
    ranks: dict[str, int | None] = {label: None for label in values}
    for i, (label, value) in enumerate(defined, start=1):
        previous = defined[i - 2] if i >= 2 else None
        ranks[label] = ranks[previous[0]] if previous and previous[1] == value else i
    return ranks


def tune_validation_gap(
    validation_scored_dir: str | Path = DEFAULT_VALIDATION_SCORED_DIR,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    tune_scored_dir: str | Path | None = None,
) -> GeneralisationResult:
    """Compare the lock file's configurations on tune and validation.

    `tune_scored_dir` defaults to the directory recorded in the lock. Raises
    `FileNotFoundError` if the lock or an artefact is missing, and `KeyError` if a
    config from the lock is missing from either artefact.
    """
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise FileNotFoundError(
            f"no lock at {lock_path}; RQ6 compares the configurations that were locked "
            "BEFORE validation was dispatched, and without the lock there is no "
            "out-of-sample claim to test"
        )
    lock = json.loads(lock_path.read_text())
    tune_dir = Path(tune_scored_dir or lock.get("tune_scored_dir") or ds.DEFAULT_SCORED_DIR)
    validation_dir = Path(validation_scored_dir)

    tune_configs = macro_by_label(tune_dir)
    validation_configs = macro_by_label(validation_dir)
    _check_consistent_with_lock(lock, tune_configs)
    _check_same_scorer(lock, tune_dir, validation_dir)

    locked_labels = [entry["label"] for entry in lock["configs"]]
    missing = [label for label in locked_labels if label not in validation_configs]
    if missing:
        raise KeyError(
            f"locked config(s) {missing} have no entry in the validation artefact at "
            f"{validation_dir}; the single-shot run did not cover every locked cell"
        )

    tune_f1 = {label: _macro(tune_configs[label], "macro_f1") for label in locked_labels}
    validation_f1 = {
        label: _macro(validation_configs[label], "macro_f1") for label in locked_labels
    }
    tune_ranks = _ranks(tune_f1)
    validation_ranks = _ranks(validation_f1)

    gaps = []
    for label in locked_labels:
        t = tune_configs[label]
        v = validation_configs[label]
        t_f1 = tune_f1[label]
        v_f1 = validation_f1[label]
        gaps.append(
            ConfigGap(
                label=label,
                tune_macro_f1=t_f1,
                tune_macro_precision=_macro(t, "macro_precision"),
                tune_macro_recall=_macro(t, "macro_recall"),
                tune_n_funders=t["n_funders"],
                validation_macro_f1=v_f1,
                validation_macro_precision=_macro(v, "macro_precision"),
                validation_macro_recall=_macro(v, "macro_recall"),
                validation_n_funders=v["n_funders"],
                gap=None if (t_f1 is None or v_f1 is None) else v_f1 - t_f1,
                tune_rank=tune_ranks[label],
                validation_rank=validation_ranks[label],
            )
        )

    comparable = [g for g in gaps if g.tune_rank is not None and g.validation_rank is not None]
    ranking_preserved = (
        None
        if len(comparable) < 2
        else all(g.tune_rank == g.validation_rank for g in comparable)
    )

    return GeneralisationResult(
        gaps=tuple(gaps),
        ranking_preserved=ranking_preserved,
        tune_scored_dir=str(tune_dir),
        validation_scored_dir=str(validation_dir),
        lock_path=str(lock_path),
        locked_at=lock.get("locked_at", ""),
        source_hash=lock.get("tune_source_hash", ""),
        unlocked_validation_configs=tuple(
            sorted(set(validation_configs) - set(locked_labels))
        ),
    )
