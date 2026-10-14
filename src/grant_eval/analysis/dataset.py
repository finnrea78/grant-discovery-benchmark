"""Loads the frozen scored grid (`eval/scored/grid/`) as one `RunRecord` per run.

Nothing is recomputed: all metrics come from `score.py` via `freeze_scores.py`.
Undefined values (precision and hallucination rate for runs that extracted nothing)
stay `None` rather than `0.0`, and paired samples list any funder they drop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_SCORED_DIR",
    "AnalysisDataset",
    "PairedSample",
    "RunRecord",
    "config_label",
    "load_dataset",
]

#: Root of the frozen scored artefact: `<dir>/{runs.jsonl,manifest.json}`.
DEFAULT_SCORED_DIR = Path("eval/scored/grid")


def config_label(model: str, prompt: str, temperature: float) -> str:
    """Readable config key used for grouping in every table.

    `agent_config` is left out because it has the single value `standard` across the
    whole grid.
    """
    return f"{model} | {prompt} | t{temperature}"


@dataclass(frozen=True)
class RunRecord:
    """One scored run as written by `freeze_scores.py`.

    `precision` and `hallucination_rate` are `None` for runs that extracted nothing.
    `n_fabricated` is `None` when the funder's mirror could not be resolved.
    """

    funder: str
    model: str
    prompt: str
    temperature: float
    replicate: int
    config_label: str
    config_hash: str
    f1: float
    precision: float | None
    recall: float
    tp: int
    fp: int
    fn: int
    n_extracted: int
    n_in_scope: int
    hallucination_rate: float | None
    n_fabricated: int | None
    fp_by_reason: dict
    true_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]
    cost_usd: float | None
    latency_s: float | None
    run_failed: bool
    abstained: bool
    split: str


@dataclass(frozen=True)
class PairedSample:
    """Two configs' per-funder means, aligned by funder for a paired test.

    `a[i]` and `b[i]` belong to `funders[i]`. `excluded` lists funders dropped for a
    missing value on either side, with the reason.
    """

    funders: tuple[str, ...]
    a: tuple[float, ...]
    b: tuple[float, ...]
    excluded: tuple[tuple[str, str], ...] = ()

    @property
    def differences(self) -> tuple[float, ...]:
        """`a - b` per funder, in the order `funders` names them."""
        return tuple(x - y for x, y in zip(self.a, self.b))

    def __len__(self) -> int:
        return len(self.funders)


def _record(payload: dict) -> RunRecord:
    """Convert one `runs.jsonl` line into a `RunRecord`.

    The frozen file stores `precision` as 0.0 for empty extractions, so `None` is
    rebuilt from the `precision_undefined` flag.
    """
    detection = payload["detection"]
    telemetry = payload["telemetry"]
    model = payload["model"]
    prompt = payload["prompt_strategy"]
    temperature = payload["temperature"]
    return RunRecord(
        funder=payload["funder"],
        model=model,
        prompt=prompt,
        temperature=temperature,
        replicate=payload["replicate"],
        config_label=config_label(model, prompt, temperature),
        config_hash=payload["config_hash"],
        f1=detection["f1"],
        precision=None if detection["precision_undefined"] else detection["precision"],
        recall=detection["recall"],
        tp=detection["tp"],
        fp=detection["fp"],
        fn=detection["fn"],
        n_extracted=detection["n_extracted"],
        n_in_scope=detection["n_in_scope"],
        hallucination_rate=detection["hallucination_rate"],
        n_fabricated=detection["n_fabricated"],
        fp_by_reason=detection["fp_by_reason"],
        true_positive_ids=tuple(detection["true_positives"]),
        false_negative_ids=tuple(detection["false_negatives"]),
        cost_usd=telemetry["cost_usd"],
        latency_s=telemetry["latency_s"],
        run_failed=telemetry["run_failed"],
        abstained=telemetry["abstained"],
        split=telemetry["split"],
    )


@dataclass
class AnalysisDataset:
    """The frozen grid as `RunRecord`s, plus its manifest.

    `funder_means` and `paired` take an optional `runs=` subset, e.g. to drop failed runs.
    """

    runs: tuple[RunRecord, ...]
    manifest: dict

    def funders(self) -> tuple[str, ...]:
        return tuple(sorted({r.funder for r in self.runs}))

    def config_labels(self) -> tuple[str, ...]:
        return tuple(sorted({r.config_label for r in self.runs}))

    def splits(self) -> tuple[str, ...]:
        """Distinct `split` values in the runs, sorted. Normally just one.

        Returned as a tuple because the loader does not enforce a single split.
        """
        return tuple(sorted({r.split for r in self.runs}))

    def cell(self, config_label: str, funder: str) -> tuple[RunRecord, ...]:
        """The replicates for one (config, funder) cell, sorted by replicate."""
        matches = [
            r for r in self.runs if r.config_label == config_label and r.funder == funder
        ]
        return tuple(sorted(matches, key=lambda r: r.replicate))

    def funder_means(
        self, value: str = "f1", runs: tuple[RunRecord, ...] | None = None
    ) -> dict[str, dict[str, float]]:
        """`{config_label: {funder: mean(value over that cell's replicates)}}`.

        `None` values are skipped, so a cell with no defined replicate has no entry.
        """
        if runs is None:
            runs = self.runs
        grouped: dict[str, dict[str, list[float]]] = {}
        for r in runs:
            got = getattr(r, value)
            if got is None:
                continue
            grouped.setdefault(r.config_label, {}).setdefault(r.funder, []).append(got)
        return {
            config: {funder: sum(vals) / len(vals) for funder, vals in by_funder.items()}
            for config, by_funder in grouped.items()
        }

    def paired(
        self,
        config_a: str,
        config_b: str,
        value: str = "f1",
        runs: tuple[RunRecord, ...] | None = None,
    ) -> PairedSample:
        """Per-funder means for `config_a` and `config_b`, aligned for a paired test.

        Funders with no value on either side are listed in `excluded`.
        """
        means = self.funder_means(value=value, runs=runs)
        means_a = means.get(config_a, {})
        means_b = means.get(config_b, {})
        funders: list[str] = []
        a: list[float] = []
        b: list[float] = []
        excluded: list[tuple[str, str]] = []
        for funder in self.funders():
            if funder not in means_a:
                excluded.append((funder, f"undefined {value} for config {config_a}"))
                continue
            if funder not in means_b:
                excluded.append((funder, f"undefined {value} for config {config_b}"))
                continue
            funders.append(funder)
            a.append(means_a[funder])
            b.append(means_b[funder])
        return PairedSample(
            funders=tuple(funders), a=tuple(a), b=tuple(b), excluded=tuple(excluded)
        )


def load_dataset(scored_dir: Path = DEFAULT_SCORED_DIR) -> AnalysisDataset:
    """Reads `<scored_dir>/{runs.jsonl,manifest.json}` into an `AnalysisDataset`.

    Raises if the runs file is missing or a record has no `telemetry` block (an old
    freeze without the failure and cost fields).
    """
    scored_dir = Path(scored_dir)
    runs_path = scored_dir / "runs.jsonl"
    manifest_path = scored_dir / "manifest.json"
    if not runs_path.exists():
        raise FileNotFoundError(
            f"No frozen artefact at {runs_path}. Run `freeze_scores` "
            "(grant_eval.scoring.freeze_scores) to produce one."
        )
    records = []
    with runs_path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "telemetry" not in payload:
                raise ValueError(
                    f"{runs_path}:{lineno} has no telemetry block — this is a stale "
                    "freeze from before telemetry was added to score.to_dict; "
                    "re-run freeze_scores against a current scorer."
                )
            records.append(_record(payload))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return AnalysisDataset(runs=tuple(records), manifest=manifest)
