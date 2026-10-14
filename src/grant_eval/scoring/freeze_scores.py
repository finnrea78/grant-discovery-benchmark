"""Score the grid once and save the results with a manifest of what produced them.

Writes `runs.jsonl` (one scored run per line), `configs.json` (one aggregate per
config) and `manifest.json`. The manifest hashes the inputs, the scorer source, the
ground-truth files and each funder's mirror, and lists any result that could not be
scored, so a later change to any of them can be detected. Output is deterministic
apart from `created_utc`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from grant_eval.scoring import aggregate as aggregatemod
from grant_eval.scoring import score as scoremod
from grant_eval.scoring.groundtruth import DEFAULT_GROUND_TRUTH_DIR, load_ground_truth
from grant_eval.scoring.mirror import DEFAULT_REGISTRY, DEFAULT_SNAPSHOT_DIR, load_mirror

#: Where the frozen scores land unless told otherwise.
DEFAULT_OUT_DIR = Path("eval/scored/grid")

#: Modules whose source is hashed into `manifest.scorer.source_hash`.
SCORER_MODULES = (
    "score.py",
    "aggregate.py",
    "identity.py",
    "groundtruth.py",
    "mirror.py",
)


@dataclass(frozen=True)
class FreezeReport:
    """What one freeze did, for the caller and the CLI to report."""

    out_dir: Path
    n_scored: int
    n_skipped: int
    n_configs: int
    manifest: dict
    #: `(filename, reason)` for every result that could not be scored.
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _content_hash(pairs) -> str:
    """Hash over sorted `(name, sha256)` pairs, so order does not matter."""
    joined = "\n".join(f"{name}  {digest}" for name, digest in sorted(pairs))
    return _sha256_bytes(joined.encode("utf-8"))


def _scorer_fingerprint() -> dict:
    """Hash the scorer modules' source, plus the git SHA if available.

    The git SHA can be missing and ignores uncommitted edits, so `source_hash` is the
    one to rely on.
    """
    here = Path(__file__).parent
    modules = {name: _sha256_file(here / name) for name in SCORER_MODULES}
    return {
        "modules": modules,
        "source_hash": _content_hash(modules.items()),
        "git": _git_description(),
    }


def _git_description() -> dict:
    """`{sha, dirty}` for the working tree, or `{}` where git cannot answer."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"sha": sha, "dirty": bool(status)}


def _ground_truth_fingerprint(gt, directory) -> dict:
    files = {Path(p).name: _sha256_file(p) for p in gt.files}
    return {
        "directory": str(directory),
        "n_files": len(gt.files),
        "n_rows": len(gt.rows),
        "n_in_scope": len(gt.in_scope),
        "n_funders": len(gt.funders),
        "files": files,
        "content_hash": _content_hash(files.items()),
        "rejected": [{"file": Path(p).name, "reason": r} for p, r in gt.rejected],
    }


def _mirror_fingerprint(mirror, registry, snapshots_dir) -> dict:
    if mirror is None:
        return {"resolved": False}
    return {
        "resolved": True,
        "registry": str(registry),
        "snapshots_dir": str(snapshots_dir),
        "snapshots": {
            funder: {
                "slug": index.slug,
                "snapshot_date": index.snapshot_date,
                "content_hash": index.content_hash,
                "n_files": index.n_files,
                "n_urls": len(index.urls),
            }
            for funder, index in sorted(mirror.by_funder.items())
        },
        "missing": [{"slug": slug, "reason": reason} for slug, reason in mirror.missing],
    }


def _expand(results) -> list[Path]:
    """Every result JSON under `results`, searching folders recursively, sorted.

    Unlike `score._expand` this recurses, so the grid root (`grid/<funder>/*.json`) works.
    """
    out: list[Path] = []
    for raw in results:
        path = Path(raw)
        out.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    return out


def _dump(path: Path, payload) -> None:
    """Deterministic JSON: sorted keys, fixed separators, trailing newline."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def freeze(
    results,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    ground_truth_dir: str | Path = DEFAULT_GROUND_TRUTH_DIR,
    registry: str | Path = DEFAULT_REGISTRY,
    snapshots_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
    use_mirror: bool = True,
) -> FreezeReport:
    """Score every result under `results` (files or folders) and write to `out_dir`.

    Results whose funder has no ground truth are listed in `skipped`.
    """
    paths = _expand(results)
    if not paths:
        # An empty output would make every figure blank with no explanation.
        raise ValueError(f"no result files found under {[str(r) for r in results]}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(ground_truth_dir)
    mirror = (
        load_mirror(registry=registry, snapshots_dir=snapshots_dir) if use_mirror else None
    )

    scores = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        try:
            result = scoremod.load_result(path)
            index = mirror.for_funder(result.funder) if mirror is not None else None
            scores.append(scoremod.score_run(result, gt, mirror=index, source_path=path))
        except (KeyError, ValueError) as exc:
            skipped.append((Path(path).name, str(exc)))

    configs = aggregatemod.aggregate_runs(scores) if scores else []

    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in scores:
            handle.write(
                json.dumps(scoremod.to_dict(run), sort_keys=True, default=str) + "\n"
            )
    _dump(out_dir / "configs.json", [aggregatemod.aggregate_to_dict(c) for c in configs])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "roots": [str(r) for r in results],
            "n_files": len(paths),
            "content_hash": _content_hash((p.name, _sha256_file(p)) for p in paths),
        },
        "scorer": _scorer_fingerprint(),
        "ground_truth": _ground_truth_fingerprint(gt, ground_truth_dir),
        "mirror": _mirror_fingerprint(mirror, registry, snapshots_dir),
        "counts": {
            "n_scored": len(scores),
            "n_skipped": len(skipped),
            "n_configs": len(configs),
        },
        "skipped": [{"file": name, "reason": reason} for name, reason in skipped],
    }
    _dump(out_dir / "manifest.json", manifest)

    return FreezeReport(
        out_dir=out_dir,
        n_scored=len(scores),
        n_skipped=len(skipped),
        n_configs=len(configs),
        manifest=manifest,
        skipped=tuple(skipped),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="freeze_scores",
        description=(
            "Score result JSONs once and freeze them, with a manifest fingerprinting "
            "the scorer, the ground truth, the mirror and the inputs, so a later change "
            "to any of them is detectable rather than assumed away."
        ),
    )
    p.add_argument("results", nargs="+", help="ExtractionResult JSON file(s), or directories")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"default: {DEFAULT_OUT_DIR}")
    p.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH_DIR))
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOT_DIR))
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not resolve the hallucination split; it is recorded as unresolved",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = freeze(
            args.results,
            args.out_dir,
            ground_truth_dir=args.ground_truth,
            registry=args.registry,
            snapshots_dir=args.snapshots,
            use_mirror=not args.no_mirror,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"froze {report.n_scored} run(s) across {report.n_configs} config(s)")
    print(f"  -> {report.out_dir}")
    mirror = report.manifest["mirror"]
    if not mirror["resolved"]:
        print("  mirror: NOT resolved — the hallucination split is undefined")
    elif mirror["missing"]:
        for entry in mirror["missing"]:
            print(f"  no mirror for {entry['slug']}: {entry['reason']}", file=sys.stderr)
    if report.skipped:
        print(f"  {report.n_skipped} result(s) skipped:", file=sys.stderr)
        for name, reason in report.skipped:
            print(f"    {name}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
