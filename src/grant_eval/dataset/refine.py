"""Amend an existing snapshot: add missed pages, or drop what the study won't read.

Both operations are recorded in ``meta/snapshot.json`` under ``amendments``, so an
amended snapshot is never mistaken for a single uninterrupted crawl.

``topup``
    Fetch a list of URLs the crawl missed into an existing snapshot. This changes
    ``content_hash``, since new pages are new input.

``prune``
    Delete files the agent never reads (images, media, CSS/JS, fonts). The hash only
    covers HTML and documents, so it should stay the same; if it changes, readable
    content was removed.

Usage::

    uv run refine topup --snapshot <dir> --urls urls.txt
    uv run refine prune --snapshot <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from grant_eval.dataset.snapshot import (
    DOC_SUFFIXES,
    PAGE_SUFFIXES,
    USER_AGENT,
    build_manifest,
    write_index,
)


def _load(snapshot_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((snapshot_dir / "meta" / "snapshot.json").read_text())
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    return meta, manifest


def _rewrite(snapshot_dir: Path, meta: dict, amendment: dict) -> str:
    """Rebuild manifest + index after a mutation and record what was done."""
    site_dir = snapshot_dir / "site"
    log_path = snapshot_dir / "meta" / "wget.log"
    manifest = build_manifest(snapshot_dir, site_dir, log_path)

    meta.setdefault("amendments", []).append(amendment)
    meta["counts"] = {
        k: sum(1 for f in manifest["files"] if f["kind"] == k)
        for k in ("page", "document", "asset")
    }
    meta["total_bytes"] = sum(f["bytes"] for f in manifest["files"])

    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (snapshot_dir / "meta" / "snapshot.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_index(snapshot_dir, meta, manifest)
    return manifest["content_hash"]


def topup(args: argparse.Namespace) -> int:
    snapshot_dir = Path(args.snapshot).resolve()
    meta, manifest = _load(snapshot_dir)
    before = manifest["content_hash"]

    urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()]
    if not urls:
        print("error: no URLs to fetch")
        return 1

    log_path = snapshot_dir / "meta" / "wget.log"
    cmd = [
        "wget",
        "--input-file=" + str(Path(args.urls).resolve()),
        # -r implies -x; a plain --input-file run does not, and without it the
        # files land beside the host directory instead of inside it.
        "--force-directories",
        "--adjust-extension",
        "--no-convert-links",
        f"--user-agent={USER_AGENT}",
        f"--wait={args.wait}",
        "--random-wait",
        "--tries=3",
        "--timeout=30",
        "--no-verbose",
        f"--append-output={log_path}",
        f"--directory-prefix={snapshot_dir / 'site'}",
    ]
    print(f"fetching {len(urls)} URL(s)…", flush=True)
    started = datetime.now(timezone.utc)
    proc = subprocess.run(cmd)
    if proc.returncode not in (0, 8):
        print(f"error: wget exited {proc.returncode}")
        return proc.returncode

    after = _rewrite(
        snapshot_dir,
        meta,
        {
            "action": "topup",
            "at_utc": started.isoformat(timespec="seconds"),
            "reason": args.reason,
            "urls": urls,
            "content_hash_before": before,
        },
    )
    print(f"\ncontent_hash {before[:12]}… -> {after[:12]}…  (expected to change: pages added)")
    return 0


def prune(args: argparse.Namespace) -> int:
    snapshot_dir = Path(args.snapshot).resolve()
    meta, manifest = _load(snapshot_dir)
    before = manifest["content_hash"]

    keep = PAGE_SUFFIXES | DOC_SUFFIXES
    site_dir = snapshot_dir / "site"
    removed, freed = 0, 0
    for path in sorted(p for p in site_dir.rglob("*") if p.is_file()):
        # Suffixes can carry a query string (wget bakes it into the filename).
        if path.suffix.lower().split("?")[0] in keep:
            continue
        freed += path.stat().st_size
        path.unlink()
        removed += 1
    for d in sorted((p for p in site_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()

    after = _rewrite(
        snapshot_dir,
        meta,
        {
            "action": "prune",
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "policy": "removed assets (images, media, css/js, fonts); kept HTML + documents",
            "removed_files": removed,
            "freed_bytes": freed,
            "content_hash_before": before,
        },
    )
    print(f"removed {removed} files, freed {freed / 1e6:.1f} MB")
    if after == before:
        print(f"content_hash unchanged ({after[:12]}…) — nothing the agent reads was removed")
    else:
        print(f"WARNING: content_hash changed {before[:12]}… -> {after[:12]}… — readable content was deleted")
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("topup", help="fetch missed URLs into an existing snapshot")
    t.add_argument("--snapshot", required=True)
    t.add_argument("--urls", required=True, help="file with one URL per line")
    t.add_argument("--wait", type=float, default=1.0)
    t.add_argument("--reason", default="coverage check found published pages the crawl missed")
    t.set_defaults(func=topup)

    r = sub.add_parser("prune", help="delete assets the study never reads")
    r.add_argument("--snapshot", required=True)
    r.set_defaults(func=prune)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
