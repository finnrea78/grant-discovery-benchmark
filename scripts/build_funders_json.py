#!/usr/bin/env python3
"""Build `data/funders.json`, one row per funder, from the snapshot files.

Combines four sources: the tune/validation split listed in `data/funders.md`, entry URLs
from the snapshot shell scripts, snapshot dates from directory names, and display names
from the ground-truth CSVs. Every disagreement between sources is printed and saved to
`data/funders.build-log.txt`.

`entry_url` is taken from `meta/wget.log` rather than `manifest.json`, because the
manifest sometimes rebuilds the URL from the filename when its log lookup fails (every
page of `fulbright` and `heritage-fund`). `leverhulme`, `henry-moore` and `rsc` were
captured by hand and are not in any script, so the script cross-check is skipped for
them. When wget followed a redirect, the first URL in wget.log is the one the mirror
was actually crawled from.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FUNDERS_MD = REPO_ROOT / "data" / "funders.md"
SNAPSHOTS_ROOT = REPO_ROOT / "data" / "snapshots"
GROUND_TRUTH_ROOT = REPO_ROOT / "data" / "ground-truth"
OUT_PATH = REPO_ROOT / "data" / "funders.json"
BUILD_LOG_PATH = REPO_ROOT / "data" / "funders.build-log.txt"

RUN_SCRIPTS = [
    REPO_ROOT / "ralph" / "run_snapshots.sh",
    REPO_ROOT / "ralph" / "run_snapshots_batch2.sh",
    REPO_ROOT / "ralph" / "run_snapshots_batch3.sh",
]

TARGET_ROW = re.compile(r'^\s*"([^"|]+)\|([^"|]+)\|([^"]*)"\s*$')

# wget's log line format: `URL:<fetched> [size] -> "<saved>"`. Kept separate from the
# same pattern in `dataset.snapshot` so this check does not share code with the
# manifest path it is checking.
WGET_LOG_LINE = re.compile(r'URL:(\S+)\s+\[[^\]]*\]\s+->\s+"([^"]+)"')


def parse_script_entry_urls() -> dict[str, str]:
    """slug -> entry URL, from the `"slug|url|flags"` TARGETS rows of each script.

    Only used to cross-check `meta/wget.log`."""
    urls: dict[str, str] = {}
    for script in RUN_SCRIPTS:
        in_targets = False
        for line in script.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*TARGETS=\(", line):
                in_targets = True
                continue
            if in_targets and line.strip() == ")":
                in_targets = False
                continue
            if not in_targets:
                continue
            m = TARGET_ROW.match(line)
            if m:
                slug, url, _flags = m.groups()
                urls[slug] = url
    return urls


def parse_split_lists() -> dict[str, str]:
    """slug -> "tune"/"validation", from the Tune (16) and Validation (9) lists in
    `data/funders.md`."""
    text = FUNDERS_MD.read_text(encoding="utf-8")

    def slugs_after(label: str) -> list[str]:
        m = re.search(rf"\*\*{label}[^:]*:\*\*(.*?)\.\n", text, re.S)
        if not m:
            raise SystemExit(f"could not find a {label!r} list in {FUNDERS_MD}")
        body = m.group(1).replace("\n", " ")
        return [s.strip(" `") for s in body.split(",") if s.strip(" `")]

    split: dict[str, str] = {}
    for slug in slugs_after("Tune"):
        split[slug] = "tune"
    for slug in slugs_after("Validation"):
        split[slug] = "validation"
    return split


def find_snapshot_dir(slug: str) -> Path:
    funder_dir = SNAPSHOTS_ROOT / slug
    dated = sorted(p for p in funder_dir.iterdir() if p.is_dir())
    if len(dated) != 1:
        raise SystemExit(
            f"{slug}: expected exactly one dated snapshot dir under {funder_dir}, "
            f"found {[p.name for p in dated]}"
        )
    return dated[0]


def first_crawled_url(slug: str, log_path: Path) -> str:
    """The first URL wget's log says it fetched, which is the URL the mirror was
    crawled from. Exits if the log is missing or unreadable rather than guessing."""
    if not log_path.is_file():
        raise SystemExit(
            f"{slug}: no wget.log at {log_path} — cannot determine the crawled entry "
            f"URL without one. This needs a human decision, not a fallback."
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = WGET_LOG_LINE.search(line)
        if m:
            return m.group(1)
    raise SystemExit(
        f"{slug}: {log_path} has no line matching {WGET_LOG_LINE.pattern!r} — "
        f"cannot determine the crawled entry URL without one."
    )


def modal_display_name(slug: str, snapshot_date: str) -> str:
    csv_path = GROUND_TRUTH_ROOT / f"{slug}_{snapshot_date}.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{slug}: no ground-truth CSV at {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as fh:
        names = [row["funder"] for row in csv.DictReader(fh)]
    if not names:
        raise SystemExit(f"{slug}: {csv_path} has no data rows")
    try:
        return statistics.mode(names)
    except statistics.StatisticsError:
        # Tie: Counter.most_common is stable on first-seen order for equal counts.
        return Counter(names).most_common(1)[0][0]


def build() -> tuple[list[dict], list[str]]:
    """Return (rows, log_lines). The log lists every funder where the script row, the
    recorded `--entry` and the wget.log entry URL disagree."""
    script_urls = parse_script_entry_urls()
    split_by_slug = parse_split_lists()

    rows: list[dict] = []
    log_lines: list[str] = [
        f"# data/funders.json reconciliation log — generated "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        f"scripts/build_funders_json.py",
        "#",
        "# entry_url is the first URL matched in meta/wget.log (the URL the",
        "# mirror was CRAWLED FROM). Lines below note every funder where that differs",
        "# from the run_snapshots*.sh TARGETS row and/or meta/snapshot.json's recorded",
        "# --entry argument, and why.",
        "",
    ]

    for slug, split in sorted(split_by_slug.items()):
        snap_dir = find_snapshot_dir(slug)
        snapshot_date = snap_dir.name

        meta_path = snap_dir / "meta" / "snapshot.json"
        requested_entry = json.loads(meta_path.read_text(encoding="utf-8"))["entry_url"]

        entry_url = first_crawled_url(slug, snap_dir / "meta" / "wget.log")

        script_entry = script_urls.get(slug)
        if script_entry is None:
            msg = (f"{slug}: not in any run_snapshots*.sh TARGETS array (hand-invoked "
                   f"outside the batch scripts); entry_url {entry_url!r} is from "
                   f"meta/wget.log")
            print(f"  {msg}")
            log_lines.append(msg)
        elif script_entry.rstrip("/") != entry_url.rstrip("/"):
            msg = (f"{slug}: TARGETS row {script_entry!r} != crawled entry "
                   f"{entry_url!r} (wget followed a redirect during the crawl); "
                   f"entry_url is the URL actually fetched")
            print(f"  {msg}")
            log_lines.append(msg)

        if requested_entry.rstrip("/") != entry_url.rstrip("/"):
            msg = (f"{slug}: meta/snapshot.json --entry {requested_entry!r} != "
                   f"crawled entry {entry_url!r} (redirect, or the entry that was "
                   f"actually invoked differs from the committed TARGETS row)")
            print(f"  {msg}")
            log_lines.append(msg)

        manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))

        rows.append(
            {
                "slug": slug,
                "display_name": modal_display_name(slug, snapshot_date),
                "split": split,
                "snapshot_date": snapshot_date,
                "entry_url": entry_url,
                "content_hash": manifest["content_hash"],
            }
        )
    return rows, log_lines


def main() -> None:
    rows, log_lines = build()
    OUT_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    BUILD_LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    n_tune = sum(1 for r in rows if r["split"] == "tune")
    n_val = sum(1 for r in rows if r["split"] == "validation")
    print(f"\nwrote {len(rows)} funders to {OUT_PATH} ({n_tune} tune / {n_val} validation)")
    print(f"wrote reconciliation log to {BUILD_LOG_PATH}")


if __name__ == "__main__":
    main()
