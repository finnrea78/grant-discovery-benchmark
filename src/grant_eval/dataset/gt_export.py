"""Write the review workbook's sheets back to the per-funder ground-truth CSVs.

The inverse of :mod:`grant_eval.dataset.gt_workbook`. Labelling happens in the
workbook, but the scorer reads the CSVs, so this brings them back in line.
Values round-trip verbatim; amounts come back as plain integers, Excel dates as
ISO ``YYYY-MM-DD`` text and empty cells as "". A row in the CSV but missing from
the sheet aborts the export unless ``--allow-row-loss`` is given, because losing an
in-scope row would quietly change every recall figure.

Usage::

    uv run gt_export --dry-run                    # report the diff, write nothing
    uv run gt_export                              # write the CSVs (backs each up first)
    uv run gt_export --workbook "data/ground-truth/my-workbook.xlsx"
    uv run gt_export --only heritage-fund         # one funder
    uv run gt_export --allow-row-loss             # accept deliberate row deletions
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import shutil
from pathlib import Path

from openpyxl import load_workbook

from grant_eval.dataset.gt_workbook import NUMERIC, gt_csvs, read_rows, sheet_name

DATE_COLS = {"snapshot_date", "verified_date", "deadlines", "next_open_date"}


def cell_text(value, column: str) -> str:
    """One sheet cell as the CSV should store it."""
    if value is None:
        return ""
    if isinstance(value, _dt.datetime):
        # Excel keeps a midnight time on a bare date; the protocol wants the date.
        if value.time() == _dt.time(0, 0):
            return value.date().isoformat()
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        # Amounts are whole pounds; openpyxl hands back 10000.0 for 10000.
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def sheet_table(ws) -> tuple[list[str], list[dict]]:
    """A sheet as (header, rows). Trailing unnamed columns are padding, not data."""
    it = ws.iter_rows(values_only=True)
    raw_header = next(it, None) or ()
    header = [str(h).strip() for h in raw_header if h is not None and str(h).strip()]
    width = len(header)

    rows: list[dict] = []
    for raw in it:
        cells = list(raw)[:width]
        if not any(c is not None and str(c).strip() for c in cells):
            continue  # blank spacer row
        cells += [None] * (width - len(cells))
        rows.append({col: cell_text(v, col) for col, v in zip(header, cells)})
    return header, rows


def diff_rows(csv_rows: list[dict], sheet_rows: list[dict], header: list[str]) -> dict:
    """Compare by ``opportunity_id``; report changed cells, added and lost rows."""
    key = "opportunity_id"
    old = {r[key]: r for r in csv_rows if r.get(key)}
    new = {r[key]: r for r in sheet_rows if r.get(key)}

    changes = []
    for oid, row in new.items():
        prev = old.get(oid)
        if prev is None:
            continue
        for col in header:
            before, after = (prev.get(col) or "").strip(), (row.get(col) or "").strip()
            if before != after:
                changes.append((oid, col, before, after))
    return {
        "changed": changes,
        "added": [oid for oid in new if oid not in old],
        "lost": [oid for oid in old if oid not in new],
    }


def backup(path: Path) -> Path:
    """``x.csv`` -> ``x.csv.bak-1``. Never overwrites an existing backup."""
    n = 1
    while (dest := path.with_suffix(path.suffix + f".bak-{n}")).exists():
        n += 1
    shutil.copy2(path, dest)
    return dest


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in header})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ground-truth", default="data/ground-truth")
    p.add_argument("--workbook", default=None, help="defaults to the newest .xlsx in --ground-truth")
    p.add_argument("--only", action="append", default=[], help="sheet name; repeatable")
    p.add_argument("--dry-run", action="store_true", help="report the diff, write nothing")
    p.add_argument("--allow-row-loss", action="store_true",
                   help="accept rows present in a CSV but absent from its sheet")
    p.add_argument("--quiet-cells", action="store_true", help="counts only, no per-cell diff")
    args = p.parse_args()

    gt_root = Path(args.ground_truth).resolve()
    if args.workbook:
        wb_path = Path(args.workbook).resolve()
    else:
        books = [b for b in gt_root.glob("*.xlsx") if ".bak-" not in b.name]
        if not books:
            print(f"error: no workbook under {gt_root}")
            return 1
        wb_path = max(books, key=lambda b: b.stat().st_mtime)

    if not wb_path.exists():
        print(f"error: workbook not found: {wb_path}")
        return 1

    print(f"workbook: {wb_path.name}")
    wb = load_workbook(wb_path, data_only=True)
    by_slug = {sheet_name(p): p for p in gt_csvs(gt_root)}

    planned: list[tuple[Path, list[str], list[dict], dict]] = []
    orphan_sheets, total_lost = [], 0

    for name in wb.sheetnames:
        if name == "Overview":
            continue
        if args.only and name not in args.only:
            continue
        target = by_slug.get(name)
        if target is None:
            orphan_sheets.append(name)
            continue

        header, rows = sheet_table(wb[name])
        csv_rows, csv_header = read_rows(target)
        if header != csv_header:
            # A column appended to the right of the CSV's own columns is a schema
            # extension and is carried through. A reorder, rename or dropped column
            # is treated as damage.
            if header[: len(csv_header)] == csv_header:
                for col in header[len(csv_header):]:
                    print(f"  + {name}: new column {col!r}")
            else:
                print(f"  ! {name}: header differs from {target.name} beyond an append")
                print(f"      sheet: {header}")
                print(f"      csv  : {csv_header}")
                return 1

        d = diff_rows(csv_rows, rows, header)
        total_lost += len(d["lost"])
        planned.append((target, header, rows, d))

    for name in orphan_sheets:
        print(f"  ! sheet {name!r} has no matching CSV — skipped")

    print()
    touched = 0
    for target, _header, rows, d in planned:
        if not (d["changed"] or d["added"] or d["lost"]):
            continue
        touched += 1
        print(f"  {target.name}  ({len(rows)} rows)")
        for oid in d["added"]:
            print(f"      + row {oid}")
        for oid in d["lost"]:
            print(f"      - row {oid}   <-- IN CSV, NOT IN SHEET")
        if d["changed"] and not args.quiet_cells:
            for oid, col, before, after in d["changed"]:
                print(f"      ~ {oid[:44]:44s} {col:20s} {before[:38]!r} -> {after[:38]!r}")
        elif d["changed"]:
            print(f"      ~ {len(d['changed'])} cell(s) changed")

    if not touched:
        print("  nothing to write — every sheet already matches its CSV")
        return 0

    if total_lost and not args.allow_row_loss:
        print(f"\nerror: {total_lost} row(s) exist in a CSV but not in the workbook.")
        print("Restore them in the workbook, or re-run with --allow-row-loss if the")
        print("deletion was deliberate. Refusing to drop labelled rows silently.")
        return 1

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    print()
    for target, header, rows, d in planned:
        if not (d["changed"] or d["added"] or d["lost"]):
            continue
        b = backup(target)
        write_csv(target, header, rows)
        print(f"  wrote {target.name}   (backup: {b.name})")

    print(f"\n{touched} CSV(s) updated from {wb_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
