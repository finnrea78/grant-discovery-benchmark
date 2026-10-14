"""Build or extend the review workbook that the ground-truth CSVs are labelled in.

One sheet per funder plus an Overview. Values are copied verbatim from the CSVs so
a sheet round-trips back to its CSV; only the amount columns are stored as numbers.

By default only CSVs with no sheet yet are added, because labelling happens in the
workbook and regenerating a worked-on sheet would lose those edits. ``--rebuild``
regenerates every sheet. Hand-added Overview columns are kept, and the old workbook
is backed up to ``<name>.bak-<n>.xlsx`` before saving.

Usage::

    uv run gt_workbook                            # add sheets for new snapshots
    uv run gt_workbook --workbook data/ground-truth/ground-truth_all-funders.xlsx
    uv run gt_workbook --rebuild                  # regenerate everything (destructive)
    uv run gt_workbook --dry-run                  # say what would change, write nothing
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

NUMERIC = {"amount_min_gbp", "amount_max_gbp"}

# column -> (width, wrap)
WIDTHS: dict[str, tuple[int, bool]] = {
    "opportunity_id": (34, False),
    "funder": (24, False),
    "split": (7, False),
    "snapshot_date": (13, False),
    "title": (40, True),
    "url": (46, False),
    "source_page": (46, False),
    "status": (10, False),
    "deadlines": (18, True),
    "next_open_date": (14, False),
    "amount_min_gbp": (13, False),
    "amount_max_gbp": (13, False),
    "amount_raw": (26, True),
    "eligibility_summary": (60, True),
    "summary": (60, True),
    "discipline_tags": (30, True),
    "in_scope": (9, False),
    "exclusion_reason": (28, True),
    "edge_case_tags": (22, True),
    "uncertain_fields": (20, True),
    "verified_date": (13, False),
    "labeller": (14, False),
    "pass_id": (8, False),
    "notes": (90, True),
}

HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(color="FFFFFF", bold=True)
UNCERTAIN_FILL = PatternFill("solid", fgColor="FFF2CC")  # amber
OUT_OF_SCOPE_FILL = PatternFill("solid", fgColor="F2F2F2")  # grey
OOS_FONT = Font(color="808080")

OVERVIEW_COLS = [
    ("Funder", "funder", 30),
    ("Sheet", "sheet", 26),
    ("Source CSV", "file", 40),
    ("Snapshot", "snapshot", 13),
    ("Rows", "rows", 8),
    ("In scope", "in_scope_yes", 10),
    ("Out of scope", "in_scope_no", 13),
    ("Rows w/ uncertain", "uncertain_rows", 18),
    ("Rows w/ edge case", "edge_case_rows", 18),
    ("Split(s)", "splits", 14),
]
SUMMED = {"rows", "in_scope_yes", "in_scope_no", "uncertain_rows", "edge_case_rows"}

LEGEND = [
    "Amber cells = the field is listed in that row's uncertain_fields.",
    "Grey rows = in_scope is not 'yes'.",
    "Columns, order and cell values are verbatim from the CSVs — nothing added or reformatted "
    "except amount_min_gbp / amount_max_gbp, which are stored as numbers.",
    "Add sheets for new snapshots (existing sheets untouched): uv run gt_workbook",
    "Regenerate every sheet from the CSVs, discarding workbook edits: uv run gt_workbook --rebuild",
]


def split_tags(value: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[;,]", value or "") if t.strip()]


def sheet_name(path: Path) -> str:
    """``leverhulme_2026-08-09.csv`` -> ``leverhulme`` (Excel caps names at 31 chars)."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", path.stem)[:31]


def gt_csvs(gt_root: Path) -> list[Path]:
    """The per-funder CSVs, excluding the template and any exported workbook dumps."""
    return sorted(
        p
        for p in gt_root.glob("*.csv")
        if not p.name.startswith(("_", "ground-truth_all-funders"))
    )


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return rows, header


def add_sheet(wb: Workbook, path: Path) -> dict:
    """Write one funder's CSV to its own sheet, styled for review."""
    rows, header = read_rows(path)
    ws = wb.create_sheet(sheet_name(path))

    ws.append(header)
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30

    col_of = {name: i + 1 for i, name in enumerate(header)}

    for r in rows:
        values = []
        for name in header:
            v = r.get(name) or ""
            if name in NUMERIC and v.strip():
                try:
                    v = int(float(v.replace(",", "")))
                except ValueError:
                    pass
            values.append(v)
        ws.append(values)
        row_idx = ws.max_row

        out_of_scope = (r.get("in_scope") or "").strip().lower() in {"no", "false", "0"}
        for name in header:
            cell = ws.cell(row=row_idx, column=col_of[name])
            _, wrap = WIDTHS.get(name, (24, True))
            cell.alignment = Alignment(vertical="top", wrap_text=wrap)
            if out_of_scope:
                cell.fill = OUT_OF_SCOPE_FILL
                cell.font = OOS_FONT

        # Colour the cells the labeller flagged as uncertain.
        for field in split_tags(r.get("uncertain_fields")):
            if field in col_of:
                ws.cell(row=row_idx, column=col_of[field]).fill = UNCERTAIN_FILL

    for name, idx in col_of.items():
        ws.column_dimensions[get_column_letter(idx)].width = WIDTHS.get(name, (24, True))[0]

    ws.freeze_panes = "B2"
    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(header))}{ws.max_row}"

    return stats_for(ws.title, path, rows)


def stats_for(title: str, path: Path, rows: list[dict]) -> dict:
    yes = sum(1 for r in rows if (r.get("in_scope") or "").strip().lower() == "yes")
    return {
        "sheet": title,
        "file": path.name,
        "funder": rows[0].get("funder", "") if rows else title,
        "snapshot": rows[0].get("snapshot_date", "") if rows else "",
        "rows": len(rows),
        "in_scope_yes": yes,
        "in_scope_no": len(rows) - yes,
        "uncertain_rows": sum(1 for r in rows if split_tags(r.get("uncertain_fields"))),
        "edge_case_rows": sum(1 for r in rows if split_tags(r.get("edge_case_tags"))),
        "splits": ", ".join(
            sorted({(r.get("split") or "").strip() for r in rows if r.get("split")})
        ),
    }


def harvest_extra_columns(wb: Workbook) -> dict[str, list]:
    """Keep whatever a human added to the right of the Overview's generated columns.

    Keyed by sheet name so it survives the rows being re-sorted when new funders
    are added.
    """
    if "Overview" not in wb.sheetnames:
        return {}
    ws = wb["Overview"]
    generated = len(OVERVIEW_COLS)
    if ws.max_column <= generated:
        return {}
    sheet_col = next(i for i, c in enumerate(OVERVIEW_COLS, start=1) if c[1] == "sheet")
    extra: dict[str, list] = {}
    for row in range(2, ws.max_row + 1):
        key = ws.cell(row, sheet_col).value
        if not key or key == "TOTAL":
            continue
        tail = [ws.cell(row, c).value for c in range(generated + 1, ws.max_column + 1)]
        if any(v is not None for v in tail):
            extra[str(key)] = tail
    return extra


def write_overview(wb: Workbook, stats: list[dict], extra: dict[str, list]) -> None:
    if "Overview" in wb.sheetnames:
        del wb["Overview"]
    ws = wb.create_sheet("Overview", 0)

    ws.append([c[0] for c in OVERVIEW_COLS])
    for cell in ws[1]:
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30

    for s in sorted(stats, key=lambda s: s["funder"].lower()):
        ws.append([s[key] for _, key, _ in OVERVIEW_COLS])
        for offset, value in enumerate(extra.get(s["sheet"], []), start=1):
            ws.cell(ws.max_row, len(OVERVIEW_COLS) + offset, value)

    total = ws.max_row + 1
    ws.cell(row=total, column=1, value="TOTAL").font = Font(bold=True)
    for i, (_, key, _) in enumerate(OVERVIEW_COLS, start=1):
        if key in SUMMED:
            ws.cell(row=total, column=i, value=sum(s[key] for s in stats)).font = Font(bold=True)

    for i, (_, _, width) in enumerate(OVERVIEW_COLS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "A2"

    row = total + 2
    for line in LEGEND:
        ws.cell(row=row, column=1, value=line).font = Font(italic=True, color="595959")
        row += 1


def order_sheets(wb: Workbook) -> None:
    """Overview first, then funder sheets alphabetically, so new ones slot in."""
    for i, name in enumerate(["Overview"] + sorted(n for n in wb.sheetnames if n != "Overview")):
        wb.move_sheet(name, offset=i - wb.sheetnames.index(name))


def backup(path: Path) -> Path:
    n = 1
    while (dest := path.with_suffix(f".bak-{n}.xlsx")).exists():
        n += 1
    shutil.copy2(path, dest)
    return dest


def sheet_rows(ws) -> list[dict]:
    """Read a sheet back as CSV-shaped dicts, so its counts reflect any edits made in it."""
    header = [c.value for c in ws[1]]
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        out.append({k: ("" if v is None else str(v)) for k, v in zip(header, row)})
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ground-truth", default="data/ground-truth")
    p.add_argument(
        "--workbook",
        default=None,
        help="workbook to extend (default: <ground-truth>/ground-truth_all-funders.xlsx)",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="regenerate every sheet from the CSVs, discarding edits made in the workbook",
    )
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = p.parse_args()

    gt_root = Path(args.ground_truth).resolve()
    out = (
        Path(args.workbook).resolve()
        if args.workbook
        else gt_root / "ground-truth_all-funders.xlsx"
    )
    csvs = gt_csvs(gt_root)
    if not csvs:
        print(f"error: no ground-truth CSVs under {gt_root}")
        return 1

    fresh = args.rebuild or not out.exists()
    if fresh:
        wb = Workbook()
        wb.remove(wb.active)
        extra: dict[str, list] = {}
        existing: set[str] = set()
        print(f"{'rebuilding' if args.rebuild else 'creating'}: {out.name}")
    else:
        wb = load_workbook(out)
        extra = harvest_extra_columns(wb)
        existing = {n for n in wb.sheetnames if n != "Overview"}
        print(f"extending: {out.name}  ({len(existing)} sheets already present)")
        if extra:
            print(f"  preserving {len(extra)} hand-edited Overview row(s) past the generated columns")

    stats: list[dict] = []
    added: list[str] = []
    for path in csvs:
        name = sheet_name(path)
        if name in existing:
            # Count from the sheet, not the CSV: the sheet is the labelled artefact.
            stats.append(stats_for(name, path, sheet_rows(wb[name])))
            continue
        rows, _ = read_rows(path)
        stats.append(stats_for(name, path, rows) if args.dry_run else add_sheet(wb, path))
        added.append(name)
        print(f"  + {name:<28} {stats[-1]['rows']:>3} rows   <- {path.name}")

    if not added:
        print("  nothing to add — every CSV already has a sheet")

    if args.dry_run:
        print(f"\ndry run: would add {len(added)} sheet(s) to {out}")
        return 0

    write_overview(wb, stats, extra)
    order_sheets(wb)

    if out.exists():
        print(f"  backup: {backup(out).name}")
    wb.save(out)
    print(
        f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB), {len(stats)} funder sheets, "
        f"{sum(s['rows'] for s in stats)} rows "
        f"({len(added)} added, {len(stats) - len(added)} left untouched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
