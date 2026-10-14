"""Load the ground-truth CSVs from `data/ground-truth/`.

The folder also holds a blank template, a per-funder summary table and an old
cp1252 export of the British Ecological Society file. The old export has a valid
header but predates two data-entry fixes and would double-count that funder. So files
pass three checks in order, and anything rejected is recorded with a reason:

1. Filename must be `<funder-slug>_<YYYY-MM-DD>.csv` (this is what rejects the old
   BES export, which check 2 would accept).
2. Header must have every column in `REQUIRED_COLUMNS`, with at least one data row.
3. `opportunity_id` must be unique across all loaded files; a duplicate raises.

Files are decoded with utf-8, then utf-8-sig, then cp1252, and the decoder used is
stored on the result.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from grant_eval.schema import GroundTruthOpportunity
from grant_eval.scoring.identity import MatchKey, build_key_index, normalise_funder

#: The per-funder ground-truth filename convention: `<funder-slug>_<snapshot-date>.csv`.
FILENAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*_\d{4}-\d{2}-\d{2}\.csv$")

#: Columns a ground-truth file must have. A subset of the full header, so optional
#: bookkeeping columns can come and go.
REQUIRED_COLUMNS = frozenset(
    {
        "opportunity_id",
        "funder",
        "title",
        "url",
        "status",
        "deadlines",
        "in_scope",
        "edge_case_tags",
        "uncertain_fields",
    }
)

#: Decoders tried in order. cp1252 is last because it decodes almost anything and
#: would otherwise hide a real utf-8 file.
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252")

DEFAULT_GROUND_TRUTH_DIR = Path("data/ground-truth")


def read_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read one CSV, trying each decoder in turn. Returns `(rows, encoding_used)`."""
    last: UnicodeDecodeError | None = None
    for encoding in _ENCODINGS:
        try:
            with open(path, newline="", encoding=encoding) as fh:
                return list(csv.DictReader(fh)), encoding
        except UnicodeDecodeError as exc:
            last = exc
    raise UnicodeDecodeError(  # pragma: no cover - cp1252 decodes everything
        last.encoding, last.object, last.start, last.end,
        f"{path}: not decodable as any of {_ENCODINGS}",
    )


@dataclass(frozen=True)
class FunderGroundTruth:
    """Every ground-truth row for one funder (in file order), indexed by match key."""

    funder: str
    rows: list[GroundTruthOpportunity]
    by_key: dict[MatchKey, GroundTruthOpportunity]
    shared_urls: frozenset[str]
    source: Path | None = None

    @property
    def in_scope(self) -> list[GroundTruthOpportunity]:
        """Rows with `in_scope = yes`; the recall denominator."""
        return [r for r in self.rows if r.in_scope]

    @property
    def distractors(self) -> list[GroundTruthOpportunity]:
        """Rows with `in_scope = no`. Matching one is a false positive of its own kind."""
        return [r for r in self.rows if not r.in_scope]


@dataclass
class GroundTruth:
    """The whole labelled corpus, indexed by normalised funder name."""

    funders: dict[str, FunderGroundTruth] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)
    #: `(path, reason)` for every `*.csv` in the directory that was NOT loaded.
    rejected: list[tuple[Path, str]] = field(default_factory=list)
    #: `path -> decoder that worked`, for provenance.
    encodings: dict[Path, str] = field(default_factory=dict)

    def for_funder(self, funder: str) -> FunderGroundTruth:
        """Look up one funder's rows by name (normalised). Raises if it is unlabelled."""
        key = normalise_funder(funder)
        try:
            return self.funders[key]
        except KeyError:
            known = ", ".join(sorted(f.funder for f in self.funders.values()))
            raise KeyError(
                f"no ground truth for funder {funder!r}. Labelled funders: {known}"
            ) from None

    @property
    def rows(self) -> list[GroundTruthOpportunity]:
        return [r for f in self.funders.values() for r in f.rows]

    @property
    def in_scope(self) -> list[GroundTruthOpportunity]:
        return [r for r in self.rows if r.in_scope]


def _reject_reason(path: Path, rows: list[dict[str, str]] | None) -> str | None:
    """Why this file is not ground truth, or None if it is."""
    if not FILENAME_RE.match(path.name):
        return "filename is not <funder-slug>_<YYYY-MM-DD>.csv"
    if rows is None:
        return "unreadable"
    if not rows:
        return "no data rows"
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        return f"header missing {sorted(missing)}"
    return None


def load_ground_truth(directory: str | Path = DEFAULT_GROUND_TRUTH_DIR) -> GroundTruth:
    """Load every ground-truth CSV directly under `directory`.

    Subfolders are not searched: `deferred-batch3/` holds two deferred funders that
    would otherwise enter the denominator. Raises `ValueError` on a duplicate
    `opportunity_id` or a match-key collision.
    """
    directory = Path(directory)
    gt = GroundTruth()
    by_funder: dict[str, list[GroundTruthOpportunity]] = defaultdict(list)
    funder_display: dict[str, str] = {}
    funder_source: dict[str, Path] = {}
    seen_ids: dict[str, Path] = {}

    for path in sorted(directory.glob("*.csv")):
        raw: list[dict[str, str]] | None = None
        encoding: str | None = None
        if FILENAME_RE.match(path.name):
            # Only decode files whose name passes, so the rejects are never parsed.
            raw, encoding = read_rows(path)
        reason = _reject_reason(path, raw)
        if reason is not None:
            gt.rejected.append((path, reason))
            continue

        gt.files.append(path)
        gt.encodings[path] = encoding  # type: ignore[assignment]
        for row in raw:  # type: ignore[union-attr]
            record = GroundTruthOpportunity.from_csv_row(row)
            if record.opportunity_id in seen_ids:
                raise ValueError(
                    f"duplicate opportunity_id {record.opportunity_id!r} in {path.name} "
                    f"and {seen_ids[record.opportunity_id].name}. `opportunity_id` is a "
                    f"stable key that is never reused (LABELLING-PROTOCOL.md), so this "
                    f"is a duplicated export, not two opportunities."
                )
            seen_ids[record.opportunity_id] = path
            key = normalise_funder(record.funder)
            by_funder[key].append(record)
            funder_display.setdefault(key, record.funder.strip())
            funder_source.setdefault(key, path)

    for key, rows in by_funder.items():
        display = funder_display[key]
        index, shared = build_key_index(rows, display)
        gt.funders[key] = FunderGroundTruth(
            funder=display,
            rows=rows,
            by_key=index,  # type: ignore[arg-type]
            shared_urls=shared,
            source=funder_source[key],
        )
    return gt
