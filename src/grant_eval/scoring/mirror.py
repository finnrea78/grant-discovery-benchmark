"""Checks whether an extracted URL exists in the frozen snapshot (the hallucination test).

`score.py` counts every extracted record that matches no ground-truth row as an
`unmatched` false positive. That mixes two things: a URL the agent invented, and a real
page the labeller never listed. Only the first is a hallucination, and on the main grid
most unmatched records turned out to be real pages, so the split matters.

Each snapshot has a `manifest.json` listing the URL of every captured file, so the
check is an offline set lookup. Both sides go through `identity.normalise_url` so the
mirror check uses the same rule as ground-truth matching. The whole manifest is used,
not only HTML pages, because citing a real PDF is not a fabrication. A URL that
normalises to "" is never in the mirror.

A funder whose manifest cannot be read is listed in `Mirror.missing` and `for_funder`
returns `None`. An empty index would wrongly call every false positive a fabrication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from grant_eval.scoring.identity import normalise_url

__all__ = [
    "DEFAULT_REGISTRY",
    "DEFAULT_SNAPSHOT_DIR",
    "Mirror",
    "MirrorIndex",
    "load_mirror",
    "normalise_url",
]

#: Funder registry with the slug, display name, split and snapshot date of each mirror.
DEFAULT_REGISTRY = Path("data/funders.json")

#: Root of the frozen snapshots: `<slug>/<snapshot_date>/manifest.json`.
DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")


@dataclass(frozen=True)
class MirrorIndex:
    """The normalised URLs in one funder's frozen snapshot.

    `urls` can be smaller than `n_files` because normalisation is many-to-one.
    `n_without_url` counts manifest entries with no URL.
    """

    funder: str
    slug: str
    snapshot_date: str
    content_hash: str
    urls: frozenset[str]
    n_files: int
    split: str = ""
    n_without_url: int = 0

    def contains(self, url: str) -> bool:
        """True if `url` is in this mirror. An empty normalised URL never is."""
        normalised = normalise_url(url)
        return bool(normalised) and normalised in self.urls


@dataclass(frozen=True)
class Mirror:
    """Every loadable funder mirror, keyed by display name (as used in results and CSVs)."""

    by_funder: dict[str, MirrorIndex]
    #: `(slug, reason)` for every registry funder whose mirror could NOT be loaded.
    missing: tuple[tuple[str, str], ...] = ()

    def for_funder(self, funder: str) -> MirrorIndex | None:
        """This funder's index, or `None` if its mirror could not be loaded."""
        return self.by_funder.get(funder)

    @property
    def funders(self) -> list[str]:
        return sorted(self.by_funder)


def _read_registry(path: Path) -> list[dict]:
    """The funder registry as a list of records (accepts either file shape)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "funders" in payload:
        return list(payload["funders"])
    return list(payload)


def _load_index(record: dict, snapshots_dir: Path) -> tuple[MirrorIndex | None, str]:
    """One funder's index, or `(None, reason)` naming why it could not be built."""
    slug = record.get("slug", "")
    snapshot_date = record.get("snapshot_date") or ""
    if not snapshot_date:
        return None, "no snapshot_date in the registry record"

    manifest_path = snapshots_dir / slug / snapshot_date / "manifest.json"
    if not manifest_path.exists():
        return None, f"no manifest at {manifest_path}"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable manifest at {manifest_path}: {exc}"

    files = manifest.get("files") or []
    urls = {normalise_url(f["url"]) for f in files if f.get("url")}
    urls.discard("")

    return (
        MirrorIndex(
            funder=record.get("display_name", ""),
            slug=slug,
            snapshot_date=snapshot_date,
            content_hash=manifest.get("content_hash", ""),
            urls=frozenset(urls),
            n_files=len(files),
            split=str(record.get("split", "")),
            n_without_url=sum(1 for f in files if not f.get("url")),
        ),
        "",
    )


def load_mirror(
    *,
    registry: str | Path = DEFAULT_REGISTRY,
    snapshots_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
) -> Mirror:
    """Index every funder's frozen snapshot from its `manifest.json`.

    Funders whose mirror cannot be read are listed in `Mirror.missing`.
    """
    snapshots_dir = Path(snapshots_dir)
    by_funder: dict[str, MirrorIndex] = {}
    missing: list[tuple[str, str]] = []

    for record in _read_registry(Path(registry)):
        index, reason = _load_index(record, snapshots_dir)
        if index is None:
            missing.append((record.get("slug", ""), reason))
            continue
        by_funder[index.funder] = index

    return Mirror(by_funder=by_funder, missing=tuple(missing))
