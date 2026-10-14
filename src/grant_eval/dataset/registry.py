"""Load the funder registry (`data/funders.json`): one record per funder.

`entry_url` is the URL the mirror was crawled from, not the funder homepage, since
most mirrors only cover a funding section. `content_hash` comes from the manifest.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from grant_eval.schema import Split

DEFAULT_REGISTRY_PATH = Path("data/funders.json")
DEFAULT_SNAPSHOT_ROOT = Path("data/snapshots")


class FunderRecord(BaseModel):
    """One funder's frozen identity: where its mirror is and which split it is in."""

    model_config = ConfigDict(frozen=True)

    slug: str
    display_name: str
    split: Split
    snapshot_date: date
    entry_url: str
    content_hash: str

    def snapshot_dir(self, root: Path = DEFAULT_SNAPSHOT_ROOT) -> Path:
        return Path(root) / self.slug / self.snapshot_date.isoformat()


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, FunderRecord]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    registry: dict[str, FunderRecord] = {}
    for row in rows:
        record = FunderRecord(**row)
        if record.slug in registry:
            raise ValueError(f"duplicate slug in {path}: {record.slug!r}")
        registry[record.slug] = record
    return registry


def funders_for_split(
    registry: dict[str, FunderRecord], split: Split
) -> list[FunderRecord]:
    return sorted(
        (r for r in registry.values() if r.split is split), key=lambda r: r.slug
    )
