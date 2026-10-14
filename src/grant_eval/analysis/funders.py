"""Funder metadata for the descriptive breakdowns by funder type and page structure.

Type groups are a hand-written mapping checked against `data/funders.json`, because the
free-text type column in `data/funders.md` is too loose to parse reliably. Structural
complexity is `rows_per_url`, the mean number of ground-truth rows sharing one
normalised URL. It tracks F1 far better than funder size, but it partly reflects the
scorer: titles are only checked when a URL is shared, so packed pages are where title
paraphrases cost matches.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from grant_eval.scoring.groundtruth import load_ground_truth
from grant_eval.scoring.identity import normalise_url

__all__ = [
    "DEFAULT_REGISTRY",
    "DEFAULT_PROSE_REGISTRY",
    "PACKED_THRESHOLD",
    "TYPE_GROUPS",
    "FunderProfile",
    "complexity_band",
    "load_profiles",
]

#: The machine-readable funder registry (`data/funders.md` is the prose audit trail).
DEFAULT_REGISTRY = Path("data/funders.json")

#: Prose registry that `raw_type` is read from, for display only.
DEFAULT_PROSE_REGISTRY = Path("data/funders.md")

#: One `funders.md` table row: `| # | Funder | `slug` | Type | Split | ... |`.
_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|[^|]*\|\s*`([a-z0-9][a-z0-9-]*)`\s*\|([^|]*)\|")

#: Slug -> coarse type group. Professional bodies, learned societies and the British
#: Academy are pooled as `membership_body` because each is too small alone.
TYPE_GROUPS = {
    "garfield-weston": "charity", "nuffield": "charity", "wolfson": "charity",
    "leverhulme": "charity", "cara": "charity",
    "carnegie-trust": "niche_trust", "henry-moore": "niche_trust",
    "nuffield-farming": "niche_trust", "paul-mellon": "niche_trust",
    "sasakawa": "niche_trust",
    "british-ecological-society": "membership_body",
    "physiological-society": "membership_body", "rsc": "membership_body",
    "genetics-society": "membership_body", "society-antiquaries": "membership_body",
    "british-academy": "membership_body",
}

#: `rows_per_url` at or above this counts as "packed". Two tune funders clear it.
PACKED_THRESHOLD = 2.0


@dataclass(frozen=True)
class FunderProfile:
    """One funder's type group and structural-complexity numbers.

    `n_rows` and `n_distinct_urls` include distractor rows, as `identity.shared_urls`
    does. `max_in_scope_on_one_page` is the most in-scope rows sharing one URL.
    """

    slug: str
    display_name: str
    split: str
    #: `None` for funders outside `TYPE_GROUPS` (the validation split), so they don't
    #: form an empty-string group.
    type_group: str | None
    raw_type: str
    n_rows: int
    n_in_scope: int
    n_distinct_urls: int
    rows_per_url: float
    max_in_scope_on_one_page: int


def complexity_band(profile: FunderProfile) -> str:
    """`"packed"` when `rows_per_url >= PACKED_THRESHOLD`, else `"one_per_page"`."""
    return "packed" if profile.rows_per_url >= PACKED_THRESHOLD else "one_per_page"


def _raw_types(prose_registry: str | Path) -> dict[str, str]:
    """slug -> free-text "Type" cell from the `funders.md` tables, for display."""
    types: dict[str, str] = {}
    for line in Path(prose_registry).read_text(encoding="utf-8").splitlines():
        match = _TABLE_ROW_RE.match(line.strip())
        if match is None:
            continue
        slug, raw_type = match.group(1), match.group(2).strip()
        types.setdefault(slug, raw_type)
    return types


def _validate_raw_types(
    raw_types: dict[str, str], tune_slugs: set[str], prose_registry: str | Path
) -> None:
    """Raise if any tune slug has no `raw_type`, which means the table format changed."""
    missing = sorted(slug for slug in tune_slugs if not raw_types.get(slug))
    if missing:
        raise ValueError(
            f"{prose_registry}: no raw type parsed for tune slug(s) {missing} — "
            "the table format may have changed; check _TABLE_ROW_RE against the file."
        )


def load_profiles(
    split: str = "tune",
    *,
    registry: str | Path = DEFAULT_REGISTRY,
    prose_registry: str | Path = DEFAULT_PROSE_REGISTRY,
) -> dict[str, FunderProfile]:
    """Profiles for every funder in `split`, keyed by display name.

    Raises:
        KeyError: a tune slug is missing from `TYPE_GROUPS`.
        ValueError: a tune slug has no type in `data/funders.md`.
    """
    records = json.loads(Path(registry).read_text(encoding="utf-8"))
    gt = load_ground_truth()
    raw_types = _raw_types(prose_registry)
    tune_slugs = {r["slug"] for r in records if r["split"] == "tune"}
    _validate_raw_types(raw_types, tune_slugs, prose_registry)

    profiles: dict[str, FunderProfile] = {}
    for record in records:
        if record["split"] != split:
            continue
        slug = record["slug"]
        if split == "tune":
            # Only tune funders need a type group; validation funders get None.
            try:
                type_group = TYPE_GROUPS[slug]
            except KeyError:
                raise KeyError(f"{slug}: no type group — add it to TYPE_GROUPS") from None
        else:
            type_group = TYPE_GROUPS.get(slug)

        display_name = record["display_name"]
        funder_gt = gt.for_funder(display_name)

        urls = Counter(normalise_url(r.url) for r in funder_gt.rows)
        in_scope_urls = Counter(normalise_url(r.url) for r in funder_gt.in_scope)
        rows_per_url = len(funder_gt.rows) / len(urls)
        max_in_scope_on_one_page = max(in_scope_urls.values(), default=0)

        profiles[display_name] = FunderProfile(
            slug=slug,
            display_name=display_name,
            split=split,
            type_group=type_group,
            raw_type=raw_types.get(slug, ""),
            n_rows=len(funder_gt.rows),
            n_in_scope=len(funder_gt.in_scope),
            n_distinct_urls=len(urls),
            rows_per_url=rows_per_url,
            max_in_scope_on_one_page=max_in_scope_on_one_page,
        )
    return profiles
