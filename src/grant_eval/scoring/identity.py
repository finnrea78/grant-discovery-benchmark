"""The match key that decides whether two records are the same opportunity.

The key is `(funder, normalised_url, normalised_title)`. The title is only used when
more than one ground-truth row of a funder shares the same normalised URL; otherwise
the URL alone decides, so a paraphrased title is never penalised. Both the scorer and
the agent loop import this module so the two cannot drift apart.

`shared_urls` is computed over every row of a funder, distractors included. Some
shared-URL groups mix in-scope and out-of-scope rows, so leaving distractors out would
give a distractor the same URL-only key as a real row.

`normalise_url`: lowercase, drop the fragment, drop the query string, strip one
trailing slash.

`normalise_title` applies five fixed steps in order (no fuzzy matching):

    1. Unicode NFKC normalisation.
    2. Map curly quotes to straight quotes and dash-like characters to a hyphen.
    3. Casefold.
    4. Replace every character that is not a letter or digit with a space.
    5. Collapse whitespace, strip, and drop a leading "the ".

All titles within each shared-URL group stay distinct after normalisation.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Container, Iterable

# Always a 3-tuple: the third slot is the normalised title when the URL is shared within
# the funder and "" otherwise, so keys from the two branches can never collide.
MatchKey = tuple[str, str, str]

_TYPOGRAPHIC = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-",
        "—": "-", "―": "-", "−": "-",
    }
)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_LEADING_ARTICLE = re.compile(r"^the ")


def normalise_funder(funder: str) -> str:
    """Funder part of the key: stripped and casefolded (guards against hand-edited cells)."""
    return (funder or "").strip().casefold()


def normalise_url(url: str) -> str:
    """URL part of the key: lowercase, no fragment, no query, no trailing slash.

    The fragment is dropped before the query string, so `page?a=1#b` and `page#b?a=1`
    both reduce to `page`.
    """
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    u = u.split("?", 1)[0]
    u = u.lower()
    if u.endswith("/"):
        u = u[:-1]
    return u


def normalise_title(title: str) -> str:
    """Title part of the key, using the five steps in the module docstring."""
    t = unicodedata.normalize("NFKC", title or "")
    t = t.translate(_TYPOGRAPHIC)
    t = t.casefold()
    t = _NON_ALNUM.sub(" ", t)
    t = " ".join(t.split())
    return _LEADING_ARTICLE.sub("", t)


def shared_urls(urls: Iterable[str], *, already_normalised: bool = False) -> frozenset[str]:
    """The normalised URLs that appear more than once in `urls`.

    For ground truth, pass every row of one funder, distractors included. Counting is
    per row, so two rows sharing both URL and title collide later in `build_key_index`,
    which raises rather than merging them.
    """
    counts = Counter(u if already_normalised else normalise_url(u) for u in urls)
    return frozenset(u for u, n in counts.items() if n > 1 and u)


def match_key(
    *,
    funder: str,
    url: str,
    title: str,
    shared: Container[str],
) -> MatchKey:
    """Build the match key for one record.

    `title` is only used when the normalised URL is in `shared` (from `shared_urls`);
    pass an empty set to key on URL alone.
    """
    curl = normalise_url(url)
    if curl in shared:
        return (normalise_funder(funder), curl, normalise_title(title))
    return (normalise_funder(funder), curl, "")


def opportunity_match_key(
    opportunity,
    funder: str,
    shared: Container[str] = frozenset(),
) -> MatchKey:
    """`match_key` for an `Opportunity`-like record, with fallbacks for an empty URL.

    Normalisation can still empty a URL (a bare "/" or just a query string). A keyless
    record would equal every other keyless record, so we fall back to the
    `opportunity_id`, then the normalised title, in the URL slot.
    """
    curl = normalise_url(opportunity.url or "")
    if curl:
        return match_key(
            funder=funder,
            url=opportunity.url or "",
            title=opportunity.title or "",
            shared=shared,
        )
    fallback = (opportunity.opportunity_id or "").strip().casefold()
    if not fallback:
        fallback = normalise_title(opportunity.title or "")
    return (normalise_funder(funder), fallback, "")


def build_key_index(rows: Iterable, funder: str) -> tuple[dict[MatchKey, object], frozenset[str]]:
    """Index one funder's ground-truth rows by match key.

    Raises on a key collision, since two rows with the same URL and title are a
    labelling error. Returns `(key -> row, shared_urls)`.
    """
    rows = list(rows)
    shared = shared_urls(r.url for r in rows)
    index: dict[MatchKey, object] = {}
    for row in rows:
        key = opportunity_match_key(row, funder, shared)
        if key in index:
            other = index[key]
            raise ValueError(
                f"{funder}: ground-truth rows {other.opportunity_id!r} and "
                f"{row.opportunity_id!r} collide on match key {key!r} — they share a "
                f"normalised URL and a normalised title, so no scorer can tell them "
                f"apart. Fix the labelling."
            )
        index[key] = row
    return index, shared
