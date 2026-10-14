"""Funder metadata for the descriptive breakdowns. Offline: committed data only."""
from __future__ import annotations

import json
import pathlib

import pytest

from grant_eval.analysis import funders as fmod


@pytest.fixture(scope="module")
def profiles():
    return fmod.load_profiles()


def test_every_tune_funder_has_a_type_group(profiles):
    assert len(profiles) == 16
    assert all(p.type_group for p in profiles.values())


def test_every_tune_profile_has_a_real_type_group_never_none(profiles):
    assert all(
        p.type_group in {"charity", "niche_trust", "membership_body"}
        for p in profiles.values()
    )


def test_type_group_is_none_never_empty_string_for_an_unmapped_split():
    """Validation funders, which have no entry in `TYPE_GROUPS`, get None rather than ""."""
    validation = fmod.load_profiles(split="validation")
    assert len(validation) == 8
    assert all(p.type_group is None for p in validation.values())


def test_the_type_mapping_cannot_drift_from_the_registry():
    """Every tune slug is in `TYPE_GROUPS` and every key in it is a real slug."""
    registry = json.loads(pathlib.Path("data/funders.json").read_text())
    slugs = {r["slug"] for r in registry}
    tune = {r["slug"] for r in registry if r["split"] == "tune"}
    assert tune <= set(fmod.TYPE_GROUPS)
    assert set(fmod.TYPE_GROUPS) <= slugs


def test_the_three_groups_have_the_recorded_sizes(profiles):
    sizes = {}
    for p in profiles.values():
        sizes[p.type_group] = sizes.get(p.type_group, 0) + 1
    assert sizes == {"charity": 5, "niche_trust": 5, "membership_body": 6}


def test_structural_complexity_is_rows_per_distinct_url(profiles):
    """Rows per URL matches hand counts from the CSVs for the two most packed funders."""
    sal = profiles["Society of Antiquaries of London"]
    assert (sal.n_rows, sal.n_distinct_urls) == (11, 3)
    assert sal.rows_per_url == pytest.approx(11 / 3)
    assert sal.max_in_scope_on_one_page == 8

    hm = profiles["Henry Moore Foundation"]
    assert (hm.n_rows, hm.n_distinct_urls) == (7, 2)
    assert hm.max_in_scope_on_one_page == 5


def test_a_funder_with_one_opportunity_per_page_scores_one(profiles):
    lev = profiles["The Leverhulme Trust"]
    assert lev.rows_per_url == pytest.approx(1.0)
    assert lev.max_in_scope_on_one_page == 1


def test_henry_moores_raw_type_is_pinned_to_actual_content(profiles):
    """The raw type matches the exact text in funders.md, so a wrong column is caught."""
    hm = profiles["Henry Moore Foundation"]
    assert hm.raw_type == "niche trust (sculpture/visual arts)"


def test_a_mangled_prose_file_raises_naming_what_is_missing(tmp_path):
    """A funders.md with no usable table raises and names the slugs with no type."""
    broken = tmp_path / "funders.md"
    broken.write_text("# Funders\n\nThe table used to be here. It is not any more.\n")
    with pytest.raises(ValueError, match=r"henry-moore"):
        fmod.load_profiles(prose_registry=broken)


def test_rows_per_url_actually_normalises_the_url(monkeypatch):
    """Rows per URL counts normalised URLs.

    The real URLs are already clean, so this forces every URL to normalise to the same value.
    """
    monkeypatch.setattr(fmod, "normalise_url", lambda url: "same")
    collapsed = fmod.load_profiles()
    lev = collapsed["The Leverhulme Trust"]
    assert lev.n_distinct_urls == 1
    assert lev.rows_per_url == pytest.approx(lev.n_rows)


def test_only_two_tune_funders_are_packed(profiles):
    """Only two funders are packed, which is why this grouping is descriptive only."""
    packed = sorted(k for k, p in profiles.items() if fmod.complexity_band(p) == "packed")
    assert packed == ["Henry Moore Foundation", "Society of Antiquaries of London"]


def test_validation_funders_can_be_requested_separately(profiles):
    validation = fmod.load_profiles(split="validation")
    assert len(validation) == 8
    assert set(validation) & set(profiles) == set()


def test_an_unmapped_slug_raises_rather_than_defaulting(monkeypatch):
    """A registry slug with no type group raises a KeyError."""
    monkeypatch.setattr(fmod, "TYPE_GROUPS", {"leverhulme": "charity"})
    with pytest.raises(KeyError, match="no type group"):
        fmod.load_profiles()
