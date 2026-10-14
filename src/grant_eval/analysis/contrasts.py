"""Paired contrasts between factor levels, built on the plain statistics in `stats.py`.

Runs are grouped by factor level and aligned by funder. The confirmatory family is the
set of within-factor pairwise contrasts (3 model + 6 prompt + 1 temperature = 10 on the
full grid), Holm-corrected as one family. Comparing all 24 configs pairwise (276 tests
over 16 funders) would leave almost nothing significant after correction.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from . import dataset as ds
from . import stats as st

__all__ = [
    "EXPECTED_FAMILY_SIZE",
    "FACTOR_ATTRIBUTES",
    "Contrast",
    "CellMeans",
    "Interaction",
    "cell_means",
    "confirmatory_family",
    "describe_largest_gap",
    "expected_family_size",
    "factor_gap_by_level",
    "factor_levels",
    "interaction",
    "marginal_means",
]

#: Family size on the full 3 x 4 x 2 tune grid. The validation split only has two
#: configs differing in prompt, so `confirmatory_family` checks against
#: `expected_family_size` instead.
EXPECTED_FAMILY_SIZE = 10

#: The factor each level belongs to, read off `RunRecord`.
FACTOR_ATTRIBUTES = {"model": "model", "prompt": "prompt", "temperature": "temperature"}


def _level(run: ds.RunRecord, factor: str) -> str:
    """The level a run carries for `factor`, as a string.

    Temperature is a float on `RunRecord`, so converting here keeps grouping and
    lookups using the same key type.
    """
    attribute = FACTOR_ATTRIBUTES[factor]
    return str(getattr(run, attribute))


def factor_levels(data: ds.AnalysisDataset) -> dict[str, tuple[str, ...]]:
    """Distinct levels of each factor, sorted, read off the data rather than assumed."""
    return {
        factor: tuple(sorted({_level(r, factor) for r in data.runs}))
        for factor in FACTOR_ATTRIBUTES
    }


def expected_family_size(levels: dict[str, tuple[str, ...]]) -> int:
    """Number of within-factor contrasts implied by `levels`: the sum of C(n, 2).

    A 3 x 4 x 2 grid gives 10 and the 1 x 2 x 1 validation split gives 1. A factor
    with one level contributes nothing.
    """
    total = 0
    for factor_levels_ in levels.values():
        n = len(factor_levels_)
        total += n * (n - 1) // 2
    return total


def marginal_means(
    data: ds.AnalysisDataset,
    factor: str,
    value: str = "f1",
    runs: tuple[ds.RunRecord, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """`{level: {funder: mean(value)}}`, averaged over every other factor and replicate.

    On the full grid a model's per-funder value is a mean over 40 runs. `None` values
    are skipped rather than treated as zero.
    """
    if runs is None:
        runs = data.runs
    grouped: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        got = getattr(r, value)
        if got is None:
            continue
        level = _level(r, factor)
        grouped.setdefault(level, {}).setdefault(r.funder, []).append(got)
    return {
        level: {funder: sum(vals) / len(vals) for funder, vals in by_funder.items()}
        for level, by_funder in grouped.items()
    }


@dataclass(frozen=True)
class Contrast:
    """One pairwise comparison of two levels of a single factor, paired on funder.

    Differences are `level_a - level_b`. Both the Wilcoxon and sign tests are kept, as
    pre-registered. `p_value` is the Wilcoxon p-value and `adjusted_p` its Holm-adjusted
    value. `excluded` lists funders dropped for a missing value on either side.
    """

    factor: str
    level_a: str
    level_b: str
    n_funders: int
    mean_difference: float
    median_difference: float | None
    wilcoxon: st.WilcoxonResult
    sign: st.SignResult
    difference_ci: st.BootstrapCI
    p_value: float | None
    adjusted_p: float | None
    rejected: bool | None
    excluded: tuple[tuple[str, str], ...] = ()


def _paired_levels(
    data: ds.AnalysisDataset,
    factor: str,
    level_a: str,
    level_b: str,
    value: str,
    runs: tuple[ds.RunRecord, ...] | None,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...], tuple[tuple[str, str], ...]]:
    """Per-funder means for two levels of one factor, aligned by funder.

    Funders missing a value on either side are excluded and listed.
    """
    means = marginal_means(data, factor, value=value, runs=runs)
    means_a = means.get(level_a, {})
    means_b = means.get(level_b, {})
    funders: list[str] = []
    a: list[float] = []
    b: list[float] = []
    excluded: list[tuple[str, str]] = []
    for funder in data.funders():
        if funder not in means_a:
            excluded.append((funder, f"undefined {value} for {factor}={level_a}"))
            continue
        if funder not in means_b:
            excluded.append((funder, f"undefined {value} for {factor}={level_b}"))
            continue
        funders.append(funder)
        a.append(means_a[funder])
        b.append(means_b[funder])
    return tuple(funders), tuple(a), tuple(b), tuple(excluded)


def confirmatory_family(
    data: ds.AnalysisDataset,
    value: str = "f1",
    runs: tuple[ds.RunRecord, ...] | None = None,
    alpha: float = 0.05,
) -> tuple[Contrast, ...]:
    """The pre-registered within-factor contrasts, Holm-corrected as one family.

    Each pair of levels is compared on the factor's marginal means, averaging over the
    other factors and replicates. The family size comes from the levels in the data, so
    the validation split gives a single prompt contrast.
    """
    levels = factor_levels(data)
    pending: list[tuple[str, str, str, tuple, tuple, tuple, tuple]] = []
    for factor, factor_levels_ in levels.items():
        for level_a, level_b in itertools.combinations(factor_levels_, 2):
            funders, a, b, excluded = _paired_levels(
                data, factor, level_a, level_b, value, runs
            )
            pending.append((factor, level_a, level_b, funders, a, b, excluded))

    differences = [
        tuple(x - y for x, y in zip(a, b)) for _f, _la, _lb, _funders, a, b, _e in pending
    ]
    wilcoxons = [st.wilcoxon_signed_rank(d) for d in differences]
    signs = [st.sign_test(d) for d in differences]
    difference_cis = [st.bootstrap_ci(d) for d in differences]

    raw_p = [w.p_value if w.p_value is not None else 1.0 for w in wilcoxons]
    holm = st.holm_bonferroni(raw_p, alpha=alpha)

    contrasts: list[Contrast] = []
    for i, (factor, level_a, level_b, funders, a, b, excluded) in enumerate(pending):
        d = differences[i]
        w = wilcoxons[i]
        p_value = w.p_value
        contrasts.append(
            Contrast(
                factor=factor,
                level_a=level_a,
                level_b=level_b,
                n_funders=len(funders),
                mean_difference=sum(d) / len(d) if d else 0.0,
                median_difference=w.median_difference,
                wilcoxon=w,
                sign=signs[i],
                difference_ci=difference_cis[i],
                p_value=p_value,
                adjusted_p=holm.adjusted[i] if p_value is not None else None,
                rejected=holm.rejected[i] if p_value is not None else None,
                excluded=excluded,
            )
        )

    expected = expected_family_size(levels)
    assert len(contrasts) == expected, (
        f"confirmatory_family produced {len(contrasts)} contrasts, expected "
        f"{expected} from the levels present in the data "
        f"({ {f: len(ls) for f, ls in levels.items()} }); a factor silently lost a "
        "level between factor_levels() and the pairing loop"
    )
    return tuple(contrasts)


@dataclass(frozen=True)
class CellMeans:
    """Per-funder means for every combination of two factors' levels.

    `means[(level_a, level_b)][funder]` averages over all other factors and replicates.
    """

    factor_a: str
    factor_b: str
    means: dict[tuple[str, str], dict[str, float]]


def cell_means(
    data: ds.AnalysisDataset,
    factor_a: str,
    factor_b: str,
    value: str = "f1",
    runs: tuple[ds.RunRecord, ...] | None = None,
) -> CellMeans:
    """Mean of `value` per `(level_a, level_b, funder)`, used for the interaction contrast."""
    if runs is None:
        runs = data.runs
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in runs:
        got = getattr(r, value)
        if got is None:
            continue
        key = (_level(r, factor_a), _level(r, factor_b))
        grouped.setdefault(key, {}).setdefault(r.funder, []).append(got)
    means = {
        key: {funder: sum(vals) / len(vals) for funder, vals in by_funder.items()}
        for key, by_funder in grouped.items()
    }
    return CellMeans(factor_a=factor_a, factor_b=factor_b, means=means)


def factor_gap_by_level(
    cells: CellMeans, level_b1: str, level_b2: str
) -> dict[str, float]:
    """`{level_a: mean(level_b1 - level_b2)}` for every level of `cells.factor_a`.

    Covers all levels, not just the two passed to `interaction()`, because the
    pre-registered prediction was about every model. Averaged over funders present on
    both sides; levels with no shared funder are skipped.
    """
    gaps: dict[str, float] = {}
    levels_a = sorted({key[0] for key in cells.means})
    for level in levels_a:
        b1 = cells.means.get((level, level_b1), {})
        b2 = cells.means.get((level, level_b2), {})
        common = sorted(set(b1) & set(b2))
        if not common:
            continue
        diffs = [b1[f] - b2[f] for f in common]
        gaps[level] = sum(diffs) / len(diffs)
    return gaps


def describe_largest_gap(cells: CellMeans, factor_b: str, level_b1: str, level_b2: str) -> str:
    """Name the level of `cells.factor_a` with the largest absolute gap, and its sign."""
    gaps = factor_gap_by_level(cells, level_b1, level_b2)
    if not gaps:
        return f"undetermined (no level of {cells.factor_a!r} has a defined {factor_b} gap)"
    largest_level = max(gaps, key=lambda level: abs(gaps[level]))
    largest_gap = gaps[largest_level]
    sign_word = "negative" if largest_gap < 0 else "positive" if largest_gap > 0 else "zero"
    return f"largest on {largest_level}, {sign_word}"


@dataclass(frozen=True)
class Interaction:
    """A pre-registered difference-in-differences between two factors.

    F1 is bounded and not normal, so this uses a per-funder DiD,
    `(A1B1 - A1B2) - (A2B1 - A2B2)`, with a bootstrap CI over funders instead of an
    ANOVA interaction term.
    """

    factor_a: str
    level_a1: str
    level_a2: str
    factor_b: str
    level_b1: str
    level_b2: str
    difference_in_differences: float
    ci: st.BootstrapCI
    n_funders: int
    per_funder: dict[str, float]
    predicted_direction: str
    observed_direction: str
    excluded: tuple[tuple[str, str], ...] = ()


def interaction(
    data: ds.AnalysisDataset,
    factor_a: str,
    level_a1: str,
    level_a2: str,
    factor_b: str,
    level_b1: str,
    level_b2: str,
    value: str = "f1",
    runs: tuple[ds.RunRecord, ...] | None = None,
    predicted_direction: str = "",
) -> Interaction:
    """The difference-in-differences `(A1B1 - A1B2) - (A2B1 - A2B2)`, per funder.

    Only funders present in all four cells are used. Others are listed in `excluded`
    with the cells they were missing from. Raises `KeyError` for an unknown level.
    """
    cells = cell_means(data, factor_a, factor_b, value=value, runs=runs)
    levels = factor_levels(data)
    for factor, level_1, level_2 in (
        (factor_a, level_a1, level_a2),
        (factor_b, level_b1, level_b2),
    ):
        for level in (level_1, level_2):
            if level not in levels.get(factor, ()):
                raise KeyError(f"no such level {level!r} for factor {factor!r}")

    keys = (
        (level_a1, level_b1),
        (level_a1, level_b2),
        (level_a2, level_b1),
        (level_a2, level_b2),
    )
    cell_funders = [set(cells.means.get(key, {})) for key in keys]
    common_funders = sorted(set.intersection(*cell_funders)) if cell_funders else []

    excluded: list[tuple[str, str]] = []
    for funder in data.funders():
        if funder in common_funders:
            continue
        missing_cells = [
            f"{factor_a}={la!r}, {factor_b}={lb!r}"
            for (la, lb), funders_here in zip(keys, cell_funders)
            if funder not in funders_here
        ]
        excluded.append((funder, f"undefined {value} for " + "; ".join(missing_cells)))

    per_funder: dict[str, float] = {}
    for funder in common_funders:
        a1b1 = cells.means[(level_a1, level_b1)][funder]
        a1b2 = cells.means[(level_a1, level_b2)][funder]
        a2b1 = cells.means[(level_a2, level_b1)][funder]
        a2b2 = cells.means[(level_a2, level_b2)][funder]
        per_funder[funder] = (a1b1 - a1b2) - (a2b1 - a2b2)

    diffs = tuple(per_funder[f] for f in common_funders)
    ci = st.bootstrap_ci(diffs)
    did = sum(diffs) / len(diffs) if diffs else 0.0

    observed_direction = describe_largest_gap(cells, factor_b, level_b1, level_b2)

    return Interaction(
        factor_a=factor_a,
        level_a1=level_a1,
        level_a2=level_a2,
        factor_b=factor_b,
        level_b1=level_b1,
        level_b2=level_b2,
        difference_in_differences=did,
        ci=ci,
        n_funders=len(common_funders),
        per_funder=per_funder,
        predicted_direction=predicted_direction,
        observed_direction=observed_direction,
        excluded=tuple(excluded),
    )
