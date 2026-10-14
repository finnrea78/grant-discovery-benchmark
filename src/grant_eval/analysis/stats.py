"""Statistical tests used by the analysis: Holm correction, sign test, Wilcoxon and
a clustered bootstrap CI.

Functions take plain numbers so they can be tested against hand-worked examples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon

__all__ = [
    "HolmResult",
    "holm_bonferroni",
    "SignResult",
    "sign_test",
    "WilcoxonResult",
    "wilcoxon_signed_rank",
    "BootstrapCI",
    "bootstrap_ci",
]


@dataclass(frozen=True)
class HolmResult:
    """A family of p-values after step-down Holm correction, in the caller's order."""

    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    n_comparisons: int


def holm_bonferroni(p_values, alpha: float = 0.05) -> HolmResult:
    """Holm (1979) step-down correction. It assumes no independence, which matters
    because the contrasts share funders.

    Adjusted p is `max_{j<=i} (m - j + 1) * p_(j)` capped at 1. The running maximum
    stops a larger raw p from adjusting below a smaller one.
    """
    p_values = list(p_values)
    m = len(p_values)
    if m == 0:
        raise ValueError("holm_bonferroni: empty family of p-values")

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)

    rejected = [adj < alpha for adj in adjusted]

    return HolmResult(
        p_values=tuple(p_values),
        adjusted=tuple(adjusted),
        rejected=tuple(rejected),
        alpha=alpha,
        n_comparisons=m,
    )


@dataclass(frozen=True)
class SignResult:
    """The exact two-sided sign test on paired differences.

    `n_used` is the sample size after ties are dropped. When it is 0 the result is
    `undefined` and `p_value` is `None` rather than 1.0.
    """

    n_positive: int
    n_negative: int
    n_ties: int
    n_used: int
    p_value: float | None
    proportion_positive: float | None
    undefined: bool


def sign_test(differences) -> SignResult:
    """Exact two-sided sign test on paired differences.

    Unlike Wilcoxon it does not assume the differences are symmetric, which F1
    differences over 16 funders may not be. Ties are dropped and counted.
    """
    n_positive = sum(1 for d in differences if d > 0)
    n_negative = sum(1 for d in differences if d < 0)
    n_ties = sum(1 for d in differences if d == 0)
    n_used = n_positive + n_negative

    if n_used == 0:
        return SignResult(
            n_positive=n_positive,
            n_negative=n_negative,
            n_ties=n_ties,
            n_used=0,
            p_value=None,
            proportion_positive=None,
            undefined=True,
        )

    p_value = binomtest(min(n_positive, n_negative), n_used, 0.5).pvalue

    return SignResult(
        n_positive=n_positive,
        n_negative=n_negative,
        n_ties=n_ties,
        n_used=n_used,
        p_value=p_value,
        proportion_positive=n_positive / n_used,
        undefined=False,
    )


@dataclass(frozen=True)
class WilcoxonResult:
    """Wilcoxon signed-rank result with a matched-pairs rank-biserial effect size
    (Kerby 2014).

    `w_plus + w_minus` should equal `n_used * (n_used + 1) / 2`. `statistic` is
    `min(w_plus, w_minus)`, as in scipy. `median_difference` includes zero differences,
    since a zero is a real result, even though zeros are dropped from the ranking.
    `undefined` is True when `n_used` is 0 or below `MIN_WILCOXON_N`.
    """

    statistic: float | None
    p_value: float | None
    w_plus: float | None
    w_minus: float | None
    n_used: int
    n_ties: int
    rank_biserial: float | None
    median_difference: float | None
    undefined: bool
    reason: str = ""


#: Below this many non-tied pairs the exact two-sided p cannot go under 0.05
#: (2 / 2^5 = 0.0625), so the test is reported as undefined.
MIN_WILCOXON_N = 6

#: Above this many non-tied pairs scipy's normal approximation is used instead of the
#: exact distribution.
MAX_EXACT_WILCOXON_N = 25


def wilcoxon_signed_rank(differences) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired differences (exact up to n=25).

    Zero differences are dropped from the ranking and counted.
    """
    differences = list(differences)
    median_difference = float(np.median(differences)) if differences else None

    non_zero = [d for d in differences if d != 0]
    n_used = len(non_zero)
    n_ties = len(differences) - n_used

    if n_used == 0:
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            w_plus=None,
            w_minus=None,
            n_used=0,
            n_ties=n_ties,
            rank_biserial=None,
            median_difference=median_difference,
            undefined=True,
            reason="wilcoxon_signed_rank: nothing left to rank once ties are dropped",
        )

    if n_used < MIN_WILCOXON_N:
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            w_plus=None,
            w_minus=None,
            n_used=n_used,
            n_ties=n_ties,
            rank_biserial=None,
            median_difference=median_difference,
            undefined=True,
            reason=(
                f"wilcoxon_signed_rank: n_used={n_used} < MIN_WILCOXON_N="
                f"{MIN_WILCOXON_N}; the smallest attainable exact two-sided p is "
                f"already above 0.05 at this n, so the test cannot reject at any "
                f"conventional alpha whatever the data are"
            ),
        )

    ranks = rankdata([abs(d) for d in non_zero])
    w_plus = float(sum(r for d, r in zip(non_zero, ranks) if d > 0))
    w_minus = float(sum(r for d, r in zip(non_zero, ranks) if d < 0))
    statistic = min(w_plus, w_minus)

    method = "exact" if n_used <= MAX_EXACT_WILCOXON_N else "approx"
    _, p_value = wilcoxon(non_zero, alternative="two-sided", method=method)

    s = n_used * (n_used + 1) / 2
    rank_biserial = (w_plus - w_minus) / s

    return WilcoxonResult(
        statistic=statistic,
        p_value=float(p_value),
        w_plus=w_plus,
        w_minus=w_minus,
        n_used=n_used,
        n_ties=n_ties,
        rank_biserial=rank_biserial,
        median_difference=median_difference,
        undefined=False,
        reason="",
    )


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile bootstrap confidence interval over cluster values.

    `n_clusters` and `seed` are kept so tables can show how the interval was built.
    """

    point: float
    lower: float | None
    upper: float | None
    standard_error: float | None
    n_clusters: int
    n_resamples: int
    level: float
    seed: int
    undefined: bool
    reason: str = ""


#: With fewer clusters than this the resampling is degenerate, so no interval is given.
MIN_BOOTSTRAP_CLUSTERS = 3

#: Fixed seed so intervals are the same on every run.
DEFAULT_SEED = 20260908


def bootstrap_ci(
    clusters,
    statistic=None,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Percentile bootstrap CI, resampling CLUSTERS with replacement.

    The caller passes one value per funder, since runs within a funder are not
    independent. With only 16 clusters the interval is probably too narrow (Davison and
    Hinkley 1997; Field and Welsh 2007). Percentile is used rather than BCa because
    BCa's jackknife step is unreliable with so few clusters.

    `statistic` defaults to the mean. Below `MIN_BOOTSTRAP_CLUSTERS` the result is
    `undefined` with no interval.
    """
    values = np.asarray(list(clusters), dtype=float)
    if values.size == 0:
        raise ValueError("bootstrap_ci: empty sample")

    n = values.size
    use_mean = statistic is None
    point = float(values.mean()) if use_mean else float(statistic(values))

    if n < MIN_BOOTSTRAP_CLUSTERS:
        return BootstrapCI(
            point=point,
            lower=None,
            upper=None,
            standard_error=None,
            n_clusters=n,
            n_resamples=n_resamples,
            level=level,
            seed=seed,
            undefined=True,
            reason=(
                f"bootstrap_ci: n_clusters={n} < MIN_BOOTSTRAP_CLUSTERS="
                f"{MIN_BOOTSTRAP_CLUSTERS}; a resample distribution over this few "
                f"clusters is degenerate, and an interval computed from it would be "
                f"theatre rather than inference"
            ),
        )

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resample_values = values[resample_indices]

    if use_mean:
        resample_stats = resample_values.mean(axis=1)
    else:
        resample_stats = np.array([statistic(row) for row in resample_values])

    standard_error = float(resample_stats.std(ddof=0))
    lower_pct = (1 - level) / 2 * 100
    upper_pct = (1 + level) / 2 * 100
    lower, upper = np.percentile(resample_stats, [lower_pct, upper_pct])

    return BootstrapCI(
        point=point,
        lower=float(lower),
        upper=float(upper),
        standard_error=standard_error,
        n_clusters=n,
        n_resamples=n_resamples,
        level=level,
        seed=seed,
        undefined=False,
        reason="",
    )
