"""Tests for the statistics helpers.

Each function is checked against a value worked out without the implementation:
hand arithmetic, exhaustive enumeration, or a published worked example.

"""
from __future__ import annotations

import itertools
import math

import pytest

from grant_eval.analysis import stats


def test_holm_matches_the_hand_worked_example():
    """Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*,
    Scandinavian Journal of Statistics 6:65-70.

    m = 4, p = [0.01, 0.02, 0.03, 0.04]. Step-down multipliers are 4, 3, 2, 1 with
    monotonicity enforced:
        i=1: 4 * 0.01 = 0.04
        i=2: 3 * 0.02 = 0.06        -> max(0.04, 0.06) = 0.06
        i=3: 2 * 0.03 = 0.06        -> max(0.06, 0.06) = 0.06
        i=4: 1 * 0.04 = 0.04        -> max(0.06, 0.04) = 0.06
    Adjusted = [0.04, 0.06, 0.06, 0.06]. At alpha = 0.05 only the first is rejected.
    """
    result = stats.holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    assert result.adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert result.rejected == (True, False, False, False)
    assert result.n_comparisons == 4


def test_holm_returns_values_in_the_callers_order():
    """Results come back in input order, so callers can zip them against their contrast names."""
    result = stats.holm_bonferroni([0.04, 0.01, 0.03, 0.02])
    assert result.adjusted == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_enforces_monotonicity():
    """Adjusted p-values never decrease, so a larger raw p cannot adjust below an earlier one."""
    result = stats.holm_bonferroni([0.02, 0.021, 0.99])
    assert result.adjusted[0] <= result.adjusted[1] <= result.adjusted[2]


def test_holm_caps_at_one():
    """Adjusted p-values are capped at 1.

    `m * p_(1) = 2 * 0.6 = 1.2`, so a missing cap would show up here.

    """
    assert stats.holm_bonferroni([0.6, 0.7]).adjusted == pytest.approx([1.0, 1.0])


def test_holm_agrees_with_statsmodels_on_a_random_family():
    """Cross-check against statsmodels, which is already a dependency."""
    from statsmodels.stats.multitest import multipletests

    import random

    rng = random.Random(20260908)
    for _ in range(50):
        pvals = [rng.random() for _ in range(rng.randint(2, 12))]
        assert stats.holm_bonferroni(pvals).adjusted == pytest.approx(
            list(multipletests(pvals, method="holm")[1])
        )


def test_holm_rejects_an_empty_family_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="empty"):
        stats.holm_bonferroni([])


def test_sign_test_matches_the_exact_binomial_by_hand():
    """Exact two-sided sign test: with n = 10 non-tied pairs and 2 negatives,

        p = 2 * [C(10,0) + C(10,1) + C(10,2)] / 2^10
          = 2 * (1 + 10 + 45) / 1024
          = 112 / 1024
          = 0.109375

    Siegel & Castellan (1988), *Nonparametric Statistics for the Behavioral Sciences*,
    section 4.1. Also checked against scipy.stats.binomtest(2, 10, 0.5).

    """
    differences = [1.0] * 8 + [-1.0] * 2
    result = stats.sign_test(differences)
    assert (result.n_positive, result.n_negative, result.n_used) == (8, 2, 10)
    assert result.p_value == pytest.approx(0.109375)
    assert result.undefined is False


def test_sign_test_drops_ties_and_counts_them():
    """Zero differences are dropped, since they carry no direction."""
    result = stats.sign_test([1.0] * 8 + [-1.0] * 2 + [0.0] * 5)
    assert result.n_ties == 5
    assert result.n_used == 10
    assert result.p_value == pytest.approx(0.109375)


def test_sign_test_is_undefined_when_every_pair_ties():
    """With no usable pairs there is no p-value, rather than p = 1.0, which would look like evidence of no difference."""
    result = stats.sign_test([0.0, 0.0, 0.0])
    assert result.undefined is True
    assert result.p_value is None
    assert result.n_used == 0


def test_sign_test_agrees_with_exhaustive_enumeration():
    """Enumerate all 2^n sign assignments under the null and count those at least as extreme."""
    import random

    rng = random.Random(11)
    for _ in range(20):
        n = rng.randint(3, 12)
        diffs = [rng.choice([-1.0, 1.0]) for _ in range(n)]
        observed = sum(1 for d in diffs if d > 0)
        extreme = sum(
            1
            for signs in itertools.product([0, 1], repeat=n)
            if abs(sum(signs) - n / 2) >= abs(observed - n / 2)
        )
        assert stats.sign_test(diffs).p_value == pytest.approx(extreme / 2**n)


# Hand-worked example, used by three tests below.
#
#   d      = [+0.10, +0.20, -0.05, +0.15, +0.30, -0.02, +0.25, +0.08]
#   |d|      0.10   0.20   0.05   0.15   0.30   0.02   0.25   0.08
#   rank      4      6      2      5      8      1      7      3
#
#   W+ = 4 + 6 + 5 + 8 + 7 + 3 = 33
#   W- = 2 + 1                 = 3
#   W+ + W- = 36 = n(n+1)/2 = 8*9/2   (checks the ranking)
#   W  = min(W+, W-) = 3
#
# Exact two-sided p: subsets of {1..8} summing to <= 3 are {}, {1}, {2}, {3}, {1,2}
# -> 5 of 2^8 = 256, so p = 2 * 5/256 = 10/256 = 0.0390625.
#
# Rank-biserial r = (W+ - W-) / (n(n+1)/2) = (33 - 3)/36 = 30/36 = 0.8333...
HAND = [0.10, 0.20, -0.05, 0.15, 0.30, -0.02, 0.25, 0.08]


def test_wilcoxon_matches_the_hand_worked_ranking():
    """Wilcoxon (1945), *Individual Comparisons by Ranking Methods*, Biometrics
    Bulletin 1(6):80-83; procedure as in Siegel & Castellan (1988), section 5.4. The
    rank arithmetic is written out above.
    """
    result = stats.wilcoxon_signed_rank(HAND)
    assert (result.w_plus, result.w_minus) == (33, 3)
    assert result.w_plus + result.w_minus == 8 * 9 / 2
    assert result.statistic == 3
    assert result.p_value == pytest.approx(0.0390625)
    assert result.undefined is False


def test_wilcoxon_exact_p_matches_exhaustive_enumeration():
    """Enumerate all 2^8 sign assignments and count those with W- at least as extreme as observed."""
    n = len(HAND)
    ranks = range(1, n + 1)
    count = sum(
        1
        for signs in itertools.product([0, 1], repeat=n)
        if (lambda w: w <= 3 or w >= 33)(sum(r for r, s in zip(ranks, signs) if s))
    )
    assert count / 2**n == pytest.approx(0.0390625)
    assert stats.wilcoxon_signed_rank(HAND).p_value == pytest.approx(count / 2**n)


def test_rank_biserial_is_the_simple_difference_formula():
    """Kerby (2014), *The Simple Difference Formula*, Comprehensive Psychology 3:1.

    r = (W+ - W-) / S with S = n(n+1)/2. Here (33 - 3)/36 = 0.8333.
    """
    result = stats.wilcoxon_signed_rank(HAND)
    assert result.rank_biserial == pytest.approx(30 / 36)
    assert -1.0 <= result.rank_biserial <= 1.0


def test_rank_biserial_sign_follows_the_direction_of_the_difference():
    negated = stats.wilcoxon_signed_rank([-d for d in HAND])
    assert negated.rank_biserial == pytest.approx(-30 / 36)
    assert negated.p_value == pytest.approx(0.0390625)


def test_median_difference_is_reported_beside_the_p_value():
    """The median difference gives the effect size in F1 points, not just whether it is non-zero.

    HAND sorted is [-0.05, -0.02, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]; for n=8 the
    median is the mean of the 4th and 5th values, 0.10 and 0.15, so 0.125. This
    matches `statistics.median(HAND)` and the Hodges-Lehmann pseudomedian.

    """
    result = stats.wilcoxon_signed_rank(HAND)
    assert result.median_difference == pytest.approx((0.10 + 0.15) / 2)


def test_wilcoxon_drops_zero_differences_and_counts_them():
    result = stats.wilcoxon_signed_rank(HAND + [0.0, 0.0])
    assert result.n_ties == 2
    assert result.n_used == 8
    assert result.p_value == pytest.approx(0.0390625)


def test_median_difference_includes_dropped_zeros():
    """Zero differences are kept in the median even though the ranking drops them.

    A zero is a real result (two runs scored the same). HAND plus two zeros, sorted, is
    [-0.05, -0.02, 0.0, 0.0, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30], so the median is
    (0.08 + 0.10)/2 = 0.09, not the 0.125 you get from the non-zero values alone.

    """
    result = stats.wilcoxon_signed_rank(HAND + [0.0, 0.0])
    assert result.median_difference == pytest.approx(0.09)


def test_wilcoxon_is_undefined_when_nothing_is_left_to_rank():
    result = stats.wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert result.undefined is True
    assert result.p_value is None
    assert result.rank_biserial is None


def test_wilcoxon_is_undefined_below_the_minimum_usable_n():
    """With n < 6 no p-value is reported.

    The smallest possible two-sided exact p is 2/2^n > 0.05, so the test can never
    reject, and a p there would be read as evidence of no effect.

    """
    result = stats.wilcoxon_signed_rank([0.1, 0.2, 0.3])
    assert result.undefined is True
    assert "cannot reject" in result.reason


def test_bootstrap_reproduces_the_exhaustive_distribution_at_n_three():
    """With n = 3 all 3^3 = 27 resamples can be listed, which gives an exact answer.

    x = [1, 2, 6] gives 27 resample means from 1.0 to 6.0, with exact standard error
    1.2472191289246470.

    """
    x = [1.0, 2.0, 6.0]
    means = sorted(sum(c) / 3 for c in itertools.product(x, repeat=3))
    mean_of_means = sum(means) / len(means)
    exhaustive_se = math.sqrt(
        sum((m - mean_of_means) ** 2 for m in means) / len(means)
    )
    assert exhaustive_se == pytest.approx(1.2472191289246470)

    result = stats.bootstrap_ci(x, n_resamples=200_000, seed=1)
    assert result.standard_error == pytest.approx(exhaustive_se, rel=0.02)
    assert result.point == pytest.approx(3.0)


def test_bootstrap_standard_error_converges_to_the_analytic_value():
    """For the mean, the bootstrap SE tends to the population SD over sqrt(n).

    Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, eq. 6.6.
    For [1, 2, 6]: deviations -2, -1, 3; squares 4, 1, 9; sum 14.
    sd_pop = sqrt(14/3), SE = sqrt(14/3)/sqrt(3) = sqrt(14)/3 = 1.2472191289246473.

    """
    x = [1.0, 2.0, 6.0]
    analytic = math.sqrt(14) / 3
    assert stats.bootstrap_ci(
        x, n_resamples=200_000, seed=2
    ).standard_error == pytest.approx(analytic, rel=0.02)


def test_bootstrap_standard_error_uses_population_ddof():
    """The reported standard error uses ddof=0.

    The convergence tests above use a 2% tolerance, which can't tell ddof=0 from
    ddof=1 at 200,000 resamples. Here a small n_resamples makes the gap about 8%, and
    the same rng draws are reproduced to compute the expected value.

    """
    import numpy as np

    x = [1.0, 2.0, 6.0, 9.0]
    seed = 42
    n_resamples = 7
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    resample_means = np.array(x)[idx].mean(axis=1)
    population_se = resample_means.std(ddof=0)
    sample_se = resample_means.std(ddof=1)
    assert population_se != pytest.approx(sample_se)  # confirm they are distinguishable here

    result = stats.bootstrap_ci(x, n_resamples=n_resamples, seed=seed)
    assert result.standard_error == pytest.approx(population_se)


def test_bootstrap_percentile_bounds_bracket_the_point_estimate():
    result = stats.bootstrap_ci([1.0, 2.0, 6.0], n_resamples=50_000, seed=3)
    assert result.lower <= result.point <= result.upper
    assert result.level == 0.95


def test_bootstrap_percentile_bounds_match_the_exhaustive_distribution():
    """Percentile bounds for [1, 2, 6] converge to the extremes 1.0 and 6.0.

    Linear interpolation over the 27 exact resample means gives 1.2167 and 5.1333, but
    each extreme mean has probability 1/27 = 3.7%, which is more than 2.5%. So the true
    2.5th and 97.5th percentiles are the extremes themselves, and a large run lands on
    them. Using 5/95 percentiles instead would not land there, so that mistake is caught.

    """
    import numpy as np

    x = [1.0, 2.0, 6.0]
    means = sorted(sum(c) / 3 for c in itertools.product(x, repeat=3))
    lower, upper = np.percentile(means, [2.5, 97.5])
    assert (lower, upper) == pytest.approx((1.2166666666666666, 5.133333333333331))

    # For large n_resamples, the probability of each extreme mean, worked out directly.
    p_extreme = (1 / 3) ** 3
    assert p_extreme == pytest.approx(1 / 27)
    assert p_extreme > 0.025  # so the true 2.5th/97.5th percentiles are the extremes

    result = stats.bootstrap_ci(x, n_resamples=200_000, seed=9)
    assert result.lower == pytest.approx(1.0, abs=0.01)
    assert result.upper == pytest.approx(6.0, abs=0.01)


def test_bootstrap_bounds_are_interior_when_the_tail_mass_is_small():
    """Percentile bounds fall strictly inside the data range.

    The other bound tests all land on the sample min or max, so a stub returning
    min/max without resampling would pass them. For [1, 2, 6, 100] each extreme has
    probability (1/4)**4 = 0.39%, under the 2.5% tail, so the true bounds are interior:
    1.5 and 75.5 by exhaustive enumeration of all 256 resamples.

    """
    x = [1.0, 2.0, 6.0, 100.0]
    n = 4
    means = sorted(sum(c) / n for c in itertools.product(x, repeat=n))
    assert len(means) == 256

    def discrete_quantile(sorted_values, p):
        """Smallest x with F(x) >= p, computed without numpy's interpolation."""
        total = len(sorted_values)
        for i, value in enumerate(sorted_values):
            if (i + 1) / total >= p:
                return value
        return sorted_values[-1]

    exhaustive_lower = discrete_quantile(means, 0.025)
    exhaustive_upper = discrete_quantile(means, 0.975)
    assert (exhaustive_lower, exhaustive_upper) == pytest.approx((1.5, 75.5))

    p_extreme = (1 / 4) ** 4
    assert p_extreme == pytest.approx(1 / 256)
    assert p_extreme < 0.025  # so the true bounds are inside the range

    for seed in (1, 7, 20260908):
        result = stats.bootstrap_ci(x, n_resamples=200_000, seed=seed)
        assert result.point == pytest.approx(27.25)
        assert result.lower == pytest.approx(1.5, abs=0.1)
        assert result.upper == pytest.approx(75.5, abs=0.5)
        # A min/max stub would return exactly (1.0, 100.0), which these exclude.
        assert result.lower > min(x)
        assert result.upper < max(x)


def test_bootstrap_is_reproducible_for_a_given_seed():
    """The same seed gives the same interval, and different seeds give different ones.

    The seed comparison uses `standard_error` because for [1, 2, 6] the bounds almost
    always land on the extremes 1.0/6.0 whatever the seed.

    """
    a = stats.bootstrap_ci([1.0, 2.0, 6.0], n_resamples=5000, seed=7)
    b = stats.bootstrap_ci([1.0, 2.0, 6.0], n_resamples=5000, seed=7)
    assert (a.lower, a.upper, a.standard_error) == (b.lower, b.upper, b.standard_error)
    c = stats.bootstrap_ci([1.0, 2.0, 6.0], n_resamples=5000, seed=8)
    assert c.standard_error != a.standard_error


def test_bootstrap_resamples_clusters_not_observations():
    """Resampling is clustered on funder, so a cluster enters a resample whole.

    Identical clusters therefore give a zero-width interval, which resampling single
    observations would not.

    """
    result = stats.bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_resamples=1000, seed=4)
    assert result.standard_error == pytest.approx(0.0)
    assert result.lower == pytest.approx(0.5)
    assert result.upper == pytest.approx(0.5)


def test_bootstrap_accepts_an_arbitrary_statistic():
    """The bootstrap works with any statistic, here `max`.

    `point` and `upper` both equal max(x) = 9.0 either way, so `lower` is the useful
    check: all four draws landing on 1.0 has probability 1/256, under 2.5%, so a real
    bootstrap gives 2.0 rather than min(x) = 1.0.

    """
    result = stats.bootstrap_ci(
        [1.0, 2.0, 6.0, 9.0], statistic=lambda v: max(v), n_resamples=5000, seed=5
    )
    assert result.point == 9.0
    assert result.upper == 9.0
    assert result.lower == pytest.approx(2.0)


def test_bootstrap_records_the_cluster_count_beside_the_interval():
    """The cluster count is returned with the interval.

    16 clusters is few and cluster-bootstrap intervals are known to be too narrow then,
    so the count is kept next to the '95%'.

    """
    result = stats.bootstrap_ci([0.1 * i for i in range(16)], n_resamples=2000, seed=6)
    assert result.n_clusters == 16


def test_bootstrap_is_undefined_below_the_minimum_cluster_count():
    result = stats.bootstrap_ci([0.4, 0.6], n_resamples=1000, seed=1)
    assert result.undefined is True
    assert result.lower is None and result.upper is None
    assert "clusters" in result.reason


def test_bootstrap_refuses_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        stats.bootstrap_ci([], n_resamples=100, seed=1)
