import numpy as np
import pytest

from parallax_bench.metrics.stats import bootstrap_ci, holm_correction, paired_wilcoxon


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=100).tolist()
    ci1 = bootstrap_ci(values)
    ci2 = bootstrap_ci(values)
    assert ci1 == ci2  # seeded — reproducible
    assert ci1.lo <= ci1.mean <= ci1.hi
    assert ci1.n == 100


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([])
    assert ci.n == 0 and np.isnan(ci.mean)


def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.05, 50)
    res = paired_wilcoxon(a + 0.1, a)
    assert res.p_value < 0.001
    assert res.median_diff == pytest.approx(0.1, abs=0.01)


def test_wilcoxon_identical_samples():
    res = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res.p_value == 1.0


def test_wilcoxon_length_mismatch():
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0], [1.0, 2.0])


def test_holm_correction_known_example():
    # classic example: p = [0.01, 0.04, 0.03, 0.005], m = 4
    adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # monotone in the sorted order and capped at 1
    assert holm_correction([0.9, 0.95]) == pytest.approx([1.0, 1.0])
