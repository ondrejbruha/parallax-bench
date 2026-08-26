"""Statistical machinery: bootstrap CIs, paired Wilcoxon, Holm correction.

Language variants of the same ``query_group`` are paired observations — that
is the entire reason ``query_group`` is mandatory in the data format.  Without
pairing only much weaker unpaired tests remain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI of the mean over queries."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        mean=float(arr.mean()),
        lo=float(np.quantile(means, alpha)),
        hi=float(np.quantile(means, 1.0 - alpha)),
        n=int(arr.size),
    )


@dataclass(frozen=True)
class PairedTest:
    statistic: float
    p_value: float
    n_pairs: int
    median_diff: float


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> PairedTest:
    """Wilcoxon signed-rank test on paired per-query metric values.

    ``a`` and ``b`` must be aligned by query_group (same order).  All-zero
    differences (identical performance) yield p = 1.0 rather than an error.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired samples differ in length: {x.shape} vs {y.shape}")
    diffs = x - y
    if np.all(diffs == 0):
        return PairedTest(statistic=0.0, p_value=1.0, n_pairs=int(x.size), median_diff=0.0)
    res = sps.wilcoxon(x, y, zero_method="wilcox")
    return PairedTest(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_pairs=int(x.size),
        median_diff=float(np.median(diffs)),
    )


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction; returns adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted
