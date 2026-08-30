"""Derived Parallax metrics over language-by-language retrieval matrices.

Rows are query languages and columns are document/index languages.  These
metrics describe language sensitivity; they complement rather than replace
the absolute IR metrics in :mod:`parallax_bench.metrics.retrieval`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParallaxSummary:
    metric: str
    origin: str
    mean_parallax: float
    mean_parallax_ci_lo: float
    mean_parallax_ci_hi: float
    parallax_rms: float
    language_robustness_std: float
    worst_language_gap: float
    mean_row_gap: float
    mean_directional_asymmetry: float
    max_directional_asymmetry: float
    max_asymmetry_pair: list[str] | None

    def as_dict(self) -> dict:
        return asdict(self)


def metric_slug(metric: str) -> str:
    """Stable, filesystem-safe metric name (for example ``ndcg_at_10``)."""
    return metric.lower().replace("@", "_at_").replace("/", "_per_").replace(" ", "_")


def build_score_matrix(
    aggregated: pd.DataFrame,
    metric: str,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """Build an ordered X×X absolute matrix; unavailable cells remain NaN."""
    required = {"query_lang", "index_lang", "metric", "mean"}
    missing = required - set(aggregated.columns)
    if missing:
        raise ValueError(f"aggregated metrics missing columns: {', '.join(sorted(missing))}")
    sub = aggregated[aggregated["metric"] == metric]
    if languages is None:
        languages = sorted(set(sub["query_lang"]) | set(sub["index_lang"]))
    if sub.duplicated(["query_lang", "index_lang"]).any():
        raise ValueError(f"duplicate cells for metric {metric!r}")
    return sub.pivot(index="query_lang", columns="index_lang", values="mean").reindex(
        index=languages, columns=languages
    )


def cross_lingual_penalty(matrix: pd.DataFrame) -> pd.DataFrame:
    """CLP(q,d) = S(q,d) - S(q,q), with an exact zero diagonal."""
    _validate_square(matrix)
    out = matrix.copy().astype(float)
    for lang in out.index:
        diagonal = out.loc[lang, lang]
        out.loc[lang] = out.loc[lang] - diagonal
        if pd.notna(diagonal):
            out.loc[lang, lang] = 0.0
    return out


def english_relative_delta(matrix: pd.DataFrame, baseline_lang: str = "en") -> pd.DataFrame:
    """EN_DELTA(q,d) = S(q,d) - S(en,en)."""
    _validate_square(matrix)
    if baseline_lang not in matrix.index:
        raise ValueError(f"baseline language {baseline_lang!r} is not in the matrix")
    baseline = matrix.loc[baseline_lang, baseline_lang]
    return matrix.astype(float) - baseline


def directional_asymmetries(clp: pd.DataFrame) -> pd.DataFrame:
    """One deterministic row per unordered pair: abs(CLP(a,b) - CLP(b,a))."""
    _validate_square(clp)
    rows = []
    for a, b in combinations(clp.index, 2):
        left, right = clp.loc[a, b], clp.loc[b, a]
        value = abs(float(left) - float(right)) if pd.notna(left) and pd.notna(right) else np.nan
        rows.append({"language_a": a, "language_b": b, "asymmetry": value})
    return pd.DataFrame(rows, columns=["language_a", "language_b", "asymmetry"])


def summarize_parallax(
    matrix: pd.DataFrame,
    metric: str,
    origin: str = "all",
    mean_parallax_ci: tuple[float, float] = (math.nan, math.nan),
) -> ParallaxSummary:
    """Compute global language-sensitivity summaries, ignoring unavailable cells."""
    clp = cross_lingual_penalty(matrix)
    off_diagonal = _off_diagonal_values(clp)
    finite_off = off_diagonal[np.isfinite(off_diagonal)]
    matrix_values = matrix.to_numpy(dtype=float)
    finite_matrix = matrix_values[np.isfinite(matrix_values)]
    row_gaps = []
    row_stds = []
    for _, row in matrix.iterrows():
        values = row.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            row_gaps.append(float(values.max() - values.min()))
            row_stds.append(float(values.std(ddof=0)))
    asym = directional_asymmetries(clp)
    finite_asym = asym.dropna(subset=["asymmetry"])
    max_pair = None
    max_asymmetry = math.nan
    if not finite_asym.empty:
        maximum = finite_asym.sort_values(
            ["asymmetry", "language_a", "language_b"], ascending=[False, True, True]
        ).iloc[0]
        max_asymmetry = float(maximum["asymmetry"])
        max_pair = [str(maximum["language_a"]), str(maximum["language_b"])]
    return ParallaxSummary(
        metric=metric,
        origin=origin,
        mean_parallax=_mean_or_nan(finite_off),
        mean_parallax_ci_lo=float(mean_parallax_ci[0]),
        mean_parallax_ci_hi=float(mean_parallax_ci[1]),
        parallax_rms=(
            float(np.sqrt(np.mean(np.square(finite_off)))) if finite_off.size else math.nan
        ),
        language_robustness_std=_mean_or_nan(np.asarray(row_stds)),
        worst_language_gap=(
            float(finite_matrix.max() - finite_matrix.min()) if finite_matrix.size else math.nan
        ),
        mean_row_gap=_mean_or_nan(np.asarray(row_gaps)),
        mean_directional_asymmetry=(
            float(finite_asym["asymmetry"].mean()) if not finite_asym.empty else math.nan
        ),
        max_directional_asymmetry=max_asymmetry,
        max_asymmetry_pair=max_pair,
    )


def bootstrap_mean_parallax_ci(
    per_query: pd.DataFrame,
    metric: str,
    languages: list[str],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Grouped bootstrap CI for mean CLP, resampling whole query groups.

    Every resample retains all language variants and matrix cells belonging to
    a selected ``query_group``.  It never treats translated variants as
    independent observations.
    """
    required = {"query_group", "query_lang", "index_lang", "metric", "value"}
    missing = required - set(per_query.columns)
    if missing:
        raise ValueError(f"per-query metrics missing columns: {', '.join(sorted(missing))}")
    sub = per_query[per_query["metric"] == metric].dropna(subset=["query_group"])
    groups = sorted(sub["query_group"].unique())
    if not groups:
        return math.nan, math.nan
    grouped = (
        sub.groupby(["query_group", "query_lang", "index_lang"], sort=True)["value"]
        .mean()
        .reset_index()
    )
    positions = {lang: pos for pos, lang in enumerate(languages)}
    cube = np.full((len(groups), len(languages), len(languages)), np.nan)
    group_positions = {group: pos for pos, group in enumerate(groups)}
    for row in grouped.itertuples(index=False):
        if row.query_lang in positions and row.index_lang in positions:
            cube[
                group_positions[row.query_group],
                positions[row.query_lang],
                positions[row.index_lang],
            ] = row.value
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        with np.errstate(invalid="ignore"):
            absolute = np.nanmean(cube[chosen], axis=0)
        clp = absolute - np.diag(absolute)[:, None]
        np.fill_diagonal(clp, 0.0)
        values = clp[~np.eye(len(languages), dtype=bool)]
        values = values[np.isfinite(values)]
        if values.size:
            samples.append(float(values.mean()))
    if not samples:
        return math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _validate_square(matrix: pd.DataFrame) -> None:
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("matrix rows and columns must contain the same ordered languages")


def _off_diagonal_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    return values[~np.eye(len(matrix), dtype=bool)]


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else math.nan
