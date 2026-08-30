"""Human-readable tables and deterministic heatmaps from scored run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from parallax_bench.metrics.parallax import metric_slug

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def load_matrix(out_dir: Path, metric: str, kind: str, origin: str = "all") -> pd.DataFrame:
    slug = metric_slug(metric)
    suffix = {"absolute": "", "clp": "_clp", "en_delta": "_en_delta"}[kind]
    root = out_dir / "matrices"
    if origin != "all":
        root = root / f"origin_{origin}"
    path = root / f"{slug}{suffix}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, index_col="query_lang")


def write_heatmaps(
    out_dir: Path,
    metric: str,
    *,
    system: str,
    data_version: str,
    origin: str = "all",
) -> tuple[Path, Path]:
    """Render absolute and CLP PNG heatmaps from precomputed matrix CSVs."""
    matrix_dir = out_dir / "matrices"
    if origin != "all":
        matrix_dir = matrix_dir / f"origin_{origin}"
    absolute = load_matrix(out_dir, metric, "absolute", origin)
    clp = load_matrix(out_dir, metric, "clp", origin)
    slug = metric_slug(metric)
    absolute_path = matrix_dir / f"absolute_{slug}_matrix.png"
    parallax_path = matrix_dir / f"parallax_{slug}_matrix.png"
    origin_label = "all queries" if origin == "all" else f"{origin} queries"
    _draw_heatmap(
        absolute,
        absolute_path,
        title=f"{system} — {metric} absolute — {data_version} ({origin_label})",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    finite = clp.to_numpy(dtype=float)
    finite = finite[pd.notna(finite)]
    extent = (
        max(abs(float(finite.min())), abs(float(finite.max()))) if finite.size else 1.0
    )
    if extent == 0.0:
        extent = 1.0
    _draw_heatmap(
        clp,
        parallax_path,
        title=f"{system} — {metric} Cross-Lingual Penalty — {data_version} ({origin_label})",
        cmap="RdBu",
        vmin=-extent,
        vmax=extent,
    )
    return absolute_path, parallax_path


def comparison_table(
    runs: list[tuple[str, Path]], metric: str, origin: str = "all"
) -> pd.DataFrame:
    """Combine already-scored run summaries without executing retrieval or scoring."""
    rows = []
    for system, out_dir in runs:
        path = out_dir / "parallax_summary.csv"
        if not path.is_file():
            continue
        summary = pd.read_csv(path)
        selected = summary[(summary["metric"] == metric) & (summary["origin"] == origin)]
        if selected.empty:
            continue
        item = selected.iloc[0]
        absolute = load_matrix(out_dir, metric, "absolute", origin)
        rows.append(
            {
                "system": system,
                metric: float(absolute.stack().mean()),
                "mean_parallax": item["mean_parallax"],
                "parallax_rms": item["parallax_rms"],
                "worst_language_gap": item["worst_language_gap"],
            }
        )
    return pd.DataFrame(rows)


def _draw_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    width = max(6.0, 0.85 * len(matrix.columns) + 2.5)
    height = max(5.0, 0.75 * len(matrix.index) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Document / index language")
    ax.set_ylabel("Query language")
    ax.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.3f}"
            ax.text(column, row, label, ha="center", va="center", color="black", fontsize=8)
            if matrix.index[row] == matrix.columns[column]:
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.0,
                    )
                )
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=160, metadata={"Software": "parallax-bench"})
    plt.close(fig)
