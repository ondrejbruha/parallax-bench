import json

import pandas as pd
import pytest

pytest.importorskip("matplotlib", reason="heatmap dependency is not installed")

from parallax_bench.reporting import comparison_table, write_heatmaps


def _scored_run(path, system, values):
    matrices = path / "matrices"
    matrices.mkdir(parents=True)
    matrix = pd.DataFrame(values, index=["cs", "en"], columns=["cs", "en"])
    matrix.to_csv(matrices / "ndcg_at_10.csv", index_label="query_lang")
    clp = matrix.subtract(pd.Series([values[0][0], values[1][1]], index=matrix.index), axis=0)
    clp.to_csv(matrices / "ndcg_at_10_clp.csv", index_label="query_lang")
    summary = pd.DataFrame(
        [
            {
                "metric": "ndcg@10",
                "origin": "all",
                "mean_parallax": -0.2,
                "parallax_rms": 0.2,
                "worst_language_gap": max(map(max, values)) - min(map(min, values)),
            }
        ]
    )
    summary.to_csv(path / "parallax_summary.csv", index=False)
    (path / "config.json").write_text(json.dumps({"system_id": system}), encoding="utf-8")


def test_heatmaps_and_comparison_are_created_from_scored_files(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _scored_run(first, "system-a", [[1.0, 0.8], [0.7, 0.9]])
    _scored_run(second, "system-b", [[0.9, 0.8], [0.8, 0.9]])

    absolute, parallax = write_heatmaps(
        first, "ndcg@10", system="system-a", data_version="v1"
    )
    assert absolute.is_file() and absolute.stat().st_size > 0
    assert parallax.is_file() and parallax.stat().st_size > 0

    comparison = comparison_table(
        [("system-a", first), ("system-b", second)], "ndcg@10"
    )
    assert comparison["system"].tolist() == ["system-a", "system-b"]
    assert comparison.loc[0, "ndcg@10"] == pytest.approx(0.85)
