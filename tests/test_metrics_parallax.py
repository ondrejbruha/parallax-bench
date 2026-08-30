import math

import pandas as pd
import pytest

from parallax_bench.metrics.parallax import (
    bootstrap_mean_parallax_ci,
    build_score_matrix,
    cross_lingual_penalty,
    directional_asymmetries,
    english_relative_delta,
    metric_slug,
    summarize_parallax,
)


@pytest.fixture()
def matrix():
    return pd.DataFrame([[1.0, 0.8], [0.7, 0.9]], index=["cs", "en"], columns=["cs", "en"])


def test_matrix_construction_and_diagonal_identification():
    aggregated = pd.DataFrame(
        [
            {"query_lang": "en", "index_lang": "cs", "metric": "ndcg@10", "mean": 0.7},
            {"query_lang": "cs", "index_lang": "en", "metric": "ndcg@10", "mean": 0.8},
            {"query_lang": "en", "index_lang": "en", "metric": "ndcg@10", "mean": 0.9},
            {"query_lang": "cs", "index_lang": "cs", "metric": "ndcg@10", "mean": 1.0},
        ]
    )
    result = build_score_matrix(aggregated, "ndcg@10", ["cs", "en"])
    assert result.loc["cs", "cs"] == 1.0
    assert result.loc["en", "en"] == 0.9
    assert list(result.index) == ["cs", "en"]


def test_cross_lingual_penalty(matrix):
    clp = cross_lingual_penalty(matrix)
    assert clp.loc["cs", "cs"] == 0.0
    assert clp.loc["en", "en"] == 0.0
    assert clp.loc["cs", "en"] == pytest.approx(-0.2)
    assert clp.loc["en", "cs"] == pytest.approx(-0.2)


def test_english_relative_delta(matrix):
    delta = english_relative_delta(matrix)
    assert delta.loc["en", "en"] == 0.0
    assert delta.loc["cs", "cs"] == pytest.approx(0.1)
    assert delta.loc["en", "cs"] == pytest.approx(-0.2)


def test_summary_metrics(matrix):
    summary = summarize_parallax(matrix, "ndcg@10")
    assert summary.mean_parallax == pytest.approx(-0.2)
    assert summary.parallax_rms == pytest.approx(0.2)
    assert summary.language_robustness_std == pytest.approx(0.1)
    assert summary.worst_language_gap == pytest.approx(0.3)
    assert summary.mean_row_gap == pytest.approx(0.2)
    assert summary.mean_directional_asymmetry == pytest.approx(0.0)
    assert summary.max_directional_asymmetry == pytest.approx(0.0)
    assert summary.max_asymmetry_pair == ["cs", "en"]


def test_directional_asymmetry():
    clp = pd.DataFrame(
        [[0.0, -0.04], [-0.13, 0.0]], index=["cs", "en"], columns=["cs", "en"]
    )
    result = directional_asymmetries(clp)
    assert result.to_dict("records") == [
        {"language_a": "cs", "language_b": "en", "asymmetry": pytest.approx(0.09)}
    ]


def test_missing_cell_is_preserved_and_ignored_in_available_summaries():
    aggregated = pd.DataFrame(
        [
            {"query_lang": "cs", "index_lang": "cs", "metric": "m", "mean": 1.0},
            {"query_lang": "en", "index_lang": "cs", "metric": "m", "mean": 0.7},
            {"query_lang": "en", "index_lang": "en", "metric": "m", "mean": 0.9},
        ]
    )
    matrix = build_score_matrix(aggregated, "m", ["cs", "en"])
    assert math.isnan(matrix.loc["cs", "en"])
    clp = cross_lingual_penalty(matrix)
    assert math.isnan(clp.loc["cs", "en"])
    assert summarize_parallax(matrix, "m").mean_parallax == pytest.approx(-0.2)


def test_origin_subset_and_grouped_bootstrap():
    rows = []
    values = {
        "q1": [[1.0, 0.8], [0.7, 0.9]],
        "q2": [[0.8, 0.6], [0.6, 0.8]],
    }
    for group, group_values in values.items():
        for q_pos, query_lang in enumerate(["cs", "en"]):
            for d_pos, index_lang in enumerate(["cs", "en"]):
                rows.append(
                    {
                        "query_group": group,
                        "query_lang": query_lang,
                        "index_lang": index_lang,
                        "metric": "ndcg@10",
                        "value": group_values[q_pos][d_pos],
                        "origin": "translated" if group == "q1" else "native",
                    }
                )
    per_query = pd.DataFrame(rows)
    translated = per_query[per_query.origin == "translated"]
    ci = bootstrap_mean_parallax_ci(
        translated, "ndcg@10", ["cs", "en"], n_resamples=50
    )
    assert ci[0] == pytest.approx(-0.2)
    assert ci[1] == pytest.approx(-0.2)


def test_metric_slug():
    assert metric_slug("nDCG@10") == "ndcg_at_10"
