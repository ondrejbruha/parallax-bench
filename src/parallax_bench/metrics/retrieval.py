"""Retrieval metrics over TREC qrels at document granularity.

Definitions follow ``trec_eval`` exactly (linear gain nDCG, log2 discount)
so results are reproducible with ``pytrec_eval`` / ``ir_measures`` — the test
suite asserts agreement with ``pytrec_eval``.

Inputs are plain dicts: ``ranking`` is a deduplicated list of doc_ids in rank
order, ``rels`` maps doc_id -> graded relevance for one query.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Ranking = Sequence[str]
Rels = Mapping[str, int]


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    gains = [float(rels.get(d, 0)) for d in ranking[:k]]
    ideal = sorted((float(r) for r in rels.values() if r > 0), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def recall_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranking[:k] if d in relevant)
    return hit / len(relevant)


def mrr_at_k(ranking: Ranking, rels: Rels, k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def success_at_1(ranking: Ranking, rels: Rels) -> float:
    return 1.0 if ranking and rels.get(ranking[0], 0) > 0 else 0.0


DEFAULT_METRICS = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "mrr@100": lambda r, q: mrr_at_k(r, q, 100),
    "success@1": lambda r, q: success_at_1(r, q),
}


def score_run(
    rankings: Mapping[str, Ranking],
    qrels: Mapping[str, Rels],
    metrics: Mapping[str, object] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-query metric values: ``{query_id: {metric: value}}``.

    Queries present in qrels but missing from ``rankings`` score 0 on every
    metric — a missing measurement is a result, not a gap to silently drop.
    """
    metrics = metrics or DEFAULT_METRICS
    out: dict[str, dict[str, float]] = {}
    for query_id, rels in qrels.items():
        ranking = rankings.get(query_id, [])
        out[query_id] = {name: fn(ranking, rels) for name, fn in metrics.items()}  # type: ignore[operator]
    return out


def aggregate(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    names = next(iter(per_query.values())).keys()
    n = len(per_query)
    return {name: sum(v[name] for v in per_query.values()) / n for name in names}
