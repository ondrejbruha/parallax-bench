"""Our metric implementations must agree with pytrec_eval (trec_eval) exactly."""

import random

import pytest

from parallax_bench.metrics.retrieval import (
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_run,
    success_at_1,
)

pytrec_eval = pytest.importorskip("pytrec_eval")


def _random_case(seed: int, graded: bool):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(50)]
    qrels = {}
    runs = {}
    for qi in range(20):
        qid = f"q{qi}"
        relevant = rng.sample(docs, rng.randint(1, 5))
        qrels[qid] = {d: (rng.randint(1, 3) if graded else 1) for d in relevant}
        ranked = rng.sample(docs, rng.randint(5, 30))
        runs[qid] = ranked
    return qrels, runs


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("graded", [False, True])
def test_agreement_with_pytrec_eval(seed, graded):
    qrels, runs = _random_case(seed, graded)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "recall_10", "recall_100", "recip_rank", "success_1"}
    )
    trec_run = {
        qid: {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        for qid, ranked in runs.items()
    }
    reference = evaluator.evaluate(trec_run)
    for qid, ranked in runs.items():
        rels = qrels[qid]
        assert ndcg_at_k(ranked, rels, 10) == pytest.approx(reference[qid]["ndcg_cut_10"])
        assert recall_at_k(ranked, rels, 10) == pytest.approx(reference[qid]["recall_10"])
        assert recall_at_k(ranked, rels, 100) == pytest.approx(reference[qid]["recall_100"])
        assert mrr_at_k(ranked, rels, 100) == pytest.approx(reference[qid]["recip_rank"])
        assert success_at_1(ranked, rels) == pytest.approx(reference[qid]["success_1"])


def test_missing_query_scores_zero():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    per_query = score_run({"q1": ["d1"]}, qrels)
    assert per_query["q1"]["ndcg@10"] == 1.0
    assert per_query["q2"]["ndcg@10"] == 0.0
    assert set(per_query) == {"q1", "q2"}


def test_perfect_and_empty_ranking():
    rels = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b"], rels, 10) == pytest.approx(1.0)
    assert ndcg_at_k([], rels, 10) == 0.0
    assert recall_at_k([], rels, 10) == 0.0
    assert success_at_1([], rels) == 0.0
