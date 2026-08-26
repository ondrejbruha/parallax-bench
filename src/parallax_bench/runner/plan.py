"""Task planning: expand a dataset into the full task list up front.

Retrieval (L1): the full matrix — every query variant against every language
index.  The diagonal is monoRAG, off-diagonal cells are cross-lingual
retrieval.  Rerank on/off is *not* planned here: ablations are separate
configured systems, so they are separate runs.

Generation (L2): an order of magnitude smaller slice — diagonal, one
query-language cut against all indexes, and MultiRAG — times the requested
modes.
"""

from __future__ import annotations

from parallax_bench.data import Dataset
from parallax_bench.runner.ingest import index_name

MULTI_SEPARATOR = ","


def plan_retrieval(ds: Dataset, prefix: str = "bench_", k: int = 100) -> list[dict]:
    tasks = []
    for q in ds.queries:
        for index_lang in ds.languages:
            tasks.append(
                {
                    "kind": "retrieval",
                    "query_id": q.query_id,
                    "query_lang": q.lang,
                    "target_index": index_name(prefix, index_lang),
                    "k": k,
                }
            )
    return tasks


def plan_generation(
    ds: Dataset,
    prefix: str = "bench_",
    modes: list[str] | None = None,
    cut_lang: str | None = "cs",
    include_multi: bool = True,
    k: int = 10,
) -> list[dict]:
    modes = modes or ["direct"]
    all_indexes = MULTI_SEPARATOR.join(index_name(prefix, lang) for lang in ds.languages)
    tasks = []
    for q in ds.queries:
        targets = {index_name(prefix, q.lang)}  # diagonal: query lang == index lang
        if cut_lang and q.lang == cut_lang:
            targets.update(index_name(prefix, lang) for lang in ds.languages)
        for mode in modes:
            for target in sorted(targets):
                tasks.append(
                    {
                        "kind": "generation",
                        "query_id": q.query_id,
                        "query_lang": q.lang,
                        "target_index": target,
                        "mode": mode,
                        "k": k,
                    }
                )
            if include_multi:
                tasks.append(
                    {
                        "kind": "generation",
                        "query_id": q.query_id,
                        "query_lang": q.lang,
                        "target_index": all_indexes,
                        "mode": mode,
                        "k": k,
                    }
                )
    return tasks
