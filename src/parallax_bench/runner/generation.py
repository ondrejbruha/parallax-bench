"""Generation phase worker: one task = one generate() call, stored verbatim."""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from parallax_bench.adapters.base import RagSystem
from parallax_bench.data import Dataset
from parallax_bench.metrics.generation import detect_language
from parallax_bench.runner import queue as q
from parallax_bench.runner.plan import MULTI_SEPARATOR


def run_generation(
    session: Session,
    system: RagSystem,
    ds: Dataset,
    run_id: str,
    max_attempts: int = 3,
    progress_every: int = 20,
    log=print,
) -> dict[str, int]:
    queries = {query.query_id: query for query in ds.queries}
    candidates = ds.languages
    done = 0
    while (task := q.claim_next(session, run_id, "generation", max_attempts)) is not None:
        query = queries.get(task.query_id)
        if query is None:
            q.fail_task(session, task, f"query_id {task.query_id!r} not in dataset", max_attempts)
            continue
        try:
            index_names = task.target_index.split(MULTI_SEPARATOR)
            t0 = time.monotonic()
            answer = system.generate(query.text, index_names, task.mode or "direct")
            latency_ms = int((time.monotonic() - t0) * 1000)
            q.store_generation(
                session,
                task,
                answer_text=answer.text,
                retrieved=answer.retrieved_doc_ids,
                cited=answer.cited_doc_ids,
                latency_ms=latency_ms,
                detected_lang=detect_language(answer.text, candidates),
                raw=answer.raw,
            )
            done += 1
            if progress_every and done % progress_every == 0:
                log(f"  generation: {done} tasks done")
        except Exception as exc:  # noqa: BLE001
            q.fail_task(session, task, f"{type(exc).__name__}: {exc}", max_attempts)
    return q.run_progress(session, run_id)
