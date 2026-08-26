"""Retrieval phase worker: drain the queue, store full rankings.

Never evaluate during the run — metrics are computed later over the finished
table (``score``), so they can be recomputed under a different definition
without repeating a forty-hour run.
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from parallax_bench.adapters.base import RagSystem
from parallax_bench.data import Dataset
from parallax_bench.runner import queue as q


def run_retrieval(
    session: Session,
    system: RagSystem,
    ds: Dataset,
    run_id: str,
    max_attempts: int = 3,
    progress_every: int = 50,
    log=print,
) -> dict[str, int]:
    queries = {query.query_id: query for query in ds.queries}
    done = 0
    started = time.monotonic()
    while (task := q.claim_next(session, run_id, "retrieval", max_attempts)) is not None:
        query = queries.get(task.query_id)
        if query is None:
            q.fail_task(session, task, f"query_id {task.query_id!r} not in dataset", max_attempts)
            continue
        try:
            doc_ids = system.search(query.text, task.target_index, task.k)
            q.store_retrieval(session, task, doc_ids)
            done += 1
            if progress_every and done % progress_every == 0:
                rate = done / (time.monotonic() - started)
                log(f"  retrieval: {done} tasks done ({rate:.1f}/s)")
        except Exception as exc:  # noqa: BLE001 — a failing task must not kill the run
            q.fail_task(session, task, f"{type(exc).__name__}: {exc}", max_attempts)
    return q.run_progress(session, run_id)
