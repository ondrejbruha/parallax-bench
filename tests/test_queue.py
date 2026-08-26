"""Queue semantics: claim/complete/fail, resumability, missing-rate accounting."""

from sqlalchemy.orm import Session

from parallax_bench.runner import queue as q


def _engine(tmp_path):
    return q.get_engine(f"sqlite:///{tmp_path}/test.db")


def _make_run(session, n_tasks=3, kind="retrieval"):
    tasks = [
        {
            "kind": kind,
            "query_id": f"q0000{i}_cs",
            "query_lang": "cs",
            "target_index": "bench_cs",
            "k": 100,
        }
        for i in range(n_tasks)
    ]
    return q.create_run(session, "test-run", {"data_version": "smoke"}, {}, None, tasks)


def test_claim_complete_cycle(tmp_path):
    engine = _engine(tmp_path)
    with Session(engine) as session:
        run = _make_run(session)
        seen = []
        while (task := q.claim_next(session, run.id, "retrieval")) is not None:
            seen.append(task.query_id)
            q.store_retrieval(session, task, ["32016R0679"])
        assert len(seen) == 3
        assert q.run_progress(session, run.id) == {"done": 3}


def test_failed_task_retries_then_fails_permanently(tmp_path):
    engine = _engine(tmp_path)
    with Session(engine) as session:
        run = _make_run(session, n_tasks=1)
        for attempt in range(3):
            task = q.claim_next(session, run.id, "retrieval", max_attempts=3)
            assert task is not None, f"attempt {attempt} should be claimable"
            q.fail_task(session, task, "boom", max_attempts=3)
        # attempts exhausted → permanently failed, run continues without it
        assert q.claim_next(session, run.id, "retrieval", max_attempts=3) is None
        assert q.run_progress(session, run.id) == {"failed": 1}


def test_resume_after_interruption(tmp_path):
    engine = _engine(tmp_path)
    with Session(engine) as session:
        run = _make_run(session, n_tasks=2)
        task = q.claim_next(session, run.id, "retrieval")
        q.store_retrieval(session, task, ["32016R0679"])
        run_id = run.id
    # simulated restart: a new session picks up exactly the remaining task
    with Session(engine) as session:
        task = q.claim_next(session, run_id, "retrieval")
        assert task is not None
        remaining_after = q.claim_next(session, run_id, "retrieval")
        assert remaining_after is None  # the claimed one is locked, nothing else pending
        q.store_retrieval(session, task, ["32016R0679"])
        assert q.run_progress(session, run_id) == {"done": 2}


def test_store_retrieval_is_idempotent_per_task(tmp_path):
    engine = _engine(tmp_path)
    with Session(engine) as session:
        run = _make_run(session, n_tasks=1)
        task = q.claim_next(session, run.id, "retrieval")
        q.store_retrieval(session, task, ["A", "B"])
        # re-claim after forced reset (e.g. operator retry) must not duplicate rows
        task.status = "pending"
        session.commit()
        task = q.claim_next(session, run.id, "retrieval")
        q.store_retrieval(session, task, ["A", "B"])
        rows = session.query(q.RetrievalResult).filter_by(task_id=task.id).all()
        assert [(r.rank, r.doc_id) for r in rows] == [(1, "A"), (2, "B")]
