"""Database-backed task queue — the reason a run survives interruption.

All tasks are generated up front when a run is created.  Progress is then
trivially observable (``SELECT status, count(*) ... GROUP BY status``) and a
restart resumes exactly where it stopped, with no loss and no duplication.

SQLite by default (the smoke quickstart needs no Postgres); any SQLAlchemy
URL works — Heroku Postgres for real runs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
import uuid
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    status: Mapped[str] = mapped_column(String(32), default="created")
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    system_json: Mapped[dict] = mapped_column(JSON, default=dict)
    lock_json: Mapped[list] = mapped_column(JSON, default=list)  # manifest.lock entries
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)

    tasks: Mapped[list[Task]] = relationship(back_populates="run")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # 'retrieval' | 'generation'
    query_id: Mapped[str] = mapped_column(String(64))
    query_lang: Mapped[str] = mapped_column(String(8))
    target_index: Mapped[str] = mapped_column(String(255))  # generation: comma-joined list
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True)  # generation only
    k: Mapped[int] = mapped_column(Integer, default=100)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="tasks")


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    doc_id: Mapped[str] = mapped_column(String(64))


class GenerationResult(Base):
    __tablename__ = "generation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), index=True)
    answer: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    retrieved_json: Mapped[str] = mapped_column(Text)   # list[str]
    cited_json: Mapped[str] = mapped_column(Text)       # list[str]
    detected_lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    raw_json: Mapped[str] = mapped_column(Text, default="{}")


def default_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        # Heroku still hands out postgres:// which SQLAlchemy 2 refuses
        return url.replace("postgres://", "postgresql://", 1)
    Path(".parallax").mkdir(exist_ok=True)
    return "sqlite:///.parallax/parallax.db"


def get_engine(url: str | None = None):
    engine = create_engine(url or default_db_url())
    Base.metadata.create_all(engine)
    return engine


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def create_run(
    session: Session,
    name: str,
    config: dict,
    system_meta: dict,
    git_sha: str | None,
    tasks: list[dict],
    lock: list[dict] | None = None,
) -> Run:
    run = Run(
        id=uuid.uuid4().hex[:12],
        name=name,
        status="created",
        config_json=config,
        system_json=system_meta,
        lock_json=lock or [],
        git_sha=git_sha,
    )
    session.add(run)
    session.flush()
    session.bulk_insert_mappings(Task, [{**t, "run_id": run.id} for t in tasks])  # type: ignore[arg-type]
    session.commit()
    return run


def claim_next(session: Session, run_id: str, kind: str, max_attempts: int = 3) -> Task | None:
    """Claim one pending task; portable across SQLite and Postgres.

    A task stuck in 'running' for over an hour is considered orphaned by a
    dead worker and is reclaimable — that is what makes restarts seamless.
    """
    stale = _utcnow() - dt.timedelta(hours=1)
    stmt = (
        select(Task)
        .where(
            Task.run_id == run_id,
            Task.kind == kind,
            Task.attempts < max_attempts,
            (Task.status == "pending")
            | ((Task.status == "running") & (Task.locked_at < stale)),
        )
        .order_by(Task.id)
        .limit(1)
    )
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    task = session.execute(stmt).scalar_one_or_none()
    if task is None:
        return None
    task.status = "running"
    task.attempts += 1
    task.locked_at = _utcnow()
    task.locked_by = worker_id()
    session.commit()
    return task


def complete_task(session: Session, task: Task) -> None:
    task.status = "done"
    task.error = None
    session.commit()


def fail_task(session: Session, task: Task, error: str, max_attempts: int = 3) -> None:
    """After exhausting attempts the task goes to 'failed' but the run goes on.

    Missing measurements are reported as a missing rate — which is itself a
    result, not noise to hide.
    """
    task.status = "failed" if task.attempts >= max_attempts else "pending"
    task.error = error[:4000]
    session.commit()


def run_progress(session: Session, run_id: str) -> dict[str, int]:
    rows = session.execute(
        select(Task.status, func.count()).where(Task.run_id == run_id).group_by(Task.status)
    ).all()
    return {status: count for status, count in rows}


def store_retrieval(session: Session, task: Task, doc_ids: list[str]) -> None:
    session.query(RetrievalResult).filter_by(task_id=task.id).delete()
    session.add_all(
        [RetrievalResult(task_id=task.id, rank=i + 1, doc_id=d) for i, d in enumerate(doc_ids)]
    )
    complete_task(session, task)


def store_generation(
    session: Session,
    task: Task,
    answer_text: str,
    retrieved: list[str],
    cited: list[str],
    latency_ms: int,
    detected_lang: str | None,
    raw: dict,
) -> None:
    session.query(GenerationResult).filter_by(task_id=task.id).delete()
    session.add(
        GenerationResult(
            task_id=task.id,
            answer=answer_text,
            latency_ms=latency_ms,
            retrieved_json=json.dumps(retrieved),
            cited_json=json.dumps(cited),
            detected_lang=detected_lang,
            raw_json=json.dumps(raw, ensure_ascii=False, default=str),
        )
    )
    complete_task(session, task)
