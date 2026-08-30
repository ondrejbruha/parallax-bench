"""parallax-bench CLI.

The quickstart contract (§6 of the methodology): ``parallax-bench run
--system baseline-local --subset smoke`` works offline, without GPU, API keys
or Postgres, in under two minutes.  CI runs exactly that command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from parallax_bench import __version__
from parallax_bench.config import get_system, redacted_config
from parallax_bench.data import load_dataset, validate_dataset
from parallax_bench.runner import queue as q
from parallax_bench.runner.generation import run_generation
from parallax_bench.runner.ingest import ingest as do_ingest
from parallax_bench.runner.ingest import parity_warnings
from parallax_bench.runner.plan import plan_generation, plan_retrieval
from parallax_bench.runner.retrieval import run_retrieval

app = typer.Typer(
    name="parallax-bench",
    help="Measuring language-induced retrieval displacement in RAG systems.",
    no_args_is_help=True,
)

_SUBSET = typer.Option("smoke", "--subset", "--data-version", help="Data version (smoke, v1, …)")
_DATA_DIR = typer.Option(None, "--data-dir", help="Override benchmark/ data directory")
_SYSTEMS_TOML = typer.Option(None, "--systems", help="Path to systems.toml")
_DB = typer.Option(None, "--db", help="SQLAlchemy database URL (default: $DATABASE_URL or SQLite)")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — not a git checkout after pip install
        return None


def _resolve_run(session: Session, run_id: str | None) -> q.Run:
    if run_id:
        run = session.get(q.Run, run_id)
        if run is None:
            run = session.execute(
                select(q.Run).where(q.Run.name == run_id).order_by(q.Run.created_at.desc())
            ).scalars().first()
        if run is None:
            raise typer.BadParameter(f"run {run_id!r} not found")
        return run
    run = session.execute(select(q.Run).order_by(q.Run.created_at.desc())).scalars().first()
    if run is None:
        raise typer.BadParameter("no runs in the database yet")
    return run


@app.callback()
def _main() -> None:
    """parallax-bench — the PARALLAX benchmark CLI."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def validate(
    subset: str = typer.Option("all", "--subset", "--data-version"),
    data_dir: Path | None = _DATA_DIR,
) -> None:
    """Dataset integrity checks (runs in CI on every PR)."""
    versions = ["smoke", "v1"] if subset == "all" else [subset]
    failed = False
    for ver in versions:
        try:
            ds = load_dataset(ver, data_dir)
        except FileNotFoundError:
            typer.echo(f"[{ver}] not present — skipped")
            continue
        rep = validate_dataset(ds)
        for w in rep.warnings:
            typer.echo(f"[{ver}] warning: {w}")
        for e in rep.errors:
            typer.echo(f"[{ver}] ERROR: {e}", err=True)
        n_groups = len(ds.query_groups)
        typer.echo(
            f"[{ver}] {'OK' if rep.ok else 'FAILED'} — {len(ds.queries)} queries "
            f"in {n_groups} groups × {len(ds.languages)} languages "
            f"({', '.join(ds.languages)}), {len(ds.qrels)} qrels, "
            f"{len(ds.manifest)} manifest entries"
        )
        failed = failed or not rep.ok
    if failed:
        raise typer.Exit(code=1)


@app.command()
def fetch(
    subset: str = typer.Option("v1", "--subset", "--data-version"),
    data_dir: Path | None = _DATA_DIR,
) -> None:
    """Download corpus texts per the manifest and verify sha256."""
    from parallax_bench.fetch import fetch_corpus

    ds = load_dataset(subset, data_dir)
    if not ds.manifest:
        typer.echo(f"[{subset}] has no manifest — nothing to fetch")
        raise typer.Exit(code=1)
    n_ok, failures = fetch_corpus(ds, log=typer.echo)
    typer.echo(f"[{subset}] {n_ok}/{len(ds.manifest)} documents present and verified")
    for f in failures:
        typer.echo(f"ERROR: {f}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def verify(
    subset: str = typer.Option("v1", "--subset", "--data-version"),
    data_dir: Path | None = _DATA_DIR,
) -> None:
    """Re-download per the manifest and report source drift. Fixes nothing.

    Source drift is a finding ("the corpus drifted by X % in N months"),
    not an error to repair. Run once when the manifest is made and again
    before submission.
    """
    from parallax_bench.fetch import verify_corpus

    ds = load_dataset(subset, data_dir)
    if not ds.manifest:
        typer.echo(f"[{subset}] has no manifest — nothing to verify")
        raise typer.Exit(code=1)
    report = verify_corpus(ds, log=typer.echo)
    typer.echo(
        f"[{subset}] {report.n_unchanged}/{report.n_checked} unchanged, "
        f"{len(report.drifted)} drifted ({report.drift_rate:.1%}), "
        f"{len(report.unreachable)} unreachable"
    )
    for entry in report.unreachable:
        typer.echo(f"  unreachable: {entry}", err=True)


@app.command()
def ingest(
    system: str = typer.Option(..., "--system"),
    subset: str = _SUBSET,
    data_dir: Path | None = _DATA_DIR,
    systems_toml: Path | None = _SYSTEMS_TOML,
    index_prefix: str = typer.Option("bench_", "--index-prefix"),
) -> None:
    """Index the corpus: one index per language, doc_id = CELEX everywhere."""
    ds = load_dataset(subset, data_dir)
    sys_cfg = get_system(system, systems_toml)
    rag = sys_cfg.instantiate()
    counts, lock = do_ingest(rag, ds, index_prefix)
    for name, n in counts.items():
        typer.echo(f"  {name}: {n} documents")
    for warning in parity_warnings(lock):
        typer.echo(f"WARNING: {warning}", err=True)
    typer.echo("ingest complete (reads only the frozen snapshot — never the network)")


@app.command()
def run(
    system: str = typer.Option(..., "--system"),
    phase: str = typer.Option("retrieval", "--phase", help="retrieval | generation"),
    subset: str = _SUBSET,
    data_dir: Path | None = _DATA_DIR,
    systems_toml: Path | None = _SYSTEMS_TOML,
    db: str | None = _DB,
    index_prefix: str = typer.Option("bench_", "--index-prefix"),
    resume: str | None = typer.Option(None, "--resume", help="Run id to resume"),
    no_ingest: bool = typer.Option(False, "--no-ingest", help="Skip the ingest step"),
    k: int = typer.Option(100, "--k", help="Ranking depth stored per retrieval task"),
    modes: str = typer.Option("direct", "--modes", help="Generation modes, comma-separated"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
) -> None:
    """Execute a phase. Resumable: the queue lives in the database."""
    if phase not in ("retrieval", "generation"):
        raise typer.BadParameter("--phase must be 'retrieval' or 'generation'")
    ds = load_dataset(subset, data_dir)
    if not ds.queries:
        typer.echo(
            f"[{subset}] has no query set yet — generate queries.jsonl first "
            f"(benchmark/build/make_queries.py)",
            err=True,
        )
        raise typer.Exit(code=1)
    rep = validate_dataset(ds, check_texts=False)
    if not rep.ok:
        for e in rep.errors:
            typer.echo(f"dataset ERROR: {e}", err=True)
        raise typer.Exit(code=1)

    sys_cfg = get_system(system, systems_toml)
    rag = sys_cfg.instantiate()
    engine = q.get_engine(db)

    with Session(engine) as session:
        if resume:
            run_row = _resolve_run(session, resume)
            typer.echo(f"resuming run {run_row.id} ({run_row.name})")
        else:
            lock_entries: list[dict] = []
            if not no_ingest:
                typer.echo("ingesting corpus (frozen snapshot only)…")
                counts, lock = do_ingest(rag, ds, index_prefix)
                for name, n in counts.items():
                    typer.echo(f"  {name}: {n} documents")
                warnings = parity_warnings(lock)
                for warning in warnings:
                    typer.echo(f"WARNING: {warning}", err=True)
                if warnings:
                    typer.echo("parity gate failed — fix extraction before running", err=True)
                    raise typer.Exit(code=1)
                lock_entries = [e.as_dict() for e in lock]
            if phase == "retrieval":
                tasks = plan_retrieval(ds, index_prefix, k)
            else:
                tasks = plan_generation(ds, index_prefix, modes=modes.split(","))
            config = {
                "system_id": sys_cfg.id,
                "adapter": sys_cfg.adapter,
                # resolved, with secrets visibly "<redacted>" and endpoints aliased
                "adapter_config": redacted_config(sys_cfg.config),
                "data_version": subset,
                "index_prefix": index_prefix,
                "phase": phase,
                "k": k,
                "modes": modes.split(",") if phase == "generation" else None,
                "languages": ds.languages,
                "harness_version": __version__,
            }
            run_row = q.create_run(
                session,
                name=f"{subset}-{sys_cfg.id}-{phase}",
                config=config,
                system_meta=rag.describe(),
                git_sha=_git_sha(),
                tasks=tasks,
                lock=lock_entries,
            )
            typer.echo(f"created run {run_row.id} with {len(tasks)} tasks")

        run_row.status = "running"
        session.commit()
        worker = run_retrieval if phase == "retrieval" else run_generation
        progress = worker(session, rag, ds, run_row.id, max_attempts, log=typer.echo)
        run_row.status = "done" if not progress.get("pending") else "interrupted"
        session.commit()
        summary = ", ".join(f"{s}={n}" for s, n in sorted(progress.items()))
        typer.echo(f"run {run_row.id}: {summary}")
        if progress.get("failed"):
            typer.echo(
                "some tasks failed permanently — they will be reported as missing rate",
                err=True,
            )
        typer.echo(f"next: parallax-bench score --run {run_row.id}")


@app.command()
def score(
    run_id: str | None = typer.Option(None, "--run", help="Run id or name (default: latest)"),
    db: str | None = _DB,
    data_dir: Path | None = _DATA_DIR,
    out: Path = typer.Option(Path("runs"), "--out", help="Output directory root"),
    baseline_lang: str = typer.Option("en", "--baseline-lang"),
) -> None:
    """Compute metrics over a finished run. Never runs during collection."""
    import pandas as pd

    from parallax_bench.scoring import (
        diagonal_stats,
        score_generation,
        score_retrieval,
        write_outputs,
    )

    engine = q.get_engine(db)
    with Session(engine) as session:
        run_row = _resolve_run(session, run_id)
        cfg = run_row.config_json
        ds = load_dataset(cfg["data_version"], data_dir)
        prefix = cfg.get("index_prefix", "bench_")
        if cfg.get("phase") == "generation":
            aggregated = pd.DataFrame(
                columns=[
                    "query_lang", "index_lang", "metric", "mean", "ci_lo", "ci_hi",
                    "n", "missing_rate",
                ]
            )
            per_query = pd.DataFrame(
                columns=[
                    "query_id", "query_group", "origin", "query_lang", "index_lang",
                    "metric", "value", "measured",
                ]
            )
            stats = pd.DataFrame()
        else:
            aggregated, per_query = score_retrieval(session, run_row, ds, prefix)
            stats = diagonal_stats(per_query, baseline_lang) if not per_query.empty else per_query
        generation = score_generation(session, run_row, ds, prefix)
        out_dir = out / f"{run_row.created_at:%Y-%m-%d}-{run_row.name}-{run_row.id}"
        write_outputs(
            run_row,
            out_dir,
            aggregated,
            per_query,
            stats,
            generation,
            ds.languages,
        )
        typer.echo(f"wrote {out_dir}/")
        typer.echo(f"next: parallax-bench report --run {run_row.id}")


@app.command()
def report(
    run_id: str | None = typer.Option(None, "--run", help="Run id or name (default: latest)"),
    db: str | None = _DB,
    metric: str = typer.Option("ndcg@10", "--metric"),
    origin: str = typer.Option("all", "--origin", help="all | translated | native"),
    compare: str | None = typer.Option(
        None, "--compare", help="Additional scored run ids/names, comma-separated"
    ),
    out: Path = typer.Option(Path("runs"), "--out"),
) -> None:
    """Render scored matrices, summaries, heatmaps, and optional run comparison."""
    import pandas as pd

    engine = q.get_engine(db)
    with Session(engine) as session:
        run_row = _resolve_run(session, run_id)
        progress = q.run_progress(session, run_row.id)
        typer.echo(f"run {run_row.id} ({run_row.name}), status={run_row.status}")
        typer.echo("tasks: " + ", ".join(f"{s}={n}" for s, n in sorted(progress.items())))
        out_dir = out / f"{run_row.created_at:%Y-%m-%d}-{run_row.name}-{run_row.id}"
        metrics_csv = out_dir / "metrics.csv"
        if not metrics_csv.is_file():
            typer.echo("no metrics.csv yet — run `parallax-bench score` first")
            raise typer.Exit(code=1)
        df = pd.read_csv(metrics_csv)
        if not df.empty:
            from parallax_bench.reporting import comparison_table, load_matrix, write_heatmaps

            if metric not in set(df.metric):
                raise typer.BadParameter(
                    f"metric {metric!r} not in metrics.csv; available: "
                    + ", ".join(sorted(df.metric.unique()))
                )
            try:
                matrix = load_matrix(out_dir, metric, "absolute", origin)
                clp = load_matrix(out_dir, metric, "clp", origin)
                en_delta = load_matrix(out_dir, metric, "en_delta", origin)
            except FileNotFoundError:
                available = ["all"] + sorted(
                    path.name.removeprefix("origin_")
                    for path in (out_dir / "matrices").glob("origin_*")
                    if path.is_dir()
                )
                typer.echo(
                    f"query origin {origin!r} is not available for this run; "
                    f"available: {', '.join(available)}"
                )
                return
            typer.echo(
                f"\n{metric} absolute ({origin}; rows: query language, "
                "columns: index language)\n"
            )
            typer.echo(matrix.round(4).to_string())
            typer.echo("\nCross-Lingual Penalty: S(q,d) - S(q,q)\n")
            typer.echo(clp.round(4).to_string())
            typer.echo("\nEnglish-relative delta: S(q,d) - S(en,en)\n")
            typer.echo(en_delta.round(4).to_string())
            summary = pd.read_csv(out_dir / "parallax_summary.csv")
            summary_row = summary[(summary.metric == metric) & (summary.origin == origin)]
            if not summary_row.empty:
                typer.echo("\nParallax summary\n")
                typer.echo(summary_row.round(4).to_string(index=False))
            images = write_heatmaps(
                out_dir,
                metric,
                system=run_row.config_json.get("system_id", run_row.name),
                data_version=run_row.config_json.get("data_version", "unknown"),
                origin=origin,
            )
            typer.echo(f"\nheatmaps: {images[0]}, {images[1]}")

            comparison_runs = [(run_row.config_json.get("system_id", run_row.name), out_dir)]
            identifiers = [
                item.strip() for item in (compare or "").split(",") if item.strip()
            ]
            for identifier in identifiers:
                other = _resolve_run(session, identifier)
                other_dir = out / f"{other.created_at:%Y-%m-%d}-{other.name}-{other.id}"
                comparison_runs.append(
                    (other.config_json.get("system_id", other.name), other_dir)
                )
            if len(comparison_runs) > 1:
                table = comparison_table(comparison_runs, metric, origin)
                typer.echo("\nsystem comparison\n")
                typer.echo(table.round(4).to_string(index=False))
        else:
            typer.echo("\nno retrieval metrics in this generation-only run")
        gen_csv = out_dir / "generation.csv"
        if gen_csv.is_file():
            gen = pd.read_csv(gen_csv)
            summary = gen.groupby(["regime", "mode"]).agg(
                n=("query_id", "count"),
                lang_correct=("lang_correct", "mean"),
                citation_relevant=("citation_relevant_rate", "mean"),
                latency_ms=("latency_ms", "median"),
            )
            typer.echo("\ngeneration (mechanical metrics)\n")
            typer.echo(summary.round(3).to_string())
        stats_csv = out_dir / "stats.csv"
        if stats_csv.is_file():
            typer.echo("\npaired Wilcoxon vs baseline language (diagonal, Holm-corrected)\n")
            typer.echo(pd.read_csv(stats_csv).round(4).to_string(index=False))


@app.command()
def experiment(
    system: str = typer.Option(..., "--system"),
    subset: str = typer.Option("v1", "--subset", "--data-version"),
    data_dir: Path | None = _DATA_DIR,
    systems_toml: Path | None = _SYSTEMS_TOML,
    db: str | None = _DB,
    out: Path = typer.Option(Path("runs"), "--out"),
    index_prefix: str = typer.Option("bench_", "--index-prefix"),
    k: int = typer.Option(100, "--k"),
    modes: str = typer.Option("direct", "--modes"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
    baseline_lang: str = typer.Option("en", "--baseline-lang"),
    skip_fetch: bool = typer.Option(False, "--skip-fetch"),
    skip_verify: bool = typer.Option(False, "--skip-verify"),
    skip_generation: bool = typer.Option(False, "--skip-generation"),
) -> None:
    """Run the complete validate→fetch→verify→run→score→report workflow."""
    typer.echo(f"=== validate {subset} ===")
    validate(subset=subset, data_dir=data_dir)
    if not skip_fetch:
        typer.echo(f"\n=== fetch {subset} ===")
        fetch(subset=subset, data_dir=data_dir)
    if not skip_verify:
        typer.echo(f"\n=== verify source drift for {subset} ===")
        verify(subset=subset, data_dir=data_dir)

    phases = ["retrieval"] if skip_generation else ["retrieval", "generation"]
    engine = q.get_engine(db)
    completed: list[tuple[str, str]] = []
    for phase in phases:
        typer.echo(f"\n=== {phase} run: {system} ===")
        run(
            system=system,
            phase=phase,
            subset=subset,
            data_dir=data_dir,
            systems_toml=systems_toml,
            db=db,
            index_prefix=index_prefix,
            resume=None,
            no_ingest=False,
            k=k,
            modes=modes,
            max_attempts=max_attempts,
        )
        with Session(engine) as session:
            created = session.execute(
                select(q.Run)
                .where(q.Run.name == f"{subset}-{system}-{phase}")
                .order_by(q.Run.created_at.desc())
            ).scalars().first()
            if created is None:  # pragma: no cover - defensive: run() just created it
                raise RuntimeError(f"could not resolve newly created {phase} run")
            created_id = created.id
        typer.echo(f"\n=== score {phase} run {created_id} ===")
        score(
            run_id=created_id,
            db=db,
            data_dir=data_dir,
            out=out,
            baseline_lang=baseline_lang,
        )
        typer.echo(f"\n=== report {phase} run {created_id} ===")
        report(
            run_id=created_id,
            db=db,
            metric="ndcg@10",
            origin="all",
            compare=None,
            out=out,
        )
        completed.append((phase, created_id))

    typer.echo("\nexperiment complete")
    for phase, created_id in completed:
        typer.echo(f"  {phase}: {created_id}")


if __name__ == "__main__":
    app()
