"""``score``: compute metrics over a finished (or partial) run.

Strictly separated from ``run`` — this reads the results tables and can be
re-executed any time with a different metric definition without repeating the
collection.  Outputs land in ``runs/<run>/`` as small, diffable CSVs:

- ``metrics.csv``    — one row per (query_lang, index_lang, metric) with mean,
  bootstrap CI and missing rate; the X×X matrix in long form
- ``per_query.csv``  — per-query metric values (the input to any custom stats)
- ``stats.csv``      — paired Wilcoxon of every language against the baseline
  language on the diagonal, Holm-corrected
- ``matrices/``      — absolute, Cross-Lingual Penalty, English-relative delta,
  and directional-asymmetry CSVs, including available query-origin subsets
- ``parallax_summary.csv/json`` — global Parallax metrics and grouped-bootstrap
  confidence intervals
- ``generation.csv`` — mechanical generation metrics, when the run has them
- ``config.json`` / ``system.json`` — the frozen provenance of every number
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from parallax_bench.data import Dataset
from parallax_bench.metrics.generation import citation_accuracy
from parallax_bench.metrics.parallax import (
    bootstrap_mean_parallax_ci,
    build_score_matrix,
    cross_lingual_penalty,
    directional_asymmetries,
    english_relative_delta,
    metric_slug,
    summarize_parallax,
)
from parallax_bench.metrics.retrieval import DEFAULT_METRICS, score_run
from parallax_bench.metrics.stats import bootstrap_ci, holm_correction, paired_wilcoxon
from parallax_bench.runner.plan import MULTI_SEPARATOR
from parallax_bench.runner.queue import GenerationResult, RetrievalResult, Run, Task


def _index_lang(index_name: str, prefix: str) -> str:
    return index_name.removeprefix(prefix)


def load_rankings(
    session: Session, run_id: str, prefix: str
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """(query_lang, index_lang) -> {query_id: ranked doc_ids} for done tasks."""
    tasks = session.execute(
        select(Task).where(Task.run_id == run_id, Task.kind == "retrieval")
    ).scalars().all()
    by_task: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in session.execute(
        select(RetrievalResult).join(Task, RetrievalResult.task_id == Task.id).where(
            Task.run_id == run_id
        )
    ).scalars():
        by_task[row.task_id].append((row.rank, row.doc_id))
    out: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for t in tasks:
        if t.status != "done":
            continue
        ranked = [d for _, d in sorted(by_task.get(t.id, []))]
        out[(t.query_lang, _index_lang(t.target_index, prefix))][t.query_id] = ranked
    return dict(out)


def score_retrieval(
    session: Session, run: Run, ds: Dataset, prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (aggregated long-form matrix, per-query values)."""
    qrels = ds.qrels_by_query()
    queries_by_lang: dict[str, list[str]] = defaultdict(list)
    for q in ds.queries:
        queries_by_lang[q.lang].append(q.query_id)
    group_of = {q.query_id: q.query_group for q in ds.queries}
    origin_of = {q.query_id: q.origin for q in ds.queries}

    rankings = load_rankings(session, run.id, prefix)
    agg_rows, pq_rows = [], []
    for query_lang in ds.languages:
        for index_lang in ds.languages:
            cell = rankings.get((query_lang, index_lang), {})
            cell_qrels = {qid: qrels[qid] for qid in queries_by_lang[query_lang] if qid in qrels}
            per_query = score_run(cell, cell_qrels)
            n_expected = len(cell_qrels)
            n_measured = sum(1 for qid in cell_qrels if qid in cell)
            for qid, values in per_query.items():
                for metric, value in values.items():
                    pq_rows.append(
                        {
                            "query_id": qid,
                            "query_group": group_of.get(qid),
                            "origin": origin_of.get(qid),
                            "query_lang": query_lang,
                            "index_lang": index_lang,
                            "metric": metric,
                            "value": value,
                            "measured": qid in cell,
                        }
                    )
            for metric in DEFAULT_METRICS:
                ci = bootstrap_ci([per_query[qid][metric] for qid in per_query])
                agg_rows.append(
                    {
                        "query_lang": query_lang,
                        "index_lang": index_lang,
                        "metric": metric,
                        "mean": ci.mean,
                        "ci_lo": ci.lo,
                        "ci_hi": ci.hi,
                        "n": n_expected,
                        "missing_rate": 1 - n_measured / n_expected if n_expected else 0.0,
                    }
                )
    return pd.DataFrame(agg_rows), pd.DataFrame(pq_rows)


def diagonal_stats(per_query: pd.DataFrame, baseline_lang: str = "en") -> pd.DataFrame:
    """Paired Wilcoxon: each language's diagonal vs the baseline diagonal.

    Pairing runs through query_group — the same underlying query in two
    languages — which is why query_group is mandatory in the data format.
    """
    diag = per_query[per_query.query_lang == per_query.index_lang]
    rows = []
    for metric in sorted(diag.metric.unique()):
        base = diag[(diag.metric == metric) & (diag.query_lang == baseline_lang)]
        base_by_group = dict(zip(base.query_group, base.value, strict=True))
        raw_ps: list[float] = []
        pending: list[dict] = []
        for lang in sorted(diag.query_lang.unique()):
            if lang == baseline_lang:
                continue
            other = diag[(diag.metric == metric) & (diag.query_lang == lang)]
            other_by_group = dict(zip(other.query_group, other.value, strict=True))
            groups = sorted(set(base_by_group) & set(other_by_group))
            if len(groups) < 5:
                continue
            test = paired_wilcoxon(
                [other_by_group[g] for g in groups], [base_by_group[g] for g in groups]
            )
            raw_ps.append(test.p_value)
            pending.append(
                {
                    "metric": metric,
                    "lang": lang,
                    "baseline": baseline_lang,
                    "n_pairs": test.n_pairs,
                    "median_diff": test.median_diff,
                    "p_raw": test.p_value,
                }
            )
        for row, p_adj in zip(pending, holm_correction(raw_ps), strict=True):
            rows.append({**row, "p_holm": p_adj})
    return pd.DataFrame(rows)


def score_generation(session: Session, run: Run, ds: Dataset, prefix: str) -> pd.DataFrame:
    qrels = ds.qrels_by_query()
    tasks = {
        t.id: t
        for t in session.execute(
            select(Task).where(Task.run_id == run.id, Task.kind == "generation")
        ).scalars()
    }
    rows = []
    for res in session.execute(
        select(GenerationResult)
        .join(Task, GenerationResult.task_id == Task.id)
        .where(Task.run_id == run.id)
    ).scalars():
        t = tasks[res.task_id]
        indexes = t.target_index.split(MULTI_SEPARATOR)
        regime = "multi" if len(indexes) > 1 else (
            "mono" if _index_lang(indexes[0], prefix) == t.query_lang else "cross"
        )
        cited = json.loads(res.cited_json)
        retrieved = json.loads(res.retrieved_json)
        cit = citation_accuracy(cited, retrieved, qrels.get(t.query_id, {}))
        rows.append(
            {
                "query_id": t.query_id,
                "query_lang": t.query_lang,
                "regime": regime,
                "index_lang": _index_lang(indexes[0], prefix) if regime != "multi" else "*",
                "mode": t.mode,
                "latency_ms": res.latency_ms,
                "detected_lang": res.detected_lang,
                "lang_correct": (
                    None if res.detected_lang is None else res.detected_lang == t.query_lang
                ),
                "citation_valid_rate": cit.valid_rate,
                "citation_relevant_rate": cit.relevant_rate,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    run: Run,
    out_dir: Path,
    aggregated: pd.DataFrame,
    per_query: pd.DataFrame,
    stats: pd.DataFrame,
    generation: pd.DataFrame | None,
    languages: list[str] | None = None,
    baseline_lang: str = "en",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(out_dir / "metrics.csv", index=False)
    per_query.to_csv(out_dir / "per_query.csv", index=False)
    if not stats.empty:
        stats.to_csv(out_dir / "stats.csv", index=False)
    if generation is not None and not generation.empty:
        generation.to_csv(out_dir / "generation.csv", index=False)
    if languages and not per_query.empty:
        write_parallax_outputs(
            out_dir, aggregated, per_query, languages, baseline_lang=baseline_lang
        )
    (out_dir / "config.json").write_text(
        json.dumps(run.config_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "system.json").write_text(
        json.dumps(run.system_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # manifest.lock: what actually went into the index in THIS run, hash by
    # hash — without it the metrics table is just a claim
    with (out_dir / "manifest.lock").open("w", encoding="utf-8") as fh:
        for entry in run.lock_json or []:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    cfg = run.config_json
    (out_dir / "README.md").write_text(
        f"# Run {run.id} — {run.name}\n\n"
        f"- created: {run.created_at:%Y-%m-%d %H:%M} UTC\n"
        f"- system: `{cfg.get('system_id')}` (`{cfg.get('adapter')}`)\n"
        f"- data version: `{cfg.get('data_version')}`, "
        f"languages: {', '.join(cfg.get('languages', []))}\n"
        f"- harness: `{cfg.get('harness_version')}`, git sha: `{run.git_sha}`\n\n"
        f"Provenance: `config.json` (resolved config, secrets redacted), "
        f"`system.json` (read from the running system), `manifest.lock` "
        f"({len(run.lock_json or [])} indexed documents, binding sha256_text "
        f"each). Aggregated metrics in `metrics.csv`; per-query values in "
        f"`per_query.csv`. Parallax matrices and summaries are in `matrices/`, "
        f"`parallax_summary.csv`, and `parallax_summary.json`. Raw rankings "
        f"live in the run database / release "
        f"assets, not in git.\n",
        encoding="utf-8",
    )
    return out_dir


def write_parallax_outputs(
    out_dir: Path,
    aggregated: pd.DataFrame,
    per_query: pd.DataFrame,
    languages: list[str],
    *,
    baseline_lang: str = "en",
) -> tuple[pd.DataFrame, Path]:
    """Write absolute/delta matrices and global summaries for all query origins."""
    if baseline_lang not in languages:
        raise ValueError(
            f"baseline language {baseline_lang!r} unavailable; choose one of {languages}"
        )
    matrices_root = out_dir / "matrices"
    matrices_root.mkdir(parents=True, exist_ok=True)
    origins = ["all", *sorted(str(v) for v in per_query["origin"].dropna().unique())]
    summary_rows: list[dict] = []
    for origin in origins:
        subset = per_query if origin == "all" else per_query[per_query["origin"] == origin]
        subset_aggregated = (
            aggregated
            if origin == "all"
            else (
                subset.groupby(["query_lang", "index_lang", "metric"], sort=True)["value"]
                .mean()
                .reset_index()
                .rename(columns={"value": "mean"})
            )
        )
        matrix_dir = matrices_root if origin == "all" else matrices_root / f"origin_{origin}"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        for metric in sorted(str(v) for v in subset_aggregated["metric"].unique()):
            absolute = build_score_matrix(subset_aggregated, metric, languages)
            clp = cross_lingual_penalty(absolute)
            en_delta = english_relative_delta(absolute, baseline_lang)
            slug = metric_slug(metric)
            _write_matrix(absolute, matrix_dir / f"{slug}.csv")
            _write_matrix(clp, matrix_dir / f"{slug}_clp.csv")
            _write_matrix(en_delta, matrix_dir / f"{slug}_en_delta.csv")
            directional_asymmetries(clp).to_csv(
                matrix_dir / f"{slug}_asymmetry.csv", index=False, float_format="%.10g"
            )
            ci = bootstrap_mean_parallax_ci(subset, metric, languages)
            summary_rows.append(summarize_parallax(absolute, metric, origin, ci).as_dict())

    summary = pd.DataFrame(summary_rows).sort_values(["origin", "metric"]).reset_index(drop=True)
    summary.to_csv(out_dir / "parallax_summary.csv", index=False, float_format="%.10g")
    records = [_json_safe(row) for row in summary.to_dict("records")]
    summary_json = out_dir / "parallax_summary.json"
    summary_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary, matrices_root


def _write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    matrix.to_csv(path, index=True, index_label="query_lang", float_format="%.10g")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value
