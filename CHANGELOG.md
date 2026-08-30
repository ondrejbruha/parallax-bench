# Changelog

## Unreleased

- Data format (`queries.jsonl`, `qrels.txt`, `manifest.jsonl`) and
  `parallax-bench validate`
- Adapter protocol (`describe` / `index` / `search` / `generate`) with two
  adapters: `baseline_local` (built-in BM25 + optional dense) and `edge_ant`
- Retrieval metrics (nDCG, Recall, MRR, Success@1) verified against
  `pytrec_eval`; bootstrap CIs, paired Wilcoxon, Holm correction
- Derived Parallax matrices and summaries: Cross-Lingual Penalty,
  English-relative delta, mean penalty with query-group bootstrap CI,
  normalized RMS norm, language robustness, gaps, and directional asymmetry
- Deterministic matrix CSV/JSON outputs, origin subsets, annotated report
  heatmaps, and comparison tables for already-scored systems
- Pure multilingual dense retrieval mode in the local adapter, including E5
  query/passage prefixes, normalized cosine similarity, and model provenance
- Mechanical generation metrics: citation accuracy, response language
  correctness
- Resumable DB-backed runner (SQLite/Postgres), `run` / `score` / `report`
- One-command `experiment` orchestration for validate, fetch, source verify,
  retrieval, generation, scoring, and reporting, with explicit skip controls
- `smoke` subset: 10 documents × 3 languages with texts, offline quickstart
- Reproducibility level 1+2: two-hash manifest (`sha256_source` informative,
  `sha256_text` binding, versioned extractor), frozen text normalization,
  content-addressed `corpus-store/`, network-free verified ingest, per-run
  `manifest.lock`, redacted resolved `config.json`, `verify` drift report
