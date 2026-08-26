# Changelog

## Unreleased

- Data format (`queries.jsonl`, `qrels.txt`, `manifest.jsonl`) and
  `parallax-bench validate`
- Adapter protocol (`describe` / `index` / `search` / `generate`) with two
  adapters: `baseline_local` (built-in BM25 + optional dense) and `edge_ant`
- Retrieval metrics (nDCG, Recall, MRR, Success@1) verified against
  `pytrec_eval`; bootstrap CIs, paired Wilcoxon, Holm correction
- Mechanical generation metrics: citation accuracy, response language
  correctness
- Resumable DB-backed runner (SQLite/Postgres), `run` / `score` / `report`
- `smoke` subset: 10 documents × 3 languages with texts, offline quickstart
- Reproducibility level 1+2: two-hash manifest (`sha256_source` informative,
  `sha256_text` binding, versioned extractor), frozen text normalization,
  content-addressed `corpus-store/`, network-free verified ingest, per-run
  `manifest.lock`, redacted resolved `config.json`, `verify` drift report
