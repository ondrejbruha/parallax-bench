# parallax-bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22123082.svg)](https://doi.org/10.5281/zenodo.22123082)

*Measuring language-induced retrieval displacement in RAG systems using
parallel corpora.*

Parallax is the effect where **the same object appears at a different position
when viewed from a different vantage point**. This benchmark measures that
**the same document lands at a different rank when you ask about it in a
different language** — with everything else (content, models, configuration,
hardware) held constant.

The PARALLAX benchmark uses parallel EU legislative corpora: the same document
exists in many languages under a stable CELEX identifier. Ingesting each
language into its own index and running every query variant against every
index yields a fully controlled X×X matrix:

- **diagonal** — monolingual RAG (query language = document language)
- **off-diagonal** — cross-lingual retrieval
- **all indexes at once** — MultiRAG

Because ground truth is at *document* level and the document keeps the same ID
across languages, **a single TREC-format qrels file is valid for every cell of
the matrix** — monolingual and cross-lingual runs are scored by identical code
over identical judgements.

## Quickstart (two minutes, offline)

```bash
pip install parallax-bench
parallax-bench run --system baseline-local --subset smoke
parallax-bench score
parallax-bench report
```

No GPU, no API key, no server, no Postgres — the `smoke` subset ships with
document texts and the local baseline (BM25, optionally + dense via
`pip install 'parallax-bench[baseline]'`) runs anywhere. CI runs exactly these
commands on every PR.

To run the complete experiment lifecycle with one command (validation, corpus
fetch, source-drift verification, retrieval and generation runs, scoring, and
reports), configure a system and run:

```bash
parallax-bench experiment --system baseline-local --subset v1
```

The workflow remains decomposable and resumable through the individual
commands. For an already fetched/frozen corpus, or a retrieval-only pass, use
`--skip-fetch`, `--skip-verify`, or `--skip-generation` as appropriate.

`score` writes the standard IR results plus absolute, Cross-Lingual Penalty,
and English-relative language matrices as CSV. `report` reads those scored
artifacts and renders annotated PNG heatmaps; it never reruns retrieval. See
[docs/parallax-metrics.md](docs/parallax-metrics.md) for definitions and file
names. Smoke output validates the pipeline only and is not research evidence.

## Benchmarking a real system

```bash
pip install parallax-bench
parallax-bench fetch    --data-version v1        # download ONCE into corpus-store/, verify sha256_text
parallax-bench ingest   --system my-system       # reads only the frozen snapshot — never the network
parallax-bench run      --system my-system --phase retrieval
parallax-bench run      --system my-system --phase generation
parallax-bench score    --run <run_id>
parallax-bench report   --run <run_id>
parallax-bench verify   --data-version v1        # measure source drift; fixes nothing
```

Systems are defined in `systems.toml` (see `systems.example.toml`) as an
adapter class plus configuration. Ablations are expressed as two configured
instances of the same adapter — never as extra protocol parameters. The
adapter protocol has exactly four methods (`describe`, `index`, `search`,
`generate`); see [docs/adding-a-system.md](docs/adding-a-system.md).

The example configuration includes four recommended retrieval runs: the
built-in lexical BM25 baseline, a pure `intfloat/multilingual-e5-large` dense
baseline, and Edge Ant with and without reranking. These are separate configured
systems/runs; reranking is not a benchmark protocol option. Install the dense
extra before running E5:

```bash
pip install 'parallax-bench[baseline]'
cp systems.example.toml systems.toml
parallax-bench run --system multilingual-e5-large --subset v1
```

For an already scored run, select a query-origin subset or compare systems:

```bash
parallax-bench report --run <run_id> --metric ndcg@10 --origin translated
parallax-bench report --run <bm25_run> --compare <e5_run>,<edge_ant_run>
```

An unavailable origin (for example `native` in the current `v1`) is reported
as unavailable and does not make the report fail.

`run` is resumable: the task queue lives in a database (SQLite by default,
any SQLAlchemy URL — e.g. Heroku Postgres — via `--db` or `$DATABASE_URL`).
Interrupt at any point; `run --resume <id>` picks up where it stopped.
`score` is separate from `run`, so metrics can be recomputed under different
definitions without repeating a forty-hour collection.

## Results & reproducibility

Aggregated metrics for our runs live in [`runs/`](runs/) as small, diffable
CSVs; raw rankings are attached to GitHub releases. Every run directory
carries full provenance: `config.json` (resolved configuration, secrets
visibly redacted), `system.json` (read from the running system, never
assumed), and `manifest.lock` (the binding `sha256_text` of every document
that went into the index).

Results are **verifiable**: every published number carries the full
provenance needed to check it (`runs/<id>/`). The corpus snapshot used for
these runs is **not yet publicly archived**, so third-party re-execution on
the identical documents is not possible at this time. The manifest records
`sha256_text` for every document, so an archive published later can be
verified against the numbers already reported. Rebuilding the corpus from
EUR-Lex is best-effort; `parallax-bench verify` measures source drift
rather than assuming its absence.

To add your own system's results, open a PR adding a `runs/<id>/` directory —
CI validates the schema. There is no submission leaderboard by design.

## Dataset

`benchmark/v1/` contains queries, TREC qrels and a corpus **manifest** —
`(celex, lang) → URL + sha256`. Document texts are **not redistributed**
(licensing, size, and trust: it is immediately visible that nobody touched
the documents). `parallax-bench fetch` downloads them from EUR-Lex and
verifies checksums; if a source document changes, the run fails loudly.

The single exception is `benchmark/smoke/` (10 documents × 3 languages,
including texts) which exists so the quickstart works offline.

## Licensing

Dual-licensed on purpose: **code** (`src/`, `tests/`, `benchmark/build/`)
under [Apache-2.0](LICENSE); **queries + qrels** (`benchmark/*/`) under
[CC BY 4.0](benchmark/LICENSE-DATA). Corpus documents are not redistributed;
EUR-Lex content is reusable under Decision 2011/833/EU with attribution.

## Citation

See [CITATION.cff](CITATION.cff). Code versions follow semver; dataset
versions (`v1`, `v2`, …) are independent, each with its own Zenodo DOI, and a
released dataset version is never modified in place.
