# Methodology

## What is measured

How a RAG pipeline's quality degrades with language, with data held
constant. Not "which embedding model is better" — "where in the system is
quality lost when moving from English to Czech, and how much".

Parallel EU corpora contain the *same document* in many languages. Ingest
each language into its own index, run every query variant against every
index, and one variable moves (language) while everything else holds
(content, models, configuration, hardware).

## The matrix

```
                      index (document language)
                  cs    en    de    pl    fr   …
         cs    [mono][ CL ][ CL ][ CL ][ CL ]
  query  en    [ CL ][mono][ CL ][ CL ][ CL ]
  lang   de    [ CL ][ CL ][mono][ CL ][ CL ]
         …
         + MultiRAG: query in L against ALL indexes at once
```

- **diagonal** = monolingual RAG
- **off-diagonal** = cross-lingual retrieval (`Δ(cross-lingual)` per language
  pair → a degradation matrix)
- **MultiRAG** = one assistant over all indexes

English remains available as a secondary common reference, not as a goal. The
primary degradation comparison uses each query language's own monolingual
diagonal, which better isolates the effect of changing document language. See
[`parallax-metrics.md`](parallax-metrics.md).

## Ground truth

Document-level, not chunk-level — chunking runs independently per language
version, so chunk boundaries do not align across languages and chunk-level
cross-language ground truth cannot exist. The same document enters every
index under the same id (CELEX), which makes relevance seamless: a query
generated from document D is relevant to document D in *any* index. See
`data-format.md` for why this means a single qrels file covers the matrix.

## Query generation

Primary set: generate from the English pivot version of each document,
translate into the remaining languages (fully parallel; constant
translationese bias — see `limitations.md` for why that is acceptable and
how it is validated). Validation sample (~10 % of documents): independent
native generation per language; not parallel, used only to check that
trends agree. The generation prompt must force answers to be unambiguously
in the one source document, and is published verbatim in `generation.md`.

## Two-level scope

Retrieval is cheap; generation is not. The design separates them:

| Level | Scope | Typical calls |
|---|---|---|
| **L1 retrieval** | full matrix X×X×N (× ablations as separate systems) | ~10⁵ |
| **L2 generation** | diagonal + one query-language cut + MultiRAG, × modes | ~10³–10⁴ |

`score` is separate from `run`, so metric definitions can change without
repeating collection; raw rankings are stored in full.

The convenience command `parallax-bench experiment --system <id>` orchestrates
validation, fetch, source verification, both experiment phases, scoring, and
reporting in that order. It ingests once before retrieval, then reuses the same
language indexes for generation. It calls the same separated phase
implementations; it does not merge collection with scoring or weaken
resumability/provenance.

## Metrics

- **Retrieval:** nDCG@10, Recall@10, Recall@100, MRR@100, Success@1 —
  trec_eval definitions, tested against `pytrec_eval`.
- **Parallax (derived retrieval):** Cross-Lingual Penalty, English-relative
  delta, mean penalty, normalized matrix norm, row-wise stability and gaps,
  and directional asymmetry. These quantify language sensitivity and are
  reported beside—not instead of—the absolute IR metrics. Full definitions
  are in [`parallax-metrics.md`](parallax-metrics.md).
- **Generation (mechanical):** citation accuracy (cited docs ∈ retrieved
  set; cited docs relevant per qrels), response language correctness
  (did the system answer in the query's language — in cross-lingual mode a
  finding of its own, not a sanity check), latency.
- **Generation (LLM-judge):** faithfulness, answer relevancy — pluggable
  `Judge` protocol; the judge model must differ from the generator model.
- **Statistics:** bootstrap CIs over queries; the mean Cross-Lingual Penalty
  CI resamples whole `query_group` units so translated variants are never
  treated as independent; paired Wilcoxon across languages via `query_group`;
  Holm correction. Missing measurements are
  reported as a missing rate — a systematically higher failure rate in one
  language is a result, not noise.

## Provenance

Every published number traces to *(query set version, manifest sha,
system.json, systems.toml config, harness git sha)*. `system.json` is read
from the running system, never assumed from configuration; the system under
test must be frozen for the duration of a run.
