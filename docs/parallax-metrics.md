# Parallax retrieval metrics

Parallax Bench reports two distinct families of retrieval measurements.
Standard IR metrics measure absolute retrieval quality. Derived Parallax
metrics measure how that quality changes when query or document language
changes while document identity and relevance judgements stay fixed.

## Cell aggregation

For an IR metric `M`, let `S(q,d)` be its mean over queries in query language
`q` evaluated against the index containing document language `d`. Every metric
produces an X×X matrix whose rows are query languages and columns are document
languages. The diagonal `S(q,q)` is monolingual; other cells are cross-lingual.

The standard metrics and their definitions are unchanged: nDCG@10, Recall@10,
Recall@100, MRR@100, and Success@1.

## Derived matrices

Cross-Lingual Penalty (CLP) is the primary degradation value:

```text
CLP(q,d) = S(q,d) - S(q,q)
CLP(q,q) = 0
```

It compares every cross-lingual cell with the monolingual performance of the
same query language. Negative values indicate degradation; positive values
indicate improvement. This is the matrix shown by the Parallax heatmap.

English-relative Delta retains a common English reference:

```text
EN_DELTA(q,d) = S(q,d) - S(en,en)
```

Unlike CLP, its diagonal is not generally zero. It answers a different
question and must not be described as the primary cross-lingual penalty.

## Global summaries

All off-diagonal means ignore unavailable cells in a partial run; missingness
remains explicit in `metrics.csv`.

- `mean_parallax` is the arithmetic mean of all off-diagonal `CLP(q,d)`.
  Its sign is retained: more negative usually means greater degradation.
- `parallax_rms = sqrt(mean(CLP(q,d)^2))` over off-diagonal cells. Zero means
  no language-induced displacement; larger values mean more instability. This
  normalized Frobenius norm is comparable across different language counts.
- `language_robustness_std` is the mean, over query-language rows, of the
  population standard deviation of `S(q,*)`. Lower is more stable; this is not
  a higher-is-better metric.
- `worst_language_gap = max(S(q,d)) - min(S(q,d))` over the full matrix.
- `mean_row_gap` is the mean of `max(S(q,*)) - min(S(q,*))` over rows.
- For an unordered language pair `(a,b)`, directional asymmetry is
  `abs(CLP(a,b) - CLP(b,a))`. Reports include its mean, maximum, and the
  lexicographically deterministic pair attaining the maximum.

The confidence interval for `mean_parallax` uses a percentile bootstrap that
resamples complete `query_group` units. Language variants of one translated
query are therefore retained together. Matrix-cell confidence intervals for
the all-query aggregation remain in `metrics.csv`.

## Outputs

`parallax-bench score --run <run>` writes deterministic artifacts:

```text
parallax_summary.csv
parallax_summary.json
matrices/
├── ndcg_at_10.csv
├── ndcg_at_10_clp.csv
├── ndcg_at_10_en_delta.csv
├── ndcg_at_10_asymmetry.csv
└── origin_translated/
    └── ...same files for that query origin...
```

The convention applies to every standard metric. `report` reads these files
and writes `absolute_<metric>_matrix.png` and
`parallax_<metric>_matrix.png` beside them. Cells are annotated, the diagonal
is outlined, and CLP uses a zero-centred colour scale.

Query origins are never fabricated. `score` emits `all` and each origin that
actually occurs in the dataset. `report --origin native` reports the subset as
unavailable without failing if no native queries exist.

To compare already-scored runs without rerunning retrieval:

```bash
parallax-bench report --run <primary> --compare <run2>,<run3> --metric ndcg@10
```

The comparison deliberately shows absolute mean quality and language
instability side by side; better absolute retrieval does not imply better
language robustness.
