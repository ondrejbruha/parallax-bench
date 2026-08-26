# Limitations

Stated here deliberately and in full, before anyone else has to point them
out.

## Translationese in translated queries

All language variants of the primary query set descend from an English
pivot, so they share translation artifacts. For *comparing languages
against each other* this bias is constant across the compared conditions
and largely cancels — isolating the language variable is the whole point of
the parallel design, realism of individual queries is not. It remains a
bias: absolute per-language numbers should not be read as "how well the
system serves real Czech users". The native-generation validation sample
(`origin: "native"`) exists to check that conclusions hold without the
shared pivot; if trends disagree there, the translated results should not
be trusted.

## Document-level ground truth

Relevance is judged at document granularity because chunk boundaries do not
align across language versions. Consequences: a system retrieving the right
document but the wrong passage scores the same as one retrieving the right
passage; passage-level quality differences between languages are invisible
to the retrieval metrics (they surface only indirectly, via generation
metrics).

## One embedder per run

A run measures one deployed pipeline configuration. Findings about, e.g., a
specific multilingual embedder or an English-only reranker apply to that
component class, not to "dense retrieval" in general. Cross-system claims
require multiple configured systems, which the harness supports but the
initial result set only partially covers.

## Legal domain as primary corpus

EU legislation is what exists in genuinely parallel form at scale. Legal
text has unusual register, long sentences, and heavy terminology;
degradation patterns may differ in other domains. A smaller general-domain
parallel control corpus is the mitigation, not a solution.

## Automatically derived relevance

Qrels follow from the generation procedure (query generated from document D
⇒ D is relevant) without exhaustive human verification of the full set.
Queries whose answer accidentally also exists in other documents create
false negatives in scoring. The generation prompt is designed to prevent
this and a sample is reviewed manually, but the guarantee is procedural,
not annotated.

## Reproducibility level, stated precisely

Results are *verifiable* (level 1): every number traces to `config.json`,
`system.json` and `manifest.lock` with binding text hashes. They are
*re-runnable* (level 2) from the local corpus snapshot — which is **not yet
publicly archived**, so level 2 currently holds for the authors, not for
third parties. *Rebuildability from EUR-Lex* (level 3) is deliberately not
claimed: source drift is measured by `parallax-bench verify` and reported
as a number, not assumed away. "Fully reproducible" would be an inaccurate
claim and is intentionally avoided.

## Smoke subset is not evidence

`benchmark/smoke/` exists so the quickstart and CI run offline in seconds.
Ten documents and twenty query groups support no conclusions; its numbers
appear in documentation only to illustrate output shape.

## Language identification heuristic

Response language correctness uses a transparent stopword/diacritics
detector, adequate for paragraph-length answers in the benchmark languages
but not for short or mixed-language answers; undecidable answers are
reported separately rather than counted as wrong.
