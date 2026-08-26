# Datasheet — the PARALLAX benchmark dataset

Following the structure of *Datasheets for Datasets* (Gebru et al., 2021),
abbreviated to what applies.

## Motivation

**Why was the dataset created?** To measure language-induced retrieval
displacement in RAG systems: how the rank of the *same document* changes when
the *same question* is asked in a different language, everything else held
constant. No existing benchmark isolates the language variable this way, and
Czech is heavily underrepresented in RAG evaluation resources.

**Who created it?** Ondřej Brůha, as part of a master's thesis on evaluating
multilingual RAG systems using parallel legislative corpora.

## Composition

- **Queries** (`queries.jsonl`) — parallel language variants of the same
  underlying question, bound by `query_group`. Each targets exactly one
  source document.
- **Relevance judgements** (`qrels.txt`) — TREC format, *document*
  granularity. Because documents keep the same CELEX id across languages, a
  single qrels file is valid for every query-language × index-language cell.
- **Corpus manifest** (`manifest.jsonl`) — `(celex, lang) → URL + sha256 +
  length`. **Document texts are not part of the dataset** (see below).
- **Smoke subset** (`smoke/`) — 10 documents × 3 languages (cs, en, de)
  *including texts*, 20 hand-authored query groups. Exists so the quickstart
  runs offline; not intended for drawing conclusions.

## Collection & generation

Corpus documents come from EUR-Lex (parallel EU legislation, stable CELEX
identifiers). Query generation is documented verbatim — prompt, model,
date — in each version's `generation.md`. Where an LLM generated or
translated queries, this is disclosed there; primary sets are
pivot-generated (English) and translated (`origin: "translated"`), with a
native-generation validation sample (`origin: "native"`).

## Distribution & licensing

Queries and qrels: **CC BY 4.0** (`LICENSE-DATA`). Corpus texts are **not
redistributed** — three reasons: upstream dataset licences vary; size; and
trust (it is immediately verifiable that documents are untouched — `fetch`
re-downloads from the source and checks sha256). EUR-Lex content itself is
reusable under Commission Decision 2011/833/EU with source attribution.

## Known limitations

See `docs/limitations.md` — translationese in translated queries,
document-level (not chunk-level) ground truth, legal domain as primary,
automatically derived relevance without full human verification.

## Maintenance

Dataset versions (`v1`, `v2`, …) are immutable once released; each gets its
own DOI via the GitHub–Zenodo integration. Corrections produce a new version.
