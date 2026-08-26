# Query set v1 — provenance

## Corpus

Four EU digital-law acts, fetched directly from EUR-Lex on **2026-08-26**
and pinned in `manifest.jsonl` (binding `sha256_text`, extractor
`parallax_bench.fetch.eurlex_html:v1`; built by
`benchmark/build/make_manifest.py` from `celex_ids.txt`):

| CELEX | Act |
|---|---|
| 32016R0679 | GDPR — General Data Protection Regulation |
| 32024R1689 | AI Act — Artificial Intelligence Act |
| 32022L2555 | NIS2 — cybersecurity directive |
| 32022R2554 | DORA — Digital Operational Resilience Act |

Languages: **cs, en, de**. EUR-Lex content reused under Commission Decision
2011/833/EU; © European Union, https://eur-lex.europa.eu/.

## Generation

- **Script:** `benchmark/build/make_queries.py` (this repository)
- **Generator model:** `claude-opus-5` (Anthropic API)
- **Translator model:** `claude-opus-5` (same run)
- **Date of generation run:** 2026-08-26
- **Parameters:** `--queries-per-doc 10`, `--pivot en`,
  `--languages cs,en,de` → 40 query groups × 3 language variants = 120
  queries, all `origin: "translated"` with `pivot_lang: "en"`
- **Pipeline:** queries generated from the **English** text of each
  document, then translated to Czech and German; the three variants share a
  `query_group` (the pairing unit for statistical tests)
- **Qrels:** derived mechanically — a query generated from document D is
  relevant to D (document-level, relevance 1); one line per query variant

Both the generator and the translator are LLMs, disclosed here per the
methodology; any LLM judge used later in evaluation must be a *different*
model and is to be disclosed in the run's provenance.

## Prompts (verbatim)

### Generation prompt

```
You are building an information-retrieval benchmark over EU legislation.

Read the document below and write {n} questions about it, in English.

Hard requirements for every question:
- The answer must be stated explicitly and unambiguously in THIS document.
- The question must NOT be answerable from general knowledge of EU law or
  from other EU acts — it must require this specific document.
- Do not mention the document's title, number or CELEX id in the question.
- Ask about substantive content (obligations, thresholds, deadlines,
  definitions, conditions), not about document metadata.

Return a JSON array of {n} strings and nothing else.

Document (CELEX {celex}):
{text}
```

### Translation prompt

```
Translate the following question into {target_language}.
Preserve the exact meaning and the register of a professional legal question.
Return only the translation, nothing else.

Question:
{text}
```

## Validation status

- [ ] `parallax-bench validate --subset v1` passes (120 queries, 40 complete
  groups, 120 qrels)
- [ ] manual review of a sample of queries (recommended ≥50) before any
  published run
- **Native validation sample (`origin: "native"`): not generated in v1.**
  The translationese check (§3.3b of the methodology) is deferred; treat
  absolute per-language numbers accordingly (see `docs/limitations.md`).

Note: the `generator`/`translator` fields inside `queries.jsonl` are the
authoritative record of what produced each row; if this document and those
fields ever disagree, the fields win.
