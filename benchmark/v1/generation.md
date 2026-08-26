# Query set v1 — provenance

> **Status: not yet generated.** This file is the template that will be
> filled in verbatim when v1 is produced; v1 is released only together with
> its `queries.jsonl`, `qrels.txt` and `manifest.jsonl`.

When generating v1, record here — literally, not paraphrased:

- **Corpus:** source (JRC-Acquis / MultiEURLEX / EUR-Lex direct), document
  selection criteria, languages, retrieval date.
- **Generator model + date:** exact model id and the date of every
  generation run.
- **Translator model + date:** ditto for the translation step.
- **Prompts:** the full generation and translation prompts, verbatim
  (the current candidates live in `benchmark/build/make_queries.py`).
- **Native validation sample:** which ~10 % of documents, per-language
  generation prompt, `origin: "native"` rows.
- **Manual review:** who reviewed which sample of queries before the set
  was accepted (recommended: ≥50 queries).

Both the query generator and any LLM judge used in evaluation must be
disclosed here — some venues require it, and it costs nothing.
