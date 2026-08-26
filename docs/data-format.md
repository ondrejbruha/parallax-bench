# Data format

A data version (`benchmark/v1/`, `benchmark/smoke/`) consists of three files.

## `queries.jsonl`

One query variant per line:

```json
{"query_id": "q00123_cs", "query_group": "q00123", "lang": "cs",
 "text": "Za jakých podmínek lze zpracovávat osobní údaje bez souhlasu?",
 "source_celex": "32016R0679", "origin": "translated", "pivot_lang": "en",
 "generator": "<model>", "translator": "<model>", "query_set_version": "v1"}
```

- `query_id` = `{query_group}_{lang}`, enforced by the schema.
- **`query_group` is mandatory.** It binds the language variants of the same
  underlying query together. Without it, paired statistical tests (same
  query across languages → paired Wilcoxon) are impossible and only much
  weaker unpaired tests remain.
- `origin` is `translated` (pivot-generated, translated — the primary,
  fully parallel set) or `native` (independently generated per language —
  the validation sample). `translated` requires `pivot_lang`.

## `qrels.txt` — TREC format

```
q00123_cs 0 32016R0679 1
q00123_de 0 32016R0679 1
q00123_en 0 32016R0679 1
```

Standard TREC qrels plug directly into `pytrec_eval` / `ir_measures`; the
bundled metrics are tested to agree with `pytrec_eval` exactly.

**Key property: one qrels file covers the whole X×X matrix.** Relevance is
at *document* level and a document keeps the same id (CELEX) in every
language index. "Document C is relevant" is true regardless of which
language index was searched — which index was used is a property of the
*run*, not of the ground truth. Monolingual and cross-lingual cells are
therefore scored by identical code over identical judgements. This is a
direct consequence of corpus parallelism and the reason the benchmark is
cheap to maintain.

## `manifest.jsonl`

```json
{"celex": "32016R0679", "lang": "cs", "url": "https://eur-lex.europa.eu/...",
 "sha256_source": "…", "sha256_text": "…",
 "extractor": "parallax_bench.fetch.eurlex_html:v1", "char_len": 412883}
```

**Two hashes, not one — the trickiest spot in the whole design.** What goes
into `index()` is not the downloaded bytes but the text *after* extraction.
If the manifest pinned only the source bytes, replaying the pipeline with a
different HTML→text converter would pass the check and silently index
different texts. Therefore:

- **`sha256_text` is binding** — the hash of the normalized text after
  extraction, exactly what is indexed. Every verification in `fetch`,
  `ingest` and `validate` checks this one.
- `sha256_source` is informative only: EUR-Lex HTML contains volatile
  elements (generation date, session), so the raw hash may jitter without a
  content change.
- `extractor` names the versioned extractor. **Changing the extractor is a
  dataset change** — a new data-version — never a silent fix.

### Text normalization (frozen)

`sha256_text` is computed over text normalized by exactly these rules,
which are never changed within a data-version:

- UTF-8 encoding, unicode NFC normalization
- line endings `\n`
- trailing whitespace stripped from every line
- runs of 3+ blank lines collapsed to 2
- no other transformation

This sounds like a detail; it is the single thing that decides whether
hashes agree across machines.

**No document texts in the repository.** Three reasons, all load-bearing:

- **Licensing.** Packaged corpora (JRC-Acquis, MultiEURLEX) carry their own
  terms; EUR-Lex content is reusable under Decision 2011/833/EU with
  attribution, but a manifest sidesteps the packaging question entirely.
- **Size.** Megabytes instead of gigabytes.
- **Trust.** It is verifiable at a glance that nobody touched the documents.

### The corpus snapshot (`corpus-store/`)

`parallax-bench fetch` downloads per the manifest **once**, verifies
`sha256_text`, and freezes the texts in a content-addressed store (not in
git):

```
corpus-store/
├── by-hash/ab/cd/abcd1234…     # file name = sha256_text of its content
└── index.jsonl                  # (celex, lang) → sha256_text
```

Content addressing gives integrity by construction (a corrupted file no
longer matches its own name), free deduplication, and exactly the shape a
later public archive (Zenodo) upload takes.

**The hard rule: `ingest` never touches the network.** It reads only the
frozen snapshot, re-verifies every text against the manifest, and fails on
mismatch — it does not repair and does not re-fetch. `fetch` is never run
after ingest has started; both experiment phases must run over identical
bytes. Separately, `parallax-bench verify` re-downloads from the source and
*reports* drift as a measured number, fixing nothing.

The one exception to no-texts-in-git is `benchmark/smoke/`, which ships
texts under `docs/<lang>/<celex>.txt` (10 documents can be redistributed; a
corpus cannot) so the quickstart works offline.

### `manifest.lock` (per run)

The dataset manifest says what the corpus *should* be; `runs/<id>/
manifest.lock` records what *actually* went into the index in that run —
`(celex, lang, index_name, sha256_text, char_len, chunk_count)` per
document, written at ingest time. It is what lets anyone say "cell (cs→de)
in table 4 was computed over these documents in these versions", and its
`chunk_count` column doubles as evidence the cross-language parity gate ran.

## Validation

`parallax-bench validate` enforces: every qrels `query_id` exists in
`queries.jsonl` (and vice versa); every `source_celex` is in the manifest
for *every* dataset language (parallelism); every `query_group` has the
same, complete set of language variants; checksums well-formed and, where
texts are present, matching. CI runs it on every PR.
