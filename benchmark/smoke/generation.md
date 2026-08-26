# Smoke query set — provenance

- **Method:** hand-authored, no LLM involved. English pivot written first,
  then hand-translated into Czech and German.
- **Author:** Ondřej Brůha, 2026-08-26.
- **Design rule:** every query's answer is stated explicitly in exactly one
  of the ten smoke documents; queries never mention document titles or
  numbers. Two query groups per document, 20 groups × 3 languages.
- **Documents:** ten EU consumer/digital-law acts fetched from EUR-Lex on
  2026-08-26 (see `manifest.jsonl`), converted to plain text by
  `parallax_bench.fetch.html_to_text` (see `benchmark/build/make_smoke.py`).
  EUR-Lex content reused under Commission Decision 2011/833/EU;
  © European Union, 1998–2026, https://eur-lex.europa.eu/.
- **Purpose:** offline quickstart and CI only — not for drawing conclusions.
