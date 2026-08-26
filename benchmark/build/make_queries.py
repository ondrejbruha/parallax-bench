"""Reproduce query-set generation (requires an Anthropic API key).

Two-stage pipeline per the methodology:

1. **Generate** K queries per document from the *English pivot* text.  The
   prompt forces queries whose answer is unambiguously in that one document —
   otherwise the document-level ground truth leaks ("what is personal data"
   would hit half the corpus).
2. **Translate** each query into the remaining languages.  All variants share
   a ``query_group``, which is what makes paired tests possible.

Additionally, for a ~10% validation sample, queries are generated *natively*
from each language version (``origin: "native"``) — not parallel, so they
never enter the main table, but trends over them defend the translationese
objection.

The exact prompts, model and date used for a released query set are recorded
verbatim in ``benchmark/<version>/generation.md``.

Usage:
    python benchmark/build/make_queries.py --data-version v1 \
        --languages cs,en,de --queries-per-doc 2 --out benchmark/v1/queries.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from parallax_bench.data import load_dataset

MODEL = "claude-opus-5"

GENERATION_PROMPT = """\
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
"""

TRANSLATION_PROMPT = """\
Translate the following question into {target_language}.
Preserve the exact meaning and the register of a professional legal question.
Return only the translation, nothing else.

Question:
{text}
"""

LANGUAGE_NAMES = {
    "cs": "Czech", "en": "English", "de": "German", "pl": "Polish",
    "sk": "Slovak", "fr": "French", "es": "Spanish", "it": "Italian",
    "hu": "Hungarian", "nl": "Dutch",
}


class _QueryList(BaseModel):
    queries: list[str]


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("pip install anthropic — needed only for query generation") from exc
    return anthropic.Anthropic()


def _text_response(client, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text").strip()


def generate_queries(client, celex: str, text: str, n: int) -> list[str]:
    raw = _text_response(client, GENERATION_PROMPT.format(n=n, celex=celex, text=text))
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        return _QueryList(queries=json.loads(raw)).queries[:n]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"unparseable generation output for {celex}: {raw[:200]}") from exc


def translate(client, text: str, target_lang: str) -> str:
    return _text_response(
        client,
        TRANSLATION_PROMPT.format(target_language=LANGUAGE_NAMES[target_lang], text=text),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--languages", default="cs,en,de")
    parser.add_argument("--pivot", default="en")
    parser.add_argument("--queries-per-doc", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qrels-out", type=Path, default=None)
    args = parser.parse_args()

    langs = args.languages.split(",")
    ds = load_dataset(args.data_version, args.data_dir)
    client = _client()
    today = dt.date.today().isoformat()

    celexes = sorted({m.celex for m in ds.manifest})
    rows: list[dict] = []
    qrels: list[str] = []
    counter = 0
    for celex in celexes:
        pivot_path = ds.doc_text_path(celex, args.pivot)
        if not pivot_path.is_file():
            raise SystemExit(f"missing pivot text for {celex}; run `parallax-bench fetch` first")
        queries = generate_queries(
            client, celex, pivot_path.read_text(encoding="utf-8"), args.queries_per_doc
        )
        for q_text in queries:
            counter += 1
            group = f"q{counter:05d}"
            for lang in langs:
                text = q_text if lang == args.pivot else translate(client, q_text, lang)
                rows.append(
                    {
                        "query_id": f"{group}_{lang}",
                        "query_group": group,
                        "lang": lang,
                        "text": text,
                        "source_celex": celex,
                        "origin": "translated",
                        "pivot_lang": args.pivot,
                        "generator": f"{MODEL} ({today})",
                        "translator": f"{MODEL} ({today})",
                        "query_set_version": args.data_version,
                    }
                )
                qrels.append(f"{group}_{lang} 0 {celex} 1")
            print(f"  {group} ({celex}): {q_text[:70]}…")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    qrels_out = args.qrels_out or args.out.parent / "qrels.txt"
    qrels_out.write_text("".join(line + "\n" for line in qrels), encoding="utf-8")
    print(f"wrote {len(rows)} queries to {args.out} and {len(qrels)} qrels to {qrels_out}")
    print("record the prompts, model and date in generation.md — verbatim")


if __name__ == "__main__":
    main()
