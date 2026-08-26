"""Build a corpus manifest from a CELEX list — this is how a data version
gets *defined*.

You choose the documents (CELEX ids) and languages; this script fetches each
one from EUR-Lex once, runs the versioned extractor, and pins the result:
``sha256_text`` in the manifest is the binding hash every later ``fetch`` /
``ingest`` verifies against. The fetched texts also land in the local
``corpus-store/`` immediately, so the machine that builds the manifest
already holds the frozen snapshot.

Documents are fetched **directly from EUR-Lex** on purpose — packaged corpora
(JRC-Acquis, MultiEURLEX) carry their own licences that a manifest would
inherit; use them at most as a source of CELEX id lists, never of texts.

Usage:
    python benchmark/build/make_manifest.py \
        --celex-file celex_ids.txt \
        --languages cs,en,de \
        --out benchmark/v1/manifest.jsonl

``celex_ids.txt`` is one CELEX id per line (blank lines and ``#`` comments
ignored). A document missing in ANY requested language is rejected as a
whole — the corpus must stay parallel, and a partial document would poison
the single-qrels-for-the-whole-matrix property.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

from parallax_bench.data import store_path
from parallax_bench.fetch import EXTRACTOR_ID, _download, html_to_text, sha256_str

MIN_CHARS = 2000  # shorter than this is an error page, not a legal act


def url_for(celex: str, lang: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"


def read_celex_list(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celex-file", type=Path, required=True,
                        help="one CELEX id per line; # comments allowed")
    parser.add_argument("--languages", required=True, help="comma-separated, e.g. cs,en,de")
    parser.add_argument("--out", type=Path, required=True,
                        help="manifest path, e.g. benchmark/v1/manifest.jsonl")
    parser.add_argument("--store", type=Path, default=Path.cwd() / "corpus-store")
    args = parser.parse_args()

    langs = args.languages.split(",")
    celex_ids = read_celex_list(args.celex_file)
    entries: list[dict] = []
    rejected: list[str] = []

    with httpx.Client(headers={"User-Agent": "parallax-bench"}) as client:
        for celex in celex_ids:
            per_doc: list[dict] = []
            for lang in langs:
                url = url_for(celex, lang)
                try:
                    raw = _download(client, url)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {celex}/{lang}: download failed ({type(exc).__name__})")
                    per_doc = []
                    break
                text = html_to_text(raw)
                if len(text) < MIN_CHARS or "requested document does not exist" in text.lower():
                    print(f"  {celex}/{lang}: not available in this language")
                    per_doc = []
                    break
                digest = sha256_str(text)
                target = store_path(args.store, digest)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                per_doc.append(
                    {
                        "celex": celex,
                        "lang": lang,
                        "url": url,
                        "sha256_text": digest,
                        "sha256_source": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "extractor": EXTRACTOR_ID,
                        "char_len": len(text),
                    }
                )
            if per_doc:
                entries.extend(per_doc)
                print(f"  {celex}: ok in {len(per_doc)} languages")
            else:
                rejected.append(celex)  # parallelism is all-or-nothing per document

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (args.store / "index.jsonl").open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps({k: e[k] for k in ("celex", "lang", "sha256_text",
                                                   "sha256_source", "extractor")}) + "\n")
    print(f"wrote {len(entries)} entries ({len(entries) // len(langs)} documents "
          f"× {len(langs)} languages) to {args.out}")
    if rejected:
        print(f"rejected (not parallel or unreachable): {', '.join(rejected)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
