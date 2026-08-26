"""Build (or rebuild) the smoke subset's document texts and manifest.

The smoke subset is the single exception to the no-texts rule: ten documents
can be redistributed (EUR-Lex content, Decision 2011/833/EU, with attribution),
a whole corpus cannot.  Texts are fetched from EUR-Lex, normalised by the same
``html_to_text`` used by ``parallax-bench fetch``, and checksummed into the
manifest.

Usage: python benchmark/build/make_smoke.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from parallax_bench.fetch import EXTRACTOR_ID, html_to_text

SMOKE = Path(__file__).resolve().parent.parent / "smoke"

# Consumer/digital-law directives and regulations: well known, moderate size,
# available in all three smoke languages including pre-2004 acts (special
# edition translations).
CELEX_IDS = [
    "31985L0374",  # product liability
    "31993L0013",  # unfair contract terms
    "32000L0031",  # e-commerce
    "32001L0029",  # copyright in the information society
    "32002L0058",  # ePrivacy
    "32005L0029",  # unfair commercial practices
    "32006L0114",  # misleading and comparative advertising
    "32011L0083",  # consumer rights
    "32019L1024",  # open data and re-use of public sector information
    "32019R1150",  # platform-to-business fairness
]
LANGS = ["cs", "en", "de"]


def url_for(celex: str, lang: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"


def main() -> None:
    entries = []
    with httpx.Client(
        follow_redirects=True, headers={"User-Agent": "parallax-bench"}, timeout=60
    ) as client:
        for celex in CELEX_IDS:
            for lang in LANGS:
                resp = client.get(url_for(celex, lang))
                resp.raise_for_status()
                text = html_to_text(resp.text)
                assert len(text) > 5000, f"suspiciously short text for ({celex}, {lang})"
                path = SMOKE / "docs" / lang / f"{celex}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                entries.append(
                    {
                        "celex": celex,
                        "lang": lang,
                        "url": url_for(celex, lang),
                        "sha256_text": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "sha256_source": hashlib.sha256(resp.content).hexdigest(),
                        "extractor": EXTRACTOR_ID,
                        "char_len": len(text),
                    }
                )
                print(f"  ({celex}, {lang}) {len(text)} chars")
    with (SMOKE / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} manifest entries")


if __name__ == "__main__":
    main()
