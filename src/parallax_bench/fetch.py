"""Fetch corpus texts per the manifest into the content-addressed store.

Reproducibility contract (see ``docs/data-format.md``):

- The corpus is downloaded **once**, frozen, and only read afterwards.
  ``fetch`` is never run after ingest has started.
- The binding hash is ``sha256_text`` — the normalized text *after*
  extraction, exactly what goes into ``index()``.  The raw-bytes hash
  (``sha256_source``) is informative only, because EUR-Lex HTML contains
  volatile elements.
- The extractor is versioned (``EXTRACTOR_ID``); changing it means a new
  data-version, never a silent fix.
- ``verify`` re-downloads and *reports* drift against the manifest; it fixes
  nothing.  Source drift is a finding, not an error to paper over.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import re
import unicodedata
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from parallax_bench.data import Dataset, ManifestEntry, store_path

EXTRACTOR_ID = "parallax_bench.fetch.eurlex_html:v1"

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def normalize_text(text: str) -> str:
    """The frozen normalization every ``sha256_text`` is computed over.

    Defined once and NEVER changed within a data-version (a change here is a
    new extractor version and therefore a new data-version):

    - UTF-8, unicode NFC
    - line endings ``\\n``
    - trailing whitespace stripped per line
    - runs of 3+ blank lines collapsed to 2
    - nothing else
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # ≥3 blank lines -> exactly 2
    return text.strip() + "\n"


def html_to_text(html: str) -> str:
    """Deterministic HTML -> normalized text (extractor ``v1``)."""
    text = _SCRIPT_RE.sub(" ", html)
    text = re.sub(r"</(p|div|tr|li|h[1-6]|table)>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return normalize_text(text)


def extract(raw: str) -> str:
    return html_to_text(raw) if "<html" in raw[:2000].lower() else normalize_text(raw)


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30), reraise=True)
def _download(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def fetch_entry(client: httpx.Client, ds: Dataset, entry: ManifestEntry) -> tuple[str, bool]:
    """Fetch one document into the store unless already present.

    Returns (state, ok): state is 'cached' | 'fetched', ok reflects whether
    the stored text matches the manifest's binding hash.
    """
    target = store_path(ds.store_root, entry.sha256_text)
    if target.is_file():
        # content-addressed: the name IS the hash; re-verify cheaply anyway
        return "cached", sha256_str(target.read_text(encoding="utf-8")) == entry.sha256_text
    raw = _download(client, entry.url)
    text = extract(raw.decode("utf-8", errors="replace"))
    digest = sha256_str(text)
    ok = digest == entry.sha256_text
    # store under the ACTUAL hash — a drifted document must not squat on the
    # manifest's address
    actual = store_path(ds.store_root, digest)
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_text(text, encoding="utf-8")
    with (ds.store_root / "index.jsonl").open("a", encoding="utf-8") as fh:
        import json

        fh.write(
            json.dumps(
                {
                    "celex": entry.celex,
                    "lang": entry.lang,
                    "sha256_text": digest,
                    "sha256_source": hashlib.sha256(raw).hexdigest(),
                    "extractor": EXTRACTOR_ID,
                }
            )
            + "\n"
        )
    return "fetched", ok


def fetch_corpus(ds: Dataset, log=print) -> tuple[int, list[str]]:
    """Fetch everything in the manifest. Returns (n_ok, failures)."""
    failures: list[str] = []
    n_ok = 0
    with httpx.Client(headers={"User-Agent": "parallax-bench"}) as client:
        for entry in ds.manifest:
            if (ds.root / "docs" / entry.lang / f"{entry.celex}.txt").is_file():
                n_ok += 1  # bundled snapshot (smoke) — nothing to fetch
                continue
            try:
                state, ok = fetch_entry(client, ds, entry)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"({entry.celex}, {entry.lang}): {type(exc).__name__}: {exc}")
                continue
            if ok:
                n_ok += 1
                if state == "fetched":
                    log(f"  fetched ({entry.celex}, {entry.lang})")
            else:
                failures.append(
                    f"({entry.celex}, {entry.lang}): sha256_text mismatch — the source "
                    f"document drifted since the manifest was made; do not run on it"
                )
    return n_ok, failures


@dataclass(frozen=True)
class DriftReport:
    n_checked: int
    n_unchanged: int
    drifted: list[str]      # "(celex, lang)" entries whose sha256_text changed
    unreachable: list[str]  # download failures

    @property
    def drift_rate(self) -> float:
        return len(self.drifted) / self.n_checked if self.n_checked else 0.0


def verify_corpus(ds: Dataset, log=print) -> DriftReport:
    """Re-download per the manifest and report drift. Fixes nothing.

    Turns "would rebuild-from-source even work?" from an assumption into a
    measured number: *the source corpus drifted by X % in N months*.
    """
    drifted: list[str] = []
    unreachable: list[str] = []
    n_unchanged = 0
    with httpx.Client(headers={"User-Agent": "parallax-bench"}) as client:
        for entry in ds.manifest:
            try:
                raw = _download(client, entry.url)
            except Exception as exc:  # noqa: BLE001
                unreachable.append(f"({entry.celex}, {entry.lang}): {type(exc).__name__}")
                continue
            digest = sha256_str(extract(raw.decode("utf-8", errors="replace")))
            if digest == entry.sha256_text:
                n_unchanged += 1
            else:
                drifted.append(f"({entry.celex}, {entry.lang})")
                log(f"  drifted: ({entry.celex}, {entry.lang})")
    return DriftReport(
        n_checked=len(ds.manifest),
        n_unchanged=n_unchanged,
        drifted=drifted,
        unreachable=unreachable,
    )
