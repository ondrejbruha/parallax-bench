"""Ingest: load the parallel corpus into one index per language.

Two invariants, both load-bearing:

- The same document goes into every language index under the same ``doc_id``
  (CELEX).  The language is carried by the index name, never by the document
  id — that is what makes a single qrels file valid for the whole matrix.
- **Ingest never touches the network.**  It reads the frozen snapshot
  (bundled smoke texts or the content-addressed ``corpus-store/``), verifies
  every text against the manifest's binding ``sha256_text``, and fails on
  mismatch — it does not repair, it does not re-fetch.

Ingest also produces the ``manifest.lock`` entries: the record of what
*actually* went into the index in this run, hash by hash.  ``chunk_count``
is filled in when the adapter exposes an optional ``chunk_counts(index_name)
-> dict[doc_id, int]`` hook (an introspection extra, deliberately not part
of the four-method protocol), and is the basis of the cross-language parity
check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from parallax_bench.adapters.base import Doc, RagSystem
from parallax_bench.data import Dataset


def index_name(prefix: str, lang: str) -> str:
    return f"{prefix}{lang}"


@dataclass(frozen=True)
class LockEntry:
    celex: str
    lang: str
    index_name: str
    sha256_text: str
    char_len: int
    chunk_count: int | None

    def as_dict(self) -> dict:
        return {
            "celex": self.celex,
            "lang": self.lang,
            "index_name": self.index_name,
            "sha256_text": self.sha256_text,
            "char_len": self.char_len,
            "chunk_count": self.chunk_count,
        }


def _load_verified(ds: Dataset, celex: str, lang: str) -> str:
    entry = ds.manifest_entry(celex, lang)
    path = ds.doc_text_path(celex, lang)
    if not path.is_file():
        raise FileNotFoundError(
            f"({celex}, {lang}) not in the local snapshot ({path}); "
            f"run `parallax-bench fetch` once, before ingest — never during"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.sha256_text:
        raise ValueError(
            f"({celex}, {lang}): snapshot text {digest[:12]}… does not match "
            f"manifest sha256_text {entry.sha256_text[:12]}… — refusing to index"
        )
    return data.decode("utf-8")


def ingest(
    system: RagSystem, ds: Dataset, prefix: str = "bench_"
) -> tuple[dict[str, int], list[LockEntry]]:
    """Index every language of the corpus from the verified snapshot.

    Returns ({index_name: doc_count}, manifest.lock entries).
    """
    counts: dict[str, int] = {}
    lock: list[LockEntry] = []
    for lang in ds.languages:
        ix = index_name(prefix, lang)
        entries = [m for m in ds.manifest if m.lang == lang]
        docs = [
            Doc(doc_id=m.celex, lang=lang, text=_load_verified(ds, m.celex, lang))
            for m in entries
        ]
        system.index(docs, ix)
        chunk_counts: dict[str, int] = {}
        counter = getattr(system, "chunk_counts", None)
        if callable(counter):
            chunk_counts = counter(ix) or {}
        for m, doc in zip(entries, docs, strict=True):
            lock.append(
                LockEntry(
                    celex=m.celex,
                    lang=lang,
                    index_name=ix,
                    sha256_text=m.sha256_text,
                    char_len=len(doc.text),
                    chunk_count=chunk_counts.get(m.celex),
                )
            )
        counts[ix] = len(docs)
    return counts, lock


def parity_warnings(lock: list[LockEntry]) -> list[str]:
    """Cross-language chunk-count parity: a language version with wildly
    fewer chunks means broken extraction and a contaminated experiment."""
    by_celex: dict[str, dict[str, int]] = {}
    for e in lock:
        if e.chunk_count is not None:
            by_celex.setdefault(e.celex, {})[e.lang] = e.chunk_count
    warnings = []
    for celex, per_lang in sorted(by_celex.items()):
        if len(per_lang) < 2:
            continue
        lo, hi = min(per_lang.values()), max(per_lang.values())
        if lo * 2 < hi:
            warnings.append(
                f"chunk-count parity: {celex} ranges {lo}–{hi} across languages "
                f"({per_lang}) — check extraction before trusting the run"
            )
    return warnings
