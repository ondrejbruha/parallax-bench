"""Dataset schemas, loaders and integrity validation.

The dataset (queries + qrels + manifest) is the contract everything else builds
on.  Three files per data version:

- ``queries.jsonl``  — one query variant per line, tied together by ``query_group``
- ``qrels.txt``      — TREC-format relevance judgements at *document* level
- ``manifest.jsonl`` — (celex, lang) -> URL + sha256; corpus texts are NOT
  redistributed, they are fetched and verified against the manifest

The ``smoke`` subset is the single exception: it ships document texts under
``docs/<lang>/<celex>.txt`` so the quickstart runs offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LANG_RE = re.compile(r"^[a-z]{2}$")
_QUERY_ID_RE = re.compile(r"^(?P<group>q[0-9]+)_(?P<lang>[a-z]{2})$")


class Query(BaseModel):
    """One language variant of a query.

    ``query_group`` binds the language variants of the same underlying query
    together; it is what makes paired statistical tests possible and is
    therefore mandatory.
    """

    query_id: str
    query_group: str
    lang: str
    text: str
    source_celex: str
    origin: str  # 'translated' | 'native'
    pivot_lang: str | None = None
    generator: str | None = None
    translator: str | None = None
    query_set_version: str

    @field_validator("lang")
    @classmethod
    def _lang_iso(cls, v: str) -> str:
        if not _LANG_RE.match(v):
            raise ValueError(f"lang must be a two-letter ISO code, got {v!r}")
        return v

    @field_validator("origin")
    @classmethod
    def _origin_enum(cls, v: str) -> str:
        if v not in ("translated", "native"):
            raise ValueError(f"origin must be 'translated' or 'native', got {v!r}")
        return v

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query text must be non-empty")
        return v

    @model_validator(mode="after")
    def _id_consistent(self) -> Query:
        m = _QUERY_ID_RE.match(self.query_id)
        if not m:
            raise ValueError(f"query_id must look like 'q00123_cs', got {self.query_id!r}")
        if m.group("group") != self.query_group or m.group("lang") != self.lang:
            raise ValueError(
                f"query_id {self.query_id!r} inconsistent with "
                f"query_group={self.query_group!r} / lang={self.lang!r}"
            )
        if self.origin == "translated" and not self.pivot_lang:
            raise ValueError(f"{self.query_id}: origin 'translated' requires pivot_lang")
        return self


class ManifestEntry(BaseModel):
    """One (document, language) pair of the corpus. No text — URL + checksums.

    Two hashes, deliberately: ``sha256_source`` pins the downloaded bytes and
    is *informative only* (EUR-Lex HTML contains volatile elements, so it may
    jitter without a content change).  ``sha256_text`` pins the normalized
    text *after* extraction — exactly what goes into ``index()`` — and is the
    **binding** one.  ``extractor`` names the versioned extractor that
    produced the text; changing the extractor is a dataset change (new
    data-version), never a silent fix.
    """

    celex: str
    lang: str
    url: str
    sha256_text: str
    sha256_source: str | None = None
    extractor: str
    char_len: int

    @field_validator("sha256_text")
    @classmethod
    def _sha_shape(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError(f"sha256_text must be 64 lowercase hex chars, got {v!r}")
        return v

    @field_validator("sha256_source")
    @classmethod
    def _source_sha_shape(cls, v: str | None) -> str | None:
        if v is not None and not _SHA256_RE.match(v):
            raise ValueError(f"sha256_source must be 64 lowercase hex chars, got {v!r}")
        return v

    @field_validator("lang")
    @classmethod
    def _lang_iso(cls, v: str) -> str:
        if not _LANG_RE.match(v):
            raise ValueError(f"lang must be a two-letter ISO code, got {v!r}")
        return v


@dataclass(frozen=True)
class QrelEntry:
    query_id: str
    doc_id: str
    relevance: int


def store_path(store_root: Path, sha256_text: str) -> Path:
    """Content-addressed location: ``by-hash/ab/cd/<sha256_text>``.

    Integrity by construction (a corrupted file no longer matches its own
    name), deduplication for free, and exactly the shape a later public
    archive upload takes.
    """
    return store_root / "by-hash" / sha256_text[:2] / sha256_text[2:4] / sha256_text


@dataclass
class Dataset:
    """A loaded data version (e.g. ``v1`` or ``smoke``)."""

    version: str
    root: Path
    queries: list[Query]
    qrels: list[QrelEntry]
    manifest: list[ManifestEntry]
    store_root: Path = field(default_factory=lambda: Path.cwd() / "corpus-store")

    @property
    def languages(self) -> list[str]:
        return sorted({q.lang for q in self.queries})

    @property
    def query_groups(self) -> dict[str, list[Query]]:
        groups: dict[str, list[Query]] = defaultdict(list)
        for q in self.queries:
            groups[q.query_group].append(q)
        return dict(groups)

    def qrels_by_query(self) -> dict[str, dict[str, int]]:
        """query_id -> {doc_id: relevance}, the shape retrieval metrics consume."""
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for e in self.qrels:
            out[e.query_id][e.doc_id] = e.relevance
        return dict(out)

    def manifest_entry(self, celex: str, lang: str) -> ManifestEntry:
        for m in self.manifest:
            if m.celex == celex and m.lang == lang:
                return m
        raise KeyError(f"({celex}, {lang}) not in manifest")

    def doc_text_path(self, celex: str, lang: str) -> Path:
        """Path of a locally available document text.

        ``smoke`` ships its texts under ``docs/<lang>/<celex>.txt`` (they are
        the snapshot); other versions live in the content-addressed
        ``corpus-store/`` written once by ``parallax-bench fetch``.
        """
        bundled = self.root / "docs" / lang / f"{celex}.txt"
        if bundled.is_file():
            return bundled
        return store_path(self.store_root, self.manifest_entry(celex, lang).sha256_text)


def find_dataset_root(version: str, data_dir: Path | None = None) -> Path:
    """Resolve where a data version lives.

    Order: explicit ``data_dir``, a ``benchmark/`` directory in the current
    working tree (git checkout), then the copy bundled inside the installed
    wheel.  Never use paths relative to ``__file__`` outside the package —
    that works from git and breaks after ``pip install``.
    """
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir) / version)
    candidates.append(Path.cwd() / "benchmark" / version)

    from importlib.resources import files

    try:
        pkg_data = Path(str(files("parallax_bench") / "_data" / version))
        candidates.append(pkg_data)
    except (ModuleNotFoundError, TypeError):  # pragma: no cover
        pass

    for c in candidates:
        if (c / "queries.jsonl").is_file():
            return c
    raise FileNotFoundError(
        f"data version {version!r} not found; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append((lineno, json.loads(line)))
    return rows


def load_qrels(path: Path) -> list[QrelEntry]:
    entries: list[QrelEntry] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 TREC fields, got {len(parts)}")
            query_id, _iter, doc_id, rel = parts
            entries.append(QrelEntry(query_id=query_id, doc_id=doc_id, relevance=int(rel)))
    return entries


def load_dataset(version: str, data_dir: Path | None = None) -> Dataset:
    root = find_dataset_root(version, data_dir)
    queries = [Query.model_validate(row) for _, row in _read_jsonl(root / "queries.jsonl")]
    qrels = load_qrels(root / "qrels.txt")
    manifest_path = root / "manifest.jsonl"
    manifest = (
        [ManifestEntry.model_validate(row) for _, row in _read_jsonl(manifest_path)]
        if manifest_path.is_file()
        else []
    )
    return Dataset(version=version, root=root, queries=queries, qrels=qrels, manifest=manifest)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(ds: Dataset, check_texts: bool = True) -> ValidationReport:
    """Integrity checks per the CI contract (§10 of the assignment).

    - every query_id in qrels exists in queries.jsonl
    - every source_celex exists in the manifest for every dataset language
    - every query_group has the same set of language variants
    - sha256 checksums have the right shape (enforced by the schema) and,
      where texts are present locally, actually match
    """
    rep = ValidationReport()

    query_ids = {q.query_id for q in ds.queries}
    if len(query_ids) != len(ds.queries):
        seen: set[str] = set()
        for q in ds.queries:
            if q.query_id in seen:
                rep.errors.append(f"duplicate query_id {q.query_id!r}")
            seen.add(q.query_id)

    qrel_qids = {e.query_id for e in ds.qrels}
    for qid in sorted(qrel_qids - query_ids):
        rep.errors.append(f"qrels references unknown query_id {qid!r}")
    for qid in sorted(query_ids - qrel_qids):
        rep.errors.append(f"query {qid!r} has no qrels entry")

    langs = set(ds.languages)
    groups = ds.query_groups
    lang_sets = {g: {q.lang for q in qs} for g, qs in groups.items()}
    expected = max(lang_sets.values(), key=len) if lang_sets else set()
    for g, ls in sorted(lang_sets.items()):
        if ls != expected:
            rep.errors.append(
                f"query_group {g!r} has language variants {sorted(ls)}, "
                f"expected {sorted(expected)} — paired tests need complete groups"
            )
        by_lang: dict[str, int] = defaultdict(int)
        for q in groups[g]:
            by_lang[q.lang] += 1
        for lang, n in by_lang.items():
            if n > 1:
                rep.errors.append(f"query_group {g!r} has {n} variants for lang {lang!r}")
        celexes = {q.source_celex for q in groups[g]}
        if len(celexes) > 1:
            rep.errors.append(f"query_group {g!r} spans several source_celex: {sorted(celexes)}")

    manifest_by_key = {(m.celex, m.lang): m for m in ds.manifest}
    if len(manifest_by_key) != len(ds.manifest):
        rep.errors.append("manifest contains duplicate (celex, lang) entries")
    if ds.manifest:
        manifest_celex = {m.celex for m in ds.manifest}
        for q in ds.queries:
            if q.source_celex not in manifest_celex:
                rep.errors.append(
                    f"{q.query_id}: source_celex {q.source_celex!r} not in manifest"
                )
        for celex in sorted(manifest_celex):
            for lang in sorted(langs):
                if (celex, lang) not in manifest_by_key:
                    rep.errors.append(
                        f"manifest is missing ({celex}, {lang}) — corpus must be "
                        f"parallel across all dataset languages"
                    )
    else:
        rep.warnings.append("no manifest.jsonl — corpus provenance is untracked")

    if check_texts:
        present = missing = 0
        for m in ds.manifest:
            p = ds.doc_text_path(m.celex, m.lang)
            if not p.is_file():
                missing += 1
                continue
            present += 1
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest != m.sha256_text:
                rep.errors.append(
                    f"sha256_text mismatch for ({m.celex}, {m.lang}): "
                    f"manifest {m.sha256_text[:12]}…, file {digest[:12]}… — "
                    f"the text is not the one the manifest pins"
                )
        if missing and present:
            rep.warnings.append(
                f"{missing} of {missing + present} corpus texts missing locally "
                f"(run `parallax-bench fetch`)"
            )
        elif missing and not present:
            rep.warnings.append(
                "no corpus texts present locally (run `parallax-bench fetch`)"
            )

    return rep
