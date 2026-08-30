"""Local baseline adapter: BM25 (built-in) + optional dense retrieval.

No server, no GPU, no API key.  This is the reference point in the results
table — the reader sees what a trivial pipeline does — and, just as
importantly, the second adapter that keeps the protocol from being shaped
around any single system.

BM25 runs over passage chunks and aggregates to document level (max over
chunks), which mirrors what real RAG pipelines do. With ``embedding_model``
set (requires the ``baseline`` extra), retrieval can be dense-only or fuse
dense and BM25 rankings via reciprocal rank fusion. Indexes persist on disk
so ``ingest`` and ``run`` can be separate CLI invocations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from parallax_bench.adapters.base import Answer, Doc

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk(text: str, target_chars: int = 1200) -> list[str]:
    """Greedy paragraph-boundary chunker; language-agnostic on purpose."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > target_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
        while len(buf) > 2 * target_chars:  # single huge paragraph
            chunks.append(buf[:target_chars])
            buf = buf[target_chars:]
    if buf:
        chunks.append(buf)
    return chunks or [""]


class _Bm25Index:
    """Plain BM25 (Okapi) over chunks. ~40 lines, deterministic, no deps."""

    def __init__(self, chunks: list[tuple[str, str]], k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self.doc_ids = [doc_id for doc_id, _ in chunks]
        self.texts = [text for _, text in chunks]
        self.tfs = [Counter(_tokenize(t)) for t in self.texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.idf = {
            term: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()
        }

    def score(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        scores = [0.0] * len(self.tfs)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))


class BaselineLocalSystem:
    name = "baseline-local"

    def __init__(
        self,
        data_dir: str = ".parallax/baseline_local",
        embedding_model: str | None = None,
        retrieval_mode: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        chunk_chars: int = 1200,
        rrf_k: int = 60,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        # Preserve the original configuration contract: setting only an
        # embedding model means BM25+dense RRF. Explicit ``dense`` selects the
        # new pure multilingual dense baseline.
        retrieval_mode = retrieval_mode or ("hybrid" if embedding_model else "bm25")
        if retrieval_mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("retrieval_mode must be 'bm25', 'dense', or 'hybrid'")
        if retrieval_mode in {"dense", "hybrid"} and not embedding_model:
            raise ValueError(f"retrieval_mode={retrieval_mode!r} requires embedding_model")
        self.retrieval_mode = retrieval_mode
        self.k1, self.b = k1, b
        self.chunk_chars = chunk_chars
        self.rrf_k = rrf_k
        self._bm25_cache: dict[str, _Bm25Index] = {}
        self._dense_cache: dict[str, tuple] = {}
        self._encoder = None

    # -- protocol -----------------------------------------------------------

    def describe(self) -> dict:
        return {
            "adapter": "baseline_local",
            "retrieval_mode": self.retrieval_mode,
            "lexical": (
                {"type": "bm25-okapi", "k1": self.k1, "b": self.b}
                if self.retrieval_mode in {"bm25", "hybrid"}
                else None
            ),
            "dense": (
                {
                    "embedding_model": self.embedding_model,
                    "similarity": "cosine",
                    "normalized_embeddings": True,
                    "e5_prefixes": self._uses_e5_prefixes,
                }
                if self.retrieval_mode in {"dense", "hybrid"}
                else None
            ),
            "fusion": (
                {"type": "rrf", "k": self.rrf_k}
                if self.retrieval_mode == "hybrid"
                else None
            ),
            "chunking": {"type": "paragraph-greedy", "target_chars": self.chunk_chars},
        }

    def index(self, docs: Iterable[Doc], index_name: str) -> None:
        index_dir = self.data_dir / index_name
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / "chunks.jsonl"
        existing: set[str] = set()
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                existing = {json.loads(line)["doc_id"] for line in fh if line.strip()}
        with path.open("a", encoding="utf-8") as fh:
            for doc in docs:
                if doc.doc_id in existing:
                    continue
                for chunk in _chunk(doc.text, self.chunk_chars):
                    fh.write(
                        json.dumps(
                            {"doc_id": doc.doc_id, "lang": doc.lang, "text": chunk},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                existing.add(doc.doc_id)
        # invalidate caches; embeddings are (re)built lazily on first search
        self._bm25_cache.pop(index_name, None)
        self._dense_cache.pop(index_name, None)
        for emb in index_dir.glob("embeddings-*.npy"):
            emb.unlink()

    def search(self, query: str, index_name: str, k: int) -> list[str]:
        if self.retrieval_mode == "dense":
            return self._dense_rank(query, index_name)[:k]
        bm25 = self._load_bm25(index_name)
        lexical = self._rank_docs(bm25, bm25.score(query))
        if self.retrieval_mode == "bm25":
            return lexical[:k]
        dense = self._dense_rank(query, index_name)
        return _rrf([lexical, dense], self.rrf_k)[:k]

    def generate(self, query: str, index_names: list[str], mode: str) -> Answer:
        """Extractive 'generation': the best chunk verbatim, cited.

        The baseline has no LLM by design; an extractive answer keeps the
        generation phase runnable offline and gives generation metrics a
        floor to compare real systems against.
        """
        per_index = [self.search(query, ix, 10) for ix in index_names]
        fused = _rrf(per_index, self.rrf_k) if len(per_index) > 1 else per_index[0]
        if not fused:
            return Answer(text="", retrieved_doc_ids=[], cited_doc_ids=[], raw={"mode": mode})
        best_doc = fused[0]
        chunk_text = self._best_chunk(query, index_names, best_doc)
        return Answer(
            text=chunk_text,
            retrieved_doc_ids=fused[:10],
            cited_doc_ids=[best_doc],
            raw={"mode": mode, "extractive": True},
        )

    def chunk_counts(self, index_name: str) -> dict[str, int]:
        """Optional introspection hook for manifest.lock (not protocol)."""
        counts: dict[str, int] = defaultdict(int)
        for doc_id, _ in self._load_chunks(index_name):
            counts[doc_id] += 1
        return dict(counts)

    # -- internals ----------------------------------------------------------

    def _load_chunks(self, index_name: str) -> list[tuple[str, str]]:
        path = self.data_dir / index_name / "chunks.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"index {index_name!r} does not exist under {self.data_dir} — run ingest first"
            )
        out: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    out.append((row["doc_id"], row["text"]))
        return out

    def _load_bm25(self, index_name: str) -> _Bm25Index:
        if index_name not in self._bm25_cache:
            self._bm25_cache[index_name] = _Bm25Index(
                self._load_chunks(index_name), self.k1, self.b
            )
        return self._bm25_cache[index_name]

    @staticmethod
    def _rank_docs(bm25: _Bm25Index, scores: list[float]) -> list[str]:
        best: dict[str, float] = {}
        for doc_id, score in zip(bm25.doc_ids, scores, strict=True):
            if score > 0 and score > best.get(doc_id, 0.0):
                best[doc_id] = score
        return sorted(best, key=lambda d: (-best[d], d))

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "embedding_model is set but sentence-transformers is not installed; "
                    "pip install 'parallax-bench[baseline]'"
                ) from exc
            self._encoder = SentenceTransformer(self.embedding_model, device="cpu")
        return self._encoder

    @property
    def _uses_e5_prefixes(self) -> bool:
        return bool(self.embedding_model and "e5" in self.embedding_model.lower())

    def _embedding_text(self, text: str, kind: str) -> str:
        if self._uses_e5_prefixes:
            prefix = "query" if kind == "query" else "passage"
            return f"{prefix}: {text}"
        return text

    def _dense_rank(self, query: str, index_name: str) -> list[str]:
        import numpy as np

        if index_name not in self._dense_cache:
            chunks = self._load_chunks(index_name)
            cache_key = hashlib.sha256(
                f"{self.embedding_model}|e5={self._uses_e5_prefixes}".encode()
            ).hexdigest()[:16]
            emb_path = self.data_dir / index_name / f"embeddings-{cache_key}.npy"
            if emb_path.is_file():
                matrix = np.load(emb_path)
            else:
                matrix = self._get_encoder().encode(
                    [self._embedding_text(text, "passage") for _, text in chunks],
                    normalize_embeddings=True,
                )
                np.save(emb_path, matrix)
            self._dense_cache[index_name] = ([doc_id for doc_id, _ in chunks], matrix)
        doc_ids, matrix = self._dense_cache[index_name]
        q = self._get_encoder().encode(
            [self._embedding_text(query, "query")], normalize_embeddings=True
        )[0]
        sims = matrix @ q
        best: dict[str, float] = {}
        for doc_id, sim in zip(doc_ids, sims, strict=True):
            if doc_id not in best or sim > best[doc_id]:
                best[doc_id] = float(sim)
        return sorted(best, key=lambda d: (-best[d], d))

    def _best_chunk(self, query: str, index_names: list[str], doc_id: str) -> str:
        best_text, best_score = "", float("-inf")
        for ix in index_names:
            bm25 = self._load_bm25(ix)
            for i, score in enumerate(bm25.score(query)):
                if bm25.doc_ids[i] == doc_id and score > best_score:
                    best_score, best_text = score, bm25.texts[i]
        return best_text
