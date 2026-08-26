"""Edge Ant adapter — a representative production RAG pipeline behind HTTP.

Componentially: multilingual bi-encoder + pgvector HNSW + Postgres FTS + RRF
fusion + cross-encoder reranking.  Everything system-specific stays inside
this adapter:

- Asynchronous indexing (POST returns 202/pending) is hidden behind a
  blocking ``index()`` that polls until ``status == "indexed"``.
- Rerank on/off is not a protocol knob: Edge Ant skips reranking when
  ``prefetch_k == result_count``, so the ablation is two configured instances
  of this adapter (see ``systems.example.toml``).
- Auth: Edge Ant's interactive e-mail login is unusable for a robot, so the
  adapter self-signs an HS256 JWT from the shared secret and re-mints it
  before expiry.  The secret comes from configuration/environment and must
  never appear in this repository.

Assistants (generation) are instance administration, not benchmark state:
create the assistants up front — identical system prompt, provider, model and
retrieval settings, differing only in their source folders — and hand their
ids to the adapter via ``assistant_ids``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from collections.abc import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from parallax_bench.adapters.base import Answer, Doc

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def sign_jwt(secret: str, user_name: str, mail: str, rights: str, ttl_hours: int = 12) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    claims = {
        "user_name": user_name,
        "mail": mail,
        "rights": rights,
        "iat": now,
        "exp": now + ttl_hours * 3600,
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(claims, separators=(",", ":")).encode())
    )
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


class _Transient(Exception):
    """5xx / timeout — worth retrying; distinguishes Cloudflare tunnel hiccups."""


class EdgeAntSystem:
    name = "edge-ant"

    def __init__(
        self,
        base_url: str,
        jwt_secret: str,
        mail: str = "benchmark@example.org",
        user_name: str = "benchmark-runner",
        rights: str = "write",
        prefetch_k: int = 50,
        result_count: int = 10,
        min_chunk_length: int = 0,
        chunk_pattern: str | None = None,
        chunk_overlap: int | None = None,
        assistant_ids: dict[str, str] | None = None,
        index_poll_interval_s: float = 2.0,
        index_timeout_s: float = 600.0,
        timeout_search_s: float = 30.0,
        timeout_generate_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._jwt_secret = jwt_secret
        self._mail = mail
        self._user_name = user_name
        self._rights = rights
        self.prefetch_k = prefetch_k
        self.result_count = result_count
        self.min_chunk_length = min_chunk_length
        self.chunk_pattern = chunk_pattern
        self.chunk_overlap = chunk_overlap
        self.assistant_ids = assistant_ids or {}
        self.index_poll_interval_s = index_poll_interval_s
        self.index_timeout_s = index_timeout_s
        self.timeout_search_s = timeout_search_s
        self.timeout_generate_s = timeout_generate_s
        self._token: str | None = None
        self._token_exp: float = 0.0
        self._chunk_counts: dict[str, dict[str, int]] = {}  # index -> doc_id -> chunks
        self._client = httpx.Client(base_url=self.base_url, http2=True)

    # -- auth ---------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._token is None or time.time() > self._token_exp - 600:
            self._token = sign_jwt(self._jwt_secret, self._user_name, self._mail, self._rights)
            self._token_exp = time.time() + 12 * 3600
        return {"Authorization": f"Bearer {self._token}"}

    # -- http ---------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_Transient),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _request(self, method: str, path: str, json_body: dict | None, timeout: float):
        try:
            resp = self._client.request(
                method, path, json=json_body, headers=self._headers(), timeout=timeout
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Transient(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 500:
            # A Cloudflare tunnel 502 serves an HTML error page, not JSON —
            # tag it so it never shows up as a system failure in results.
            source = "tunnel" if "text/html" in resp.headers.get("content-type", "") else "system"
            raise _Transient(f"HTTP {resp.status_code} ({source})")
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # -- protocol -----------------------------------------------------------

    def describe(self) -> dict:
        """Read what is actually deployed, never assume from configuration."""
        info: dict = {"adapter": "edge_ant", "base_url": self.base_url}
        for path in ("/api/v1/info", "/api/v1/health"):
            try:
                info["instance"] = self._request("GET", path, None, 10)
                break
            except Exception:  # noqa: BLE001 — endpoint availability varies by version
                continue
        info["search"] = {
            "prefetch_k": self.prefetch_k,
            "result_count": self.result_count,
            "min_chunk_length": self.min_chunk_length,
            "rerank_active": self.prefetch_k != self.result_count,
        }
        info["chunking"] = {"pattern": self.chunk_pattern, "overlap": self.chunk_overlap}
        return info

    def index(self, docs: Iterable[Doc], index_name: str) -> None:
        pending: list[tuple[str, str]] = []  # (edge-ant doc ref, our doc_id)
        for doc in docs:
            body: dict = {"name": doc.doc_id, "content": doc.text, "index_name": index_name}
            if self.chunk_pattern is not None:
                body["pattern"] = self.chunk_pattern
            if self.chunk_overlap is not None:
                body["overlap"] = self.chunk_overlap
            created = self._request("POST", "/api/v1/document", body, 120)
            doc_ref = created.get("id") or created.get("document_id")
            if doc_ref is None:
                raise RuntimeError(f"document POST returned no id: {created}")
            pending.append((str(doc_ref), doc.doc_id))
        deadline = time.monotonic() + self.index_timeout_s
        counts = self._chunk_counts.setdefault(index_name, {})
        for doc_ref, doc_id in pending:
            while True:
                status = self._request("GET", f"/api/v1/document/{doc_ref}", None, 30)
                if status.get("status") == "indexed":
                    if isinstance(status.get("chunk_count"), int):
                        counts[doc_id] = status["chunk_count"]
                    break
                if status.get("status") == "failed":
                    raise RuntimeError(f"document {doc_ref} failed to index: {status}")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"document {doc_ref} not indexed within {self.index_timeout_s}s"
                    )
                time.sleep(self.index_poll_interval_s)

    def chunk_counts(self, index_name: str) -> dict[str, int]:
        """Optional introspection hook for manifest.lock (not protocol)."""
        return dict(self._chunk_counts.get(index_name, {}))

    def search(self, query: str, index_name: str, k: int) -> list[str]:
        body = {
            "query": query,
            "index_name": index_name,
            "result_count": max(k, self.result_count),
            "prefetch_k": (
                # keep the rerank-off invariant (prefetch_k == result_count)
                max(k, self.result_count)
                if self.prefetch_k == self.result_count
                else max(self.prefetch_k, k)
            ),
            "min_chunk_length": self.min_chunk_length,
        }
        payload = self._request("POST", "/api/v1/search", body, self.timeout_search_s)
        chunks: list = (
            payload if isinstance(payload, list)
            else payload.get("results") or payload.get("chunks") or []
        )
        seen: list[str] = []
        for chunk in chunks:
            name = chunk.get("name")
            if name and name not in seen:
                seen.append(name)
        return seen[:k]

    def generate(self, query: str, index_names: list[str], mode: str) -> Answer:
        key = ",".join(sorted(index_names))
        assistant_id = self.assistant_ids.get(key)
        if assistant_id is None:
            raise KeyError(
                f"no assistant configured for indexes {key!r}; add it to "
                f"assistant_ids in systems.toml (create assistants with an "
                f"identical system prompt, differing only in source folders)"
            )
        payload = self._request(
            "POST",
            f"/api/v1/assistant/{assistant_id}/call",
            {"query": query, "mode": mode},
            self.timeout_generate_s,
        )
        answer_text = payload.get("answer", "")
        chunks = payload.get("chunks", [])
        retrieved: list[str] = []
        for chunk in chunks:
            name = chunk.get("name")
            if name and name not in retrieved:
                retrieved.append(name)
        cited: list[str] = []
        for m in _CITATION_RE.finditer(answer_text):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(chunks):
                name = chunks[idx].get("name")
                if name and name not in cited:
                    cited.append(name)
        return Answer(
            text=answer_text,
            retrieved_doc_ids=retrieved,
            cited_doc_ids=cited,
            raw={
                "question_id": payload.get("question_id"),
                "mode": mode,
                "n_chunks": len(chunks),
            },
        )
