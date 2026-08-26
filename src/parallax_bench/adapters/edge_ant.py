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

Wire contracts below were read off the Edge Ant source, not its documentation,
because two of them are easy to get wrong in ways that fail silently:

- ``POST /api/v1/search`` returns ``{results: [{text, name, document_id}]}``
  (``ChunkResp`` in ``src/api/search.rs``) — the chunk's document is ``name``.
- ``POST /api/v1/assistant/{id}/call`` returns ``{answer, chunks, question_id}``
  where each chunk is a ``ChunkRef``: ``{document_id, document_name,
  index_name, chunk_content}`` (``src/api/assistants.rs``). The document is
  ``document_name`` here — **not** ``name``. Reading ``name`` yields ``None``
  for every chunk, so retrieval and citation lists come back empty and every
  generation metric silently scores zero.
- ``result_count`` and ``prefetch_k`` are both validated to ``1..=100`` and
  ``prefetch_k >= result_count`` (``src/api/search.rs``); violating either is a
  422.
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

#: Both search parameters are validated to 1..=100 server-side.
_API_PARAM_MAX = 100


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
        embeddings_model: str | None = None,
        reranker_model: str | None = None,
        index_poll_interval_s: float = 2.0,
        index_timeout_s: float = 600.0,
        timeout_search_s: float = 30.0,
        timeout_generate_s: float = 300.0,
    ) -> None:
        if not 1 <= result_count <= _API_PARAM_MAX:
            raise ValueError(f"result_count must be 1..{_API_PARAM_MAX}, got {result_count}")
        if not 1 <= prefetch_k <= _API_PARAM_MAX:
            raise ValueError(f"prefetch_k must be 1..{_API_PARAM_MAX}, got {prefetch_k}")
        if prefetch_k < result_count:
            raise ValueError(
                f"prefetch_k ({prefetch_k}) must be >= result_count ({result_count}); "
                f"Edge Ant rejects the request otherwise"
            )
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
        # Operator-declared, because Edge Ant exposes neither over HTTP. Kept
        # separate from probed facts in describe().
        self.embeddings_model = embeddings_model
        self.reranker_model = reranker_model
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
        """Read what is actually deployed, never assume from configuration.

        Edge Ant has no configuration endpoint, so what can genuinely be probed
        over HTTP is the API version from the public OpenAPI document. The
        embedder and reranker are process-level environment (``EMBEDDINGS_MODEL``
        / ``RERANKER_MODEL``) and are not exposed at all — they are recorded
        under ``declared`` so a reader can never mistake an operator's claim for
        a measurement. Freeze the instance for the duration of a run.
        """
        # describe() lands in runs/<id>/system.json, which is committed — the
        # real endpoint must never appear there (assignment §9), so it gets the
        # same stable alias as config.json.
        info: dict = {"adapter": "edge_ant", "base_url": "<sut-endpoint>"}
        probed: dict = {}
        try:
            doc = self._request("GET", "/api-docs/openapi.json", None, 15)
            probed["api_version"] = (doc or {}).get("info", {}).get("version")
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            pass
        try:
            self._request("GET", "/api/v1/liveness", None, 10)
            probed["liveness"] = "ok"
        except Exception:  # noqa: BLE001
            probed["liveness"] = "unreachable"
        info["probed"] = probed
        if self.embeddings_model or self.reranker_model:
            info["declared"] = {
                "embeddings_model": self.embeddings_model,
                "reranker_model": self.reranker_model,
                "note": "not exposed by any Edge Ant endpoint; operator-declared",
            }
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
        for doc_ref, _doc_id in pending:
            while True:
                status = self._request("GET", f"/api/v1/document/{doc_ref}", None, 30)
                # Note: the response carries no chunk count. Edge Ant's
                # ``Document`` struct is {id, name, content, index_name, status,
                # restricted_to, owner_mail} and no endpoint exposes chunking
                # results, so ``chunk_counts`` stays empty unless a subclass
                # computes it (see chunk_counts below).
                if status.get("status") == "indexed":
                    break
                if status.get("status") == "failed":
                    raise RuntimeError(f"document {doc_ref} failed to index: {status}")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"document {doc_ref} not indexed within {self.index_timeout_s}s"
                    )
                time.sleep(self.index_poll_interval_s)

    def chunk_counts(self, index_name: str) -> dict[str, int]:
        """Optional introspection hook for manifest.lock (not protocol).

        Empty for this adapter: Edge Ant reports no chunk count anywhere. A
        subclass can fill ``self._chunk_counts`` — chunking is a pure function
        of (content, pattern, overlap), so it can be reproduced locally — and
        the cross-language parity check then has data to work with.
        """
        return dict(self._chunk_counts.get(index_name, {}))

    def _search_params(self, k: int) -> tuple[int, int]:
        """(result_count, prefetch_k) for a request that wants depth ``k``.

        The configuration defines the system; ``k`` only says how deep the
        benchmark wants to look. The two cases differ because of what Edge Ant
        actually does:

        - **Reranking off** (``prefetch_k == result_count``): the returned list
          *is* the RRF-fused ranking, so asking for more is simply looking
          deeper at the same ranking. Both values grow with ``k``, which keeps
          the rerank-off invariant intact.
        - **Reranking on**: the cross-encoder scores exactly ``prefetch_k``
          candidates, so growing the pool would be a *different* system. The
          pool stays as configured and ``result_count`` may rise only to
          ``prefetch_k - 1`` — at ``prefetch_k`` Edge Ant skips reranking
          altogether, which would silently turn the rerank arm into the
          no-rerank arm mid-run.

        A configured pool of P therefore has no ranking deeper than P-1 with
        reranking on. That is a property of the system, not a limitation to
        paper over: to compare the two arms at depth 100, configure both with
        ``prefetch_k = 100``.
        """
        want = max(1, min(k, _API_PARAM_MAX))
        if self.prefetch_k == self.result_count:
            return want, want
        prefetch_k = min(self.prefetch_k, _API_PARAM_MAX)
        return min(want, prefetch_k - 1), prefetch_k

    def search(self, query: str, index_name: str, k: int) -> list[str]:
        result_count, prefetch_k = self._search_params(k)
        body = {
            "query": query,
            "index_name": index_name,
            "result_count": result_count,
            "prefetch_k": prefetch_k,
            "min_chunk_length": self.min_chunk_length,
        }
        payload = self._request("POST", "/api/v1/search", body, self.timeout_search_s)
        chunks: list = (
            payload if isinstance(payload, list)
            else payload.get("results") or payload.get("chunks") or []
        )
        # Several chunks of one document routinely occupy several positions;
        # relevance is at document level, so they collapse to the first.
        seen: list[str] = []
        for chunk in chunks:
            name = chunk.get("name") or chunk.get("document_name")
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

        # ChunkRef carries `document_name`, not `name` — see the module
        # docstring. `name` is kept as a fallback only so a differently shaped
        # response is not silently read as "no documents".
        def doc_of(chunk: dict) -> str | None:
            return chunk.get("document_name") or chunk.get("name")

        retrieved: list[str] = []
        for chunk in chunks:
            name = doc_of(chunk)
            if name and name not in retrieved:
                retrieved.append(name)
        cited: list[str] = []
        for m in _CITATION_RE.finditer(answer_text):
            # Edge Ant renders each chunk inside <context id="N"> with N being
            # the 1-based index into this same `chunks` array, and its prompts
            # require inline [N] citations — so an N outside the array is a
            # dangling citation, which is directly measurable.
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(chunks):
                name = doc_of(chunks[idx])
                if name and name not in cited:
                    cited.append(name)
        # `index_name` per chunk is free language provenance: for a MultiRAG
        # assistant spanning every language index, this says which language the
        # answer's evidence actually came from.
        source_indexes: dict[str, int] = {}
        for chunk in chunks:
            ix = chunk.get("index_name")
            if ix:
                source_indexes[ix] = source_indexes.get(ix, 0) + 1
        return Answer(
            text=answer_text,
            retrieved_doc_ids=retrieved,
            cited_doc_ids=cited,
            raw={
                "question_id": payload.get("question_id"),
                "mode": mode,
                "n_chunks": len(chunks),
                "source_indexes": source_indexes,
            },
        )
