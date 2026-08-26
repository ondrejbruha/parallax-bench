"""Edge Ant adapter against a mocked HTTP transport — no real instance needed.

The mocks reproduce Edge Ant's **actual** wire shapes, taken from the Rust
structs rather than from documentation:

- ``/api/v1/search``  -> ``{"results": [{"text", "name", "document_id"}]}``
- ``/assistant/{id}/call`` -> ``{"answer", "question_id", "chunks": [
  {"document_id", "document_name", "index_name", "chunk_content"}]}``

The two endpoints name the document differently, and a mock that gets that
wrong lets the adapter return empty document lists against the real system
while every test passes.
"""

import base64
import hashlib
import hmac
import json

import httpx
import pytest

from parallax_bench.adapters.base import Doc
from parallax_bench.adapters.edge_ant import EdgeAntSystem, sign_jwt


def _decode_jwt(token: str, secret: str) -> dict:
    signing_input, _, sig = token.rpartition(".")
    expected = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    assert sig == expected, "signature mismatch"
    payload = signing_input.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def test_jwt_shape_and_signature():
    token = sign_jwt("s3cret", "benchmark-runner", "bench@example.org", "write")
    claims = _decode_jwt(token, "s3cret")
    assert claims["user_name"] == "benchmark-runner"
    assert claims["rights"] == "write"
    assert claims["exp"] > claims["iat"]


def _system(handler, **kwargs) -> EdgeAntSystem:
    system = EdgeAntSystem(base_url="http://edge-ant.test", jwt_secret="s", **kwargs)
    system._client = httpx.Client(
        base_url="http://edge-ant.test", transport=httpx.MockTransport(handler)
    )
    return system


def test_search_dedupes_and_truncates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Bearer ")
        body = json.loads(request.content)
        assert body["index_name"] == "bench_cs"
        # SearchResponse wraps the list — ChunkResp is {text, name, document_id}.
        return httpx.Response(
            200,
            json={
                "results": [
                    {"text": "…", "name": "32016R0679", "document_id": 1},
                    {"text": "…", "name": "32016R0679", "document_id": 1},  # 2nd chunk
                    {"text": "…", "name": "32011L0083", "document_id": 2},
                    {"text": "…", "name": "31993L0013", "document_id": 3},
                ]
            },
        )

    system = _system(handler)
    assert system.search("dotaz", "bench_cs", 2) == ["32016R0679", "32011L0083"]


def test_search_parameters_are_clamped_to_the_api_limit():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    system = _system(handler, prefetch_k=100, result_count=100)
    system.search("q", "bench_cs", 500)
    # Both parameters are validated 1..=100 server-side; asking for more is a 422.
    assert captured["result_count"] == 100
    assert captured["prefetch_k"] == 100


def test_rerank_stays_on_when_a_deeper_k_is_requested():
    """Growing result_count up to prefetch_k would silently disable reranking."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    system = _system(handler, prefetch_k=50, result_count=10)
    system.search("q", "bench_cs", 100)
    # The pool the cross-encoder scores is the system's identity — it must not
    # grow with k — and result_count must stay strictly below it.
    assert captured["prefetch_k"] == 50
    assert captured["result_count"] == 49
    assert captured["prefetch_k"] != captured["result_count"]


def test_invalid_search_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="prefetch_k"):
        EdgeAntSystem(base_url="http://x", jwt_secret="s", prefetch_k=10, result_count=20)
    with pytest.raises(ValueError, match="1..100"):
        EdgeAntSystem(base_url="http://x", jwt_secret="s", prefetch_k=500, result_count=500)


def test_norerank_invariant_survives_large_k():
    """prefetch_k == result_count means rerank OFF; raising k must keep them equal."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=[])

    system = _system(handler, prefetch_k=10, result_count=10)
    system.search("q", "bench_cs", 100)
    assert captured["prefetch_k"] == captured["result_count"] == 100


def test_index_blocks_until_indexed():
    states = iter(["pending", "pending", "indexed"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "doc-1"})
        return httpx.Response(200, json={"id": "doc-1", "status": next(states)})

    system = _system(handler, index_poll_interval_s=0.01)
    system.index([Doc("32016R0679", "cs", "text")], "bench_cs")  # returns without raising


def _call_response() -> dict:
    """A CallResponse exactly as Edge Ant serialises it."""
    return {
        "answer": "Lhůta je 14 dnů [1]. Vrácení plateb upravuje [3].",
        "question_id": 42,
        "chunks": [
            {
                "document_id": 1,
                "document_name": "32011L0083",
                "index_name": "bench_cs",
                "chunk_content": "…",
            },
            {
                "document_id": 2,
                "document_name": "31993L0013",
                "index_name": "bench_en",
                "chunk_content": "…",
            },
            {
                "document_id": 1,
                "document_name": "32011L0083",
                "index_name": "bench_cs",
                "chunk_content": "…",
            },
        ],
    }


def test_generate_parses_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_call_response())

    system = _system(handler, assistant_ids={"bench_cs": "7"})
    answer = system.generate("Kolik dnů…?", ["bench_cs"], "direct")
    # [1] and [3] both point at the first document; [3] is the third chunk.
    assert answer.cited_doc_ids == ["32011L0083"]
    assert answer.retrieved_doc_ids == ["32011L0083", "31993L0013"]
    assert answer.raw["question_id"] == 42


def test_generate_reads_document_name_not_name():
    """Regression: ChunkRef has no `name` field.

    Reading `name` returns None for every chunk, so both document lists come
    back empty and every generation metric scores zero — against a system that
    is working perfectly.
    """
    payload = _call_response()
    assert all("name" not in c for c in payload["chunks"])

    system = _system(lambda r: httpx.Response(200, json=payload),
                     assistant_ids={"bench_cs": "7"})
    answer = system.generate("q", ["bench_cs"], "direct")
    assert answer.retrieved_doc_ids, "no documents parsed out of the response"
    assert answer.cited_doc_ids, "no citations resolved to documents"


def test_generate_records_source_index_per_chunk():
    """MultiRAG language provenance: which index the evidence came from."""
    system = _system(lambda r: httpx.Response(200, json=_call_response()),
                     assistant_ids={"bench_cs,bench_en": "9"})
    answer = system.generate("q", ["bench_en", "bench_cs"], "direct")
    assert answer.raw["source_indexes"] == {"bench_cs": 2, "bench_en": 1}


def test_describe_probes_the_running_instance():
    """Provenance must come from the instance, not from configuration."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api-docs/openapi.json":
            return httpx.Response(200, json={"info": {"version": "1.0.8"}})
        if request.url.path == "/api/v1/liveness":
            return httpx.Response(200, json={"status": "OK"})
        return httpx.Response(404, json={})

    system = _system(handler, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    info = system.describe()
    assert info["probed"]["api_version"] == "1.0.8"
    assert info["probed"]["liveness"] == "ok"
    # Edge Ant exposes no endpoint for the models, so they stay clearly marked
    # as an operator's claim rather than a measurement.
    assert info["declared"]["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert "operator-declared" in info["declared"]["note"]


def test_generate_without_assistant_is_a_config_error():
    system = _system(lambda r: httpx.Response(200, json={}))
    with pytest.raises(KeyError, match="assistant"):
        system.generate("q", ["bench_cs"], "direct")
