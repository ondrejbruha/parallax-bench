"""Edge Ant adapter against a mocked HTTP transport — no real instance needed."""

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
        chunks = [
            {"text": "…", "name": "32016R0679", "document_id": 1},
            {"text": "…", "name": "32016R0679", "document_id": 1},  # same doc, 2nd chunk
            {"text": "…", "name": "32011L0083", "document_id": 2},
            {"text": "…", "name": "31993L0013", "document_id": 3},
        ]
        return httpx.Response(200, json=chunks)

    system = _system(handler)
    assert system.search("dotaz", "bench_cs", 2) == ["32016R0679", "32011L0083"]


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


def test_generate_parses_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answer": "Lhůta je 14 dnů [1]. Vrácení plateb upravuje [3].",
                "question_id": 42,
                "chunks": [
                    {"name": "32011L0083", "text": "…"},
                    {"name": "31993L0013", "text": "…"},
                    {"name": "32011L0083", "text": "…"},
                ],
            },
        )

    system = _system(handler, assistant_ids={"bench_cs": "7"})
    answer = system.generate("Kolik dnů…?", ["bench_cs"], "direct")
    assert answer.cited_doc_ids == ["32011L0083"]
    assert answer.retrieved_doc_ids == ["32011L0083", "31993L0013"]
    assert answer.raw["question_id"] == 42


def test_generate_without_assistant_is_a_config_error():
    system = _system(lambda r: httpx.Response(200, json={}))
    with pytest.raises(KeyError, match="assistant"):
        system.generate("q", ["bench_cs"], "direct")
