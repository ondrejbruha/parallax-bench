from parallax_bench.adapters.base import Doc, RagSystem, load_adapter
from parallax_bench.adapters.baseline_local import BaselineLocalSystem, _rrf


def _docs():
    return [
        Doc("D1", "en", "The withdrawal period for distance contracts is fourteen days.\n\n"
                        "The trader shall reimburse all payments."),
        Doc("D2", "en", "Comparative advertising is permitted when it compares goods "
                        "meeting the same needs.\n\nMisleading advertising is prohibited."),
        Doc("D3", "en", "Traffic data may be processed for marketing with prior consent.\n\n"
                        "Cookies require clear and comprehensive information."),
    ]


def test_index_search_roundtrip_and_persistence(tmp_path):
    system = BaselineLocalSystem(data_dir=str(tmp_path))
    system.index(_docs(), "bench_en")
    assert system.search("withdrawal period distance contracts", "bench_en", 3)[0] == "D1"
    assert system.search("comparative advertising permitted", "bench_en", 3)[0] == "D2"

    # a fresh instance (new CLI invocation) sees the same index on disk
    reloaded = BaselineLocalSystem(data_dir=str(tmp_path))
    assert reloaded.search("cookies consent", "bench_en", 3)[0] == "D3"


def test_reingest_is_idempotent(tmp_path):
    system = BaselineLocalSystem(data_dir=str(tmp_path))
    system.index(_docs(), "bench_en")
    system.index(_docs(), "bench_en")  # run --no-ingest not passed → happens in practice
    ranking = system.search("withdrawal period", "bench_en", 10)
    assert ranking.count("D1") == 1


def test_generate_is_extractive_and_cites(tmp_path):
    system = BaselineLocalSystem(data_dir=str(tmp_path))
    system.index(_docs(), "bench_en")
    answer = system.generate("withdrawal period distance contracts", ["bench_en"], "direct")
    assert answer.cited_doc_ids == ["D1"]
    assert "fourteen days" in answer.text
    assert answer.retrieved_doc_ids[0] == "D1"


def test_protocol_conformance(tmp_path):
    system = load_adapter(
        "parallax_bench.adapters.baseline_local:BaselineLocalSystem",
        {"data_dir": str(tmp_path)},
    )
    assert isinstance(system, RagSystem)
    assert system.describe()["lexical"]["type"] == "bm25-okapi"


def test_rrf_fusion_order():
    fused = _rrf([["a", "b", "c"], ["b", "a", "c"]])
    assert fused[:2] == ["a", "b"] or fused[:2] == ["b", "a"]
    assert fused[2] == "c"


def test_dense_mode_requires_model():
    import pytest

    with pytest.raises(ValueError, match="requires embedding_model"):
        BaselineLocalSystem(retrieval_mode="dense")


def test_embedding_model_alone_preserves_hybrid_mode():
    system = BaselineLocalSystem(embedding_model="example/model")
    assert system.describe()["retrieval_mode"] == "hybrid"


def test_e5_dense_mode_uses_prefixes_and_cosine(tmp_path):
    import numpy as np

    class FakeEncoder:
        def __init__(self):
            self.inputs = []

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            self.inputs.extend(texts)
            vectors = []
            for text in texts:
                vectors.append([1.0, 0.0] if "withdrawal" in text else [0.0, 1.0])
            return np.asarray(vectors)

    system = BaselineLocalSystem(
        data_dir=str(tmp_path),
        retrieval_mode="dense",
        embedding_model="intfloat/multilingual-e5-large",
    )
    system.index(_docs(), "bench_en")
    encoder = FakeEncoder()
    system._encoder = encoder
    assert system.search("withdrawal", "bench_en", 1) == ["D1"]
    assert any(text.startswith("passage: ") for text in encoder.inputs)
    assert encoder.inputs[-1].startswith("query: ")
    description = system.describe()
    assert description["retrieval_mode"] == "dense"
    assert description["lexical"] is None
    assert description["dense"]["embedding_model"] == "intfloat/multilingual-e5-large"
    assert description["dense"]["e5_prefixes"] is True
