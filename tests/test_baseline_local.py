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
