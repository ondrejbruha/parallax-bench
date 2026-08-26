"""Dataset schema and integrity validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from parallax_bench.data import Query, load_dataset, validate_dataset

REPO = Path(__file__).resolve().parent.parent


def _q(group="q00001", lang="cs", **overrides) -> dict:
    row = {
        "query_id": f"{group}_{lang}",
        "query_group": group,
        "lang": lang,
        "text": "Testovací dotaz?",
        "source_celex": "32016R0679",
        "origin": "translated",
        "pivot_lang": "en",
        "query_set_version": "test",
    }
    row.update(overrides)
    return row


def test_query_schema_roundtrip():
    q = Query.model_validate(_q())
    assert q.query_id == "q00001_cs"


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_id": "nonsense"},
        {"query_id": "q00002_cs"},           # group mismatch
        {"lang": "cs", "query_id": "q00001_de"},  # lang mismatch
        {"origin": "invented"},
        {"origin": "translated", "pivot_lang": None},
        {"text": "   "},
    ],
)
def test_query_schema_rejects(overrides):
    with pytest.raises(ValidationError):
        Query.model_validate(_q(**overrides))


def _write_dataset(root: Path, queries: list[dict], qrels: list[str], manifest: list[dict]):
    (root / "queries.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8"
    )
    (root / "qrels.txt").write_text("".join(line + "\n" for line in qrels), encoding="utf-8")
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8"
    )


def _manifest(celex="32016R0679", lang="cs"):
    return {
        "celex": celex,
        "lang": lang,
        "url": "https://example.org/doc",
        "sha256_text": "0" * 64,
        "sha256_source": "1" * 64,
        "extractor": "parallax_bench.fetch.eurlex_html:v1",
        "char_len": 100,
    }


def test_validate_catches_incomplete_group_and_dangling_refs(tmp_path):
    root = tmp_path / "test"
    root.mkdir()
    _write_dataset(
        root,
        queries=[_q(lang="cs"), _q(lang="en", query_id="q00001_en"),
                 _q(group="q00002", lang="cs", query_id="q00002_cs")],
        qrels=["q00001_cs 0 32016R0679 1", "q00001_en 0 32016R0679 1",
               "q00002_cs 0 32016R0679 1", "q99999_cs 0 32016R0679 1"],
        manifest=[_manifest(lang="cs"), _manifest(lang="en")],
    )
    ds = load_dataset("test", tmp_path)
    rep = validate_dataset(ds, check_texts=False)
    assert not rep.ok
    assert any("q99999_cs" in e for e in rep.errors)          # dangling qrel
    assert any("q00002" in e and "variants" in e for e in rep.errors)  # incomplete group


def test_smoke_dataset_is_valid():
    ds = load_dataset("smoke", REPO / "benchmark")
    rep = validate_dataset(ds, check_texts=True)
    assert rep.ok, rep.errors
    assert ds.languages == ["cs", "de", "en"]
    assert len(ds.query_groups) == 20
    # every group complete across all languages — the paired-test invariant
    for group, variants in ds.query_groups.items():
        assert {q.lang for q in variants} == {"cs", "de", "en"}, group
