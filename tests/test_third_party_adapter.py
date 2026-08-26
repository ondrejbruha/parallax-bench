"""The third-party flow: pip install, one adapter file next to systems.toml."""

from pathlib import Path

from parallax_bench.config import get_system

ADAPTER_SRC = '''
from parallax_bench.adapters.base import Answer

class ToyRag:
    name = "toy"

    def __init__(self, greeting: str):
        self.greeting = greeting

    def describe(self):
        return {"adapter": "toy", "greeting": self.greeting}

    def index(self, docs, index_name):
        pass

    def search(self, query, index_name, k):
        return ["32016R0679"][:k]

    def generate(self, query, index_names, mode):
        return Answer(text=self.greeting, retrieved_doc_ids=[], cited_doc_ids=[])
'''

SYSTEMS_TOML = '''
[[systems]]
id = "toy-system"
adapter = "toy_adapter:ToyRag"
config = { greeting = "hello" }
'''


def test_single_file_adapter_next_to_systems_toml(tmp_path: Path):
    (tmp_path / "toy_adapter.py").write_text(ADAPTER_SRC, encoding="utf-8")
    toml_path = tmp_path / "systems.toml"
    toml_path.write_text(SYSTEMS_TOML, encoding="utf-8")

    cfg = get_system("toy-system", toml_path)
    system = cfg.instantiate()  # imports toy_adapter without PYTHONPATH tricks
    assert system.describe()["greeting"] == "hello"
    assert system.search("q", "bench_cs", 10) == ["32016R0679"]
