"""The adapter protocol — the core of reusability.

Three rules that must hold (see ``docs/adding-a-system.md``):

(a) ``index()`` blocks until documents are searchable.  Asynchronous indexing
    is a system detail, not a protocol concern.
(b) No system-specific knobs in the protocol.  Ablations are expressed as two
    configured instances of the same adapter in ``systems.toml``.
(c) If the protocol ever needs a fourth capability, something is wrong.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Doc:
    doc_id: str          # CELEX — identical across languages
    lang: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    retrieved_doc_ids: list[str]   # in rank order, deduplicated
    cited_doc_ids: list[str]       # from inline citations, if the system emits them
    raw: dict = field(default_factory=dict)  # anything extra, for traceability


@runtime_checkable
class RagSystem(Protocol):
    name: str

    def describe(self) -> dict:
        """Metadata for system.json: embedder, reranker, LLM, versions.

        Read from the *running* system where possible, never assumed.
        """
        ...

    def index(self, docs: Iterable[Doc], index_name: str) -> None:
        """Returns only once the documents are searchable."""
        ...

    def search(self, query: str, index_name: str, k: int) -> list[str]:
        """Returns doc_ids in relevance order, deduplicated."""
        ...

    def generate(self, query: str, index_names: list[str], mode: str) -> Answer:
        """``index_names`` is a list — one element = mono, several = MultiRAG."""
        ...


def load_adapter(spec: str, config: dict) -> RagSystem:
    """Instantiate an adapter from a ``module.path:ClassName`` spec.

    The class is called with the config dict as keyword arguments — the whole
    of ``[[systems]].config`` from ``systems.toml``.
    """
    module_path, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"adapter spec must look like 'pkg.module:ClassName', got {spec!r}")
    cls = getattr(import_module(module_path), class_name)
    system = cls(**config)
    if not isinstance(system, RagSystem):
        raise TypeError(f"{spec} does not implement the RagSystem protocol")
    return system
