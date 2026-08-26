# Adding a system

A system is an adapter class plus configuration in `systems.toml`. The
adapter protocol (`parallax_bench.adapters.base`) has exactly four methods:

```python
class RagSystem(Protocol):
    name: str
    def describe(self) -> dict: ...
    def index(self, docs: Iterable[Doc], index_name: str) -> None: ...
    def search(self, query: str, index_name: str, k: int) -> list[str]: ...
    def generate(self, query: str, index_names: list[str], mode: str) -> Answer: ...
```

## The three rules

**(a) `index()` blocks until documents are searchable.** If your system
indexes asynchronously, poll inside the adapter. Async indexing is a system
detail, not a protocol concern — an `await_ready()` in the protocol would
leak one system's shape into everyone's interface.

**(b) No system-specific knobs in the protocol.** Toggling reranking,
choosing an embedder, chunking parameters — all of that is *adapter
configuration*. An ablation is two configured instances of the same adapter:

```toml
[[systems]]
id = "my-system-rerank"
adapter = "my_pkg.adapter:MySystem"
config = { base_url = "${MY_BASE_URL}", rerank = true }

[[systems]]
id = "my-system-norerank"
adapter = "my_pkg.adapter:MySystem"
config = { base_url = "${MY_BASE_URL}", rerank = false }
```

**(c) If the protocol needs a fourth capability, something is wrong.** The
protocol stayed useful across a local BM25 baseline and a production HTTP
RAG stack with exactly these four methods. Extend your adapter's
configuration, not the protocol. (This rule is written down precisely so it
survives its author.)

## Contract details

- `Doc.doc_id` is the cross-language document identifier (CELEX in the
  bundled datasets). Your system must return these ids from `search()` —
  in rank order, deduplicated (first occurrence wins if your system returns
  chunks).
- `search()` should honor `k` up to at least 100 (Recall@100 is a default
  metric).
- `describe()` should report what is *actually deployed* — read versions
  from the running system where possible, don't echo your own config.
- `generate()` gets a list of index names: one element means monolingual
  RAG, several mean MultiRAG over all of them. `mode` is passed through
  from the run configuration (e.g. `direct` / `agentic`); ignore it if your
  system has one mode.
- `Answer.cited_doc_ids` — if your system emits inline citations, resolve
  them to doc_ids; otherwise return `[]` and citation metrics will be
  skipped for you.

## Checklist

1. Implement the class; config dict = constructor kwargs. A single
   `my_adapter.py` next to `systems.toml` is enough — that directory is put
   on `sys.path`, so `adapter = "my_adapter:MyClass"` works without
   packaging. No inheritance or registration: `RagSystem` is a Protocol,
   structural conformance is all that is checked.
2. Add it to `systems.toml` (secrets via `${ENV_VAR}`, never literals).
3. `parallax-bench ingest --system my-system --subset smoke`
4. `parallax-bench run --system my-system --subset smoke`
5. `parallax-bench score && parallax-bench report` — the smoke diagonal
   should be near-perfect for any reasonable system; if it is ~0, your
   doc_ids don't match the manifest's CELEX ids.
