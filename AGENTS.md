# AGENTS.md

This file defines the working conventions for AI coding agents contributing to this repository. It applies to the entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## Project Overview

Parallax Bench is a Python benchmark for measuring language-induced retrieval displacement in RAG systems using parallel corpora. The project emphasizes reproducibility, frozen benchmark inputs, explicit provenance, and a clean separation between data collection and scoring.

The package targets Python 3.12 and newer, uses a `src/` layout, and exposes the `parallax-bench` CLI through Typer.

## Repository Structure

- `src/parallax_bench/` — application and library code.
  - `adapters/` — system integration protocol and concrete adapters.
  - `runner/` — ingestion, retrieval, generation, planning, and queue execution.
  - `metrics/` — retrieval, generation, and statistical metrics.
  - `cli.py` — command-line interface and command wiring.
  - `config.py`, `data.py`, and `fetch.py` — configuration and benchmark data handling.
  - `scoring.py` — scoring orchestration.
- `tests/` — pytest test suite. Test files generally mirror a module or end-to-end behavior.
- `benchmark/` — versioned benchmark definitions, manifests, queries, qrels, smoke data, and dataset build scripts.
- `corpus-store/` — locally fetched corpus content indexed by hash. Treat it as data, not source code.
- `runs/` — reproducible benchmark outputs and their provenance.
- `docs/` — methodology, data formats, system integration, limitations, and release documentation.
- `deploy/` — deployment-specific configuration.
- `systems.example.toml` — example system adapter configuration.
- `pyproject.toml` — package metadata, dependencies, and tool configuration.

## Before Making Changes

1. Read the relevant implementation, tests, documentation, and nearby conventions before editing.
2. Inspect the current working tree and preserve all unrelated user changes.
3. Confirm the requested outcome, scope, constraints, and acceptance criteria.
4. Ask focused clarification questions whenever any material part of the task is ambiguous. Continue asking until the assignment is fully understood; do not guess when different interpretations could produce meaningfully different results.
5. For small ambiguities that are safely reversible and do not affect behavior or scope, state the assumption and proceed.

## Architecture and Design Rules

- Preserve the existing architecture and separation of responsibilities.
- Extend established abstractions before introducing parallel mechanisms.
- Keep adapter-specific behavior inside `adapters/`; do not leak it into generic runner, scoring, or metric code.
- Keep collection/execution separate from scoring so metrics can be recomputed without rerunning collection.
- Keep CLI handlers thin. Put reusable domain logic in focused modules.
- Maintain the four-method adapter protocol (`describe`, `index`, `search`, and `generate`) unless the user explicitly requests a protocol change.
- Treat benchmark versions as immutable once released. Never silently modify frozen benchmark data, qrels, queries, manifests, or published run provenance.
- Preserve reproducibility: configuration must be explicit, secrets must be redacted, and outputs must retain enough provenance to be verified.
- Prefer the smallest coherent change. Avoid unrelated refactors, formatting churn, or speculative abstractions.
- Maintain backward compatibility unless a breaking change is explicitly requested and documented.

## Implementation Standards

- Write clear, typed, idiomatic Python compatible with Python 3.12+.
- Follow the Ruff configuration in `pyproject.toml`, including the 100-character line length.
- Match existing naming, module organization, error handling, and test patterns.
- Favor simple, explicit code over cleverness.
- Validate inputs at system boundaries and produce actionable error messages.
- Do not add dependencies unless they are necessary. Explain and document any new dependency.
- Whenever code changes, review all related documentation across the repository and update every affected location in the same change. This includes READMEs, files in `docs/`, docstrings, examples, CLI help, configuration samples, comments, changelog entries, and any other text that describes the changed behavior or interface.
- Do not leave documentation that is stale, contradictory, or only partially updated. Search the repository for references to renamed concepts, commands, options, APIs, defaults, and behavior before considering the change complete.
- Never expose credentials, tokens, private URLs, or other secrets in source code, logs, fixtures, or generated artifacts.

## Testing and Verification

Every change must be verified with tests.

- Add or update tests for every behavioral change and bug fix.
- Run the narrowest relevant tests during development, then run the full test suite before considering the task complete.
- Run linting and type checks for changed Python code.
- Use the project-configured commands:

```bash
pytest
ruff check .
mypy src
```

- When appropriate, also run the offline smoke workflow described in `README.md` to verify CLI or end-to-end changes.
- Do not claim that a change works unless it has been verified.
- If a required check cannot be run, clearly report which check was skipped, why it was skipped, and the remaining risk.
- Never fix a failing test by weakening or deleting valid assertions unless the intended behavior has explicitly changed.

## Git and Change Safety

- Never commit, amend, rebase, tag, push, or open a pull request on the user's behalf unless the user explicitly asks for that exact action.
- Do not stage files unless explicitly requested.
- Never discard or overwrite unrelated user changes.
- Do not use destructive Git commands such as `git reset --hard` or forced checkout without explicit permission.
- Keep changes scoped to the task and make diffs easy to review.
- Before handing off, inspect the final diff and summarize the files changed and the verification performed.

## Data and Generated Artifacts

- Do not edit generated files when the source generator can be changed instead.
- Do not regenerate benchmark data, manifests, corpus content, or run outputs unless the task explicitly requires it.
- Preserve stable document identifiers, hashes, language mappings, and TREC-compatible formats.
- Avoid committing caches, local databases, temporary files, downloaded models, or machine-specific configuration.
- Respect the repository's split licensing: code and benchmark data may have different licenses.

## Definition of Done

A task is complete only when:

- the requested behavior is implemented and the scope is fully addressed;
- the implementation follows the existing architecture and conventions;
- relevant tests were added or updated;
- the relevant tests pass, followed by the full test suite where feasible;
- linting and type checks pass for the affected code;
- all documentation and examples affected by code changes are updated consistently across the repository;
- the final diff contains no unrelated changes; and
- the handoff clearly states what changed, what was verified, and any remaining limitations.
