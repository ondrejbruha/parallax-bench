# Releasing

Short version: git tag → GitHub release → workflow → PyPI, with no token
stored anywhere. Publishing uses PyPI **Trusted Publishing** (OIDC).

## Versioning

- **Code**: semver from the git tag via `hatch-vcs` — the version is written
  nowhere in the repo. Tag `v0.1.0` ⇒ package `parallax-bench 0.1.0`; the
  tag and `pyproject.toml` cannot drift apart.
- **Dataset**: `v1`, `v2`, … — independent of code versions, each with its
  own Zenodo DOI. A released dataset version is never modified in place.

## One-time setup

1. **PyPI pending publisher** (account needs 2FA): Your account → Publishing
   → Add a pending publisher:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `parallax-bench` |
   | Owner | GitHub user/org |
   | Repository name | `parallax-bench` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   "Pending" is exactly for a project that does not exist on PyPI yet — the
   first successful workflow run creates it. No secret lives in GitHub.

2. **Zenodo**: enable the GitHub–Zenodo integration for the repo so every
   GitHub release mints a DOI; put the concept DOI into `CITATION.cff`.

## Packaging gotcha: bundled data

`benchmark/` lives outside `src/`, so the wheel only contains it thanks to
`force-include` in `pyproject.toml` (mapped to `parallax_bench/_data/`).
Without it, the quickstart breaks after `pip install`. Code accesses it via
`importlib.resources`, never via paths relative to the repo — relative paths
work from git and break after installation.

Check the wheel content before burning a version number:

```bash
pipx run build
python -m zipfile -l dist/parallax_bench-*.whl | grep _data
pipx run twine check dist/*
```

## Dry run on TestPyPI

TestPyPI is a separate instance with its own account and its own pending
publisher; in a copy of the workflow add
`with: { repository-url: https://test.pypi.org/legacy/ }`. Then verify the
two things that most often break:

```bash
pip install -i https://test.pypi.org/simple/ parallax-bench
parallax-bench --help
parallax-bench run --system baseline-local --subset smoke
```

The second command proves the `force-include` data actually made it in.

## The release ritual

```bash
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```

One tag triggers everything: `release.yml` builds and publishes to PyPI via
OIDC, Zenodo mints a DOI for the release. The citation then points at an
installable version — exactly the traceability an examiner will ask for.

Notes:

- `fetch-depth: 0` in the workflow is required — a shallow checkout has no
  tags and `hatch-vcs` would produce a `0.1.devN+…` version.
- PyPI never allows re-uploading an existing version. A broken 0.1.0 means
  releasing 0.1.1 — hence the TestPyPI dry run.
- Release early: once `baseline_local` + smoke work, 0.1.0 is a legitimate
  alpha, and it claims the name.
