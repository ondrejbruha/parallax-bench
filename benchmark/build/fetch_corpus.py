"""Standalone corpus fetch — thin wrapper over ``parallax-bench fetch``.

Kept as a script so the dataset is usable even without installing the CLI
entry point (e.g. from a bare checkout):

    python benchmark/build/fetch_corpus.py --data-version v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parallax_bench.data import load_dataset
from parallax_bench.fetch import fetch_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    ds = load_dataset(args.data_version, args.data_dir)
    n_ok, failures = fetch_corpus(ds)
    print(f"{n_ok}/{len(ds.manifest)} documents present and verified")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
