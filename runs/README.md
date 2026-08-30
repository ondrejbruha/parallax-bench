# runs/

Results as data, not as claims. Each committed run directory contains:

```
runs/<date>-<name>-<id>/
├── config.json      # resolved run configuration, secrets visibly redacted
├── system.json      # what actually ran on the other side (read, not assumed)
├── manifest.lock    # (celex, lang) → binding sha256_text of every indexed document
├── metrics.csv      # aggregated metrics — small and diffable
├── per_query.csv    # per-query metric values
├── parallax_summary.csv/json  # global derived metrics, including grouped-bootstrap CI
├── matrices/        # absolute, CLP, EN-delta, asymmetry CSVs and report PNGs
├── stats.csv        # paired tests vs the baseline language
└── README.md        # provenance summary + link to raw rankings (release asset)
```

Only aggregated outputs go to git; raw rankings are attached to GitHub
releases so anyone can recompute metrics under their own definitions.
Origin-specific matrix files are nested under `matrices/origin_<origin>/` and
are created only for origins present in the query data.

To contribute your own system's results, open a PR adding a run directory —
CI validates the schema. There is no submission leaderboard by design.
