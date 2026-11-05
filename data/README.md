# Curated data sources

These files are the committed, offline inputs for [`scripts/export_results.py`](../scripts/export_results.py):

- `results-source.json` contains project metadata, selector job IDs, leaderboard snapshot values, account aggregates, source identity, narrative, and references.
- `runs-source.json` contains the complete sanitized 79-job campaign ledger using the same public allowlist as the generated site data.

The exporter derives selected metrics and campaign counts from `runs-source.json`, verifies all five kernel hashes and Python syntax, validates rankings and account totals, and writes:

- [`docs/data/results.json`](../docs/data/results.json)
- [`docs/data/runs.json`](../docs/data/runs.json)

Rebuild and verify without network or GPU access:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --check
```

`--archive-root` is an explicit maintainer-only refresh path for a private official-job archive. It copies only allowlisted public fields; raw grader output is never a committed input.
