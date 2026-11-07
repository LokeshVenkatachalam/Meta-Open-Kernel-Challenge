# Curated data sources

These committed files are the offline inputs for [`scripts/export_results.py`](../scripts/export_results.py):

- `results-source.json` is schema version 2 and contains the mandatory fully-agentic disclosure, four included unaudited result selectors, one excluded prefill record, source identity, and historical narrative.
- `runs-source.json` remains schema version 1 with the unchanged 79-job, 11-field historical ledger.
- `withdrawal-evidence.json` contains only the two prefill scored job IDs, withdrawal actions and timestamps, and the verification that no active prefill entry remained.

The exporter verifies all five unchanged kernel hashes and Python syntax, derives historical measurements from the run ledger, nulls active prefill metrics, preserves them under `historical`, and writes:

- [`docs/data/results.json`](../docs/data/results.json)
- [`docs/data/runs.json`](../docs/data/runs.json)
- [`docs/data/withdrawal.json`](../docs/data/withdrawal.json)

Rebuild and verify without network or GPU access:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --check
```

`--archive-root` can refresh only the sanitized historical run ledger from a private archive. It cannot remove the mandatory disclosure or restore prefill as an active result.
