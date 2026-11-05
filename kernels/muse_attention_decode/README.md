# Attention decode

[`submission.py`](submission.py) is the exact retained source for `muse_attention_decode`.

| Field | Verified value |
| --- | --- |
| Scored job | `b45e9263f36b4141` |
| Speedup | 1.151975903562266x |
| 95% CI | [1.1346962650088321, 1.1692555421157] |
| Score / baseline | 3269.426026 us / 3766.3 us |
| Provisional rank | 1278 of 1379; leader 3.3389x |
| SHA-256 | `ac3a509b872f775619e91084c71e08eea1cfc99aad7630790842b2334ce4ca19` |
| Exact-source check | `31188b3e4b2d482b`, 1.1524727843278246x |

## Design

One Triton preparation kernel fuses Q/K RMS normalization, optional interleaved RoPE, FP8 query conversion, and scaled K/V cache stores. FlashAttention-3 remains responsible for exact paged attention. The retained variant uses fixed uniform-query shapes where valid and reuses compatible scheduler metadata to avoid another prepare step.

## Limits

The fixed-query path depends on the challenge's query and cache contract. A correct custom Triton split-KV replacement reached only 0.3974x, so the campaign did not replace FlashAttention-3. Full B16/B32 attention remained the dominant cost, and the result did not approach the provisional leader.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [public result dataset](../../docs/data/results.json).
