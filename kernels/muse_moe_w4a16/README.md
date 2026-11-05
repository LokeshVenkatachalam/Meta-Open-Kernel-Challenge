# Routed experts

[`submission.py`](submission.py) is the exact retained source for `muse_moe_w4a16`.

| Field | Verified value |
| --- | --- |
| Scored job | `9c6a4cb6794448ba` |
| Speedup | 1.3402887683867017x |
| 95% CI | [1.3201844368609013, 1.3603930999125022] |
| Score / baseline | 10034.031708 us / 13448.5 us |
| Provisional rank | 1131 of 1425; leader 2.8438x |
| SHA-256 | `157e907cd37e9060e04a6dfda88cc3bde9cb2af2b4a2824717cecbaf3031106d` |
| Exact-source check | `7cbc989dbf43455b`, 1.340147019786917x |

## Design

The kernel retains SGLang's lossless Marlin repack and chooses block 8 through 16 routed tokens, then block 16 for larger batches. A Triton kernel maps global IDs to the local expert range. Larger shapes remove invalid routes before non-EP Marlin, and a fused finalizer applies per-expert RMS normalization, route weights, and FP32 accumulation.

## Limits

The verified preparation assumes EP8 with local experts 0-31, and the measured timing range ends at 256 routed tokens. H200 has no Blackwell native NVFP4 tensor-core path, so weights are dequantized for supported Hopper compute. The two Marlin GEMMs remained roughly 94-95% of large-shape GPU time.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [public result dataset](../../docs/data/results.json).
