# Decoder layer

[`submission.py`](submission.py) is the exact retained source for `muse_decode_layer`.

| Field | Verified value |
| --- | --- |
| Scored job | `9187326d522142a2` |
| Speedup | 1.1855883514963717x |
| 95% CI | [1.167804526223926, 1.203372176768817] |
| Score / baseline | 19079.387860 us / 22620.3 us |
| Provisional rank | 1172 of 1299; leader 2.6144x |
| SHA-256 | `34ae067a20a1c69dd52acfcb52d74e164499de9af7971e78f4e0717a71beddaa` |
| Exact-source check | `027f7d1e62ed4951`, 1.162747432039166x |

## Design

The whole-layer path fuses Q/K normalization, RoPE, FP8 query conversion, and cache stores around the proven attention implementation. Its expert section uses shape-tuned Marlin blocks, a Triton pre-expert RMS normalization, fused global-to-local expert mapping, and the existing post-expert normalization and accumulation boundary.

The highest check, `e6bcc1d8cd1a44a7` at 1.1643985977658828x, was accepted before the cutoff but finished after close. It remained off-board and did not exceed the scored result.

## Limits

Whole-layer timing depends on both attention and expert shapes: FlashAttention dominates long full-context cases, while Marlin dominates sliding and verify cases. Primitive improvements are not claimed as layer improvements without an official layer check.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [public result dataset](../../docs/data/results.json).
