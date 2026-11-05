# BF16 projections

[`submission.py`](submission.py) is the exact retained source for `muse_bf16_proj`.

| Field | Verified value |
| --- | --- |
| Scored job | `c9e908a969fc4e40` |
| Speedup | 1.131876248733277x |
| 95% CI | [1.1148981050022777, 1.148854392464276] |
| Score / baseline | 4267.957743 us / 4830.8 us |
| Provisional rank | 1510 of 1746; leader 1.7277x |
| SHA-256 | `af9c07caaa1227dfe0c6213071f342c2582e02c96440b287a1a8b03e869adfb5` |
| Exact-source check | `84c15c3cb1bb4f73` |

## Design

The retained dispatch uses a pinned Hopper BF16 GEMV for selected M=1 projections, custom split-K paths for the profitable pooler and sliced-unpooler shapes, and the SGLang BF16 dispatcher elsewhere. The router reconstructs BF16 high-plus-low weights, accumulates in FP32, and uses split-major partial storage for a coalesced reduction.

The highest campaign check, `cb3815b0cc0f42c9` at 1.1277672580904239x, came from a different candidate. It is not the source-association check for these bytes.

## Limits

The shape cutovers are specific to the official H200 workload. Atomic router reduction and several broader custom dispatches regressed. This result beat SGLang but did not win the provisional leaderboard.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [public result dataset](../../docs/data/results.json).
