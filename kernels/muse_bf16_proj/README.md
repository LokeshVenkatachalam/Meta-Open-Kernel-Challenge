# BF16 projections

> **Agentic/originality disclosure:** AI agents drove this source assembly/adaptation, experimentation, submission, documentation, and website workflow. This file has not received a comprehensive originality audit and may contain copied or adapted material. Its inclusion is not a certification of original authorship. See [DISCLOSURE.md](../../DISCLOSURE.md).

[`submission.py`](submission.py) is the exact published source for `muse_bf16_proj`.

| Field | Historical measured value |
| --- | --- |
| Publication status | Included result; originality not comprehensively audited |
| Scored job | `c9e908a969fc4e40` |
| Speedup | 1.131876248733277x |
| 95% CI | [1.1148981050022777, 1.148854392464276] |
| Score / baseline | 4267.957743 us / 4830.8 us |
| Historical provisional rank | 1510 of 1746; leader 1.7277x |
| SHA-256 | `af9c07caaa1227dfe0c6213071f342c2582e02c96440b287a1a8b03e869adfb5` |
| Exact-source check | `84c15c3cb1bb4f73` |

## Technical record

The source dispatches selected M=1 projections to Hopper BF16 GEMV, uses custom split-K paths for measured pooler and sliced-unpooler shapes, and uses the SGLang BF16 dispatcher elsewhere. The router reconstructs BF16 high-plus-low weights and accumulates in FP32.

The highest campaign check, `cb3815b0cc0f42c9` at 1.1277672580904239x, came from a different candidate. Hash and judge association establish identity and measurement, not originality.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [schema-version-2 result dataset](../../docs/data/results.json).
