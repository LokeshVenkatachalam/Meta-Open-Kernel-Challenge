# Routed experts

> **Agentic/originality disclosure:** AI agents drove this source assembly/adaptation, experimentation, submission, documentation, and website workflow. This file has not received a comprehensive originality audit and may contain copied or adapted material. Its inclusion is not a certification of original authorship. See [DISCLOSURE.md](../../DISCLOSURE.md).

[`submission.py`](submission.py) is the exact published source for `muse_moe_w4a16`.

| Field | Historical measured value |
| --- | --- |
| Publication status | Included result; originality not comprehensively audited |
| Scored job | `9c6a4cb6794448ba` |
| Speedup | 1.3402887683867017x |
| 95% CI | [1.3201844368609013, 1.3603930999125022] |
| Score / baseline | 10034.031708 us / 13448.5 us |
| Historical provisional rank | 1131 of 1425; leader 2.8438x |
| SHA-256 | `157e907cd37e9060e04a6dfda88cc3bde9cb2af2b4a2824717cecbaf3031106d` |
| Exact-source check | `7cbc989dbf43455b`, 1.340147019786917x |

## Technical record

The source uses SGLang's Marlin path, shape-dependent block sizes, local expert-ID mapping, invalid-route filtering, and a fused post-expert normalization and accumulation kernel. The historical timing range ends at 256 routed tokens.

H200 has no Blackwell native NVFP4 tensor-core path, so the file dequantizes for supported Hopper compute. These technical and measurement statements do not establish that the included source is original.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [schema-version-2 result dataset](../../docs/data/results.json).
