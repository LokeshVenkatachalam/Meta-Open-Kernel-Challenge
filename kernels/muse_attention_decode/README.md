# Attention decode

> **Agentic/originality disclosure:** AI agents drove this source assembly/adaptation, experimentation, submission, documentation, and website workflow. This file has not received a comprehensive originality audit and may contain copied or adapted material. Its inclusion is not a certification of original authorship. See [DISCLOSURE.md](../../DISCLOSURE.md).

[`submission.py`](submission.py) is the exact published source for `muse_attention_decode`.

| Field | Historical measured value |
| --- | --- |
| Publication status | Included result; originality not comprehensively audited |
| Scored job | `b45e9263f36b4141` |
| Speedup | 1.151975903562266x |
| 95% CI | [1.1346962650088321, 1.1692555421157] |
| Score / baseline | 3269.426026 us / 3766.3 us |
| Historical provisional rank | 1278 of 1379; leader 3.3389x |
| SHA-256 | `ac3a509b872f775619e91084c71e08eea1cfc99aad7630790842b2334ce4ca19` |
| Exact-source check | `31188b3e4b2d482b`, 1.1524727843278246x |

## Technical record

The source fuses Q/K RMS normalization, optional interleaved RoPE, FP8 query conversion, and scaled K/V cache stores around FlashAttention-3. It uses fixed uniform-query shapes where valid and reuses scheduler metadata.

A correct custom Triton split-KV alternative historically measured 0.3974x. These technical and measurement statements do not establish that the included source is original.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [schema-version-2 result dataset](../../docs/data/results.json).
