# Decoder layer

> **Agentic/originality disclosure:** AI agents drove this source assembly/adaptation, experimentation, submission, documentation, and website workflow. This file has not received a comprehensive originality audit and may contain copied or adapted material. Its inclusion is not a certification of original authorship. See [DISCLOSURE.md](../../DISCLOSURE.md).

[`submission.py`](submission.py) is the exact published source for `muse_decode_layer`.

| Field | Historical measured value |
| --- | --- |
| Publication status | Included result; originality not comprehensively audited |
| Scored job | `9187326d522142a2` |
| Speedup | 1.1855883514963717x |
| 95% CI | [1.167804526223926, 1.203372176768817] |
| Score / baseline | 19079.387860 us / 22620.3 us |
| Historical provisional rank | 1172 of 1299; leader 2.6144x |
| SHA-256 | `34ae067a20a1c69dd52acfcb52d74e164499de9af7971e78f4e0717a71beddaa` |
| Exact-source check | `027f7d1e62ed4951`, 1.162747432039166x |

## Technical record

The source combines attention preparation, shape-tuned Marlin blocks, pre-expert RMS normalization, local expert mapping, and post-expert normalization and accumulation. Long full-context cases and routed-expert cases have different dominant costs.

The highest check, `e6bcc1d8cd1a44a7` at 1.1643985977658828x, finished after close and remained off-board. These technical and measurement statements do not establish that the included source is original.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), [provenance](../../PROVENANCE.md), and the [schema-version-2 result dataset](../../docs/data/results.json).
