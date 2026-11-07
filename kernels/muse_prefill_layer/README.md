# Prefill layer - excluded confirmed-copy history

> **Confirmed verbatim copying:** both scored prefill submissions were complete unchanged copies of pinned public kernels. This problem is excluded from active result summaries, and both scored jobs were withdrawn. The entire project workflow was AI-agent-driven. See [DISCLOSURE.md](../../DISCLOSURE.md).

[`submission.py`](submission.py) remains byte-identical to the final copied public source only as historical evidence. Its bytes have not been modified by this correction.

## Current disposition

| Field | Value |
| --- | --- |
| Included in active summary | No |
| Active score / check / ranking / progression | `null` |
| Originality status | `confirmed-verbatim-copy` |
| Active prefill entry after verification | None at 2026-11-07 09:04:37 UTC |
| Published historical-source SHA-256 | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` |

## Confirmed copied scored sources

| Historical scored job | Historical measured speedup | Exact copied source | Bytes | Plain SHA-256 | Withdrawal confirmed |
| --- | ---: | --- | ---: | --- | --- |
| `c43f7aba5ae84778` | 1.785899219x | [Madhavan113 pinned submission](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py) | 347010 | `76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4` | 2026-11-07 09:04:20 UTC |
| `23b1cc7e2a6040af` | 1.919190067x | [team keell0renz pinned kernel](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py) | 370116 | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` | 2026-11-07 09:04:22 UTC |

These were outright byte-for-byte copies, not merely inspired or adapted implementations and not speedups authored by this project. The historical judge measurements remain in [`runs.json`](../../docs/data/runs.json), but schema-version-2 [`results.json`](../../docs/data/results.json) places the old selected score, check, ranking, and first-score value only under `historical`.

Applicable public-source licenses and notices remain in [`LICENSE`](../../LICENSE) and [`NOTICE`](../../NOTICE). Licensing permission does not establish independent authorship or organizer approval.

No claim is made that organizers imposed a ban or disqualification, made an originality ruling, approved eligibility, or received a private message. Public-safe withdrawal evidence is in [`withdrawal.json`](../../docs/data/withdrawal.json).

See [performance disposition](../../PERFORMANCE.md), [experiment history](../../EXPERIMENTS.md), and [provenance](../../PROVENANCE.md).
