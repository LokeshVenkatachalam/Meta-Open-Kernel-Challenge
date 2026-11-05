# Prefill layer

[`submission.py`](submission.py) is the exact retained source for `muse_prefill_layer`.

> **Public-source reuse:** this kernel is adapted/reused public Apache-licensed work. It is not presented as an independently original kernel. See [provenance](../../PROVENANCE.md), [`LICENSE`](../../LICENSE), and [`NOTICE`](../../NOTICE).

| Field | Verified value |
| --- | --- |
| Scored job | `23b1cc7e2a6040af` |
| Speedup | 1.9191900674402353x |
| 95% CI | [1.8904022164286318, 1.9479779184518387] |
| Score / baseline | 201858.016344 us / 387403.9 us |
| Provisional rank | 107 of 898; leader 2.1471x |
| SHA-256 | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` |
| Exact-source check | `1342300976314ba1`, 1.9125244996506285x |

## Design and lineage

The pinned [keell0renz source](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py) combines ZhimaoL N72 FP8-query FMHA, pinned cuBLASLt projection plans, fused layer boundaries, and a persistent grouped W4A16 implementation. Its retained notices identify vjeux, alygg77, ZhimaoL, and other incorporated lineages.

## Limits

Later local tuning did not beat these retained public bytes: spread expert ordering reached 1.9069x, a fresh-thread context wrapper reached 1.9023x, and split large-M expert phases reached 1.8915x. None is claimed as an improvement. The ranking remains provisional.

See [performance](../../PERFORMANCE.md), [experiments](../../EXPERIMENTS.md), and the [public result dataset](../../docs/data/results.json).
