# Performance

The closed five-agent campaign ran from **2026-11-05 06:26:06.830 UTC** to the **08:00 UTC** kernel-track cutoff. Every kernel was measured by the official remote judge on **NVIDIA H200** hardware. This workstation was used only for editing, hashing, JSON export, and Python syntax checks; no local GPU was used.

All five retained scored kernels beat the SGLang baseline. None won its leaderboard. Rankings are provisional public metadata retrieved at **2026-11-05T20:28:37Z**; `placed` is the leaderboard API's `ranked` count, not the length of its returned rows.

## Retained scored results

| Problem | Scored job | Speedup vs SGLang | 95% confidence interval | Score (us) | Baseline (us) | Provisional rank | Leader |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| [BF16 projections](kernels/muse_bf16_proj/README.md) | `c9e908a969fc4e40` | 1.131876249x | [1.114898105, 1.148854392] | 4267.958 | 4830.8 | 1510 / 1746 | 1.7277x |
| [Attention decode](kernels/muse_attention_decode/README.md) | `b45e9263f36b4141` | 1.151975904x | [1.134696265, 1.169255542] | 3269.426 | 3766.3 | 1278 / 1379 | 3.3389x |
| [Routed experts](kernels/muse_moe_w4a16/README.md) | `9c6a4cb6794448ba` | 1.340288768x | [1.320184437, 1.360393100] | 10034.032 | 13448.5 | 1131 / 1425 | 2.8438x |
| [Decoder layer](kernels/muse_decode_layer/README.md) | `9187326d522142a2` | 1.185588351x | [1.167804526, 1.203372177] | 19079.388 | 22620.3 | 1172 / 1299 | 2.6144x |
| [Prefill layer](kernels/muse_prefill_layer/README.md) | `23b1cc7e2a6040af` | 1.919190067x | [1.890402216, 1.947977918] | 201858.016 | 387403.9 | 107 / 898 | 2.1471x |

A speedup above 1.0x means the official aggregate score was faster than SGLang. The interval is the judge-reported 95% confidence interval for that ratio. The score is an aggregate in microseconds, not a single kernel-launch latency.

## Scored progression and best checks

A `run` is a scored leaderboard submission. A `check` uses the same remote correctness and timing path but stays off-board. The best check can therefore differ from both the best score and the check paired with the exact retained source.

| Problem | First campaign score | Final score | Highest passing check | Exact-source paired check |
| --- | ---: | ---: | --- | --- |
| BF16 projections | 1.108101647x | 1.131876249x | `cb3815b0cc0f42c9`, 1.127767258x | `84c15c3cb1bb4f73` |
| Attention decode | 1.129312485x | 1.151975904x | `31188b3e4b2d482b`, 1.152472784x | `31188b3e4b2d482b` |
| Routed experts | 1.167268453x | 1.340288768x | `7cbc989dbf43455b`, 1.340147020x | `7cbc989dbf43455b` |
| Decoder layer | 1.081701868x | 1.185588351x | `e6bcc1d8cd1a44a7`, 1.164398598x | `027f7d1e62ed4951` |
| Prefill layer | 1.785899219x | 1.919190067x | `1342300976314ba1`, 1.912524500x | `1342300976314ba1` |

The decoder's highest check was accepted before the cutoff and finished at 08:01:34 UTC. It remained an off-board check and did not exceed the retained scored result. The BF16 highest check belongs to an earlier candidate; source association uses the paired check instead.

## Campaign scope versus account scope

The public run ledger contains the complete **79-job campaign**: 18 scored runs and 61 checks. The account-wide totals for 2026-11-05 are **92 jobs**: 20 scored runs and 72 checks, including 13 earlier jobs that are not attributed to this campaign and are not expanded into public run records.

| Problem | Campaign scored | Campaign checks | Campaign jobs | Account scored | Account checks | Account jobs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 projections | 4 | 17 | 21 | 5 | 20 | 25 |
| Attention decode | 3 | 9 | 12 | 3 | 12 | 15 |
| Routed experts | 2 | 14 | 16 | 2 | 16 | 18 |
| Decoder layer | 7 | 13 | 20 | 8 | 14 | 22 |
| Prefill layer | 2 | 8 | 10 | 2 | 10 | 12 |
| **Total** | **18** | **61** | **79** | **20** | **72** | **92** |

Per-problem account aggregates reconcile the official end-of-track quota snapshots with the supplied account-wide totals. They are counts only. Earlier source bytes are not promoted to verified campaign artifacts.

## Data files

- [`docs/data/results.json`](docs/data/results.json) contains the five retained results, rankings, counts, source hashes, narrative, and references.
- [`docs/data/runs.json`](docs/data/runs.json) contains all 79 campaign jobs using only the public allowlist.
- [`verified_sources.json`](verified_sources.json) identifies the exact retained source, plain SHA-256, scored job, and paired check.

The three compile-failed checks have `passed: false` and `null` for speedup, confidence interval, score, and baseline. No failed or unreported measurement is represented as 0x or 1x.
