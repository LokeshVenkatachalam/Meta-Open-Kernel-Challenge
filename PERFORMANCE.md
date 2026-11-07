# Performance

> **Agentic/originality disclosure:** the entire workflow - code assembly and adaptation, experiments, submissions, documentation, and website - was AI-agent-driven. The four included results below have not received a comprehensive originality audit and may contain copied or adapted material. Prefill is excluded because both scored submissions were confirmed verbatim copies and withdrawn. See [DISCLOSURE.md](DISCLOSURE.md).

The closed campaign ran from **2026-11-05 06:26:06.830 UTC** to the **08:00 UTC** kernel-track cutoff on the official remote **NVIDIA H200** judge. Local work was limited to editing, hashing, export, and syntax checks.

## Included measured results

These four historical measurements remain included. Inclusion reports judge data; it does not certify original authorship or challenge eligibility. None was first on the provisional leaderboard snapshot retrieved at **2026-11-05T20:28:37Z**.

| Problem | Scored job | Speedup vs SGLang | 95% confidence interval | Score (us) | Baseline (us) | Historical provisional rank | Leader at snapshot |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| [BF16 projections](kernels/muse_bf16_proj/README.md) | `c9e908a969fc4e40` | 1.131876249x | [1.114898105, 1.148854392] | 4267.958 | 4830.8 | 1510 / 1746 | 1.7277x |
| [Attention decode](kernels/muse_attention_decode/README.md) | `b45e9263f36b4141` | 1.151975904x | [1.134696265, 1.169255542] | 3269.426 | 3766.3 | 1278 / 1379 | 3.3389x |
| [Routed experts](kernels/muse_moe_w4a16/README.md) | `9c6a4cb6794448ba` | 1.340288768x | [1.320184437, 1.360393100] | 10034.032 | 13448.5 | 1131 / 1425 | 2.8438x |
| [Decoder layer](kernels/muse_decode_layer/README.md) | `9187326d522142a2` | 1.185588351x | [1.167804526, 1.203372177] | 19079.388 | 22620.3 | 1172 / 1299 | 2.6144x |

A speedup above 1.0x means the judge's historical aggregate was faster than SGLang. The interval and microsecond values are preserved official measurements.

## Included progression and checks

| Problem | First campaign score | Selected score | Highest passing check | Exact-source paired check |
| --- | ---: | ---: | --- | --- |
| BF16 projections | 1.108101647x | 1.131876249x | `cb3815b0cc0f42c9`, 1.127767258x | `84c15c3cb1bb4f73` |
| Attention decode | 1.129312485x | 1.151975904x | `31188b3e4b2d482b`, 1.152472784x | `31188b3e4b2d482b` |
| Routed experts | 1.167268453x | 1.340288768x | `7cbc989dbf43455b`, 1.340147020x | `7cbc989dbf43455b` |
| Decoder layer | 1.081701868x | 1.185588351x | `e6bcc1d8cd1a44a7`, 1.164398598x | `027f7d1e62ed4951` |

The decoder's highest check was accepted before the cutoff and finished afterward. It remained off-board and did not exceed the selected scored measurement.

## Excluded prefill history

Prefill has no active score, check, ranking, or progression value in schema-version-2 results. The historical ledger is retained without attributing the copied kernels as project achievements.

| Historical scored job | Historical measured speedup | Confirmed source disposition | Withdrawal |
| --- | ---: | --- | --- |
| `c43f7aba5ae84778` | 1.785899219x | Complete unchanged copy of the pinned Madhavan113 public kernel | Confirmed 2026-11-07 09:04:20 UTC |
| `23b1cc7e2a6040af` | 1.919190067x | Complete unchanged copy of the pinned team keell0renz public kernel | Confirmed 2026-11-07 09:04:22 UTC |

No active prefill leaderboard entry remained when verified at 09:04:37 UTC. This statement is a verification result, not a claim of organizer punishment or an eligibility ruling.

## Historical attempt scope

Withdrawal changes result disposition, not the immutable attempt ledger. The public run data still contains all **79 campaign jobs**: 18 scored runs and 61 checks. The account snapshot remains 92 jobs, including 13 earlier attempts.

| Problem | Campaign scored | Campaign checks | Campaign jobs | Publication disposition |
| --- | ---: | ---: | ---: | --- |
| BF16 projections | 4 | 17 | 21 | Included; originality not comprehensively audited |
| Attention decode | 3 | 9 | 12 | Included; originality not comprehensively audited |
| Routed experts | 2 | 14 | 16 | Included; originality not comprehensively audited |
| Decoder layer | 7 | 13 | 20 | Included; originality not comprehensively audited |
| Prefill layer | 2 | 8 | 10 | Excluded historical records; both scored jobs withdrawn |
| **Total** | **18** | **61** | **79** | Four included problems, one excluded problem |

## Data files

- [`docs/data/results.json`](docs/data/results.json) is schema version 2 with the mandatory agentic disclosure, four included results, and one excluded historical problem.
- [`docs/data/runs.json`](docs/data/runs.json) remains schema version 1 with the unchanged 79 records and 11-field allowlist.
- [`docs/data/withdrawal.json`](docs/data/withdrawal.json) contains public-safe withdrawal evidence.
- [`verified_sources.json`](verified_sources.json) records source hashes, audit status, and both confirmed prefill byte matches.

The three compile-failed checks still have `passed: false` and null timing metrics. No measurement fact was deleted, replaced with zero, or otherwise falsified.
