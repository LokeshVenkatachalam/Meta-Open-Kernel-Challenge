# Agentic workflow and originality correction

> **To organizers and readers:** the entire project workflow - kernel assembly and adaptation, experiment selection, challenge checks and scored submissions, documentation, and website production - was driven by AI agents. This repository is not a claim of independent human authorship.

## Confirmed prefill copying and withdrawal

Both scored submissions for `muse_prefill_layer` were confirmed as complete, byte-for-byte copies of pinned public kernels. They were not merely inspired by those sources and are not presented as speedups authored by this project.

| Historical scored job | Exact copied public source | Bytes | Plain SHA-256 | Withdrawal confirmed |
| --- | --- | ---: | --- | --- |
| `c43f7aba5ae84778` | [Madhavan113/ParadigmKernelChallenge at `721c627...`](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py) | 347010 | `76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4` | 2026-11-07 09:04:20 UTC |
| `23b1cc7e2a6040af` | [team keell0renz at `890809c...`](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py) | 370116 | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` | 2026-11-07 09:04:22 UTC |

### Where it came from and who is credited

The earlier file was copied from **Madhavan113's `ParadigmKernelChallenge`**
repository. Its source header explicitly identifies **vjeux** as the base
copyright holder and **Madhavan113** as the author of modifications, under
Apache-2.0.

The final file was copied from the integration published by **team keell0renz**.
The pinned upstream [NOTICE](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/NOTICE)
identifies **team keell0renz (Bohdan Agarkov)**. The copied file's headers also
credit **vjeux**, **alygg77**, and **ZhimaoL** for incorporated work.

Repository ownership is not a claim of sole authorship. Those upstream
contributors receive credit for their work; neither this project nor its AI
agents claim authorship of the copied implementations. No upstream endorsement
of these submissions is claimed. Their applicable notices remain preserved in
[NOTICE](NOTICE).

Authenticated leaderboard verification at **2026-11-07 09:04:37 UTC** found no active prefill entry remaining for this account. The public safe evidence is [`docs/data/withdrawal.json`](docs/data/withdrawal.json). It contains only job IDs, actions, timestamps, and the verification result.

Prefill is therefore excluded from active result summaries, rankings, leader comparisons, progression claims, and baseline-win counts. Its source, license notices, and original judge measurements remain available only as clearly labeled historical evidence.

## Remaining four results

BF16 projections, attention decode, routed experts, and decoder layer remain included as measured historical results. **They have not received a comprehensive originality audit.** AI agents could have copied or adapted additional material in those files, and this correction does not certify any of the four as original.

For those four sources:

- `originalityStatus` is `not-comprehensively-audited`;
- `reusedPublicSource` is `null`, not `false`; and
- source hashes and passing tests establish identity and execution facts, not authorship.

## What this correction does not claim

This publication does **not** claim that organizers issued a ban, disqualification, originality ruling, or eligibility approval. It does not claim that a private message was sent to organizers. It is a public disclosure addressed to organizers and readers, and future eligibility remains outside the scope of this repository.

Licensing permission is also separate from originality and challenge eligibility. The copied prefill sources retain their applicable licenses and notices, but a permissive license does not make the copied submissions independently authored.

## Historical records

The 79-job campaign ledger remains byte-faithful: 18 scored runs and 61 checks, including the original prefill measurements. Those records state what the judge measured at the time. Disposition is overlaid in schema-version-2 result metadata and the website; history is not deleted or rewritten.

See [performance disposition](PERFORMANCE.md), [experiment history](EXPERIMENTS.md), [source provenance](PROVENANCE.md), and [`verified_sources.json`](verified_sources.json).
