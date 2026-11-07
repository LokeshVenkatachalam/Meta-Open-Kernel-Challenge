# Provenance and originality status

> **To organizers and readers:** this entire publication was produced through an AI-agent-driven workflow: code assembly and adaptation, experiments, challenge submissions, documentation, and website generation. It is not a claim of independent human authorship. See the full [disclosure](DISCLOSURE.md).

Source identity, measured execution, licensing, originality, and challenge eligibility are separate facts. A matching hash proves byte identity. A passing judge run proves execution and measurement. Neither proves original authorship.

## Current disposition

| Problem | Published source SHA-256 | Summary status | Originality status |
| --- | --- | --- | --- |
| BF16 projections | `af9c07caaa1227dfe0c6213071f342c2582e02c96440b287a1a8b03e869adfb5` | Included measured result | Not comprehensively audited; copied/adapted material may exist |
| Attention decode | `ac3a509b872f775619e91084c71e08eea1cfc99aad7630790842b2334ce4ca19` | Included measured result | Not comprehensively audited; copied/adapted material may exist |
| Routed experts | `157e907cd37e9060e04a6dfda88cc3bde9cb2af2b4a2824717cecbaf3031106d` | Included measured result | Not comprehensively audited; copied/adapted material may exist |
| Decoder layer | `34ae067a20a1c69dd52acfcb52d74e164499de9af7971e78f4e0717a71beddaa` | Included measured result | Not comprehensively audited; copied/adapted material may exist |
| Prefill layer | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` | Excluded historical evidence | Confirmed verbatim public-source copy |

For the remaining four, `reusedPublicSource` is intentionally `null`, not `false`. No comprehensive audit supports a no-copy guarantee.

## Confirmed prefill copies

Both scored entries for one problem, `muse_prefill_layer`, were confirmed as complete unchanged copies:

| Historical scored job | Bytes | Plain SHA-256 | Exact pinned public source | Disposition |
| --- | ---: | --- | --- | --- |
| `c43f7aba5ae84778` | 347010 | `76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4` | [Madhavan113/ParadigmKernelChallenge at `721c627...`](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py) | Withdrawn; excluded historical measurement |
| `23b1cc7e2a6040af` | 370116 | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` | [team keell0renz at `890809c...`](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py) | Withdrawn; excluded historical measurement |

These are byte matches against pinned public sources, not inferences from similar structure or license text. The final published `submission.py` remains unchanged only to preserve historical source evidence and applicable attribution.

The scored withdrawals were confirmed at 09:04:20 and 09:04:22 UTC on 2026-11-07. Verification at 09:04:37 UTC found no active prefill entry remaining. Public-safe evidence is in [`docs/data/withdrawal.json`](docs/data/withdrawal.json).

## Remaining four sources

The four included files call or incorporate public SGLang, Triton, FlashAttention, Marlin, and other ecosystem paths. This publication previously described them as campaign-developed; that wording did not amount to a comprehensive originality audit and is no longer used as an originality conclusion.

Their exact hashes, checks, and scored jobs remain useful identity and measurement evidence. They are **not** certificates that AI agents did not copy complete or partial public material elsewhere.

## Source association method

For each published final file, source association uses retained bytes, plain SHA-256, paired judge records, and CPU-only `ast.parse`. The highest check may differ from the check paired with the exact selected source. [`verified_sources.json`](verified_sources.json) records these facts and the current audit status.

Historical reconstructed or overwritten candidates are not published as authenticated original sources. Pre-campaign source-unavailable jobs remain context only.

## License and notices

The repository is distributed under [`LICENSE`](LICENSE), Apache License 2.0. [`NOTICE`](NOTICE) preserves applicable integrated attribution and the MIT text for transitively included NathenFernandes portions. Licensing permission does not establish independent authorship, organizer approval, or challenge eligibility.

## Claims not made

This correction does not claim that organizers imposed a ban or disqualification, made an originality ruling, approved eligibility, or received a private message from this project. It is a public disclosure addressed to organizers and readers.

## Performance evidence

[`docs/data/runs.json`](docs/data/runs.json) remains the unchanged schema-version-1 transformation of 79 official archived jobs. It retains historical prefill measurements rather than rewriting the past. [`docs/data/results.json`](docs/data/results.json) schema version 2 overlays current inclusion, exclusion, withdrawal, and originality status.
