# Experiments

This is the curated experiment record for the closed 2026-11-05 kernel campaign. The canonical machine-readable history is [`docs/data/runs.json`](docs/data/runs.json); it contains every campaign job in chronological order and excludes raw logs, evaluator internals, machine identifiers, payload references, credentials, and private filesystem paths.

## Method

1. Read the official task contract and public implementation references.
2. Make one reviewable source change at a time where practical.
3. Run CPU-only `ast.parse` checks locally without importing Torch or executing a kernel.
4. Use the official hosted H200 check path for correctness and timing.
5. Promote a candidate to a scored run only after a passing check when the deadline allowed.
6. Retain the exact source bytes associated with the best verified score, even when a later experiment regressed.

A check is off-board; a run is scored. Point estimates inside overlapping confidence intervals were treated cautiously. None of the five campaigns reached first place.

## BF16 projections

The retained design combines pinned Hopper GEMV dispatch for selected M=1 shapes, custom split-K projection paths only where they measured well, and an FP32-accurate router over BF16 high-plus-low weights.

- Direct scalar Triton reduction: `9101ab6c5b394e67`, **0.4877x**. Rejected.
- Output-major router partials: `245960c4a7534597`, **0.9761x**. The strided reduction motivated split-major storage.
- Public Hopper geometry and full-M router tiles established the first scored win: `870d8dbcd1534ec6`, **1.108101647x**.
- Eight independent reduction chains raised register pressure: `e4575d8c59b747cd`, **1.0610x**. Rejected.
- Exact retained-source check: `84c15c3cb1bb4f73`, **1.1272x**; retained score: `c9e908a969fc4e40`, **1.131876249x**.
- A later atomic router reduction passed but fell to `d192f26ec76c411b`, **1.1134x**. The scored source was restored.

## Attention decode

The successful direction kept FlashAttention-3 for the dense paged reduction and reduced the preparation and scheduling overhead around it.

- Initial fused preparation: `a6c9bcb9634840d0`, compile failure. Performance fields remain null.
- Corrected fusion of Q/K norm, RoPE, FP8 Q conversion, and cache stores: `10cdd1efb0344bd8`, **1.1348x**.
- A custom Triton split-KV gather was correct but reached only `d7e1a97d2ef944bb`, **0.3974x**. Rejected.
- Fixed uniform-query invocation: `4646cb6b5bda40c4`, **1.1502x**. Shape policies, PDL, and extra split-count variants did not prove better.
- Metadata reuse removed another preparation step: paired check `31188b3e4b2d482b`, **1.152472784x**; retained score `b45e9263f36b4141`, **1.151975904x**.

## Routed experts

The campaign retained SGLang's lossless Marlin repack and improved local routing, block selection, invalid-route handling, and finalization around the two expert GEMMs.

- Initial candidate: `9141fc0077224fd3`, compile failure because a Triton epsilon was not constexpr.
- Fixed candidate established a baseline win: `cff57273a7c843c3`, **1.1703x**; first score `cf00286f1f494f3a`, **1.167268453x**.
- Block 16 improved the mix to `4d4efec69d244155`, **1.2695x**; block 32 regressed to **1.0738x**.
- Piecewise block sizes and local-route filtering reached **1.3215x**.
- Direct persistent SIMT NVFP4 GEMV passed correctness but collapsed to `5ee9869dc1b94137`, **0.2660x**. Rejected.
- Fused global-to-local expert mapping produced paired check `7cbc989dbf43455b`, **1.340147020x**, and retained score `9c6a4cb6794448ba`, **1.340288768x**.

After routing and finalization work, profiles attributed roughly 94-95% of large-shape GPU time to the two Marlin GEMMs. Further substantial gain would require a better Hopper W4A16 mainloop rather than more Python-side fusion.

## Decoder layer

Whole-layer validation combined attention preparation, dense projections, routing, expert GEMMs, and residual/norm boundaries.

- Initial fused candidate: `1b847423150d47b7`, compile failure; no fabricated timing is attached.
- Corrected fusion established a first check at **1.0781x** and first score at `71ce659ab40a4538`, **1.081701868x**.
- Marlin block sweeps improved T=128 and T=256, while block 8 everywhere, T=256 block 16, and T=32 block 32 regressed important shapes.
- Triton pre-expert RMS normalization reached `6c138b21813a4697`, **1.1550x**.
- Fused local expert mapping produced exact-source check `027f7d1e62ed4951`, **1.162747432x**, followed by retained score `9187326d522142a2`, **1.185588351x**.
- A four-warp pre-expert check, `e6bcc1d8cd1a44a7`, finished after close at **1.164398598x**. It stayed off-board and the higher scored source remained final.

## Prefill layer

The retained source is adapted/reused public Apache-licensed work, not an independently original kernel. The exact pinned keell0renz integration combines N72 Hopper FMHA, cuBLASLt projection plans, fused boundaries, and a persistent grouped W4A16 expert path.

- Public persistent-MoE/Hopper-attention base: `18bcf4766eec460b`, **1.7914x**.
- First scored candidate: `c43f7aba5ae84778`, **1.785899219x**.
- Attention autotuning selected the same fixed variants; the apparent aggregate change was not a timed-path innovation.
- Pinned N72/keell0renz source check: `1342300976314ba1`, **1.912524500x**; retained score `23b1cc7e2a6040af`, **1.919190067x**.
- Spread expert ordering (**1.9069x**), a fresh-thread context wrapper (**1.9023x**), and split large-M expert phases (**1.8915x**) all regressed.

The later experiments did not beat the retained public source. They are documented as regressions, not as evidence of a new winner.

## Historical uncertainty

Only the five final retained files are published as verified kernel sources. Historical candidate files described as reconstructed, overwritten, unavailable, or source-uncertain are not republished as authenticated submissions. Pre-campaign account jobs are included only in aggregate counts or explicitly labeled context; they are not credited to this campaign.

## Rebuild the public datasets

The default rebuild is fully offline and uses only committed curated data:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --check
```

A maintainer with the private raw archive can deliberately refresh the sanitized ledger:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --archive-root /path/to/private/runs
```

The exporter reads only an explicit allowlist from raw jobs and fails on count, schema, hash, syntax, ranking, or null-metric inconsistencies.
