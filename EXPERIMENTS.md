# Experiments

> **Agentic/originality disclosure:** AI agents drove the complete workflow, including kernel assembly and adaptation, experiment choices, checks, scored submissions, documentation, and website content. The remaining four included sources have not been comprehensively audited for originality and may contain copied or adapted material. Prefill is excluded after both scored submissions were confirmed as verbatim public-source copies. See [DISCLOSURE.md](DISCLOSURE.md).

The canonical historical ledger is [`docs/data/runs.json`](docs/data/runs.json). It remains a chronological, byte-faithful 79-job record and excludes raw logs, evaluator internals, machine identifiers, payload references, credentials, and private paths.

## Historical method

The agentic workflow read public task and implementation material, assembled or adapted candidates, ran CPU-only syntax checks, and used the official hosted H200 path for challenge checks and scored submissions. A check was off-board; a run was scored. The experiment record describes measured events but is not evidence that the submitted source was original.

## BF16 projections - included, not originality-audited

- Direct scalar Triton reduction: `9101ab6c5b394e67`, **0.4877x**.
- Output-major router partials: `245960c4a7534597`, **0.9761x**.
- First scored measurement: `870d8dbcd1534ec6`, **1.108101647x**.
- Eight reduction chains: `e4575d8c59b747cd`, **1.0610x**.
- Exact selected-source check: `84c15c3cb1bb4f73`, **1.1272x**; selected score: `c9e908a969fc4e40`, **1.131876249x**.
- Later atomic reduction: `d192f26ec76c411b`, **1.1134x**.

These measurements remain included, but no comprehensive audit establishes the source as original.

## Attention decode - included, not originality-audited

- Initial fused preparation: `a6c9bcb9634840d0`, compile failure with null performance fields.
- Corrected preparation fusion: `10cdd1efb0344bd8`, **1.1348x**.
- Custom Triton split-KV gather: `d7e1a97d2ef944bb`, **0.3974x**.
- Fixed uniform-query invocation: `4646cb6b5bda40c4`, **1.1502x**.
- Selected-source check: `31188b3e4b2d482b`, **1.152472784x**; selected score: `b45e9263f36b4141`, **1.151975904x**.

These measurements remain included, but no comprehensive audit establishes the source as original.

## Routed experts - included, not originality-audited

- Initial candidate: `9141fc0077224fd3`, compile failure with null performance fields.
- First scored measurement: `cf00286f1f494f3a`, **1.167268453x**.
- Block 16 check: `4d4efec69d244155`, **1.2695x**; block 32 regressed to **1.0738x**.
- Direct persistent SIMT NVFP4 GEMV: `5ee9869dc1b94137`, **0.2660x**.
- Selected-source check: `7cbc989dbf43455b`, **1.340147020x**; selected score: `9c6a4cb6794448ba`, **1.340288768x**.

These measurements remain included, but no comprehensive audit establishes the source as original.

## Decoder layer - included, not originality-audited

- Initial fused candidate: `1b847423150d47b7`, compile failure with null performance fields.
- First scored measurement: `71ce659ab40a4538`, **1.081701868x**.
- Triton pre-expert RMS normalization: `6c138b21813a4697`, **1.1550x**.
- Exact selected-source check: `027f7d1e62ed4951`, **1.162747432x**; selected score: `9187326d522142a2`, **1.185588351x**.
- Four-warp check `e6bcc1d8cd1a44a7` finished after close at **1.164398598x** and remained off-board.

These measurements remain included, but no comprehensive audit establishes the source as original.

## Prefill layer - excluded confirmed-copy history

The two scored prefill submissions were complete unchanged copies of public kernels:

- `c43f7aba5ae84778`, historically measured at **1.785899219x**, is byte-for-byte identical to [Madhavan113's pinned submission](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py), SHA-256 `76fb7501371fd3331858795e462ce5d0877e4b6a789a10a7c189fb7a8e5aaaa4`.
- `23b1cc7e2a6040af`, historically measured at **1.919190067x**, is byte-for-byte identical to [team keell0renz's pinned kernel](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py), SHA-256 `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92`.

Both scored jobs were withdrawn on 2026-11-07, and verification found no active prefill entry remaining. The prefill check and other timing rows remain historical facts, but none is presented as an authored project achievement. Later tuning measurements do not change the copied disposition of the scored sources.

## Historical uncertainty

Only prefill has completed copy verification for the scored submissions. The absence of a confirmed copy finding for the other four problems is not proof of originality. Historical candidate files described as reconstructed, overwritten, or unavailable remain source-uncertain and are not elevated to authenticated originals.

This correction makes no claim of an organizer ban, disqualification, private contact, or eligibility approval.

## Rebuild the public datasets

The default rebuild is offline and enforces schema-version-2 disposition:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --check
```

A maintainer with the private raw archive can refresh only the unchanged sanitized run ledger:

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/export_results.py --archive-root /path/to/private/runs
```

The exporter requires four included results, one excluded prefill record, exact source hashes, the mandatory agentic disclosure, safe withdrawal evidence, and the unchanged 79-run schema-version-1 history.
