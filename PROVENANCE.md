# Provenance

This publication separates three claims that are easy to conflate:

1. **Measured result:** an official remote H200 job reported the recorded correctness and timing.
2. **Source association:** the published bytes match a retained source whose judge content hash was paired with a passing check and scored run.
3. **Authorship and license:** source lineage is disclosed independently of performance. A verified byte match is not an originality claim.

## Verified retained sources

| Problem | Published source | Plain SHA-256 | Paired check | Scored job |
| --- | --- | --- | --- | --- |
| BF16 projections | [`submission.py`](kernels/muse_bf16_proj/submission.py) | `af9c07caaa1227dfe0c6213071f342c2582e02c96440b287a1a8b03e869adfb5` | `84c15c3cb1bb4f73` | `c9e908a969fc4e40` |
| Attention decode | [`submission.py`](kernels/muse_attention_decode/submission.py) | `ac3a509b872f775619e91084c71e08eea1cfc99aad7630790842b2334ce4ca19` | `31188b3e4b2d482b` | `b45e9263f36b4141` |
| Routed experts | [`submission.py`](kernels/muse_moe_w4a16/submission.py) | `157e907cd37e9060e04a6dfda88cc3bde9cb2af2b4a2824717cecbaf3031106d` | `7cbc989dbf43455b` | `9c6a4cb6794448ba` |
| Decoder layer | [`submission.py`](kernels/muse_decode_layer/submission.py) | `34ae067a20a1c69dd52acfcb52d74e164499de9af7971e78f4e0717a71beddaa` | `027f7d1e62ed4951` | `9187326d522142a2` |
| Prefill layer | [`submission.py`](kernels/muse_prefill_layer/submission.py) | `f1807a75ec056b7f3874d4fce02331ca0043ac38a45af7710a8287a252421a92` | `1342300976314ba1` | `23b1cc7e2a6040af` |

These are plain file SHA-256 values. The judge's `content_hash` uses a different representation and must not be substituted for a file digest. [`verified_sources.json`](verified_sources.json) is the machine-readable manifest.

## Source association method

The retained source for each problem was selected from an immutable local snapshot or a byte-identical final `submission.py`. The association uses:

- byte comparison among retained snapshots and the final working file;
- a plain SHA-256 over the published bytes;
- a passing official check associated with the same server content hash;
- the retained scored job; and
- CPU-only `ast.parse` validation after publication.

The highest campaign check is not automatically the paired check. BF16 and decoder have later or different checks with higher point estimates, while source association remains pinned to the exact retained bytes.

## Four campaign-developed submissions

BF16 projections, attention decode, routed experts, and decoder layer are campaign-developed submissions that incorporate or call public Apache-licensed SGLang kernel paths and APIs. Their `reusedPublicSource: false` flag distinguishes them from the prefill file's wholesale pinned public-source reuse; it does not claim that they are isolated from upstream libraries or baseline code.

Relevant public lineages include:

- [SGLang at commit `a55ab7d5...`](https://github.com/sgl-project/sglang/tree/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f), including Hopper BF16 GEMV, FlashAttention wrappers, Muse normalization/routing kernels, and Marlin MoE support;
- [Triton tutorials and source](https://github.com/triton-lang/triton) for public programming patterns; and
- [NVIDIA CUTLASS examples and documentation](https://github.com/NVIDIA/cutlass) for public Hopper scheduling and grouped-GEMM references.

The exact retained files keep their own source headers and imports. See each kernel README for the narrower implementation summary.

## Prefill public-source reuse

The prefill winner is adapted/reused public work and is **not represented as independently original**. The retained file is pinned to:

- team keell0renz, [`kernels/muse_prefill_layer.py` at `890809cee421500b00126d084e81b338ef9181ab`](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py);
- the integrated [upstream NOTICE at the same commit](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/NOTICE);
- [vjeux/okc-2026 at `f35a41e...`](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393);
- [alygg77/okc-muse-spark-kernels at `6cae49a...`](https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0); and
- [ZhimaoL/sglang](https://github.com/ZhimaoL/sglang), including the `okc/cand12pre` and `okc/x-n72-naw` lineages; the retained header pins the N72 work to `b0a2c3433b`.

Later local experiments did not surpass this exact retained public source. Spread tile ordering, a fresh-thread context wrapper, and split large-M expert phases all measured lower. They are not claimed as improvements over the upstream integration.

## License and notices

The repository is distributed under [`LICENSE`](LICENSE), Apache License 2.0. [`NOTICE`](NOTICE) preserves the complete applicable integrated upstream notice, including the MIT license text for transitively included NathenFernandes portions. Attribution comments embedded in the prefill file remain unchanged.

## Historical source limits

The public repository does not dump reconstructed or overwritten historical candidate source as proven winning code. Some learning logs reconstructed hypotheses after experiments from job JSON and profiles; those reconstructions are useful context but not byte-authentication. Pre-campaign jobs with unavailable source remain aggregate or research context only.

## Performance evidence

[`docs/data/runs.json`](docs/data/runs.json) is a sanitized transformation of 79 official archived jobs. Its allowlist is limited to job ID, official problem, mode, timestamps, terminal status, correctness, speedup, confidence interval, score, and baseline. Raw logs, evaluator files, node data, payload references, profiles, credentials, and private paths remain outside the publication.
