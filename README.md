# Meta-Open-Kernel-Challenge

Five verified Muse Spark Open kernels, an honest performance record, and a
static technical blog about optimizing GPU code without using a local GPU.

**Results blog:** [Hopper Field Notes](https://lokeshvenkatachalam.github.io/Meta-Open-Kernel-Challenge/)

All five problems beat the SGLang baseline on the official hosted NVIDIA H200
judge. None reached first place before the kernel track closed on November 5,
2026 at 08:00 UTC. Rankings are provisional, not final awards.

| Kernel | Best scored speedup | Verified source |
| --- | ---: | --- |
| BF16 projections | 1.1319x | [submission.py](kernels/muse_bf16_proj/submission.py) |
| Attention decode | 1.1520x | [submission.py](kernels/muse_attention_decode/submission.py) |
| Routed experts | 1.3403x | [submission.py](kernels/muse_moe_w4a16/submission.py) |
| Decoder layer | 1.1856x | [submission.py](kernels/muse_decode_layer/submission.py) |
| Prefill layer | 1.9192x | [submission.py](kernels/muse_prefill_layer/submission.py) |

## What is included

- [PERFORMANCE.md](PERFORMANCE.md): measured speedups, confidence intervals,
  rank/number-ranked comparisons, leaders, and the difference between checks
  and scored submissions.
- [EXPERIMENTS.md](EXPERIMENTS.md): all **79 campaign runs**, including failed
  and regressing experiments. The campaign used 18 scored runs and 61 checks;
  the account-wide November 5 totals also include 13 earlier runs.
- [PROVENANCE.md](PROVENANCE.md) and
  [verified_sources.json](verified_sources.json): source associations,
  plain-file SHA-256 digests, research references, and reuse disclosures.
- `kernels/`: all five exact retained Python submissions and per-kernel notes.
- `docs/`: the complete static website, including the public JSON datasets.
- `data/`: curated inputs for reproducible, offline report generation.

The kernel files are GPU programs, not CPU implementations. Publication,
preview, and validation commands below use the CPU only; they do not import or
execute the kernels. The closed challenge must not receive new submissions.

## Preview the blog locally

Python 3.10 or newer is sufficient to build and serve the site. No production
Node dependencies, framework, database, external fonts, or analytics are needed.

```bash
export CUDA_VISIBLE_DEVICES=""
python3 scripts/build_site.py
python3 scripts/serve_preview.py --port 8000
```

Open `http://127.0.0.1:8000/`. To reproduce GitHub Pages project-path hosting:

```bash
python3 scripts/serve_preview.py --port 8000 --prefix /Meta-Open-Kernel-Challenge/
```

Then open `http://127.0.0.1:8000/Meta-Open-Kernel-Challenge/`.

## Validate the publication

The browser tooling is a development dependency only. Node.js 24 and Python
3.10 are the versions used for this publication.

```bash
npm ci --ignore-scripts
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright npx playwright install chromium --only-shell

CUDA_VISIBLE_DEVICES="" \
LIBGL_ALWAYS_SOFTWARE=1 \
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright \
npm run check
```

This rebuilds the site, checks source hashes and data scope, checks local links
and publication boundaries, runs unit tests, and exercises desktop/mobile
pages with GPU acceleration disabled. No remote benchmark is started.

The data checks include all 79 records rather than the CLI's truncated
recent-job list; failed runs cannot acquire invented speedups. Kernel source
files are parsed without importing PyTorch or initializing CUDA.

## Reproduce the result exports

Committed curated inputs make report generation independent of a challenge
account or an active GPU service:

```bash
python3 scripts/export_results.py
python3 scripts/export_results.py --check
```

To sanitize a separately retained private evidence archive, use:

```bash
python3 scripts/export_results.py --archive-root /path/to/private/archive
```

Only allowlisted performance fields are published. Raw grader logs, evaluator
internals, machine identifiers, payload paths, authentication files, and tokens
must remain outside this repository.

## Host the website

The generated `docs/` directory can be uploaded to any static host. It is
committed and includes `.nojekyll`, so GitHub Pages does not need a custom build.

For GitHub Pages, use **Settings > Pages > Deploy from a branch**, select
**main** and **/docs**, and save. Changes pushed to `main` then update the site.
The expected URL is:

`https://lokeshvenkatachalam.github.io/Meta-Open-Kernel-Challenge/`

When editing curated data or the website generator, regenerate the exports
and site, run the checks, and commit the generated `docs/` files alongside
the inputs. Deployment is static; local preview tooling is not a production
server.

## Attribution and limitations

The retained prefill winner is adapted/reused public licensed work from the
keell0renz integration, with vjeux, alygg77, and ZhimaoL lineage. It is not
represented as an independently original kernel. Later local tuning did not
surpass that retained source. SGLang, Triton, FlashAttention and other primary
references are identified in the per-kernel documentation.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [PROVENANCE.md](PROVENANCE.md).
Verified source bytes and their original attribution are preserved. Do not
run formatters over those files without recording that the digest and source
association have changed.

Measurements are specific to the pinned H200 judge environment. Small
differences inside overlapping confidence intervals are not proven gains, and
standalone primitive improvements are not measured whole-layer improvements.
This is an independent participation report, not an official Meta, Paradigm,
or Andromeda website.