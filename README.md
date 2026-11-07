# Meta-Open-Kernel-Challenge

## Correction: fully agentic work, with confirmed copying

**This entire project workflow was AI-agent-driven:** kernel generation and
assembly, experiments, submissions, documentation, and the website. It is not
a claim of independently human-authored or comprehensively originality-audited
work. AI agents can select, reproduce, copy, or adapt other people's code.

**Confirmed: both scored submissions for the prefill problem copied complete
public kernels unchanged, byte-for-byte.** Those two entries have been
voluntarily withdrawn from the official leaderboard and the prefill result is
excluded from our result summaries. Its performance is not an optimization we
authored.

**The other four implementations have not undergone a comprehensive originality
audit and may also contain additional copied or adapted material.** Passing
correctness checks, matching source hashes, and retaining license notices do
not establish originality or contest eligibility.

See the [public disclosure to organizers and readers](DISCLOSURE.md) for the
exact upstream sources, byte-match evidence, withdrawal records, and limits of
what has been verified. This is a voluntary correction, not an organizer ruling
or a guarantee about future eligibility.

**Results blog:** [Hopper Field Notes](https://lokeshvenkatachalam.github.io/Meta-Open-Kernel-Challenge/)

## Where the copied files came from

- **Earlier prefill entry:** [Madhavan113's `ParadigmKernelChallenge`](https://github.com/Madhavan113/ParadigmKernelChallenge/blob/721c62729fa0b407799692f311790643706f292d/muse_prefill_layer/submission.py),
  file `muse_prefill_layer/submission.py`, commit `721c627...`. Its header credits
  the base to **vjeux** and modifications to **Madhavan113**.
- **Final prefill entry:** [team keell0renz's `muse-open-kernel-challenge`](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/kernels/muse_prefill_layer.py),
  file `kernels/muse_prefill_layer.py`, commit `890809c...`. Its upstream
  [NOTICE](https://github.com/keell0renz/muse-open-kernel-challenge/blob/890809cee421500b00126d084e81b338ef9181ab/NOTICE)
  identifies **team keell0renz (Bohdan Agarkov)** and credits incorporated work.
  The copied file's headers credit **vjeux, alygg77, and ZhimaoL**.

These are publication/integration sources, not assertions that either repository
owner is the sole author. Credit for the copied work belongs to its upstream
contributors, not this project or its AI agents. No endorsement by those
contributors is claimed. Full notices are retained in [NOTICE](NOTICE).

## Retained benchmark observations, not originality claims

The four remaining summary results were measured against SGLang on the official
hosted NVIDIA H200 judge. None reached first place before the kernel track
closed on November 5, 2026 at 08:00 UTC. Rankings and measurement records do not
certify authorship, originality, or final eligibility.

| Kernel | Recorded scored speedup | Recorded source |
| --- | ---: | --- |
| BF16 projections | 1.1319x | [submission.py](kernels/muse_bf16_proj/submission.py) |
| Attention decode | 1.1520x | [submission.py](kernels/muse_attention_decode/submission.py) |
| Routed experts | 1.3403x | [submission.py](kernels/muse_moe_w4a16/submission.py) |
| Decoder layer | 1.1856x | [submission.py](kernels/muse_decode_layer/submission.py) |

Prefill is deliberately omitted. Its source and past measurements remain
available only as clearly labeled historical evidence of the removed result.

## What is included

- [PERFORMANCE.md](PERFORMANCE.md): measured speedups, confidence intervals,
  rank/number-ranked comparisons, leaders, and the difference between checks
  and scored submissions.
- [EXPERIMENTS.md](EXPERIMENTS.md): all **79 campaign runs**, including failed
  and regressing experiments and the excluded prefill history. The campaign
  attempted 18 scored runs and 61 checks; withdrawing two entries does not erase
  those attempts. The account-wide November 5 totals also include 13 earlier runs.
- [DISCLOSURE.md](DISCLOSURE.md): the fully agentic workflow, confirmed copying
  for one problem, official withdrawals, and uncertainty about remaining code.
- [PROVENANCE.md](PROVENANCE.md) and
  [verified_sources.json](verified_sources.json): source associations,
  plain-file SHA-256 digests, research references, and reuse disclosures.
- `kernels/`: four retained AI-assembled implementations and the excluded
  prefill source kept for audit, with per-kernel notes.
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

The removed prefill result was an exact copy, not merely inspiration or an
independently authored adaptation. Its final scored source came unchanged from
the keell0renz integration; the earlier scored source was also an unchanged
public kernel from Madhavan113. Both scored entries were withdrawn so the
older one could not automatically replace the removed best result.

SGLang, Triton, FlashAttention and other public sources were also used by the
agentic workflow. Their presence and license notices are disclosed, but the
other four implementations have not been comprehensively checked for
additional copying. No original-contribution guarantee is made.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [PROVENANCE.md](PROVENANCE.md).
Verified source bytes and their original attribution are preserved. Do not
run formatters over those files without recording that the digest and source
association have changed.

Measurements are specific to the pinned H200 judge environment. Small
differences inside overlapping confidence intervals are not proven gains, and
standalone primitive improvements are not measured whole-layer improvements.
This is an independent participation report, not an official Meta, Paradigm,
or Andromeda website.