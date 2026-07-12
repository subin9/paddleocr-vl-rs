# Future work

Honest roadmap. Each item lists why it is valuable and what the hard part actually is.

## Formula CDM −2.44 vs published — the one metric not at parity

**Why:** every other benchmark metric reaches published parity (text, reading-order, table); formula
CDM is **91.77 vs 94.21** and is therefore the whole of the −0.75 `Overall` deficit. It is **not** a
scoring artifact — the CDM renderer is gated by `cdm_smoke.py`, which proves it discriminates
(identical formula F1 = 1.0, truncated F1 = 0.6) rather than silently returning 0. **Hard part:** the
worst slices are *degraded inputs* (`fuzzy_content` 0.307, `with_watermark` 0.564), which points at
the crop/preprocessing path (resize, interpolation, normalization) rather than the LM — but the
cross-stack A/B complicates that story: llama.cpp, with an entirely different image pipeline, scores
formula edit-distance **worse** than us (0.1927 vs 0.1833), not better. So a naive "our preprocessing
is lossy" hypothesis does not survive first contact. Next probe: score the *reference transformers*
implementation's formula CDM on the same crops — that separates "our crops are worse" from "this
0.9B model is simply weak on degraded formulas".

## Cross-stack residual: llama.cpp is −0.30 TEDS / +0.0094 formula-edit vs the Rust port

**Why:** the two bf16 stacks agree to 0.0003 on text and 0.0002 on reading order — so the table and
formula deltas, while small, are the *only* place two faithful implementations visibly disagree, and
that makes them a cheap probe into which stack's image path is lossier. **Hard part:** attribution
needs a per-crop diff (dump both stacks' text for the same table/formula crops and look at where they
diverge), not another aggregate score. Deliberately left **unattributed** in BENCHMARKS.md rather than
hand-waved to "numeric noise".

## Root-cause the runaway generation (why does a crop never emit EOS?)

**Why:** `magazine_TheEconomist.2023.12.02_page_062` reproducibly generated until it was killed —
hours, pre-cap. It is now *bounded* (`MAX_NEW_TOKENS=2048` → the page completes in 45s, 6339 bytes)
and the 120s per-region + 600s per-page guards catch the class in general, but **bounding is not
explaining**: something about that region makes a 0.9B model refuse to stop, and 8 crops across the
full run still hit the 120s guard and get **empty text** recorded (a small, real accuracy cost the
port pays and llama.cpp — 0 trips — does not). **Hard part:** the interesting question is whether the
degenerate loop reproduces in the *reference* transformers implementation on the same crop. If yes it
is a model property (and the guard is the right answer forever); if no, it is a port bug in sampling
or EOS handling and the guard is masking it. That single experiment is the whole task — do it before
touching any generation code.

## Purge the accidentally-committed venv from git history (before any push)

**Why:** `bench/omnidocbench/paddle-venv/` (19,585 files, 1.2GB of PaddlePaddle/modelscope wheels and
binaries) was committed in `512f17b8` — it was added before the `.gitignore` rule covering it landed.
`e2ac8bf3` stops tracking it, but **`git rm --cached` does not remove the blobs from history**; they
are still in every clone's object store (`.git` is 361MB). **Hard part:** none technically
(`git filter-repo --path bench/omnidocbench/paddle-venv --invert-paths`), but it **rewrites every
commit SHA**, so it must happen before the repo is ever pushed or shared — which is exactly why it is
recorded here instead of being done silently mid-benchmark.

## DONE — Skip visual-only regions in assembly (`VISUAL_ONLY_CLASSES`, measured §2.3-step-1)

Implemented: `assemble_markdown` drops `chart`/`image`/`header_image`/`footer_image`/`seal`. In-session
A/B on the §2.2 slice moved academic text_block 0.9953→0.0000, table TEDS 0.6883→0.9969, reading_order
0.1333→0.0000 (overall smoke5 text_block 0.276→0.077). Kept below: the *recognition* and *formula*
follow-ups it exposed.

## Skip RECOGNITION of visual-only crops (speed, not accuracy)

**Why:** the assembly-side skip above drops the junk from the *output*, but `paddleocr_vl_recognize`
still runs the VLM on every `chart`/`image`/`seal` crop first — wasted GPU time (charts especially
emit hundreds of tokens). The layout stage already knows the class in `manifest.json`, so recognition
could skip the same `VISUAL_ONLY_CLASSES`. **Hard part:** it lives in the mistral.rs example
(`paddleocr_vl_recognize`), not this crate; keep the `results.json` contract intact (emit the region
with empty text, or omit it) so `assemble` behavior is unchanged. Fold into the §2.4 load-once mode.

## book text_block 0.339 — standalone `inline_formula` wrapped as display `\[…\]`

**Why:** the one non-chart residual on the smoke5 slice. A standalone `inline_formula` region is
recognized and emitted wrapped in display delimiters `\[…\]`; the GT expects it inline, so the scorer
mismatches the block. **Hard part:** deciding wrapping by class is easy (`inline_formula` → `\(…\)`),
but confirming it doesn't regress the display-formula metric needs a before/after on a subset with
both kinds — measure, don't blind-apply.

## table Edit_dist 0.434 vs TEDS 0.997 gap

**Why:** the academic tables score near-perfect on TEDS (structure+content tree edit) but 0.43 on the
raw table Edit_dist. TEDS is OmniDocBench's headline table metric, so this is low-priority, but the
gap is worth understanding before the full run (likely OTSL→pipe cell-text normalization differences).
**Hard part:** diagnostic only — compare normalized GT vs pred table strings on the 2 academic tables.

## DONE — Load-once page-iterating recognize mode (§2.8, measured)

Implemented: `paddleocr_vl_recognize <dir>... | --list <file>` builds the engine once and iterates
page dirs; `run_pipeline.sh` runs layout per page → **one** recognition process over all pending pages
→ assembly per page. Iterate, not serve — the binary already loaded once and looped crops, so this was
~30 lines; a server would have needed HTTP + image upload + prompt marshalling for the same win.

Measured on the clean box, same 118 pages as §2.7: median page **10.0s → 8.42s**, checkpoint load
**1.76s/page → 2.02s once per run**, llama.cpp gap **3.2x → 2.7x**. Recognition per crop is unchanged
(0.52 → 0.50s/crop), which is the point: this deleted a harness artifact and nothing else.

Gated on byte-identical output (`loadonce_parity.sh`, 24 pages / 189 crops / all 22 classes, 24/24
identical) — a usability mode that changes a token is a bug. The runaway guard survived the refactor:
per-region tokio timeout unchanged, and the old per-page `timeout` process kill became an OS-thread
watchdog + `TIMEOUT_SKIP` page marker, so a wedged engine costs one page, not the run.

**The remaining speed lever is kernels, not harness** — see "LM-prefill / vision-GEMM residual kernel
work" below. That is what the 2.7x is, and it is upstream in candle.

## OmniDocBench full-benchmark accuracy-preservation run

**Why:** elevates the current 9-fixture token-parity check to a standard document-parsing benchmark,
turning "matches the reference on 9 hand-picked crops" into a defensible score on OmniDocBench v1.5.
**Hard part:** running mistral.rs vs HF/transformers vs vLLM on the same box with the official
scoring script, and framing it correctly -- the result is accuracy-preservation (the port is
token-faithful, so it should match the reference), with the port's edge being single-binary
deployment, not serving throughput. Same-hardware caveat, not a SOTA-speed claim. See
[BENCHMARKS.md](BENCHMARKS.md) "Planned".

## Assembler class-mapping expansion

**Why:** PP-DocLayoutV3 emits **25 layout classes** but the assembler only maps 3 to structure today
(`doc_title` -> `#`, `paragraph_title`/`figure_title` -> `##`, `table` -> markdown grid); everything
else renders as plain text. Mapping more classes (abstract, footnote, reference, header/footer,
formula numbering, aside/vertical text, etc.) to appropriate markdown/handling is low-cost,
high-value polish. **Hard part:** mostly taste -- deciding the right markdown for each class and
whether some (headers/footers, page numbers) should be dropped rather than rendered.

## LM-prefill / vision-GEMM residual kernel work — THE remaining speed lever

**Why:** after moving vision + LM attention to fused Sdpa/flash, the remaining prefill gap vs torch
is candle's dense GEMM/MLP (linear projections) on the vision encoder. With the per-page reload now
deleted (§2.8), this is **all that is left** of the llama.cpp gap: **0.50s/crop vs 0.12s/crop**, i.e.
the whole measured **2.7x**. No harness change can touch it. **Hard part:** it is a candle maturity
ceiling (generic-Rust / MKL-call-overhead GEMM vs torch's tuned oneDNN/cuBLAS), not a bug in this
repo — the win is bounded by candle's kernel quality, which is upstream.

## Batching Approach A (cu_seqlens packed vision)

**Why:** engine batching (Approach B) was measured leakage-free but with **no** throughput win,
because vision runs per-image regardless (see [BENCHMARKS.md](BENCHMARKS.md) "Region batching"). The
only path to a real vision-batch speedup is Approach A: block-diagonal `cu_seqlens` single-kernel
vision packing so N crops share one vision forward. **Hard part:** the attention masking is the
highest-leakage-risk change in the whole pipeline, and Approach B already showed the LM/scheduler
side can't turn a vision batch into throughput on its own -- so Approach A is only worth it if the
scheduler ceiling is addressed too. Data-backed deferred, not abandoned.
