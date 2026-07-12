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

## DONE — Purge the accidentally-committed venv from git history (before any push)

`bench/omnidocbench/paddle-venv/` (19,585 files, 1.2GB of PaddlePaddle/modelscope wheels) was committed
in `512f17b8`, before the `.gitignore` rule covering it landed. `e2ac8bf3` untracked it, but
`git rm --cached` does not remove blobs from *history* — they stayed in the object store.

Purged with `git filter-repo --path bench/omnidocbench/paddle-venv/ --invert-paths`, gated on the tree
being byte-identical afterwards: the non-venv `git ls-files` sha1 matched exactly, 21,569 venv objects →
**0**, `.git` **359M → 856K**, `cargo build`/`cargo test` green. 115 → 114 commits, because the
"untrack the venv" commit became empty and was pruned. A full pre-purge bundle is retained off-repo
(`/home/sb/paddleocr-vl-rs-prepurge.bundle`) — note filter-repo rewrites *branches* too, so a backup
branch is not a backup; the bundle is. Every SHA changed, so the first push must be a force-push of a
divergent history (the old remote only ever held 2 commits and never saw the venv).

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
with empty text, or omit it) so `assemble` behavior is unchanged. Still open: load-once (§2.8) shipped
without it, so every chart/image/seal crop is still recognized and then thrown away at assembly.

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

**The remaining speed lever is kernels, not harness** — the per-crop time did not move, so no further
harness change can touch the residual 2.7x. See "LM-prefill / vision-GEMM residual kernel work" below
for the (still unmeasured) kernel-level attribution.

## DONE — OmniDocBench full-benchmark accuracy-preservation run

Ran, scored with the official script, and written up in [BENCHMARKS.md](BENCHMARKS.md). On the
1355-page like-for-like v1.5 slice: text-edit **0.0328**, reading-order **0.0415**, TEDS **92.75**,
formula CDM **91.77** → `Overall` **93.75** vs the paper's **94.50**. Text, reading-order and table
reach published parity; the whole −0.75 deficit is formula CDM, which is the open item at the top of
this file.

**What it did not become:** a same-box run against HF/transformers or vLLM. The cross-stack arm that
*was* run is **llama.cpp** (bf16 both sides), so there is no transformers end-to-end floor anywhere in
this repo and none is claimed. That is a permanent gap, not work in flight.

## Assembler class-mapping expansion

**Why:** PP-DocLayoutV3 emits **25 layout classes** but the assembler only maps 3 to structure today
(`doc_title` -> `#`, `paragraph_title`/`figure_title` -> `##`, `table` -> markdown grid); everything
else renders as plain text. Mapping more classes (abstract, footnote, reference, header/footer,
formula numbering, aside/vertical text, etc.) to appropriate markdown/handling is low-cost,
high-value polish. **Hard part:** mostly taste -- deciding the right markdown for each class and
whether some (headers/footers, page numbers) should be dropped rather than rendered.

## LM-prefill / vision-GEMM residual kernel work — THE remaining speed lever

**Why:** after moving vision + LM attention to fused Sdpa/flash, the residual gap sits in the dense
GEMM/MLP (linear projections) of the vision encoder. With the per-page reload deleted (§2.8), this is
**all that is left** of the llama.cpp gap: **0.50s/crop vs 0.12s/crop**, i.e. the whole measured
**2.7x**. No harness change can touch it.

**Status of the attribution: best-supported hypothesis, not a measurement.** "Candle's vision GEMM is
the ceiling" is *inferred* from where the whole-pipeline time goes (0.50 vs 0.12 s/crop with attention
already on the fused path), not from timing a kernel. It is consistent with candle being a younger
kernel stack than ggml's hand-tuned per-architecture CUDA — a **maturity** gap, upstream of this repo,
not a bug in it — but nothing here has isolated the kernel from the workload. The micro-benchmark
below is what would turn the inference into a fact, and it is the honest prerequisite to filing
anything upstream. **Hard part:** if it *is* candle's GEMM, the fix is upstream and bounded by candle's
kernel quality, which this repo does not control.

## GEMM micro-benchmark: `candle.matmul` vs ggml `MUL_MAT`, bf16 both sides — settle the 2.7x

**Why:** the 2.7x is a *whole-pipeline* number, so it confounds two things — the kernel and the
workload. This bench removes the workload: same GPU, same shapes, same dtype, one op. It is the
experiment [BENCHMARKS.md](BENCHMARKS.md) ("What would actually settle it") points at, and the only
thing that upgrades "candle's vision GEMM is the ceiling" from inference to measurement.

**Shapes — the vision tower's actual GEMMs** (config: hidden 1152, intermediate 4304, 27 layers,
patch 14, spatial-merge 2; no fused QKV — q/k/v/out are four separate projections):

| GEMM | shape `[M,K]×[K,N]` | per crop |
|---|---|---|
| vision q / k / v / out_proj | `[M, 1152] × [1152, 1152]` | ×4 × 27 layers |
| vision MLP fc1 | `[M, 1152] × [1152, 4304]` | ×27 |
| vision MLP fc2 | `[M, 4304] × [4304, 1152]` | ×27 |
| connector linear_1 | `[M/4, 4608] × [4608, 4608]` | ×1 |
| connector linear_2 | `[M/4, 4608] × [4608, 1024]` | ×1 |

`M` = patch tokens, and it is the variable that matters: crops are variable-resolution (NaViT-style),
so `M` moves with crop size. One **real, checkable** value to anchor the sweep: the repo's `ocr`
fixture crop is a `1×14×46` patch grid → **M = 644** (161 merged tokens; see
`inputs_processor.rs:6` in the mistral.rs fork). Sweep `M ∈ {64, 256, 644, 1024, 4096}` to span a
text-line crop through a full-page one. Secondary: the LM's own prefill shapes (hidden 1024, inter
3072 → q `1024×2048`, k/v `1024×256`, o `2048×1024`, gate/up `1024×3072`, down `3072×1024`) — the LM is
*not* where the gap is believed to live, so it is the control arm, not the treatment.

**Method:** a ~40-line candle bin timing `Tensor::matmul` per shape (warm-up, then N iters, CUDA-synced),
against llama.cpp's existing `test-backend-ops perf -o MUL_MAT` on the same box. **Report GFLOPS per
shape** (`2·M·K·N / seconds`), not a single ratio — a per-shape table is what tells you whether candle
is uniformly behind or only behind on the tall-skinny/small-M shapes that OCR crops actually produce.
Sweep bf16 **and** f16/f32 so the dtype axis is visible rather than assumed.

**Honesty framing this bench must preserve (and can falsify):**
- **bf16 held equal on both sides.** That is the whole point — it is what rules out "ggml is fast
  because it quantized". If ggml's CUDA `MUL_MAT` turns out to lack a real bf16 path for some shape
  and upcasts or reroutes, **that is a finding, report it** — do not silently compare against f16.
- **The 2.7x is workload-specific** (compute-bound vision prefill + short OCR decode). The micro-bench
  is the *kernel* number; it should not be expected to reproduce 2.7x, and if it comes out at, say,
  1.3x, that is informative: it would mean most of the pipeline gap is *not* raw GEMM throughput and
  the hypothesis above is wrong.
- **ggml's remaining trade-off is portability, not precision.** Hand-tuned per-architecture kernels are
  real work candle has not done; a like-for-like bf16 same-shape bench is exactly what makes that
  trade-off legible instead of rhetorical.
- **Single GPU (RTX 4070 Ti Super), single box.** A kernel gap moves with the architecture it was tuned
  for. State it; do not generalize past it.

**Optional deeper probe (not required):** an `nsys`/`ncu` trace of one vision forward would name the
exact slow kernel and turn "candle maturity ceiling" from an inference into a named symbol. Caveat: GPU
profiling under WSL is frequently restricted (counter access is gated), so this may not be runnable on
this box at all. The matmul micro-bench needs none of it — it is pure wall-clock and works anywhere.

## Batching Approach A (cu_seqlens packed vision)

**Why:** engine batching (Approach B) was measured leakage-free but with **no** throughput win,
because vision runs per-image regardless (see [BENCHMARKS.md](BENCHMARKS.md) "Region batching"). The
only path to a real vision-batch speedup is Approach A: block-diagonal `cu_seqlens` single-kernel
vision packing so N crops share one vision forward. **Hard part:** the attention masking is the
highest-leakage-risk change in the whole pipeline, and Approach B already showed the LM/scheduler
side can't turn a vision batch into throughput on its own -- so Approach A is only worth it if the
scheduler ceiling is addressed too. Data-backed deferred, not abandoned.
