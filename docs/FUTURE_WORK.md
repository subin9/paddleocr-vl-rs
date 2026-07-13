# Future work

Honest roadmap. Each item lists why it is valuable and what the hard part actually is.

## Formula CDM — **the "not a port defect" verdict was WRONG. 1.48 of the 2.44 was ours.**

**This entry previously read "ROOT-CAUSED as a CJK-formula gap; not a port defect" and closed the cause
as genuine model difficulty. That is withdrawn.** Porting `crop_margin` (the item below) — a crop step
upstream performs on formula blocks and this port skipped — moved CDM **91.77 → 93.25**, closing
**1.48 of the 2.44 (61%)**. The gap to published is now **0.96** (`Overall` −0.22).

| page-avg CDM, `subset: v1.5` | overall | english (148 pg) | simplified_chinese (57 pg) |
|---|---|---|---|
| port, base | 0.9177 | 0.9349 | 0.8730 |
| port, **+`crop_margin`** | **0.9325** | **0.9460** | **0.8976** |
| | **+1.48** | +1.11 | **+2.46** |
| llama.cpp, base | 0.9032 | 0.9155 | 0.8719 |
| llama.cpp, **+`crop_margin`** | **0.9220** | 0.9279 | 0.9068 |

**Why the old verdict failed, and it is not hindsight.** The argument was: *llama.cpp reproduces the
CJK penalty over the same crops, so a defect in our port could not survive that.* But "the same crops"
means llama.cpp re-recognizes **this pipeline's own crop PNGs** — it is an independent control on the
**decode** path and a **common-mode** one on the **crop** path. It exonerates the decoder and says
nothing whatsoever about the cropper. A missing crop step is exactly the class of bug it is built not
to see. The falsifiable prediction that follows — a crop-side fix must lift *both* stacks — is what
happened: +1.48 CDM on the port, +1.88 on llama.cpp.

**And the CJK story was carrying the bug on its back.** `crop_margin` is language-blind by construction
(contrast-normalize, threshold, ink bbox), yet CJK formulas gained **+2.46 against English's +1.11** —
more than twice. The CJK-vs-English split narrows from 6.2 CDM points to 4.8. The old entry used that
split to *rule out* a crop-path cause; the split was partly produced by one.

**What is actually left.** The residual is now **0.96 CDM**, and the CJK-vs-English gap (4.8 points) is
the whole of it. It is still unattributed, and after this it is **no longer safe to assume it is the
model's**:
- **The published 94.21 has no language split**, so we still cannot check whether the reference pipeline
  also drops on CJK formulas. Settling it needs the reference transformers implementation scored
  per-language on the same crops — the experiment this entry has been deferring since it was written.
- **12 formulas (0.66%) our pipeline emitted nothing for** — a layout miss; the GT formula region was
  never detected. Ours. Too small to move the metric, but it is ours.
- `tokenize_figure_of_table` is still unported (upstream runs it on table blocks). Table TEDS is at
  parity, so it is costing nothing measurable — but that is the same shape of reasoning that just
  failed here, so it deserves an A/B rather than an assumption.

**The entry before *that* one was also wrong and was also withdrawn** (it blamed degraded inputs —
`fuzzy_content`, `with_watermark` — by reading all-1651 attribute rows against a v1.5-only deficit).
Three readings of this number, two of them wrong, and both times the error was the same: explaining the
gap instead of measuring the difference.

**Do not chase:** the scorer's own `\begin{array}` merge emits invalid nested `\[…\]` on 224/1807
predictions. It is upstream (it hits the reference model identically — our delimiter choice provably
cannot affect it), and it is **worth +0.03 CDM**: re-running CDM on all 217 affected predictions with
the nesting removed and the content held fixed moves the mean 0.6587 → 0.6608, **+0.0022**, with 29
improved and 20 *worse*. xelatex recovers from it. Those 224 score low (0.6479) because they are
**over-segmented**, not because of the nesting. Documented in BENCHMARKS.md, deliberately not
actioned.

## DONE — Formula crops skipped upstream's `crop_margin`. Ported, and it is worth −6.6% formula edit.

The port handed the VLM the raw layout box for a formula. Upstream does not: `PaddleX
pipeline.py` runs **`crop_margin(block_img)` on formula blocks only** (`"formula" in block_label and
block_label != "formula_number"`) — contrast-normalize with a LUT, threshold at 200 into an ink mask,
crop to the ink's bounding box, keep it only if >2px on both sides. It tightens the crop onto the
glyphs so `smart_resize` spends the pixel budget on the formula and not on the whitespace the detector
included. It was the only per-class crop preprocessing upstream had and we didn't.

Ported as `assemble::crop_margin`, applied through `assemble::crop_for_class` on upstream's class test
verbatim. Not applied to text/table/chart/seal — upstream does not, and a text region trimmed to its
ink loses the layout cue the model reads.

**Measured** (`subset: v1.5`, same layout regions, official scorer; re-recognized the 1,685 formula
crops and spliced them into the scored run — 911 of them changed text):

| | formula CDM ↑ | formula Edit ↓ |
|---|---|---|
| Rust port, guard only | 0.9186 | 0.1817 |
| Rust port, **+ `crop_margin`** | **0.9325** | **0.1697** |
| | **+1.39** | −0.0120 (−6.6%) |
| llama.cpp, baseline | 0.9032 | 0.1913 |
| llama.cpp, **+ `crop_margin`** | **0.9220** | **0.1805** |
| | **+1.88** | −0.0108 (−5.6%) |

Against the port's *original* baseline (no guard): CDM **91.77 → 93.25, +1.48** — which is **61% of
the −2.44 gap to published**, and it is why the formula item above is now withdrawn.

Nothing else moves (text 0.0323→0.0322, table TEDS unchanged, reading order unchanged) — only the
formula crops changed, and only the formula metric responds.

**And the cross-stack control finally earns its keep.** The scope note says llama.cpp is blind to
crop-path defects because it eats *this pipeline's crop PNGs*. That predicts a crop fix must improve
**both** stacks, and it does: +1.48 and +1.88 CDM on two independent decoders. A port-side decode bug
could not do that; a crop-side fix is exactly what does.

**Run `cdm_smoke.py` before trusting any CDM number.** `CDM.evaluate` swallows every exception into
`{"F1_score": 0}`, so a broken environment fabricates a 0.0 per formula that reads as "the model got
every formula wrong". Two silent failures hit this run — missing `scikit-image`, then
`ransac(random_state=…)` removed in scikit-image ≥ 0.23. `clone_eval_code.sh` now patches the second on
clone. Also note that the scorer's `CDM_plain` metric does not compute CDM; it only dumps the formula
pairs for an external pass.

## Cross-stack residual: llama.cpp is −0.30 TEDS / +0.0094 formula-edit vs the Rust port

**Why:** the two bf16 stacks agree to 0.0003 on text and 0.0002 on reading order — so the table and
formula deltas, while small, are the *only* place two faithful implementations visibly disagree, and
that makes them a cheap probe into which stack's image path is lossier. **Hard part:** attribution
needs a per-crop diff (dump both stacks' text for the same table/formula crops and look at where they
diverge), not another aggregate score. Deliberately left **unattributed** in BENCHMARKS.md rather than
hand-waved to "numeric noise".

## Runaway generation — MITIGATED (upstream's truncator, ported); the port is exonerated on sampling

`magazine_TheEconomist.2023.12.02_page_062` reproducibly generated until it was killed — hours,
pre-cap. It was already *bounded* (`MAX_NEW_TOKENS=2048`, plus the 120s per-region / 600s per-page
guards), but bounding is not explaining, and the open question was whether a crop that never emits
EOS is a **model property** (guard is the right answer forever) or a **port bug in sampling/EOS**
(guard is masking it).

**It is a model property, and the original codebase says so by what it does.** PaddleOCR-VL ships a
`generation_config.json` carrying nothing but `eos_token_id`/`pad_token_id` — no penalties, no
sampling — and PaddleX's local predictor *explicitly warns-and-ignores* `repetition_penalty` /
`temperature` / `top_p` ("not supported by the local model"). Upstream decodes greedily with no
repetition guard in the sampler at all. Its entire defence is (1) a per-region token cap and (2)
`truncate_repetitive_content` (`paddlex/inference/pipelines/paddleocr_vl/uilts.py`), a *string*
truncator run on every decoded region after the fact. There is no retry, no fallback, no n-gram
constraint anywhere in the stack; the maintainers' stated remedy for hallucinated runaway is "use the
full layout pipeline, not the VLM standalone".

That settles the sampling half of the question: **this port's decode already matches upstream's
exactly** (greedy, unpenalized, capped), so a sampling divergence cannot be the cause. It also means
the port was missing a piece upstream has — now ported: `assemble::truncate_repetitive_content`,
applied per class in `read_results` (text floor 50 chars, table floor 5000, upstream's own two
values). `results.json` deliberately keeps the raw string; the guard runs on ingest.

Measured, on 8,636 Korean AI-Hub crops through this engine: **2 crops** ran to the cap
(`\(f_{0}f_{0}…`, `川川川…`) and produced **51% of the whole slice's edit distance**. Both are cut by
the ported truncator. It is a known failure of the model *family*, not of this one — Nougat reports
it on 1.5% of pages ("non-Latin script languages result in instant repetitions"), olmOCR calls it
"the most common failure we experience", and the two vLLM PRs proposing a loop detector for
PaddleOCR-VL were closed unmerged.

**Two holes in upstream's own heuristic, found by running it on real output and now fixed here.**
Both are divergences from PaddleX, deliberate and measured:
1. A runaway that dies on the token cap can be cut *mid-character*, and the detokenizer leaves a
   trailing **U+FFFD**. Upstream's phrase check anchors on the exact suffix, so that one char makes
   every candidate unit mismatch and the check silently no-ops — on precisely the outputs it exists
   to catch. `truncate_repetitive_content` trims it before anchoring.
2. Upstream runs its phrase check **only when the whole output is one line**. A region that emits two
   honest lines and then loops on a third slips past every check it has (whole-string checks skipped
   for containing a newline; the line-level check needs ten near-identical lines). `truncate_repeating_lines`
   runs the phrase check per line. Safe on tables and measured, not assumed: OTSL marks a row with a
   `<nl>` *token*, so a table is a single line and the pass reduces to the check already run on it —
   0 of 1,590 real table blocks change beyond the upstream rule.

Together they took the Korean line-level CER **0.1591 → 0.1268**, one prediction changed, none worse.

**Scored A/B on OmniDocBench** (same `results.json`, re-assembled, official scorer — full table in
BENCHMARKS.md): every `v1.5` metric improves — text Edit 0.0327→0.0323, table TEDS 92.75→92.82,
formula Edit 0.1833→0.1817. 204 of 78,710 blocks altered, all degenerate.

**Still open (small):** the guard cannot help the ~8 crops that trip the 120s region guard and record
**empty text** — they never return a string to truncate. **Re-run CDM on the guarded predictions**:
18 `display_formula` blocks are degenerate and now get cleaned, formula `Edit_dist` already improved,
and CDM is the one metric still short of published — this is the cheapest shot at part of the −2.44
and it needs no VLM run. And the definitive per-crop A/B (does the *reference* transformers
implementation loop on `page_062`'s crop?) was never run; the evidence above narrows it to the model
but does not close it by direct observation. Worth one run before any in-flight detector (Nougat's
logit-variance `StoppingCriteriaScores` is the design to copy if it ever becomes necessary — it stops
the loop without distorting honest tokens, which a repetition penalty does).

## DONE — Skip visual-only regions in assembly (`VISUAL_ONLY_CLASSES`, measured)

Implemented: `assemble_markdown` drops `chart`/`image`/`header_image`/`footer_image`/`seal`. An A/B on
the 5-page plumbing slice moved academic text_block 0.9953→0.0000, table TEDS 0.6883→0.9969, reading_order
0.1333→0.0000 (overall smoke5 text_block 0.276→0.077). Kept below: the *recognition* and *formula*
follow-ups it exposed.

## Skip RECOGNITION of visual-only crops (speed, not accuracy)

**Why:** the assembly-side skip above drops the junk from the *output*, but `paddleocr_vl_recognize`
still runs the VLM on every `chart`/`image`/`seal` crop first — wasted GPU time (charts especially
emit hundreds of tokens). The layout stage already knows the class in `manifest.json`, so recognition
could skip the same `VISUAL_ONLY_CLASSES`. **Hard part:** it lives in the mistral.rs example
(`paddleocr_vl_recognize`), not this crate; keep the `results.json` contract intact (emit the region
with empty text, or omit it) so `assemble` behavior is unchanged. Still open: load-once shipped
without it, so every chart/image/seal crop is still recognized and then thrown away at assembly.

## book text_block 0.339 — standalone `inline_formula` wrapped as display `\[…\]`

**Why:** the one non-chart residual on the smoke5 slice. A standalone `inline_formula` region is
recognized and emitted wrapped in display delimiters `\[…\]`; the GT expects it inline, so the scorer
mismatches the block. **Hard part:** deciding wrapping by class is easy (`inline_formula` → `\(…\)`),
but confirming it doesn't regress the display-formula metric needs a before/after on a subset with
both kinds — measure, don't blind-apply.

## DONE — table Edit_dist 0.434 vs TEDS 0.997 gap (it was the pipe-table renderer)

The guess recorded here — "likely OTSL→pipe cell-text normalization differences" — was right, and the
full run made it unmissable: table Edit_dist was **0.5628** across 1651 pages while TEDS sat at 0.836.
Root cause (BENCHMARKS.md, "The table gap"): `otsl_to_markdown` flattened every span marker to `<fcel>` and emitted a
GitHub pipe-table, a format that cannot express a merged cell — and **34% of our tables carry a span**.
Replacing it with the reference's own OTSL→HTML converter (parity-pinned to PaddleX on all 739 tables)
took table Edit_dist **0.5628 → 0.0801** and TEDS **0.8362 → 0.9036**. No separate diagnostic was ever
needed; the format defect explained both metrics at once.

## DONE — Load-once page-iterating recognize mode (measured)

Implemented: `paddleocr_vl_recognize <dir>... | --list <file>` builds the engine once and iterates
page dirs; `run_pipeline.sh` runs layout per page → **one** recognition process over all pending pages
→ assembly per page. Iterate, not serve — the binary already loaded once and looped crops, so this was
~30 lines; a server would have needed HTTP + image upload + prompt marshalling for the same win.

Measured on the clean box, same 118 pages as the speed run: median page **10.0s → 8.42s**, checkpoint load
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
GEMM/MLP (linear projections) of the vision encoder. With the per-page reload deleted, this is
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
