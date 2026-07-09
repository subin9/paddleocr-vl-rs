# Future work

Honest roadmap. Each item lists why it is valuable and what the hard part actually is.

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

## Load-once page-iterating recognize mode (before the full 1651-page run)

**Why:** `paddleocr_vl_recognize` builds one engine (loads the checkpoint) per process invocation,
and the driver invokes it once per page. Model load is only ~1.5s (small 0.9B ckpt) so the full run
is still ~2.3h, but a load-once mode that iterates page dirs removes 1651 redundant loads and is the
clean way to run + time the full set. **Hard part:** none structural — add an arg mode to the
example that loops over multiple manifest dirs reusing one built model; keep the per-page contract
identical so `run_pipeline.sh` stays the fallback.

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

## LM-prefill / vision-GEMM residual kernel work

**Why:** after moving vision + LM attention to fused Sdpa/flash, the remaining prefill gap vs torch
is candle's dense GEMM/MLP (linear projections) on the vision encoder. **Hard part:** it is a candle
maturity ceiling (generic-Rust / MKL-call-overhead GEMM vs torch's tuned oneDNN/cuBLAS), not a bug
in this repo. Low priority: attention is no longer the bottleneck and the win here is bounded by
candle's kernel quality, which is upstream.

## Batching Approach A (cu_seqlens packed vision)

**Why:** engine batching (Approach B) was measured leakage-free but with **no** throughput win,
because vision runs per-image regardless (see [BENCHMARKS.md](BENCHMARKS.md) "Region batching"). The
only path to a real vision-batch speedup is Approach A: block-diagonal `cu_seqlens` single-kernel
vision packing so N crops share one vision forward. **Hard part:** the attention masking is the
highest-leakage-risk change in the whole pipeline, and Approach B already showed the LM/scheduler
side can't turn a vision batch into throughput on its own -- so Approach A is only worth it if the
scheduler ceiling is addressed too. Data-backed deferred, not abandoned.
