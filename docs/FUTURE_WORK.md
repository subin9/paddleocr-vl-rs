# Future work

Honest roadmap. Each item lists why it is valuable and what the hard part actually is.

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
