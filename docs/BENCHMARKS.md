# Benchmarks

Cross-stack measurements of the **recognition VLM** (PaddleOCR-VL-1.5 via mistral.rs / candle) vs
the un-ported **transformers** reference, on one box, CPU-f32 and GPU-bf16 separately. This is a
practical "does the Rust port run faster/leaner on THIS box for THIS model" measurement, not a
controlled kernel claim: candle and PyTorch are different stacks (kernels, memory layout, no quant).
See the caveats at the bottom.

The layout stage (ONNX / this repo) is not the subject here; these numbers are about the VLM
recognition step, which dominates end-to-end cost.

## Hardware / config

- CPU: Intel Core i7-14700K, 28 threads (torch `set_num_threads(28)`, `RAYON_NUM_THREADS=28`)
- GPU: RTX 4070 Ti Super, dtype bf16, driver CUDA 13.1, nvcc 12.9
- mistral.rs build: cpu (f32) / cuda (bf16); transformers 5.13, torch 2.12.1 (+cpu / +cu126)
- Method: warmup discarded, N=20 timed iters, report median + p90, fixed seed, greedy, token/length
  parity asserted so both engines do equal work. Time only the inference region; peak memory
  reported alongside (RSS for CPU, device VRAM for GPU).

## Fair baseline (read this first)

An earlier "1.44x / 1.88x faster" headline was an **unfair-baseline artifact and is retracted on
both axes.** That baseline ran with the checkpoint's shipped `use_cache: false` default, which
disables KV caching AND makes the reference re-encode the vision tower on **every decode token**
(proven: `use_cache=False` -> 6 vision calls for 6 tokens; `use_cache=True` -> 1 call, identical
output). It is a harness/config issue, not a modeling bug. Every number below uses the **fair**
baseline (`use_cache=True`): vision encoded once per request, KV-cached decode.

## Latency, fair baseline (short OCR output, natural EOS ~6 tokens)

Speedup = port / baseline; >1 means the port is faster.

### CPU-f32 (`ocr.png`, `OCR:`, greedy, N=20)

| metric | mistral.rs (med / p90) | transformers (med / p90) | speedup |
|--------|------------------------|--------------------------|---------|
| prefill / TTFT (ms) | 4677.5 / 4898.5 | 1285.4 / 1422.3 | **0.27x (port ~3.6x slower)** |
| decode (tok/s)      | 14.45 / 14.82   | 16.02 / 23.98   | **0.90x (parity, slightly slower)** |
| total latency (ms)  | 5022.0 / 5242.2 | 1588.1 / 2129.7 | **0.32x (port ~3.2x slower)** |
| peak RSS (MB)       | 5203.8          | 6008.7          | 1.15x (port leaner) |

Token parity TRUE (both emit `[16276,93919,4,5,6,2]`). The prefill gap is candle's generic CPU GEMM
vs torch's MKL/oneDNN; it dominates this short-output total. Decode is at parity. Only memory favors
the port.

### GPU-bf16 (`ocr.png`, `OCR:`, greedy, N=20)

| metric | mistral.rs (med / p90) | transformers (med / p90) | speedup |
|--------|------------------------|--------------------------|---------|
| prefill / TTFT (ms) | 62.5 / 67.0     | 36.6 / 43.2     | **0.59x (port ~1.7x slower)** |
| decode (tok/s)      | 92.59 / 106.38  | 66.67 / 89.03   | **1.39x (modest win)** |
| total latency (ms)  | 116.0 / 142.4   | 114.8 / 129.3   | **0.99x (neutral / parity)** |
| peak VRAM (MB)      | ~2120           | ~2125           | ~1.00x (neutral) |

Token parity TRUE (and token-identical to CPU-f32, so bf16 does not diverge from f32 greedy here).
Peak VRAM is the fair device number (both engines re-run under a `nvidia-smi memory.used` delta
probe; the raw JSON's host-RSS-vs-torch-VRAM fields are not comparable). The 1.7x prefill loss and
the 1.4x decode win nearly cancel at 6 tokens.

## Kernel optimization work (P2): closing the prefill gap

The prefill regression is the truthful, baseline-independent gap. Work to close it, all with token
parity re-asserted TRUE after each change:

- **Vision-encoder attention -> fused Sdpa/flash.** GPU (`cuda flash-attn`): dense page prefill
  **1309 -> 276 ms (4.7x)**, tiny `ocr.png` **62.5 -> 33.0 ms (1.9x)**. vs torch: small inputs
  **closed** (0.59x -> 1.03x, port ties/beats); dense pages **narrowed** (0.12x -> 0.58x).
- **LM-prefill attention -> Sdpa/causal-flash.** GPU dense page prefill **284 -> 203 ms (1.40x)**;
  vs torch **0.57x -> 0.80x** on the dense page. Attention (vision + LM) is no longer the
  bottleneck; the residual is vision GEMM/MLP linear projections, which flash does not touch.
- **CPU vision Sdpa (fused CPU-flash).** `ocr.png` prefill **4677.5 -> 3110.5 ms (1.50x)**; vs torch
  **0.27x -> 0.45x** (still ~2.2x slower; candle has no BLAS backend on this box by default).
- **CPU candle `mkl` BLAS backend.** On top of Sdpa: prefill **3110.5 -> 2023.5 ms** (2.31x over
  naive), and it also accelerates the LM decode GEMMs so CPU decode **flips 0.90x -> 1.50x (port now
  beats torch)**; total **0.32x -> 0.67x**. **Ceiling:** MKL LP64 is an f32-path build only -- the
  f16 CPU parity harness fails to link (`undefined symbol hgemm_`, MKL ships sgemm/dgemm only), so
  MKL can't be the shipped default CPU backend without a candle f16-gemm patch. Not a numerics
  divergence: parity on the golden stays TRUE.

## Output-length sweep: no crossover

Dense page (1000x1800), natural greedy output past 1024 tokens (every point an unforced greedy
prefix, no EOS forcing).

- **GPU-bf16:** total speedup rises monotonically **0.24x (16 tok) -> 0.48x (64) -> 0.80x (256) ->
  0.96x (1024)** but never overtakes: a fixed ~1.1 s vision-prefill deficit is slowly amortized by a
  modest (~1.05-1.17x) decode-rate lead. It asymptotes toward parity, not past it, within any
  realistic transcription length on this dense page. (On the tiny `ocr.png`, where the prefill
  deficit is only ~26 ms, parity is already reached by 6 tokens.)
- **CPU-f32:** provably no crossover at any length. The port's own decode is 6.17 tok/s vs the
  baseline's 12.14 tok/s (~2x slower), so even in the pure-decode limit the total speedup can only
  approach that ratio (~0.508x). The port loses on both halves (prefill ~3.7x slower AND decode ~2x
  slower). Measured at L=16; longer lengths are analytic from the measured per-token rates.

Takeaway: the vision prefill is the single highest-leverage target, which is exactly what the P2
kernel work above attacks.

## Region batching: leakage-free, but no throughput win

Engine-batched (N concurrent requests) vs serialized (N sequential K=1) on GPU-bf16, best-case
identical crops forming a real B=N batch (confirmed by trace):

| N  | batched wall (med/p90) | serialized wall (med/p90) | speedup | peak VRAM | parity |
|----|------------------------|---------------------------|---------|-----------|--------|
| 1  | 98 / 100 ms            | 97 / 138 ms               | 0.99x   | 3286 MiB  | ok |
| 4  | 415 / 419 ms           | 367 / 696 ms              | 0.88x   | 3339 MiB  | ok |
| 8  | 1185 / 1255 ms         | 731 / 1109 ms             | 0.62x   | 3556 MiB  | ok |
| 20 | 7506 / 7567 ms         | 1826 / 4403 ms            | 0.24x   | 4324 MiB  | ok |

Engine batching (Approach B) **regresses** throughput here (0.24x at N=20), worsening with N, for
two structural reasons: (1) vision is the dominant cost and is NOT batched -- a B=N batch still runs
N sequential per-image vision forwards, so the ceiling is ~1.0x; (2) the batched LM adds per-row
padding + per-step mrope rebuild overhead that grows with B. VRAM is never the limit (4324 MiB at
N=20 on a 16 GB card). Token parity holds at every N (leakage-free). **Recommended batch size = 1.**
The only path to a real vision-batch win is Approach A (block-diagonal `cu_seqlens` single-kernel
vision packing), deferred with this data.

## Honest residual

After the P2 work, the remaining gap is candle's vision GEMM/MLP (dense linear projections) vs
torch's oneDNN/cuBLAS-class kernels -- a candle-maturity ceiling, not a design flaw and not claimed
closed. Attention (vision + LM) is fully on the fused Sdpa/flash path.

## Correctness (not a speed number, but the primary result)

Token-for-token greedy parity vs the transformers-5.13 reference across a 9-item validation corpus
(plain text, tables, formulas, spotting, seal, chart, CJK, low-quality scan, 2-column), on
**both** CPU-f32 and GPU-bf16. 9/9 match golden token ids; bf16 rounding never flipped a greedy
argmax on this corpus. See the repo's parity harness.

# OmniDocBench v1.5 accuracy-preservation run

The 9-item corpus proves token parity but is not a standard benchmark. This section tracks a full
OmniDocBench v1.5 run scored with the **official** evaluation code. Our own measured numbers are
`PENDING` until the runs land; nothing here is fabricated or extrapolated.

**Framing (honest, held fixed as numbers land):** the primary result is **accuracy-preservation** --
the port is token-for-token faithful to transformers on the 9-item corpus, so on OmniDocBench it
should reproduce the reference's document-parsing scores within noise. A *divergence* would be a real
and valuable finding (report the failing doc types, don't hide it). The port's edge is **deployment**
(a single self-contained Rust binary, no Python/Paddle runtime), **not** serving throughput; any
speed comparison is same-box and explicitly not a SOTA-speed claim.

## Methodology (recon — recorded before any run)

**Benchmark.** OmniDocBench — [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench)
(CVPR 2025). Code license **Apache-2.0**. Bilingual zh/en, 9 document types (academic papers,
textbooks, financial reports, exam papers, ...).

- **Actual page count (measured, not from prose): 1651 pages.** The pinned GT JSON at dataset rev
  `aa1ee96` (`OmniDocBench.json`) contains **1651** entries — that is what the scorer iterates, so
  it is the real denominator. The eval README's prose still says "1355 PDF pages" (v1.0 981 + 374
  new), which is **stale**; the shipped v1.5 dataset is larger. We report against the 1651 we
  actually score and flag the discrepancy rather than trusting the README number.
- **Strata (from the GT `page_attribute`, for the stratified subset):** language — simplified_chinese
  765, english 755, en_ch_mixed 116, traditional_chinese 13, other 2. layout — single_column 887,
  other_layout 372, double_column 184, 1andmore_column 155, three_column 53. data_source (9 types) —
  book 276, PPT2PDF 253, academic_literature 215, exam_paper 193, colorful_textbook 159, newspaper
  151, magazine 149, research_report 132, note 118, historical_document 5.

- **Eval-code pin:** branch `v1_5` @ `59b103c4b47d3a01fada83491585d6512a40c0bc` (2026-04-10). `main`
  @ `2b161d0` (2026-06-26) is the moving tip; we pin the explicit `v1_5` branch for reproducibility.
- **Dataset:** [huggingface.co/datasets/opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
  / [opendatalab.com/OpenDataLab/OmniDocBench](https://opendatalab.com/OpenDataLab/OmniDocBench).
  **License/terms: research use only, non-commercial.** → dataset pages are **gitignored, never
  committed**; only our scripts/configs and result JSON/markdown are committed.
- **Metrics (v1.5):** Text = normalized **Edit distance** (also BLEU/METEOR); Tables = **TEDS**
  (+ TEDS-S structure-only); Formulas = **CDM** (Character Detection Matching via LaTeX render +
  image compare); Reading order = **Edit distance**. v1.5 uses hybrid text/formula matching.
- **Overall formula (official):** `Overall = ((1 − text_Edit) × 100 + table_TEDS + formula_CDM) / 3`.
- **Scorer invocation:** `python pdf_validation.py --config configs/end2end.yaml`; prediction path
  is a **folder of per-page markdown files**, one `.md` per image. GT + prediction paths set in the
  end2end config. Our assembler already emits per-region reading-order markdown, so the conversion is
  "one `.md` per page image" — the integration risk (verified on the 5-page slice, not assumed) is
  table/formula markdown dialect.
- **Exact filename mapping (verified from `dataset/end2end_dataset.py:162-174` @ pin 59b103c):** for
  each GT page, `img_name = basename(page_info.image_path)` (e.g. `foo.pdf_7.jpg`); the scorer looks
  for `<pred_folder>/<img_name[:-4]>.md` — i.e. **strip the 4-char image extension, append `.md`**
  (`foo.pdf_7.jpg` → `foo.pdf_7.md`). Fallbacks it also tries: `.mmd` and `.pdf`-stripped `.md`
  (nougat/marker) and `img_name + '.md'` (mineru). A **missing** prediction prints
  `!!!WARNING: No prediction for <img>` and **skips** that page (contributes nothing) — so a crashed
  page silently drops from the denominator; our runner must guarantee one `.md` per GT image.
- **GT schema:** GT is a single JSON list; each element = `{layout_dets[], extra, page_info}`.
  `page_info` has `image_path`, `height`, `width`, `page_no`, and `page_attribute`
  (`data_source`, `language` en/zh, `layout`, `special_issue[]`) — the strata for the subset. Each
  `layout_dets[]` entry has `category_type`, `poly`, `order`, `text`, `ignore`, `attribute`. The
  scorer parses our full-page `.md` into text/formula/table blocks and matches them to `layout_dets`
  via `match_method: quick_match`.
- **Scorer runs in its own venv (isolation risk).** `requirements.txt` pins an old, conflicting
  stack (numpy 1.24.4, pandas 2.0.3, scikit-learn 1.1.2, plus `apted`, `Levenshtein`, `mmeval`,
  `pylatexenc`, `func-timeout`) — incompatible with the inference venv's torch 2.12 / numpy 2.x.
  The scorer is pure-CPU post-processing, so it gets a **separate** venv; it never shares the
  inference env. (CDM formula metric needs an extra env; `CDM_plain` in the config exports the CDM
  input JSON without it — decide at subset time whether to stand up full CDM or accept `CDM_plain`.)

## Reference score (verified, primary source)

**PaddleOCR-VL-1.5 (0.9B)** on OmniDocBench v1.5, from the paper
[*PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM ...*](https://arxiv.org/abs/2601.21957) (arxiv
2601.21957v1, Table 2). Not quoted from memory; cross-checked against the official overall formula.

| model | Overall | Text-Edit ↓ | Formula-CDM ↑ | Table-TEDS ↑ | Table-TEDS-S ↑ | ReadOrder-Edit ↓ |
|-------|---------|-------------|---------------|--------------|----------------|------------------|
| **PaddleOCR-VL-1.5** | **94.50** | 0.035 | 94.21 | 92.76 | 95.79 | 0.042 |
| PaddleOCR-VL (v1.0)  | 92.86 | 0.035 | 91.22 | 90.89 | 94.76 | 0.043 |

Consistency check: `((1 − 0.035) × 100 + 92.76 + 94.21) / 3 = 94.49 ≈ 94.50` ✓ — the reported
overall and the formula agree, so the pinned numbers are self-consistent.

**This is the target the Rust port must land within noise of.** PRESERVED = overall within scorer
noise of 94.50 (noise band to be quantified from the subset run); DIVERGES = otherwise, reported
with the per-doc-type breakdown.

## Our measured scores — PENDING

| run | Overall | Text-Edit | Formula-CDM | Table-TEDS | ReadOrder-Edit | verdict |
|-----|---------|-----------|-------------|------------|----------------|---------|
| 5-page plumbing slice | PENDING | PENDING | PENDING | PENDING | PENDING | — |
| stratified subset (~100–200) | PENDING | PENDING | PENDING | PENDING | PENDING | — |
| full v1.5 (1,355) | PENDING | PENDING | PENDING | PENDING | PENDING | — |

Speed table (secondary; Rust GPU-bf16 vs transformers floor, per-stage) also PENDING — see the
existing latency sections above for the single-crop microbenchmarks already measured.

## Caveats

- Different stacks (candle/mistral.rs vs PyTorch/transformers): kernels, memory layout, no quant.
- The TTFT/decode split has a minor methodology asymmetry (the port reports an exact
  prefill/decode split from its own `Usage`; the reference times a separate prefill-only forward and
  a separate `generate`). Total latency, the headline, is directly wall-clock comparable.
- Decode tok/s is computed identically for both engines (`(tokens-1)/(total-ttft)`), not from each
  engine's self-report.
- Short-output decode (6 tokens) has wide error bars; p90 over 20 iters bounds it.
