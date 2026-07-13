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

The 9-item corpus proves token parity but is not a standard benchmark. This section is a full
OmniDocBench v1.5 run scored with the **official** evaluation code. **The runs have landed** — the
result is in the HEADLINE table below (text, reading-order and table at parity; formula −2.44 CDM).
Sections are kept in the order they were written, so the pre-run methodology and the two superseded
verdicts stay readable beneath the result rather than being edited into hindsight.

**Framing, pre-registered before any number was known (and it is worth checking it against what
landed):** the primary result is **accuracy-preservation** — the port is token-for-token faithful to
transformers on the 9-item corpus, so on OmniDocBench it should reproduce the reference's
document-parsing scores within noise. A *divergence* would be a real and valuable finding (report the
failing doc types, don't hide it). The port's edge is **deployment** (a single self-contained Rust
binary, no Python/Paddle runtime), **not** serving throughput; any speed comparison is same-box and
explicitly not a SOTA-speed claim.

**How that held up:** preservation confirmed on three of four metrics, one real gap (formula CDM)
reported rather than buried — and the deployment-not-throughput claim survived contact with the
llama.cpp cross-check, which the port **loses** by 2.7x (see the speed sections). Two divergences we
*did* find turned out to be defects in our own glue, not in the ported weights (layout
post-processing, table HTML), which is exactly the outcome the pre-registration said to look for.

## Repetition guard: measured A/B on the scored run

`assemble::truncate_repetitive_content` + `truncate_repeating_lines` (upstream's own post-hoc string
truncator, ported — see FUTURE_WORK) landed *after* the numbers above were measured, and it touches
scored output, so it was A/B'd rather than assumed. Method: take the **same** `work_reflayout`
`results.json` (no VLM re-run — the guard is pure post-processing), re-assemble with and without it,
score both with the official scorer. The baseline column reproduces the headline numbers, which is
what makes the delta trustworthy.

| `subset: v1.5` | baseline | + guard | |
|---|---|---|---|
| text Edit ↓ | 0.0327 | **0.0323** | −0.0005 |
| formula Edit ↓ | 0.1833 | **0.1817** | −0.0016 |
| table TEDS ↑ | 0.9275 | **0.9282** | +0.0007 |
| table Edit ↓ | 0.0568 | **0.0556** | −0.0012 |
| reading-order Edit ↓ | 0.0415 | **0.0414** | −0.0001 |

Every metric moves the right way. Scope: **204 of 78,710** recognized blocks are altered (143 of them
`image`, which assembly drops anyway), across **23 of 1649** pages.

**The one thing that looks like a regression, and is not.** Exactly 2 of 665 tables move: one gains
**+0.70 TEDS** (0.11 → 0.81 — the model emitted a real table then looped on `<ecel>` forever; cutting
the loop recovers `Beef meat / Chicken meat / Pork meat`), and one *loses* **−0.33** (0.38 → 0.05).
The loser is degenerate end to end: **zero `<nl>` in 7,173 chars** — the model emitted 1,024 cells and
never broke a row. TEDS was paying partial credit for a big garbage grid resembling a big real grid,
and the shorter garbage resembles it less. No legitimate table content is destroyed; it is a scoring
artifact of a page that was lost either way. It sits in the `*_hard` subset, which is why `ALL` table
TEDS wobbles −0.0002 while `v1.5` gains.

CDM is not in this A/B (it needs the CDM environment); formula `Edit_dist` stands in, and it improves.
Re-running CDM on the guarded predictions is the open item — 18 `display_formula` blocks are degenerate
and get cleaned, which is the right shape to move the −2.44.

## `crop_margin`: the formula crop parity fix, A/B'd on both stacks

Upstream trims a formula crop to its ink before recognizing it (`crop_margin`, formula blocks only);
this port did not. Ported, then measured the same way — re-recognize the **1,685** formula crops with
the tightened crop, splice them into the scored run's `results.json`, re-assemble, re-score. 911 of
them come back with different text; 291 of 1649 pages change.

| `subset: v1.5` | port +guard | port +guard+`crop_margin` | llama.cpp base | llama.cpp +`crop_margin` |
|---|---|---|---|---|
| text Edit ↓ | 0.0323 | 0.0322 | 0.0310 | 0.0309 |
| **formula Edit ↓** | 0.1817 | **0.1697** | 0.1913 | **0.1805** |
| table TEDS ↑ | 0.9282 | 0.9282 | 0.9252 | 0.9252 |
| reading-order ↓ | 0.0414 | 0.0414 | 0.0412 | 0.0413 |

**−0.0120 (−6.6%) formula edit on the port, −0.0108 (−5.6%) on llama.cpp.** Only the formula metric
moves, because only the formula crops changed.

The cross-stack column is doing real work here, and it is the reason the llama.cpp control was re-run
rather than left standing. llama.cpp reads **this pipeline's crop PNGs** — it is independent on the
decode path and *common-mode* on the crop path. So a crop-side fix must improve both stacks by a
similar margin, and a port-side bug could not. Both improved, by 6.6% and 5.6%. That is the signature
of a crop fix, and it is what the earlier "llama.cpp reproduces the gap, so it is not ours" argument
could never have detected.

## Methodology (recorded before any run)

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
- **CDM decision — RESOLVED: scored with `Edit_dist`, not CDM.** Probed the scorer-venv:
  `CDM` is in `METRIC_REGISTRY` and `latex2bbox_color` imports OK, **but the box has no LaTeX
  toolchain** — `pdflatex`/`xelatex`/`latex`/`dvisvgm` all missing (only `node` present), so CDM's
  render step would fail at runtime. Upstream requires a dedicated CDM environment (README recommends
  Docker); standing up a full TeX distribution is a heavyweight, out-of-scope side quest. **Decision:**
  formula is scored with `display_formula Edit_dist` (as the committed `subset150.end2end.yaml`
  already does). **Consequence for the compare:** Text-Edit (paper 0.035) and Table-TEDS (paper 92.76)
  are **directly comparable** to the reference; formula is a **different metric** than the paper's CDM
  (94.21), so the paper's official CDM-based `Overall` (94.50) **cannot be reproduced exactly**. We
  report per-metric + an **Edit-only proxy Overall** using `((1−text_Edit)×100 + table_TEDS +
  (1−formula_Edit)×100)/3`, **clearly flagged as a proxy** (substitutes formula-Edit for formula-CDM),
  never presented as the official 94.5-comparable Overall.
- **Scorer venv stood up + smoke-tested (PASSED).** `bench/omnidocbench/setup_scorer_venv.sh` builds
  an isolated **Python 3.10** venv via `uv` from the pinned `requirements.txt` (py3.10 chosen because
  the old pins have cp310 manylinux wheels — no source builds; installed clean). Ran the scorer
  unchanged on its shipped 18-page demo (`pdf_validation.py -c configs/end2end.yaml`, GT + preds both
  from `demo_data/`): **exit 0**, all four metric families computed and written to `result/*.json` —
  text_block Edit_dist (ALL_page_avg 0.351), display_formula Edit_dist (0.319) + `CDM_plain` export
  (writes `..._display_formula_formula.json`, the CDM-input JSON, **without** needing the CDM render
  env), table TEDS (0.926) + TEDS-S (0.915), reading_order Edit_dist (0.161). These demo numbers are
  a mixed sample, **not** PaddleOCR-VL output — the point is only that the scorer stack runs
  end-to-end and emits a score before we feed it our predictions. Scorer install is now de-risked
  independently of our output conversion.

## Reference score (verified, primary source)

**PaddleOCR-VL-1.5 (0.9B)** on OmniDocBench v1.5, from the paper
[*PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM ...*](https://arxiv.org/abs/2601.21957) (arxiv
2601.21957v1, Table 2). Not quoted from memory; cross-checked against the official overall formula.

| model | Overall | Text-Edit ↓ | Formula-CDM ↑ | Table-TEDS ↑ | Table-TEDS-S ↑ | ReadOrder-Edit ↓ |
|-------|---------|-------------|---------------|--------------|----------------|------------------|
| **PaddleOCR-VL-1.5** | **94.50** | 0.035 | 94.21 | 92.76 | 95.79 | 0.042 |
| PaddleOCR-VL (v1.0)  | 92.86 | 0.035 | 91.22 | 90.89 | 94.76 | 0.043 |

**Independently re-verified 2026-07-11** — not trusted from this doc or from memory. The rows
above were re-extracted **verbatim from the raw arXiv HTML table markup** of
[2601.21957v1](https://arxiv.org/html/2601.21957v1) (Table 2, *"Comprehensive evaluation on
OmniDocBench v1.5"*), by parsing `<tr>` cells rather than reading a summary:

```
Model Type | Methods | Parameters | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | TableTEDS-S↑ | Reading OrderEdit↓
PaddleOCR-VL [cui2025paddleocrvl] | 0.9B | 92.86 | 0.035 | 91.22 | 90.89 | 94.76 | 0.043
PaddleOCR-VL-1.5              | 0.9B | 94.50 | 0.035 | 94.21 | 92.76 | 95.79 | 0.042
```

**The checkpoint we run is PaddleOCR-VL-1.5, confirmed by content hash — not by filename.**
`ref/weights/model.safetensors` sha256 = `d557c9d8997ae57ed3b1b33bdf347be878cc335687f32ca105341c16973f8958`,
which equals the LFS oid of `model.safetensors` in
[PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) (HF API
`/tree/main`) and **not** that of `PaddleOCR-VL` (`3085f104…`) or `PaddleOCR-VL-1.6` (`85a479d5…`).
So the **94.50 row is the correct target**, and the v1.0 row is not.

**Official `Overall` definition** (from the scorer's own repo, not the paper —
[OmniDocBench README](https://github.com/opendatalab/OmniDocBench) `README.md:414`, eval pin `59b103c`):

$$\text{Overall} = \frac{(1-\text{Text Edit Distance}) \times 100 + \text{Table TEDS} + \text{Formula CDM}}{3}$$

The same README (`:71`) pins the **aggregation**: v1.5 *"removed the Chinese/English grouping, now
calculating the average score across all pages"* → the published `TextEdit` is the **all-pages
average** (`ALL_page_avg`), not an English-only or corpus-concatenated (`edit_whole`) figure. This
matters: it fixes which of our three scorer aggregates is the apples-to-apples one.

Consistency check (this validates the definition, the aggregation, *and* the transcribed row):
`((1 − 0.035)×100 + 92.76 + 94.21) / 3 = 94.49 ≈ **94.50**` ✓ and for v1.0
`((1 − 0.035)×100 + 90.89 + 91.22) / 3 = 92.87 ≈ **92.86**` ✓. Both published Overalls reproduce from
their own per-metric cells to ±0.01, so the numbers are self-consistent and correctly transcribed.

**This is the target the Rust port must land within noise of.** PRESERVED = overall within scorer
noise of 94.50; DIVERGES = otherwise, reported with the per-doc-type breakdown.

## Plumbing validation (n=5) — integration proven, NOT an accuracy verdict

The 5-page slice exists to de-risk the *integration* (our markdown → the official scorer's input),
not to measure accuracy. **Do not read these numbers as the reference comparison** — n=5,
hand-picked English pages, one deliberately hard academic double-column. The verdict comes from the
stratified subset and full run.

**How our output is fed to the scorer (the contract):**
- Pipeline per page: `paddleocr-layout <img> <dir>` (layout → crops + `manifest.json`) →
  `paddleocr_vl_recognize <dir>` (PaddleOCR-VL bf16 on GPU → `results.json`) →
  `paddleocr-layout assemble <dir>/results.json` → one markdown doc on stdout.
- We write that markdown to `preds/<stem>.md` where `<stem>` = the GT `image_path` with its **last 4
  chars stripped** — byte-for-byte what `end2end_dataset.py` looks up (`img_name[:-4] + '.md'`).
  Verified against both `.png` and `.jpg` GT names.
- Scorer: `pdf_validation.py --config <subset>.end2end.yaml`, `match_method: quick_match`, from the
  pinned eval repo (`59b103c`) in the isolated `scorer-venv`.
- Repro: `bench/omnidocbench/{run_pipeline.sh, make_subset.py}`, stem list
  `subsets/smoke5.txt`; scorer stdout captured at `bench/omnidocbench/results/smoke5.scorer.log`.

**Markdown-dialect conversion — surfaced on these 5 (the point of the slice):**
- **JPEG decode was a hard blocker (fixed).** The layout binary was built `image` = png-only;
  **981/1651 (59%) GT pages are `.jpg`** → `Unsupported(Jpeg)`. Fix: add the `jpeg` feature
  (`Cargo.toml`), rebuild. Now decodes both. Without this the majority of the benchmark can't run.
- **Tables:** OTSL (`<fcel>`/`<nl>`) → GitHub-markdown pipe tables; the scorer's `md_table_reg`
  matches them and converts to HTML for TEDS. Aligned — no fix.
- **Formulas:** the VL model *itself* emits `\[…\]` / `\(…\)` delimiters; `assemble` passes them
  through verbatim; the scorer's `display_reg`/`inline_reg` catch them. Aligned — no fix. (Earlier
  worry that formulas were unwrapped was wrong; the model wraps them.)
- **Figure/chart regions OCR to junk → FIXED (`VISUAL_ONLY_CLASSES` skip in `src/assemble.rs`).**
  Chart crops (scatter plots) transcribe as long `col | val` numeric dumps. Measured effect, direct
  A/B (assemble the SAME `results.json` with vs without the skip, score both back-to-back — isolates the
  one code change; only the academic page has charts, so only it moves):

  | metric (academic page) | no-skip | **skip** |
  |---|---|---|
  | text_block Edit_dist | 0.9953 | **0.0000** |
  | table TEDS | 0.6883 | **0.9969** |
  | reading_order Edit_dist | 0.1333 | **0.0000** |

  The chart's pipe-formatted rows were being parsed by the scorer both as a *table* (wrecking TEDS)
  and as *text* (wrecking text_block + reading order). Dropping `chart`/`image`/`seal`/`*_image` (all
  visual-only, no GT text/table/formula counterpart) fixes it. Overall smoke5 text_block 0.276 → 0.077.
  (The reference emits image placeholders the scorer strips; a skip is equivalent for scoring.)
  Recognition still *runs* on these crops today; skipping that too is a later speed win.
- **book text_block 0.339 (remaining, separate nuance):** standalone `inline_formula` regions come
  back wrapped as display `\[…\]` — a minor dialect mismatch, not chart pollution. Tracked in FUTURE_WORK.

**METHODOLOGY GOTCHA (recorded so it doesn't bite the subset/full run):** `pdf_validation.py` names
its output `result/<basename(prediction_dir)>_<match_method>_metric_result.json` (`pdf_validation.py:47`),
NOT a fixed name. The pinned eval repo also ships a committed *demo-reference* `end2end_quick_match_*`
in `result/`. Reading the wrong file silently compares against the demo, not your run — always read the
file named after YOUR prediction dir (here `smoke5_quick_match_metric_result.json`).

**5-page raw scores (official scorer, quick_match, WITH visual-skip; edit ↓ lower better, TEDS ↑ higher):**

| metric | ALL_page_avg | edit_sample_avg | note |
|--------|--------------|-----------------|------|
| text_block Edit_dist | 0.077 | 0.024 | per-source: academic 0.00 / PPT 0.00 / exam 0.01 / newspaper 0.03 / book 0.34 |
| display_formula Edit_dist | 0.110 | 0.110 | formulas extracted correctly (unchanged by skip) |
| table Edit_dist | 0.434 | — | 2 tables (academic); strict, TEDS is the headline |
| table TEDS / TEDS-structure | 0.997 / — | — | n=2 tables only |
| reading_order Edit_dist | 0.000 | — | — |

**Reproducibility confirmed:** the pipeline (layout+recognize) is deterministic (re-ran newspaper page:
0/17 regions differ, manifest identical) and the scorer is deterministic (2× identical). The
committed no-skip numbers reproduce byte-for-byte on re-run. Numbers are sane and non-degenerate.
This slice proves the integration AND the visual-skip fix; it is NOT the accuracy verdict (n=5).

## Our measured scores

### HEADLINE (2026-07-12) — the benchmark is 1355 pages, not 1651. On it, the port is at parity.

**Read this before any number below it.** Every score in the sections that follow was computed over
all **1651** pages in the shipped `OmniDocBench.json`. That file is a **superset of the benchmark**:
it bundles 296 pages belonging to three *adversarial* diagnostic subsets, tagged in the GT's own
`page_attribute.subset` field:

| `subset` tag | pages | in the published leaderboard? |
|---|---|---|
| `v1.5` | **1355** | **yes — this IS the benchmark** |
| `equation_hard` | 100 | no |
| `layout_hard` | 99 | no |
| `table_hard` | 97 | no |
| shipped JSON total | 1651 | — |

Evidence it is 1355, not 1651: the official README states "**This benchmark includes 1355 PDF pages**"
(twice: `README.md:13`, `:88`, eval pin `59b103c`), and the v1.5 changelog's own arithmetic closes
exactly — v1.0 was 981 pages, v1.5 "**Added 374 new pages**", and 981 + 374 = **1355**, the precise
count carrying the `v1.5` tag. The three `*_hard` sets ride along in the same JSON as extra
attribute-tagged diagnostics.

This mattered enormously because **the hard pages are wildly over-represented exactly where they
hurt**: they are 18% of pages overall but **29% of all formula-bearing pages** (91 of 313) and **21%
of all table-bearing pages** (97 of 458). Scoring the superset therefore dragged the formula and
table numbers down hard, and every "gap" we chased was partly this.

No re-run was needed to correct it: the official scorer already emits a per-attribute breakdown using
the **identical reduction** as the headline (`metrics/show_result.py:133` — `page_avg`: mean within a
page, then mean across pages, grouped by attribute). So its `subset: v1.5` row *is* the leaderboard
number restricted to the benchmark's 1355 pages. Reading that row off the runs we already have:

| metric | **ours — `subset: v1.5` (1355 pg) LIKE-FOR-LIKE** | ours — all 1651 (superset) | published PaddleOCR-VL-1.5 | verdict |
|---|---|---|---|---|
| text Edit ↓          | **0.0328** | 0.0368 | 0.035 | **PARITY** (marginally better) |
| table TEDS ↑         | **92.75**  | 90.36  | 92.76 | **PARITY** (−0.01) |
| table TEDS-S ↑       | **95.95**  | 94.33  | 95.79 | **PARITY** (+0.16) |
| reading-order Edit ↓ | **0.0415** | 0.0434 | 0.042 | **PARITY** (marginally better) |
| formula CDM ↑        | **91.77**  | 80.90  | 94.21 | **GAP −2.44** ← the one real divergence |
| **Overall** ↑        | **93.75**  | —      | 94.50 | **−0.75**, entirely from formula |

`Overall = ((1 − TextEdit)×100 + TableTEDS + FormulaCDM) / 3`, the scorer's own definition
(`README.md:414`). Ours: `((1−0.0328)×100 + 92.75 + 91.77)/3 = 93.75`.

**The 1355-page reading is also what the data independently corroborates.** On that slice *three*
metrics land within noise of published *simultaneously* (TEDS 92.75 vs 92.76, TEDS-S 95.95 vs 95.79,
RO 0.0415 vs 0.042). If the leaderboard were really scored on 1651, a faithful port would have to
match there — and we would be 2.4 TEDS points off while coincidentally landing dead-on across three
metrics on an arbitrary 1355-page subset. That is not a coincidence a wrong hypothesis produces.

**Caveat, stated plainly:** the leaderboard does not publish its page list, so "the benchmark = the
1355 `v1.5`-tagged pages" is an *inference* from the README, the changelog arithmetic, and the
three-metric agreement above — not a statement we can read off an official manifest. Both columns are
therefore printed above; nothing is hidden. Every downstream section still carries its original
1651-page numbers, which are the **pessimistic** ones.

**Verdict per metric:** text **PRESERVED** · reading-order **PRESERVED** · table **PRESERVED** ·
formula **GAP of −2.44 CDM** (the only remaining divergence; see the formula section below).

The formula gap is **root-caused** (formula section, below): it is a **CJK-formula** gap — within v1.5, Chinese formula
pages score CDM 0.8730 vs 0.9349 for English, and at 28% of formula-bearing pages they account for
1.72 of the 2.44 points. It is **not a port defect**: llama.cpp, an independent stack on the same
crops, reproduces the same CJK penalty. So all four metrics are consistent with a faithful port; the
formula deficit is the model's, not the translation's.

### Historical (1651-superset scoring — superseded by the table above, kept for audit)

| run | Overall | Text-Edit | Formula(metric) | Table-TEDS / -S | ReadOrder-Edit | verdict |
|-----|---------|-----------|-------------|------------|----------------|---------|
| 5-page slice (visual-skip) | n/a (n=5) | 0.077 pg-avg | edit 0.110 (no CDM env) | 0.997 / — (n=2) | 0.000 | plumbing + skip OK |
| **stratified subset (n=150)** | **84.1 (Edit-proxy)** | **0.0709** pg-avg | **edit 0.2724** (NOT CDM) | **0.8659 / 0.9112** | **0.0919** | **SANE — proceed to full** |
| **full 1651 superset** | **≤ 91.80** (see below) | **0.0797** pg-avg | edit 0.2559 (**NOT CDM**) | **0.8336 / 0.8761** | **0.0929** | DIVERGES — **artifact of the superset + 2 assembly bugs since fixed** |
| **paper reference** | 94.50 | 0.035 | CDM 94.21 | 92.76 / 95.79 | 0.042 | target (on **1355**, not 1651) |

Speed (secondary) was measured, but **against llama.cpp, not against a transformers full-page floor**
— see the speed sections for the per-page and per-stage numbers. The transformers reference was never run over
the full page set; only the single-crop latency microbenchmarks above exist for it. That remains a
**gap we state rather than estimate**, and it is why no "vs transformers, end-to-end" row appears
anywhere in this doc.

### Stratified subset (n=150) — result + verdict

Scored by the official `pdf_validation.py` (eval pin `59b103c`, `quick_match`) on the 150-page
stratified subset (`subset150.txt`; simplified_chinese 71 / english 68 / other 11), GPU-bf16, K=1
serial recognition, visual-skip on. Result JSON: `bench/OmniDocBench/result/subset150_quick_match_*`.
Values are `ALL_page_avg` (edit ↓ lower better, TEDS ↑ higher better).

**Edit-proxy Overall = ((1−0.0709)×100 + 86.59 + (1−0.2724)×100) / 3 = 84.1** — a **proxy**, not the
paper's 94.5: it substitutes **formula-Edit (0.2724)** for the paper's **formula-CDM (94.21)**, which
alone costs the proxy ~7 pts vs the CDM-based official Overall. Not presented as 94.5-comparable.

**Why the aggregate runs above the paper, and why that is expected, not a regression:** the subset is
stratified — it *oversamples* hard minority categories the full set is not dominated by. The
`data_source`/`language` breakdown shows the port essentially **matches the paper on clean English
pages** and degrades only on the hard-weighted tail:

| slice | Text-Edit | ReadOrder-Edit | Table-TEDS | read |
|-------|-----------|----------------|------------|------|
| **language: english** | **0.0338** | **0.0461** | 0.8840 | ≈ paper (text 0.035, RO 0.042) |
| subset: v1.5 (standard items) | 0.0653 | 0.0907 | 0.8989 | close |
| language: simplified_chinese | 0.1101 | 0.1325 | 0.8626 | CJK harder |
| data_source: fuzzy_scan | 0.5025 | 0.0 | — | degraded scan (tail) |
| data_source: research_report | 0.2918 | 0.2407 | — | dense multi-col (tail) |
| data_source: historical_document | 0.1724 | 0.2857 | — | tail |
| watermark | 0.2244 | 0.1821 | 1.0 | tail |
| subset: table_hard | 0.2322 | 0.2031 | 0.7589 | hard tables (tail) |

**English text-edit 0.0338 vs paper 0.035 and English read-order 0.0461 vs paper 0.042 are within
noise** — direct evidence the port preserves recognition accuracy where the subset composition
matches the paper's dominant mix. The aggregate gap is **subset-composition-driven, not a port
defect.** No layout/reading-order collapse (RO 0.092 aggregate, 0.046 EN); no degenerate metric.
Table-TEDS 0.866 aggregate vs paper 0.928 is the widest standard-category gap (academic_literature
TEDS 0.804, note 0.613 n-small); worth watching on the full set but not a STOP condition.

**Verdict: SANE — proceed to the full 1651 run** (the only apples-to-apples comparison to the paper).
Not "diverges badly": metrics are non-degenerate, English lands on the paper, and the
elevated aggregate is explained by intentional hard-case stratification. The full-set overall vs
94.50 (with the same Edit-proxy caveat, or full CDM if a LaTeX env is stood up) is the accuracy verdict.

### Full set (n=1651) vs published — side-by-side and VERDICT

Scored by the official `pdf_validation.py` (eval pin `59b103c` + the upstream dangling-anno-id guard,
`quick_match`, config `full1651.end2end.yaml`), GPU-bf16, K=1 serial recognition. Result JSON:
`bench/omnidocbench/results/full1651_quick_match_metric_result.json`. **1649/1651 pages scored** — the
2 empty-layout pages have no `.md`, and the scorer *skips* them (`end2end_dataset.py:172` → `continue`,
confirmed by exactly 2 `!!!WARNING: No prediction` lines in the run log), so they are excluded from the
denominator rather than 0-scored. Our numbers are therefore, if anything, marginally *flattered*.

Aggregation is `ALL_page_avg` for the edit metrics, matching the published convention pinned above.

| metric | ours (1651) | published PaddleOCR-VL-1.5 | delta | comparable? |
|--------|-------------|----------------------------|-------|-------------|
| Text-Edit ↓ | **0.0797** | **0.035** | **2.28× worse** (+0.045) | ✅ same metric, same aggregation |
| Table-TEDS ↑ | **83.36** | **92.76** | **−9.40 pts** | ✅ |
| Table-TEDS-S ↑ | **87.61** | **95.79** | **−8.18 pts** | ✅ |
| ReadOrder-Edit ↓ | **0.0929** | **0.042** | **2.21× worse** (+0.051) | ✅ |
| Formula | edit-dist 0.2559 | **CDM 94.21** | — | ❌ **NOT COMPARABLE** — different metric |
| **Overall** | **≤ 91.80** | **94.50** | **≥ −2.70** | bounded, see below |

**The Overall is bounded, not extrapolated.** We have not run CDM (it needs a LaTeX render environment;
our formula edit-distance is a *proxy*, not a substitute). But CDM is bounded by 100 by construction, so
plugging our two measured components into the official formula gives a hard **upper bound**:

- ours, granting a *perfect* formula score (CDM = 100): `((1−0.0797)×100 + 83.36 + 100)/3` = **91.80**
- ours, granting the *published* formula score verbatim (CDM = 94.21): `(92.03 + 83.36 + 94.21)/3` = **89.87**

Even in the physically-impossible best case the port cannot reach 94.50. **The verdict does not depend
on the missing CDM number.**

**VERDICT: DIVERGES.** Text edit distance is 2.3× the published figure, table TEDS is 9.4 points below,
reading order is 2.2× worse, and the Overall ceiling (91.80) sits 2.70 points under the published 94.50.
This is not scorer noise — it is a consistent gap across three independent metrics on the full
1651-page benchmark, i.e. the same set the paper reports on. **The accuracy-preservation claim for this
port is NOT supported.**

**Correction to the subset150 verdict above — its central argument does not survive the full run.**
That section argued the gap was "subset-composition-driven, not a port defect", resting on
*"English text-edit 0.0338 vs paper 0.035 … within noise"*. Two problems, both visible only now:

1. **That comparison was never apples-to-apples.** The published 0.035 is an **all-pages** average (the
   v1.5 leaderboard removed the EN/ZH split — see the README pin above). Comparing our *English-only*
   slice against their *all-pages* figure is a category error; it silently gave the port the easiest
   slice and the paper the full mix.
2. **The number itself did not hold.** On the full set, `language: english` text-edit is **0.0575**, not
   0.0338 — the subset's 68 English pages were an easy draw, not a representative one.

The composition explanation is therefore withdrawn. The gap is **present in the aggregate the paper
actually reports**, and the attribution work below must localize it (layout/reading-order vs in-crop
recognition vs assembly), not explain it away. Known assembly-side contributor already found on the
5-page slice: the assembler
**emits no text for `image`-class regions** (page `yanbaopptmerge_yanbaoPPT_5885` recognized "流水声" and
then dropped it), which costs text-edit on every page whose content the layout model labels `image`.

**Worst slices on the full set** (text-edit ↓, page-avg) — the attribution targets:

| slice | Text-Edit | ReadOrder-Edit | note |
|-------|-----------|----------------|------|
| handwriting | **0.4740** | **0.4423** | worst by far; both metrics collapse together |
| text_O-shaped_wrap | 0.3417 | 0.5625 | reading-order failure ⇒ text failure |
| transparent_pages | 0.2251 | 0.1586 | |
| text_embedded_in_image | 0.1674 | 0.4470 | consistent with the `image`-region drop above |
| data_source: historical_document | 0.1551 | 0.3368 | |
| data_source: fuzzy_scan | 0.1502 | 0.1043 | (much milder than subset150's 0.5025) |
| language: traditional_chinese | 0.1489 | 0.2536 | |
| language: simplified_chinese | 0.1017 | 0.1154 | vs english 0.0575 — CJK ~1.8× worse |
| data_source: research_report | 0.1016 | 0.1254 | |

The tight coupling of text-edit and reading-order edit on the worst slices (handwriting,
O-shaped-wrap, text-embedded-in-image) is the strongest available hint that a **large share of the gap
is layout/assembly-side, not in-crop recognition** — but that is a hypothesis for the sections below
to test, not a
finding. It is recorded here as such.

### REF_LAYOUT: the root cause of the text/reading-order gap, and the default flip

The divergence above was mostly **not** a VLM port defect. Our layout stage shipped the raw detector's
own defaults — score threshold **0.5**, no NMS, no per-class box merge — while the reference pipeline
(PaddleX `LayoutDetection`) post-processes at threshold **0.3** + `layout_nms` + per-class bbox merge +
`filter_boxes`. The hardcoded 0.5 silently discarded every detection in the 0.3–0.5 band, so whole
regions never reached the VLM. We ported that post-processing (`ref_postprocess`, parity-tested against
PaddleX in `tests/ref_postproc_parity.rs`) and re-ran + re-scored the **full 1651 pages**.

**A/B, official scorer (pin `59b103c`, `quick_match`), same 1649 scored pages, same binary, GPU-bf16,
K=1 serial recognition, only the layout post-processing differs:**

| metric (page_avg unless noted) | baseline `nonest2` (was default) | **`reflayout` (NEW default)** | delta | published PaddleOCR-VL-1.5 |
|---|---|---|---|---|
| text_block Edit_dist ↓ | 0.0722 | **0.0428** | **−40.7% rel** | 0.035 (**1.22×** gap, was 2.07×) |
| text_block edit_whole ↓ | 0.0662 | 0.0497 | −0.0165 | — |
| reading_order Edit_dist ↓ | 0.0917 | **0.0500** | **−45.5% rel** | 0.042 (**1.19×** gap, was 2.21×) |
| table TEDS ↑ | 0.8340 | 0.8362 | +0.0022 | 92.76 (**−9.14 pts**) |
| table TEDS_structure_only ↑ | 0.8765 | 0.8793 | +0.0028 | 95.79 (−7.86 pts) |
| display_formula Edit_dist ↓ (PROXY, **not** CDM) | 0.2564 | 0.2495 | −0.0069 | CDM 94.21 — **not comparable** |

**Which TEDS reduction (this bit the doc once, so it is pinned here).** The scorer emits table TEDS
under **two** reductions: a **per-table** average (printed as `all` — every table in the corpus weighted
equally) and a **page average** (`page` → `ALL` — mean within a page, then across pages). They differ by
~0.005 here. The published figure is a page average (the v1.5 README pin above), so **every TEDS number
in this doc is the page reduction**, read from `table.page.TEDS.ALL` in the result JSON — including these
A/B rows, which earlier quoted the per-table `all` figure against the page-average 92.76 and so were
comparing two different aggregations. Corrected; the direction and every verdict are unchanged.

Better on **every** metric, orders of magnitude outside the pre-registered ±0.0005 noise band, so the
rule registered *before* the number was known fires: **the default is now `ref_postprocess`.** The raw
detector path is retained bit-identically as an ablation behind `PADDLEOCR_VL_RAW_LAYOUT=1` — it is the
scored baseline this flip was measured against, so it must stay runnable.

The omitted post-processing accounts for **79% of the text-edit gap** and **84% of the reading-order
gap** to the published numbers. Both are now *near* parity but not *at* it. **Table TEDS barely moved
(+0.002) and is now the dominant remaining divergence (−9.14 pts) — a separate root cause the layout
fix does not touch** (diagnosis: the table-gap section, below). Attribution is unchanged and worth stating plainly: this
was a **layout-glue defect in our port, not a change to the ported VLM weights**, and it corroborates
the independent error budget (layout accounted for 53.4% of text edits; in-crop recognition
substitutions only 8.1%).

Failed/skipped pages are unchanged across both arms: the same **2** empty-layout pages emit no `.md`
and are skipped by the scorer (2 `!!!WARNING: No prediction` lines in both logs), so the A/B compares
the same 1649 pages. Reproduce: `PADDLEOCR_VL_RAW_LAYOUT=1` for the baseline arm, default for the new
one; scorer logs in `bench/omnidocbench/results/{full1651_nonest2,reflayout1651}.scorer.log`, and both
arms' scores in the matching `*_quick_match_metric_result.json` beside them.

### The table gap: an assembly/format defect in our port, not a recognition gap

After the layout flip, table TEDS (0.8362 vs published 92.76, **−9.14 pts**) was the dominant remaining
divergence — and the layout fix had not touched it (+0.002), so it had a separate root cause. Diagnosed
in the order the evidence allows, all at **zero GPU**:

**(a) Is the OTSL structure reaching the assembler, or is recognition flattening tables?** It is
reaching it. The VLM's raw output (`work_reflayout/*/results.json`) carries real OTSL on **754 table
regions across 473 pages**: 39,583 `<fcel>`, 7,487 `<nl>`, and — the load-bearing number — **34.1% of
tables carry a span marker** (1356 `<lcel>` + 2182 `<ucel>` + 67 `<xcel>`). Recognition is fine.

**(b) Does assembly emit what TEDS expects?** **No — this was the defect.** `otsl_to_markdown` rewrote
every `<ecel>/<lcel>/<ucel>/<xcel>` to a plain `<fcel>` and rendered a **GitHub pipe-table**, a format
that *cannot express a merged cell at all*. TEDS compares the cell tree, so **a third of our tables
reached the scorer with the wrong structure by construction.** It was never a hard format rejection —
the scorer's `utils/extract.py:259` does accept pipe-tables — it silently scored a structurally-wrong
table. **(c) transformers-on-the-same-crops was not needed**: (b) is sufficient and (a) exonerates
recognition.

**Classification: ASSEMBLY/FORMAT defect. Ours, cheap, fixed.** Not a port defect in the VLM, not model
difficulty. The fix ports the reference's own converter — PaddleX `pipelines/paddleocr_vl/uilts.py`
(`otsl_pad_to_sqr_v2` → `otsl_parse_texts` → `export_to_html`) — so spans become `rowspan`/`colspan`.
Pinned to the reference by `tests/otsl_html_parity.rs`: every table OTSL string the full run produced,
through both implementations — **739 distinct tables / 256 with spans / byte-identical to PaddleX.**
(A reference *quirk* is reproduced deliberately — an empty `<fcel>` swallows the next tag and renders
its literal string into the cell, firing on 5/739. It can only *add* junk, so reproducing it costs us a
little TEDS rather than winning any, and it keeps the parity assert exact.)

**A/B, official scorer, same 1649 pages, re-assembled from the EXISTING recognition output** (zero
GPU, like the keepvis/nonest A/Bs; **473 pages differ — exactly the table pages, nothing else moved**):

| metric (page_avg unless noted) | `reflayout` (pipe-table) | **`otslhtml` (NEW default)** | delta | published PaddleOCR-VL-1.5 |
|---|---|---|---|---|
| table **TEDS** ↑ | 0.8362 | **0.9036** | **+0.0674** | 92.76 (**−2.40 pts**, was −9.14) |
| table TEDS_structure_only ↑ | 0.8793 | **0.9433** | +0.0640 | 95.79 (−1.46 pts, was −7.86) |
| table Edit_dist ↓ | 0.5628 | **0.0801** | **−0.4827** | — |
| text_block Edit_dist ↓ | 0.0428 | **0.0368** | −0.0060 | 0.035 (**1.05×** gap, was 1.22×) |
| reading_order Edit_dist ↓ | 0.0500 | **0.0434** | −0.0066 | 0.042 (**1.03×** gap, was 1.19×) |
| display_formula Edit_dist ↓ (PROXY, **not** CDM) | 0.2495 | 0.2490 | −0.0005 | CDM 94.21 — **not comparable** |

TEDS moves **+0.0674**, an order of magnitude outside the pre-registered ±0.005 band, so the rule
registered *before* the number fires: **the HTML converter ships as the default** (unconditional — there
is no opt-in flag; the pipe-table path is deleted, not kept as an ablation, because it is simply wrong).

**Text and reading-order improved too, which is not a coincidence and is worth stating:** a pipe-table's
rows are *text lines*, so every mis-structured table was also leaking spurious text into the text_block
and reading_order comparisons. `research_report` — the most table-dense category — drops from 0.0675 to
**0.0144** on text-edit alone (see the per-category table below). One defect, three metrics.

**Verdict after the fix: text 1.05× and reading-order 1.03× of published — parity within noise; table
TEDS −2.40 pts — a real but much smaller residual.** That residual is a **1651-superset** number, like
everything in this section; on the 1355-page benchmark slice the same run scores TEDS **92.75 vs 92.76 —
parity** (headline). The superset's table-heavy `table_hard` pages are what the −2.40 is made of.

### Per-category preservation vs the published per-category numbers

Generated by `bench/omnidocbench/per_category_table.py text_block <log>...` (the scorer's
`*_metric_result.json` ships an **empty** `group` dict, so its printed log is the only source for the
per-attribute breakdown).

⚠️ **Read the task difference first — the last column is a different task.** The published
per-category figures (arXiv 2603.24326 Table 6, PaddleOCR-VL-L) are an **OCR-block** score: the model
is *given* the ground-truth layout blocks and only has to read the text. Ours are **end-to-end** (we
detect the layout too, and a layout miss costs us text-edit). They also aggregate differently — the
published per-category mean (0.053) is *above* their own end-to-end page_avg (0.035), so the two
columns do not even share a scale. **A lower number in our column is therefore NOT evidence the port
beats the paper**, and an equal number is not parity. Only the *shape* of the profile is comparable:
`rel. delta` normalises each column by its own mean, so a **positive** value marks a category where our
end-to-end pipeline is disproportionately worse than the model is at merely reading that category's
text — i.e. a genuine end-to-end gap — and a negative one marks no such gap.

| category | baseline `nonest2` | `reflayout` | **`otslhtml` (shipped)** | published (OCR-block) | rel. delta |
|---|---|---|---|---|---|
| newspaper | 0.0640 | 0.0515 | 0.0511 | 0.035 | **+0.69** |
| magazine | 0.0675 | 0.0422 | 0.0377 | 0.020 | **+0.62** |
| academic_literature | 0.0230 | 0.0199 | 0.0199 | 0.021 | +0.13 |
| colorful_textbook | 0.1188 | 0.0695 | 0.0632 | 0.082 | +0.12 |
| book | 0.0592 | 0.0384 | 0.0340 | 0.047 | +0.01 |
| note | 0.0730 | 0.0518 | 0.0518 | 0.077 | −0.08 |
| research_report | 0.1016 | 0.0675 | **0.0144** | 0.031 | −0.20 |
| PPT2PDF | 0.0727 | 0.0226 | 0.0226 | 0.049 | −0.33 |
| exam_paper | 0.0911 | 0.0498 | 0.0455 | 0.115 | **−0.97** |
| **ALL (page_avg)** | **0.0722** | **0.0428** | **0.0368** | 0.035 (end-to-end) | |

**Result: preservation holds on 7 of 9 categories.** Each fix improves every single category over the
one before it, and on seven of them our end-to-end profile is at or below the published *block-level*
difficulty profile — meaning whatever error remains there is the model's own (it is that good or that
weak on those pages), not something our port adds. `rel. delta` is recomputed per column, so it moves
even where the absolute number does not: it measures a category's *share* of the remaining error, and
as the total shrinks the survivors' shares grow.

**The table fix resolved `research_report`** — the most table-dense category, 0.0675 → **0.0144**
— which is exactly what a table-rendering defect predicts, and is the strongest independent confirmation
of that diagnosis: the pipe-table's rows were being read as text lines.

**Two categories remain true end-to-end gaps: `newspaper` and `magazine`.** These are the dense,
multi-column, figure-and-caption-heavy classes where the layout stage does the most work — and precisely
the work the published OCR-block task never has to do, because it is handed the blocks. The residual gap
is therefore **layout/reading-order-shaped, not recognition-shaped**, consistent with the error budget
(layout 53.4% of edits; in-crop substitutions only 8.1%). This is the target for any further layout
work; it is *not* evidence of a VLM port defect.

### Formula: scored with CDM, the published metric

The published formula number is **CDM** (Character Detection Matching), not edit distance. Our earlier
0.2490 was edit distance and was always labelled NOT-COMPARABLE; it is now retired as a proxy. CDM is
measured directly.

**Result (official scorer, CDM metric, `display_formula`):**

| | ours — `subset: v1.5` (1355 pg, like-for-like) | ours — all 1651 | published |
|---|---|---|---|
| formula **CDM** ↑ | **91.77** | 80.90 | **94.21** |

**−2.44 CDM against published — the one metric where the port does not reach parity.** The 1651-page
figure (80.90) is *not* the comparable one: 91 of the 313 formula-bearing pages in that JSON come from
`equation_hard` (CDM 57.21), a diagnostic subset the leaderboard excludes.

**How it was scored.** No scorer modification — CDM is a config switch the official harness already
ships (`METRIC_REGISTRY` carries `CDM`; `configs/end2end.yaml` merely defaults to `CDM_plain`).
Config `data/subsets/cdm1651.end2end.yaml`, `display_formula: [Edit_dist, CDM]`, same raw GT, same
`quick_match`, preds = the shipped default. Env fix is in `setup_cdm_env.sh`.

**Where the evidence is** (every number above is read off a committed file, not off this prose):
`results/cdm1651_quick_match_metric_result.json` — its `display_formula.page.CDM` block carries the
all-pages `ALL` (0.809011 → 80.90) *and* the per-attribute rows, including `subset: v1.5`
(0.91769 → **91.77**) and `equation_hard` (0.572077). One naming trap, since the scorer names its
output after the **prediction** dir and not the config: the CDM run reused the shipped `otslhtml1651`
predictions, so the scorer wrote `otslhtml1651_quick_match_metric_result.json` — the same name the
earlier full-metric run used, but containing **only** `display_formula`. Overwriting would have
destroyed the text/table/reading-order results. Both are therefore kept, under distinct names:
`otslhtml1651_*` (text/table/RO + formula-Edit) and `cdm1651_*` (formula Edit + CDM). Their shared
`Edit_dist` keys are identical, which is the internal control described below.

**The gate that makes this number trustworthy.** CDM renders LaTeX to a raster and recovers per-token
boxes by *exact pixel-colour* lookup, so a broken TeX env silently returns **CDM ≈ 0** through a bare
`except` — a fabricated, *pessimistic* self-own that every downstream `Overall` would inherit. So CDM
is gated behind `cdm_smoke.py`, which must pass before any score is believed:

```
identical formula  F1 = 1.0   (expect 1.0)          <- renderer+matcher work end-to-end
truncated formula  F1 = 0.6   (expect 0 < F1 < 1)   <- it is really comparing, not stubbing a constant
PASS: CDM renders and discriminates -> CDM scores are trustworthy.
```

The env fix was small in the end: OmniDocBench's `\mathcolor` needs **xcolor 3.x**, and the CTAN zip
ships *both* `xcolor.sty` and its rollback `xcolor-2022-06-12.sty`. Copying only the first into
`TEXMFHOME` triggers the 2021 kernel's rollback (`File 'xcolor-2022-06-12.sty' not found`) and makes
this look like it needs a 5 GB TeX Live install; with both files the **stock distro `xelatex`**
compiles with 0 undefined control sequences and emits exact `(255,0,0)`/`(0,255,0)` pixels.

**Internal control.** `Edit_dist` was scored in the same run and landed on **0.248989** — the
already-scored 0.2490 — proving the CDM run scored the pipeline we think it did, not a stale or
mismatched prediction set.

**The gap is real and is NOT a scoring artifact** (the gate rules that out).

#### Root cause of the −2.44: it is a **CJK-formula** gap, and it is the model's, not the port's

An earlier draft of this section attributed the gap to *"degraded/low-contrast inputs — `fuzzy_content`
0.307, `equation_hard` 0.572, `with_watermark` 0.564"*. **That was a like-for-like error and is
withdrawn.** Those are `ALL`-row (1651-page) attribute figures, but the −2.44 is a deficit *within*
`subset: v1.5`. `subset` is an **exclusive partition** of the 1651 pages — v1.5 (1355) / equation_hard
(100) / layout_hard (99) / table_hard (97), verified: zero pages carry two values — so
**`equation_hard` and `table_hard` contribute *nothing* to the −2.44 by construction.** They cannot be
its cause.

Re-derived from `results/otslhtml1651_quick_match_display_formula_per_sample_CDM.json` (n=1807
formulas) joined to the GT page attributes, using the scorer's own reduction (mean within page, then
across pages). The reconstruction reproduces the scorer **exactly** — `ALL` 0.8090 and `subset: v1.5`
0.9177, both to 4 dp — which is what licenses the breakdown below.

Within v1.5, `language` is a clean partition of the 205 formula-bearing pages (148 + 57 = 205):

| slice (within v1.5) | formula-bearing pages | CDM ↑ |
|---|---|---|
| `language: english` | 148 (72%) | **0.9349** |
| `language: simplified_chinese` | 57 (28%) | **0.8730** |
| **weighted → `subset: v1.5`** | **205** | **0.9177** ✓ *reproduces the scorer* |

**CJK formulas score 6.2 CDM points below English ones, and they are 28% of the benchmark's
formula-bearing pages — so they account for 1.72 of the 2.44 points (70%).** Had our CJK matched our
own English, v1.5 CDM would be 93.49, leaving a 0.72-pt residual (30%) spread across English pages.
No degraded-input slice within v1.5 is large enough to matter: the low ones are `data_source: note`
(0.654, **n=5**) and `special_issue: fuzzy_scan` (0.820, **n=4**); `watermark` is 0.9057 (n=8) —
*above* the v1.5 mean, i.e. the opposite of the withdrawn story.

**Classification: (iv) genuine model difficulty. A port defect (iii) is ruled out by cross-stack
evidence.** llama.cpp — an entirely independent implementation of the same weights — was run over the
**same crops** (cross-stack section), and it was scored **on CDM, the published metric itself**, not on
a proxy:

| stack | formula **CDM** ↑ — `subset: v1.5` (like-for-like) | CDM ↑ — all 1651 | vs published 94.21 |
|---|---|---|---|
| **published PaddleOCR-VL-1.5** | 94.21 | — | — |
| **ours (Rust port)** | **91.77** | 80.90 | **−2.44** |
| **llama.cpp** (independent) | **90.21** | 79.17 | **−4.00** |

**llama.cpp misses the published CDM by 4.00 points — a *wider* miss than our port's 2.44.** An
independent C++ stack, sharing nothing with this code but the checkpoint, therefore not only reproduces
the formula gap but reproduces it worse. A defect in *our* port cannot explain a deficit that appears —
larger — in an implementation that contains none of our code.

This was scored with `data/subsets/cdm_llamacpp1649.end2end.yaml`: the `cdm1651` config with only the
predictions path swapped to the llama.cpp preds, so the two CDM columns differ in the recognition
backend and in **nothing else** — same GT, same `quick_match`, same CDM renderer, same `cdm_smoke.py`
gate (F1 1.0 / 0.6, PASS) run beforehand. Evidence:
`results/cdm_llamacpp1649_quick_match_metric_result.json`. **Internal control:** that run also re-scored
`Edit_dist`, which came back **bit-identical** to the already-committed llama.cpp edit-distance result
(`ALL` 0.259969, `subset: v1.5` 0.192738) — proving the CDM number was computed over the same formula
set as the earlier run, not a drifted or partially-matched one.

The same conclusion holds on the edit-distance proxy, with the language split CDM's `subset` rows do
not expose — and it is the CJK slice that carries the penalty in both stacks:

| stack | display_formula Edit ↓ (english) | (simplified_chinese) | CJK penalty |
|---|---|---|---|
| **ours (Rust port)** | 0.2378 | 0.2765 | **+0.0387** |
| **llama.cpp** (independent) | 0.2507 | 0.2819 | **+0.0311** |

Both stacks degrade on Chinese formulas by a comparable margin, and llama.cpp is *slightly worse* on
formulas overall (`ALL` 0.2600 vs our 0.2490). The weakness is in the model's formula head on CJK
content.

**Honest limit on this claim:** the published 94.21 is an **all-pages** figure with no language split,
so we cannot check whether the *reference implementation* also drops on CJK formulas. What the evidence
establishes is narrower, and is all we assert: **the CJK formula weakness is not introduced by this
port.** Whether the published pipeline somehow avoids it is not something the published number can
answer.

#### One format artifact was found, measured, and is **upstream — not ours to fix**

The per-formula CDM distribution is bimodal — 57.7% score a perfect 1.0, but **67 (3.7%) are exact
zeros**. Chasing the zeros surfaced a real defect, though not one we can act on. When more than one
predicted formula matches a single GT formula, the scorer merges them
(`utils/match_quick.py:568-570`):

```python
mutli_formula = ' \\\\ '.join(['{'+ori_pred_lines[_].strip('$$').strip('\n')+'}' ...])
mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \end{array}'
```

`.strip('$$')` strips **`$`** characters only — it does not strip `\[`/`\]`. The merged string is
therefore `\begin{array}{l} {\[x\]} \\ {\[y\]} \end{array}`, and **`\[` nested inside an array is
invalid LaTeX**. This fires on **224 of 1807** predictions (12.4%), and those 224 average CDM **0.6479**
against 0.8101 overall — which looks exactly like a smoking gun.

**It is not one, on either of two independent counts.**

**(1) Our delimiter is irrelevant — switching it is a provable no-op.** The reference pipeline rewrites
`\[`→`$$` (PaddleX `pipelines/paddleocr_vl/pipeline.py:589-590`) while we emit the model's raw `\[...\]`
verbatim (`src/assemble.rs:75`), so the bug *looks* like ours. But `utils/extract.py:220` normalizes
`$$x$$` **back to** `\[x\]`, and `utils/match.py:57` feeds that *extracted* content — not the raw
markdown — into the merge. Checked against the real scorer *before* any code was written; both forms
arrive identically:

```
OURS  (\[..\]) extracted -> ['\[E = mc^2\]', '\[F = ma\]']
REF   ($$..$$) extracted -> ['\[ E = mc^2 \]', '\[ F = ma \]']
both merge to  ->  \begin{array}{l} {\[E = mc^2\]} \\ {\[F = ma\]} \end{array}    # invalid, BOTH
```

The artifact hits the **reference model identically** and so cannot be part of a gap *against* it. (It
also corrupts **26 GT** strings — everyone's problem.)

**(2) The invalid nesting costs almost nothing anyway — measured, not assumed.** We re-ran CDM on all
**217** affected predictions (those with a nested pred and a clean GT), holding gt and pred content
fixed and changing *only* the nesting:

| | mean CDM | exact zeros |
|---|---|---|
| as-is (`\[` nested in the array, what the scorer builds) | 0.6587 | 14 |
| un-nested (valid array, identical content) | **0.6608** | 12 |
| **delta** | **+0.0022** | −2 |

29 improved, 20 got *worse*, 168 unchanged. **xelatex recovers from the bad nesting**, so the construct
is cosmetically invalid but numerically harmless — worth **+0.03 CDM** across the whole set if it were
fixed. **The 0.6479 is therefore NOT caused by the nesting.** Those 224 score low because they are the
**over-segmented** cases — our pipeline emitted more than one formula region where the GT annotates one
— and over-segmented predictions are simply harder to match. That is a layout-stage property, and it is
shared by the llama.cpp run (same crops, same layout), which lands slightly *worse* on formulas overall.

**Recorded as an upstream (i) metric artifact. Not chased, and no code changed on the strength of it** —
fixing it would mean patching the official scorer to raise our own number, and it would buy +0.03.

**Residual, stated plainly:** 12 of the 67 zeros are formulas our pipeline emitted **nothing** for (a
layout miss — the GT formula region was never detected). That is ours, but it is 0.66% of formulas and
is not what the −2.44 is made of. Logged to FUTURE_WORK.

### Cross-stack: the same crops through llama.cpp

**Question this answers.** Every number above says the port matches the *published* figures. It cannot,
on its own, say the port matches *the model* — a shared-cause error (a subtly wrong crop, prompt, or
detok convention) would move our score and the reference's together. So: swap ONLY the recognition
backend, hold everything else fixed, and see whether an independent implementation of the same weights
lands in the same place.

**What is held fixed — by construction, not by convention.** `llamacpp_recognize.py` reads the crop
PNGs and manifests **the scored REF_LAYOUT run already wrote** (`work_reflayout/<stem>/`) and never
re-runs the detector; per-class prompts come from the manifest verbatim; the output contract is the
Rust binary's own `results.json`, fed to **the same `paddleocr-layout assemble`**. Layout, crops,
prompts, assembly, GT, scorer, config and page set are therefore *identical objects*, and the
recognition backend is the only free variable.

| | Rust port (mistral.rs/candle) | llama.cpp | delta |
|---|---|---|---|
| weights | `PaddleOCR-VL-1.5` safetensors, **bf16** | same model, **bf16 GGUF** (`general.file_type=32`) | — |
| decoding | greedy | greedy (`--temp 0 --top-k 1`) | — |
| text Edit ↓ | 0.0328 | **0.0325** | −0.0003 |
| reading-order Edit ↓ | 0.0415 | **0.0414** | −0.0002 |
| table TEDS ↑ | **92.75** | 92.45 | −0.30 |
| table TEDS-S ↑ | **95.95** | 95.59 | −0.36 |
| display-formula Edit ↓ | **0.1833** | 0.1927 | +0.0094 |

`subset: v1.5` slice (the 1355 benchmark pages), official scorer, pin `59b103c`, `quick_match`,
config `data/subsets/llamacpp1649cmp.end2end.yaml`. **bf16 vs bf16 — no quantization caveat:** the
GGUF's `general.file_type` is 32 = `LLAMA_FTYPE_MOSTLY_BF16`, for both the LM and the mmproj (read
out of the GGUF headers directly, not inferred from the filename).

**Page-set parity.** The Rust run has predictions for 1649 of 1651 pages; llama.cpp has all 1651. The
scorer *skips* a page with no prediction, so scoring each stack on its own set would silently score
them on **different pages**. Both are therefore scored on the **1649-page intersection**
(`preds/llamacpp1649cmp/`, symlinks, set-identical to the Rust set — asserted, not assumed).

**Verdict: MUTUALLY CONFIRMED.** Two independent implementations of these weights — different
framework, different kernels, different image preprocessing, different tokenizer plumbing — agree to
**0.0003 edit distance on text and 0.0002 on reading order**. That is far tighter than the distance to
the published figure, and it is the strongest available evidence that the port reproduces the *model*
rather than merely reproducing a number. The residual table (−0.30 TEDS) and formula (+0.0094 edit)
deltas favour the Rust port, are small, and are consistent with ordinary numeric/preprocessing
divergence between two bf16 stacks; they are **not** attributed further without evidence.

**Robustness, side by side.** Same crops, same guards:

| | Rust port | llama.cpp |
|---|---|---|
| per-region timeouts (120s → empty text) | **8 crops** | **0** |
| pages lost to the whole-page guard | **2** (one SIGKILLed at the 600s cap on a 74-crop page; one `rc=1`) | **0** |
| pages recovered despite a guard trip | 1 (`jiaocai_needrop_en_2496`: 600s cap tripped during teardown, `results.json` already complete → assembled) | — |
| runaway generation | none (the `TheEconomist p062` hang that motivated the cap now completes in **45s**, 6339 bytes, under `MAX_NEW_TOKENS=2048`) | none |

The 8 Rust region-timeouts each record **empty text** for that region — a small, real accuracy cost the
llama.cpp run does not pay, and one that slightly *understates* the port. It is left in rather than
patched out.

**One llama.cpp region-timeout did occur and was thrown away, not scored.** During the run the box went
unresponsive (llama-server's 8 GB default prompt cache, `-cram`, on a 15.9 GB box); one table crop
tripped the 120s guard and produced a 106-byte page. That page was
**quarantined and regenerated** on the healthy box: 5 crops, 1845 bytes, 2.3s. Scoring the corrupted
version would have booked an infrastructure stall as a *llama.cpp accuracy miss that never happened* —
the guard's job is to make that distinction visible, and the fix is to re-run the page, not to keep it.

### Speed, honestly — llama.cpp is 2.7x faster per page, and the port's edge is not throughput

**Secondary result, and it does not flatter the port.** Same box, same 118-page stratified sample
(`speed120.stems`), same crops, bf16 both sides, K=1 serial recognition by design.

**Current number: 2.7x** (load-once recognition, below). The 3.2x in the tables that follow was
measured with the old harness, which reloaded the checkpoint once per page; that reload has since been
deleted, which is worth **1.6s/page** and nothing else. The earlier material is kept because the two
mistakes it caught — a memory-leak "speedup" and an uncharged layout stage — are the reason any of
these numbers can be trusted.

**The first number here was wrong, and the reason it was wrong matters.** The Rust full-run timings
(median 17s/page) were taken on the box that was *later* found to be thrashing — rust-analyzer holding
6.4GB and swap at 100%. llama.cpp's were taken after that cleanup. Publishing
17s vs 2.2s would have been a **7.7x speedup manufactured out of a memory leak**. So the Rust pipeline
was re-run on the cleaned box, with llama-server stopped, over the same pages:

| crops/page | n | Rust end-to-end | llama.cpp + layout | speedup |
|---|---|---|---|---|
| 1–5 | 11 | 4.0s | 1.3s | 3.1x |
| 6–10 | 22 | 6.5s | 2.1s | 3.1x |
| 11–15 | 22 | 9.5s | 2.7s | 3.5x |
| 16–25 | 22 | 14.0s | 3.3s | 4.2x |
| 26–40 | 12 | 19.5s | 5.1s | 3.8x |
| 41–200 | 14 | 35.5s | 12.8s | 2.8x |
| **all** | **103** | **10.0s** | **3.1s** | **3.2x** |

Per-page medians over the 103 pages both stacks timed on the clean box. Rust's honest median is
**10.0s**, not the 17s the degraded run reported.

**Per-stage split (measured, `stage_split.py`), so the 3.2x is attributed and not just quoted:**

| stage | cost | who pays it |
|---|---|---|
| ONNX layout (PP-DocLayoutV3) | **0.88s**/page | both — llama.cpp *reuses the Rust run's crops*, so it never re-runs the detector; **0.88s is added back to every llama.cpp page above**, or the comparison would be a lie |
| process spawn + bf16 model load | **1.76s**/page | **Rust only** — the old `run_pipeline.sh` invoked the recognize binary once per *page*. A harness artifact, not a port property — **now deleted, see below** |
| recognition | **0.52s/crop** (Rust) vs **0.12s/crop** (llama.cpp) | the real gap |

**The gap is kernels, not harness.** Removing the per-page model reload entirely (the load-once mode)
should take a median page from 10.0s to ~8.2s — still ~2.6x slower than llama.cpp. The **0.52 vs 0.12
s/crop** is where the time actually goes, and it is consistent with the "Honest residual" section
above: candle's dense GEMM/MLP on the vision encoder is the ceiling, and that is an upstream maturity
gap, not a bug in this repo.

### Load-once recognition: the reload is gone, and it bought exactly what was predicted

The claim above was a prediction (10.0s → ~8.2s, 3.2x → ~2.6x). It has now been implemented and
measured, and the prediction is what shipped: **8.42s median, 2.7x**.

`paddleocr_vl_recognize` now takes many page dirs (`--list <file>`) and loads the ~1.9GB checkpoint
**once per run** instead of once per page; `run_pipeline.sh` drives layout (ONNX, per page) → one
recognition process over every pending page → assembly (per page). Same crops, same prompts, same
greedy sampler, same engine.

| | old (per-page reload) | **load-once** |
|---|---|---|
| checkpoint load | 1.76s **per page** | **2.02s once for the whole 118-page run** (0.02s/page amortized) |
| ONNX layout | 0.88s/page | 0.87s/page |
| recognition | 0.52s/crop | **0.50s/crop** (unchanged — as it must be) |
| assembly | — | 0.001s/page |
| **median page, end-to-end** | **10.0s** | **8.42s** |
| **vs llama.cpp + layout (3.1s)** | **3.2x** | **2.7x** |

| crops/page | n | Rust load-once | llama.cpp + layout | speedup |
|---|---|---|---|---|
| 1–5 | 11 | 1.7s | 1.3s | 1.3x |
| 6–10 | 22 | 4.6s | 2.1s | 2.2x |
| 11–15 | 22 | 7.5s | 2.7s | 2.8x |
| 16–25 | 22 | 10.7s | 3.3s | 3.2x |
| 26–40 | 12 | 17.6s | 5.1s | 3.4x |
| 41–200 | 14 | 32.6s | 12.8s | 2.6x |
| **all** | **103** | **8.4s** | **3.1s** | **2.7x** |

**Attribution — what this did and did not fix.** It removed a harness artifact worth **1.6s/page**
(measured 10.0 → 8.42s; the reload's own cost was 1.76s/page) and **nothing else**. Recognition still
costs **0.50s/crop** against llama.cpp's 0.12s: the per-crop kernel time did not move, because nothing
about the kernels changed. The residual **2.7x is therefore kernel-time, not harness-time** — no
further harness change can touch it — and it is best explained by the candle vision-GEMM/MLP ceiling
described under "Honest residual". That attribution is an *inference* from where the time goes, not a
kernel-level measurement; the micro-benchmark that would confirm it directly is specified in
FUTURE_WORK.md, and the limits on how far this 2.7x generalizes are spelled out below.

The predicted residual was 2.6x and the measured one is 2.7x. The 0.1x is the reload being slightly
cheaper to delete than it was to pay (1.6s recovered vs 1.76s charged) — page-level medians over
different bucket mixes, not a new effect. The gap shrinks most on *small* pages (1–5 crops: 3.1x →
1.3x), which is exactly the signature of removing a **fixed** per-page cost: it was the whole page on
a 1-crop page and a rounding error on a 50-crop one.

**Correctness gate, not a claim.** A speed mode that changes output is a bug, so load-once is gated on
byte-identical results: `loadonce_parity.sh` runs both arms with the same binary over 24 pages / 189
crops covering all 22 layout classes and diffs `results.json`. **24/24 byte-identical.** The gate runs
both arms *now* rather than diffing against stored outputs, so a rebuild cannot be mistaken for a mode
change.

**Runaway guard survives.** The per-region tokio timeout (empty text, continue) is unchanged; the
outer per-page `timeout` process kill — which load-once has no per-page process for — became an
OS-thread watchdog that kills a wedged engine at 2x the region budget and marks the page `TIMEOUT_SKIP`
so the resumable runner steps over it. Verified live: a crop forced to time out records empty text, its
page's `results.json` stays complete, and the run continues to the next page. One hung crop costs one
page, not the run.

**Methodology (clean box, same as the run above).** 118-page stratified sample, verified before timing: 4.1GB
of 15GB used, **no swap**, no rust-analyzer, llama-server stopped, GPU idle. llama.cpp's timings are
unchanged from the run above (server mode, bf16 GGUF, same box) and still carry the +0.88s layout add-back,
since it reuses the Rust run's crops and never runs the detector. Reproduce:
`speed_loadonce.py` → `speed_stats.py --rust-csv logs/speed_loadonce.csv`.

**What did not change: the conclusion.** llama.cpp is still faster per page on this box, and this doc
still says so. Deleting the reload made the port's *usability* honest (one load per run, not 1651 of
them — ~48 min of pure reload over the full set); it did not make the port fast, and it was never going
to.

**What this is not.** Not a SOTA-speed claim, and not a claim the port is fast. On this box, for this
workload, **llama.cpp wins on throughput and we report that plainly**. The port's actual edge is what
it was always claimed to be: a **Python-free single static binary** (no torch, no venv, no CUDA-python
stack) that reproduces the model's accuracy — which the cross-stack run shows it does, to 0.0003 edit
distance.

### What the 2.7x does and does not generalize to (read before quoting it)

The number is real and it is ours to own, but it is easy to over-read in either direction. Three
limits, stated so nobody has to infer them:

1. **The magnitude is workload-specific.** 2.7x is *this* workload: OCR crops, so a compute-bound
   **vision prefill** followed by a *short* decode (a page's regions transcribe to tens of tokens, not
   thousands). That mix maximally exposes the one thing candle is weak at here (dense vision GEMM/MLP)
   and barely exercises decode, where the port is competitive (it *beats* torch 1.39x on GPU decode,
   see the latency section). A long-generation or decode-dominated workload would weight the same two
   engines completely differently, and 2.7x should not be quoted for it. It is a per-page ratio on a
   1355-page document benchmark, not a property of candle.
2. **The direction is more general than the magnitude.** In any *compute-bound* regime dominated by
   dense GEMM, a mature hand-tuned kernel stack beats a younger generic one, and ggml's kernels are
   the more mature of the two here. So "llama.cpp is ahead on this axis" is the safe reading;
   "llama.cpp is 2.7x faster" is the unsafe one — that constant belongs to this workload and this GPU.
   **Single GPU (RTX 4070 Ti Super), single box** — no claim is made about other hardware, and a
   kernel gap is exactly the kind of thing that moves with the architecture it is tuned for.
3. **ggml's speed is not free, but the usual discount does not apply here.** The standard caveat
   against a ggml comparison is that its wins often come with trade-offs — quantized/reduced-precision
   paths, and hardware-specific hand-tuned kernels that buy speed with portability. The first of those
   is **ruled out by construction in this comparison: both sides are bf16** (the GGUF's
   `general.file_type` = 32 = `MOSTLY_BF16`, read from the header, for the LM *and* the mmproj),
   and the two stacks agree to 0.0003 edit distance, so llama.cpp is not buying speed with accuracy on
   this run. The second trade-off is real and stands: some of ggml's advantage is hand-written
   per-architecture kernel work that candle has not (yet) done. That is a **maturity** gap, not a
   design flaw in candle, and it is upstream of this repo.

**What would actually settle it.** All of the above still infers a kernel-level cause from a
whole-pipeline measurement. The clean test is to take the workload out of the picture entirely and
time `candle.matmul` against ggml `MUL_MAT` in isolation, bf16 both sides, on the vision tower's own
GEMM shapes. That micro-benchmark is specified in FUTURE_WORK.md; until it is run, "candle's vision
GEMM is the ceiling" remains the **best-supported hypothesis** (the 0.50 vs 0.12 s/crop split, and the
fact that attention is already fused, both point at it) rather than a directly measured fact, and this
doc marks it as such.

The transformers reference was **not** run over the full page set (only the single-crop latency
microbenchmarks earlier in this doc). Stated as a gap rather than estimated.

## Caveats

- Different stacks (candle/mistral.rs vs PyTorch/transformers): kernels, memory layout, no quant.
- **The published pipeline is not only the VLM.** The paper's number is end-to-end PaddleOCR-VL-1.5 with
  *its own* layout stage, prompts and post-processing; our port swaps in our ONNX PP-DocLayoutV3 port,
  our crop glue and our markdown assembler. A divergence therefore does **not** localize to the ported
  VLM weights by itself — attribution is exactly the job of the sections that follow.
- Formula CDM **is** measured now (91.77 on the benchmark slice; see the CDM section) and is what the
  `Overall` uses. The older `display_formula` **edit-distance** figures kept in this doc are a *proxy*
  and are never comparable to the paper's CDM — they are retained only as an internal control and for
  the cross-stack A/B, where both sides are scored with the same metric.
- The TTFT/decode split has a minor methodology asymmetry (the port reports an exact
  prefill/decode split from its own `Usage`; the reference times a separate prefill-only forward and
  a separate `generate`). Total latency, the headline, is directly wall-clock comparable.
- Decode tok/s is computed identically for both engines (`(tokens-1)/(total-ttft)`), not from each
  engine's self-report.
- Short-output decode (6 tokens) has wide error bars; p90 over 20 iters bounds it.
