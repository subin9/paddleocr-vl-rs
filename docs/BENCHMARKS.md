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
- **CDM decision — RESOLVED (Iter-12): scored with `Edit_dist`, not CDM.** Probed the scorer-venv:
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

**Independently re-verified 2026-07-11 (§2.4)** — not trusted from this doc or from memory. The rows
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

## Plumbing validation (§2.2, n=5) — integration proven, NOT an accuracy verdict

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
  Chart crops (scatter plots) transcribe as long `col | val` numeric dumps. Measured effect, in-session
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
  Recognition still *runs* on these crops today; skipping that too is a later speed win (§2.4/2.5).
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
0/17 regions differ, manifest identical) and the scorer is deterministic (2× identical). Iter-5's
committed no-skip numbers reproduce byte-for-byte in-session. Numbers are sane and non-degenerate.
This slice proves the integration AND the visual-skip fix; it is NOT the accuracy verdict (n=5).

## Our measured scores

| run | Overall | Text-Edit | Formula(metric) | Table-TEDS / -S | ReadOrder-Edit | verdict |
|-----|---------|-----------|-------------|------------|----------------|---------|
| 5-page slice (visual-skip) | n/a (n=5) | 0.077 pg-avg | edit 0.110 (no CDM env) | 0.997 / — (n=2) | 0.000 | plumbing + skip OK |
| **stratified subset (n=150)** | **84.1 (Edit-proxy)** | **0.0709** pg-avg | **edit 0.2724** (NOT CDM) | **0.8659 / 0.9112** | **0.0919** | **SANE — proceed to full** |
| **full v1.5 (1651)** | **≤ 91.80** (see below) | **0.0797** pg-avg | edit 0.2559 (**NOT CDM**) | **0.8336 / 0.8761** | **0.0929** | **DIVERGES** |
| **paper reference (full 1651)** | 94.50 | 0.035 | CDM 94.21 | 92.76 / 95.79 | 0.042 | target |

Speed table (secondary; Rust GPU-bf16 vs transformers floor, per-stage) also PENDING — see the
existing latency sections above for the single-crop microbenchmarks already measured.

### Stratified subset (n=150) — result + verdict (§2.3)

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
Not "diverges badly" per §2.3: metrics are non-degenerate, English lands on the paper, and the
elevated aggregate is explained by intentional hard-case stratification. The full-set overall vs
94.50 (with the same Edit-proxy caveat, or full CDM if a LaTeX env is stood up) is the accuracy verdict.

### Full set (n=1651) vs published — side-by-side and VERDICT (§2.4)

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
actually reports**, and §2.5 must attribute it (layout/reading-order vs in-crop recognition vs
assembly), not explain it away. Known assembly-side contributor already found in §2.2: the assembler
**emits no text for `image`-class regions** (page `yanbaopptmerge_yanbaoPPT_5885` recognized "流水声" and
then dropped it), which costs text-edit on every page whose content the layout model labels `image`.

**Worst slices on the full set** (text-edit ↓, page-avg) — §2.5 targets:

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
is layout/assembly-side, not in-crop recognition** — but that is a hypothesis for §2.5 to test, not a
finding. It is recorded here as such.

## Caveats

- Different stacks (candle/mistral.rs vs PyTorch/transformers): kernels, memory layout, no quant.
- **The published pipeline is not only the VLM.** The paper's number is end-to-end PaddleOCR-VL-1.5 with
  *its own* layout stage, prompts and post-processing; our port swaps in our ONNX PP-DocLayoutV3 port,
  our crop glue and our markdown assembler. A divergence therefore does **not** localize to the ported
  VLM weights by itself — attribution is exactly the job of §2.5.
- Formula CDM is **not measured**. `display_formula` edit-distance is reported as a proxy and is not
  comparable to the paper's CDM. Any Overall we state is an upper bound, never a point estimate.
- The TTFT/decode split has a minor methodology asymmetry (the port reports an exact
  prefill/decode split from its own `Usage`; the reference times a separate prefill-only forward and
  a separate `generate`). Total latency, the headline, is directly wall-clock comparable.
- Decode tok/s is computed identically for both engines (`(tokens-1)/(total-ttft)`), not from each
  engine's self-report.
- Short-output decode (6 tokens) has wide error bars; p90 over 20 iters bounds it.
