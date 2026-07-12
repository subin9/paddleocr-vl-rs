# paddleocr-vl-rs

A Python-free Rust pipeline that turns a PDF or image into structured markdown using
[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5).

Layout detection (PP-DocLayoutV3 via ONNX) finds the regions of a page; each region is cropped and
recognized by the PaddleOCR-VL vision-language model (running on [mistral.rs](https://github.com/EricLBuehler/mistral.rs));
the results are reassembled in reading order into markdown, with tables rendered from the model's
OTSL output. No Python or Paddle runtime at inference time.

## Architecture

```
PDF/image
   |  render pages (pdftoppm)
   v
PP-DocLayoutV3 (ONNX Runtime)         <- this repo: src/lib.rs
   |  Vec<Region> in reading order (25 layout classes)
   v
region crops + manifest.json          <- this repo: src/main.rs, src/assemble.rs
   |  one crop PNG + resolved task prompt per region
   v
PaddleOCR-VL VLM (via mistral.rs)     <- examples/recognize.rs (links mistral.rs)
   |  SigLIP/NaViT vision encoder -> Adaptive-MLP connector -> ERNIE-4.5-0.3B LM
   |  results.json = [{read_order, class, text}]
   v
reading-order assembly -> markdown    <- this repo: src/assemble.rs (incl. OTSL -> table)
```

The layout + assembly half (this crate) builds standalone with no GPU or engine dependency. The
recognition VLM is a separate mistral.rs build; the two stages talk only through `manifest.json` and
`results.json`.

## Upstream contributions

The recognition model and a general engine fix were contributed **upstream to mistral.rs**; this
repo is the document pipeline built on top of them.

- **[mistral.rs #2320](https://github.com/EricLBuehler/mistral.rs/pull/2320)** -- `feat(models): Support PaddleOCR-VL`.
  The recognition VLM itself (SigLIP/NaViT vision encoder -> `mlp_AR` connector -> ERNIE-4.5-0.3B),
  loaded via `--arch paddleocr_vl`. Closes [#2128](https://github.com/EricLBuehler/mistral.rs/issues/2128).
- **[mistral.rs #2319](https://github.com/EricLBuehler/mistral.rs/pull/2319)** -- `fix(llg): honor tokenizer special flag in toktrie detok`.
  PaddleOCR-VL emits tables as OTSL tokens `<fcel>` / `<nl>` (flagged `special=false`); the completion
  detok was dropping them and collapsing tables to run-on text. A general fix, not model-specific.
- **[llguidance #361](https://github.com/guidance-ai/llguidance/issues/361)** -- the root cause: a
  `<...>`-name heuristic in `toktrie_hf_tokenizers` overrides an explicit `special=false`.

## Status / correctness

**OmniDocBench v1.5, full run** — the primary result. On the benchmark's 1355 `v1.5` pages, the port
reaches published parity on every metric except formula:

| metric | this port | published PaddleOCR-VL-1.5 | |
|---|---|---|---|
| text Edit ↓ | **0.0328** | 0.035 | parity |
| table TEDS ↑ | **92.75** | 92.76 | parity |
| reading-order Edit ↓ | **0.0415** | 0.042 | parity |
| formula CDM ↑ | **91.77** | 94.21 | **−2.44 — the one gap** |
| **Overall** ↑ | **93.75** | 94.50 | −0.75, entirely from formula |

The shipped `OmniDocBench.json` is a 1651-page *superset* (it bundles 296 adversarial `*_hard` pages
that are not on the leaderboard); scoring the superset instead gives the pessimistic text 0.0368 /
TEDS 90.36 / RO 0.0434. Both columns, the evidence for the 1355-page reading, and the caveat that it
is an inference (the leaderboard does not publish its page list) are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

- Token-for-token greedy parity vs the transformers-5.13 reference across a **9-item corpus** (plain
  text, tables, formulas, spotting, seal, chart, CJK, low-quality scan, 2-column), on **both
  CPU-f32 and GPU-bf16**. 9/9 match golden token ids.
- Layout stage: Rust preprocess+run+decode matches an onnxruntime reference within ~0.05 px on the
  sample page (resampler drift only). See `tests/parity_layout.rs`.
- Load-once recognition is gated on byte-identical output vs the per-page path (24 pages / 189 crops /
  all 22 layout classes, **24/24 identical**) — a speed mode that changes a token is a bug.

## Quick start

Prerequisites (the pipeline itself is Python-free at inference; you still need the artifacts):

1. **ONNX Runtime** shared library. Any recent 1.2x build works. Point the loader at it:
   `export ORT_DYLIB_PATH=/path/to/libonnxruntime.so` (e.g. from a `pip install onnxruntime`).
2. **PP-DocLayoutV3 ONNX graph** (the layout model). Export its path:
   `export PADDLEOCR_LAYOUT_MODEL=/path/to/PP-DocLayoutV3.onnx`.
3. **PaddleOCR-VL-1.5 checkpoint** (the recognition weights) from Hugging Face.
4. **poppler-utils** for `pdftoppm` if you start from PDFs.

Build the layout binary (standalone, no GPU/engine deps):

```bash
cargo build --release          # produces target/release/paddleocr-layout
```

Run the layout stage on one page (writes crops + `manifest.json`):

```bash
./target/release/paddleocr-layout page.png out/
```

Build the recognition step against mistral.rs. The recognition stage needs **both** upstream PRs:
#2320 (the PaddleOCR-VL model) and #2319 (the OTSL detok fix — without it the table tokens
`<fcel>`/`<nl>` are dropped and tables collapse to run-on text). Both are still open, and the fork
carries **one each**, so you need both applied:

```bash
git remote add subin9 https://github.com/subin9/mistral.rs.git && git fetch subin9
git checkout subin9/paddleocr-vl-upstream        # PR #2320 — the model
git merge subin9/fix-toktrie-special-flag        # PR #2319 — the detok fix
```

They touch disjoint code and combine without conflict. Then drop `examples/recognize.rs` from this
repo into a small binary crate that depends on `mistralrs` (or into `mistralrs/examples/`) — it is
what provides the `PADDLEOCR_VL_GPU` toggle and the load-once `--list` mode used below:

```bash
# CPU/f32 (deterministic parity path):
PADDLEOCR_VL_WEIGHTS=/path/to/PaddleOCR-VL-1.5 recognize out/
# GPU/bf16 (needs a --features cuda,flash-attn mistral.rs build):
PADDLEOCR_VL_GPU=1 PADDLEOCR_VL_WEIGHTS=/path/to/PaddleOCR-VL-1.5 recognize out/
# Many pages: load the ~1.9GB checkpoint once, not once per page.
PADDLEOCR_VL_GPU=1 PADDLEOCR_VL_WEIGHTS=... recognize --list pages.txt   # one page dir per line
```

Reassemble the reading-order markdown:

```bash
./target/release/paddleocr-layout assemble out/results.json
```

Or run the whole PDF -> markdown flow end to end:

```bash
examples/pdf_to_markdown.sh input.pdf out/     # see the script header for the env vars it needs
```

## Performance

Full cross-stack numbers (fair baseline, both axes, kernel-optimization work, batching, honest
residual) are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Honest summary:

- The transformers reference is a **correctness floor here, not a competitor**; the port is
  token-faithful to it.
- **GPU-bf16: ~parity** end-to-end on short OCR output (a ~1.7x prefill loss offset by a ~1.4x
  decode win), VRAM neutral. **CPU-f32: slower** end-to-end on short output (prefill-bound), leaner
  on memory.
- Fused vision + LM attention (Sdpa/flash) closes most of the GPU prefill gap; the residual is
  candle's dense vision GEMM/MLP vs torch's oneDNN/cuBLAS -- a candle-maturity ceiling, not claimed
  closed.
- **llama.cpp is faster per page, and we report that plainly: 2.7x.** Same box, same 118-page
  sample, same crops, **bf16 on both sides**, layout cost charged to both. Median page **8.42s**
  (this port, load-once) vs **3.1s** (llama.cpp + layout). Deleting the per-page checkpoint reload
  took us 10.0s -> 8.42s and 3.2x -> 2.7x; the rest is **0.50s/crop vs 0.12s/crop** of recognition,
  i.e. the candle vision-GEMM ceiling above. Magnitude is workload-specific (compute-bound vision
  prefill, short OCR outputs) — the port's edge is a Python-free single binary that reproduces the
  model's accuracy, not throughput.
- Region **batching buys nothing** here (vision runs per-image regardless); recommended batch size
  is 1. Leakage-free, but no throughput win. Details and data in the benchmarks doc.

An earlier "1.44x / 1.88x faster" claim was withdrawn -- it measured an unfair, uncached baseline.
An earlier 3.2x-vs-llama.cpp figure was superseded by 2.7x after the reload was deleted, and a 17s/page
Rust median was retracted as a thrashing-box artifact.

## Roadmap

See [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md): the formula CDM gap (the one metric off parity),
root-causing the runaway generation, the candle vision-GEMM lever (and a micro-benchmark to size it),
assembler class-mapping expansion, and `cu_seqlens` packed-vision batching.

## License and attribution

Apache-2.0 (see [LICENSE](LICENSE)).

- [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) and PP-DocLayoutV3 are Apache-2.0
  (PaddlePaddle). This pipeline follows their model architecture and preprocessing recipes.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) is MIT (Eric Buehler). The recognition
  stage runs on it.
