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

## Planned: OmniDocBench accuracy-preservation run (TODO -- no numbers yet)

The 9-item corpus proves parity but is not a standard benchmark. The planned evaluation:

- Full **OmniDocBench v1.5**, official scoring script, on the same box.
- Compare mistral.rs (this port) vs HF/transformers vs vLLM serving the same PaddleOCR-VL checkpoint.
- **Framing (honest, to be preserved when numbers land):** the primary result is
  **accuracy-preservation** -- the port should match the reference's document-parsing scores, since
  it is token-for-token faithful. The port's edge is **deployment** (a single self-contained Rust
  binary, no Python/Paddle runtime), NOT serving throughput. Any speed comparison is same-hardware
  and explicitly not a SOTA-speed claim.

This section is a plan. It intentionally contains no fabricated numbers; it will be filled in when
the run is done.

## Caveats

- Different stacks (candle/mistral.rs vs PyTorch/transformers): kernels, memory layout, no quant.
- The TTFT/decode split has a minor methodology asymmetry (the port reports an exact
  prefill/decode split from its own `Usage`; the reference times a separate prefill-only forward and
  a separate `generate`). Total latency, the headline, is directly wall-clock comparable.
- Decode tok/s is computed identically for both engines (`(tokens-1)/(total-ttft)`), not from each
  engine's self-report.
- Short-output decode (6 tokens) has wide error bars; p90 over 20 iters bounds it.
