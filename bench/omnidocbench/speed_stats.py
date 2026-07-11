#!/usr/bin/env python3
"""§2.7 speed. Per-page wall-clock for both recognition backends on the SAME box, SAME crops.

The two harnesses are NOT directly comparable page-for-page and pretending otherwise would be the
easiest way to publish a fake speedup:

  * Rust  (run_pipeline.sh): spawns `paddleocr_vl_recognize` ONCE PER PAGE -> every page pays a full
    model load (bf16 weights -> GPU) before it recognizes a single crop, plus the ONNX layout stage.
  * llama.cpp: llama-server is resident; the client's per-page timer covers recognition + assemble
    only, because the crops were already on disk (it reuses the Rust run's layout output verbatim).

So a raw median-vs-median is apples-to-oranges. Instead regress

    seconds(page) ~= intercept + slope * n_crops(page)

over every page. `slope` is the per-crop recognition cost -- the only quantity the two stacks share a
definition of -- and `intercept` is each harness's fixed per-page overhead (Rust: model load +
layout; llama.cpp: assemble + HTTP). Report both, and report medians separately labelled.

Ordinary least squares, no dependencies. Robustness: trim pages above `--trim-pct` of the time
distribution, since a handful of guard-tripping pages (600s Rust timeout, swap-era stalls) would
otherwise dominate the fit.
"""
import argparse, json, pathlib, re, statistics as st

AP = argparse.ArgumentParser()
AP.add_argument("--rust-log", default="logs/reflayout1651.run.log")
AP.add_argument("--rust-work", default="work_reflayout")
AP.add_argument("--llama-log", default="logs/llamacpp_recognize.log")
AP.add_argument("--llama-after", type=int, default=0,
                help="ignore llama.cpp pages before this index (pages run under the pre -cram 0 "
                     "server were on a thrashing box and are not honest timings)")
AP.add_argument("--trim-pct", type=float, default=99.0)
A = AP.parse_args()
HERE = pathlib.Path(__file__).resolve().parent


def ols(pts):
    """(intercept, slope) for y = a + b*x."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx
    return my - b * mx, b


def trim(pts):
    """Drop the slowest --trim-pct tail by y (guard trips / stalls), so they can't drive the fit."""
    cut = sorted(p[1] for p in pts)[int(len(pts) * A.trim_pct / 100) - 1]
    return [p for p in pts if p[1] <= cut], cut


def report(name, pts, note):
    kept, cut = trim(pts)
    a, b = ols(kept)
    ys = sorted(p[1] for p in pts)
    print(f"\n{name}  (n={len(pts)} pages, {note})")
    print(f"  per-page wall : median {st.median(ys):.1f}s  mean {st.mean(ys):.1f}s  "
          f"p90 {ys[int(len(ys)*.9)]:.1f}s  max {ys[-1]:.1f}s")
    print(f"  crops/page    : median {st.median([p[0] for p in pts]):.0f}  "
          f"total {sum(p[0] for p in pts)} crops")
    print(f"  fit (n={len(kept)}, tail >{cut:.1f}s trimmed): "
          f"{a:.2f}s fixed/page + {b:.2f}s/crop")
    return a, b


# --- Rust: "[n] == stem ==" then "[n] wrote <path> (N bytes) in Ns"; crops from the manifest -------
rust = []
stem_of = {}
for line in open(HERE / A.rust_log, errors="replace"):
    m = re.match(r"\[(\d+)\] == (.+) ==$", line.strip())
    if m:
        stem_of[m.group(1)] = m.group(2)
        continue
    m = re.match(r"\[(\d+)\] wrote .* in (\d+)s$", line.strip())
    if m and m.group(1) in stem_of:
        man = HERE / A.rust_work / stem_of[m.group(1)] / "manifest.json"
        if man.exists():
            rust.append((len(json.load(open(man))), float(m.group(2))))

# --- llama.cpp: "[i/1651] stem: N crops, B bytes, Ts" ---------------------------------------------
llama = []
for line in open(HERE / A.llama_log, errors="replace"):
    m = re.match(r"\[(\d+)/\d+\] .*: (\d+) crops, \d+ bytes, ([\d.]+)s$", line.strip())
    if m and int(m.group(1)) >= A.llama_after:
        llama.append((int(m.group(2)), float(m.group(3))))

ra, rb = report("Rust (mistral.rs, bf16, CUDA)", rust,
                "whole page: model load + ONNX layout + recognize + assemble")
la, lb = report("llama.cpp (bf16 GGUF, resident server)", llama,
                f"recognize + assemble only, crops reused; pages >= {A.llama_after}")

print(f"\nper-crop recognition (the like-for-like number): "
      f"Rust {rb:.2f}s vs llama.cpp {lb:.2f}s  -> {rb/lb:.2f}x")
print(f"fixed per-page overhead: Rust {ra:.1f}s (dominated by the per-page model load our harness "
      f"pays) vs llama.cpp {la:.1f}s (server resident)")
