#!/usr/bin/env bash
# Cross-stack (§2.6): serve PaddleOCR-VL-1.5 from llama.cpp so the SAME crops our Rust pipeline
# recognizes can be re-recognized by a second, independent stack.
#
# Precision: the first-party GGUF is **BF16** (gguf header general.file_type=32; tensors BF16+F32) --
# the repo ships no quant ladder. So this is a like-for-like precision comparison with our bf16 port,
# NOT a quantized run. Anything else would need saying so out loud.
#
# ponytail: llama-server (load once, N requests) instead of llama-mtmd-cli per crop -- 34,097 crops
# would otherwise mean 34,097 model loads.
set -euo pipefail
BUILD="${BUILD:-/home/sb/mistral-paddle/llamacpp-build}"
GGUF="${GGUF:-$BUILD/gguf/PaddleOCR-VL-1.5.gguf}"
MMPROJ="${MMPROJ:-$BUILD/gguf/PaddleOCR-VL-1.5-mmproj.gguf}"
TMPL="${TMPL:-$BUILD/gguf/chat_template.jinja}"
PORT="${PORT:-8081}"
# -np 1: the client is strictly serial (K=1, by design, to match the Rust port), so llama.cpp's
# default of 4 parallel slots is pure waste, and it splits -c 8192 into 4 slots of 2048 ctx each --
# a big crop's image tokens + a long generation overrun 2048 and context-shift. One slot is what we
# use, and it gets the whole 8192.
#
# THE SLOW-PAGE TAIL WAS NEVER llama.cpp's. Two runs died to it and I misattributed it twice (first
# to runaway generation, then to -np 4 context-shift). It is HOST MEMORY PRESSURE, and on this box
# the other party is the editor: rust-analyzer indexing this repo holds ~5.5GB, llama-server's CUDA
# host buffers hold ~5.3GB (under WSL2 the GPU allocation is host-backed -- which is why nvidia-smi
# showed an near-empty GPU while RAM was exhausted), and the box has 15GB. The kernel then either
#   * swaps the server out -> it goes unresponsive for 200-500s while its OWN request timings stay
#     at ~400ms (the tell: median 4.1s/page, but 5 pages of 52 ate 94% of the wall clock), or
#   * OOM-kills it outright (twice; the second took this supervisor with it).
# So: make the benchmark the one thing the OOM killer will not touch. If the box gets tight the
# kernel now eats rust-analyzer (VS Code just respawns it) instead of a 34,097-crop run.
oom_immune() { sudo -n sh -c "echo -1000 > /proc/$1/oom_score_adj" 2>/dev/null || true; }

# Supervised restart: an OOM kill (or any crash) must not silently end a 34,097-crop run. The client
# blocks on /health and resumes; it never fabricates output for a dead server.
trap 'exit 0' TERM INT
oom_immune $$          # ... and not the supervisor either: the last OOM killed it, ending the run.
while true; do
  "$BUILD/llama.cpp/build/bin/llama-server" \
    -m "$GGUF" --mmproj "$MMPROJ" \
    --jinja --chat-template-file "$TMPL" \
    -ngl 99 -c 8192 -np 1 --temp 0 --top-k 1 \
    --host 127.0.0.1 --port "${PORT}" \
    --no-warmup &
  srv=$!
  oom_immune "$srv"
  wait "$srv"
  echo "=== llama-server exited ($?) -- restarting in 5s ===" >&2
  sleep 5
done
