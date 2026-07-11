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
# default of 4 parallel slots is pure waste -- and actively harmful twice over:
#   * it splits -c 8192 into 4 slots of 2048 ctx each. A big crop's image tokens + a long generation
#     overrun 2048 and trigger context-shift thrashing (observed: pages taking 1311s / 1343s against
#     a 3.2s median). With -np 1 a request gets the whole 8192 and that tail disappears.
#   * 4 slots hold 4 concurrent image-encode + compute buffers. Host anon-rss reached 4.4GB and the
#     kernel OOM-killed the server mid-run (15GB box, 2026-07-12 03:42). One slot is what we use.
#
# Supervised restart: an OOM kill (or any crash) must not silently end a 34,097-crop run. The client
# blocks on /health and resumes; it never fabricates output for a dead server.
trap 'exit 0' TERM INT
while true; do
  "$BUILD/llama.cpp/build/bin/llama-server" \
    -m "$GGUF" --mmproj "$MMPROJ" \
    --jinja --chat-template-file "$TMPL" \
    -ngl 99 -c 8192 -np 1 --temp 0 --top-k 1 \
    --host 127.0.0.1 --port "${PORT}" \
    --no-warmup
  echo "=== llama-server exited ($?) -- restarting in 5s ===" >&2
  sleep 5
done
