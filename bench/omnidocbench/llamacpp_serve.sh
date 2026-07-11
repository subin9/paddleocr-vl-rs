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
exec "$BUILD/llama.cpp/build/bin/llama-server" \
  -m "$GGUF" --mmproj "$MMPROJ" \
  --jinja --chat-template-file "$TMPL" \
  -ngl 99 -c 8192 --temp 0 --top-k 1 \
  --host 127.0.0.1 --port "${PORT}" \
  --no-warmup
