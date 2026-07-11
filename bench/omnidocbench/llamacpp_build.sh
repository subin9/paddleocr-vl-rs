#!/usr/bin/env bash
# Cross-stack (§2.6): build llama.cpp from source with CUDA.
#
# Why source: the box's Homebrew llama-mtmd-cli is b5720 (8308f98c, mid-2025), which PREDATES
# PR #18825 (PaddleOCR-VL support, merged 2026-02-19) -- `strings` finds no `paddleocr` projector
# in it at all. Pinned to 4f37f519, the SHA the support audit in CHECKLIST_ODB §2.6 was done against.
#
# The PATH nvcc is 11.5 and CANNOT target sm_89 (Ada, RTX 4070 Ti SUPER), so we pin CUDACXX to the
# 12.9 toolchain explicitly. Getting this wrong yields a CPU-only or unsupported-arch build.
set -euo pipefail
BUILD="${BUILD:-/home/sb/mistral-paddle/llamacpp-build}"
PIN="${PIN:-4f37f519}"
CUDA="${CUDA:-/usr/local/cuda}"
mkdir -p "$BUILD"
cd "$BUILD"

[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch --all --tags
git checkout "$PIN"
git rev-parse HEAD > "$BUILD/BUILT_SHA"

# ponytail: only llama-server is needed -- llamacpp_recognize.py drives it over HTTP so the model
# loads once for all 34k crops. Building the full target list would cost minutes for binaries we
# never call.
CUDACXX="$CUDA/bin/nvcc" cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j"$(nproc)" --target llama-server

echo "BUILT $(cat "$BUILD/BUILT_SHA") -> $BUILD/llama.cpp/build/bin/llama-server"
"$BUILD/llama.cpp/build/bin/llama-server" --version 2>&1 | head -3
# The whole point of the source build: both must be non-zero, or PaddleOCR-VL cannot load.
# Grep the SHARED LIBS, not llama-server -- the server binary is a thin shim that links these, so
# `strings llama-server | grep paddleocr` returns 0 even on a perfectly good build.
B="$BUILD/llama.cpp/build/bin"
echo "LM arch  (libllama.so, want llama_model_paddleocr): $(strings "$B/libllama.so" | grep -c paddleocr)"
echo "projector (libmtmd.so, want clip_graph_paddleocr):  $(strings "$B/libmtmd.so"  | grep -c paddleocr)"
