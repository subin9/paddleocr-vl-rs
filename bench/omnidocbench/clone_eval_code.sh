#!/usr/bin/env bash
# Clone the official OmniDocBench evaluation code, pinned to the v1_5 branch tip.
# Code is Apache-2.0; the clone is gitignored (bench/OmniDocBench/), not vendored.
set -euo pipefail

PIN="59b103c4b47d3a01fada83491585d6512a40c0bc"  # v1_5 branch @ 2026-04-10
DEST="$(cd "$(dirname "$0")/.." && pwd)/OmniDocBench"

git clone -b v1_5 https://github.com/opendatalab/OmniDocBench.git "$DEST"
git -C "$DEST" checkout "$PIN"
echo "Eval code at $DEST @ $PIN"
