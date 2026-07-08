#!/usr/bin/env bash
# End-to-end: PDF -> structured markdown, Python-free at inference time.
#
# Pipeline stages (see README for the architecture):
#   1. render each PDF page to a PNG            (pdftoppm, from poppler-utils)
#   2. layout detection -> crops + manifest     (this repo's `paddleocr-layout` bin, ONNX)
#   3. recognize each crop -> results.json      (PaddleOCR-VL via mistral.rs; examples/recognize.rs)
#   4. reassemble reading-order markdown        (this repo's `paddleocr-layout assemble`)
#
# Prereqisites (see README "Quick start"):
#   - poppler-utils (`pdftoppm`)
#   - ONNX Runtime shared lib; export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
#   - the PP-DocLayoutV3 ONNX graph; export PADDLEOCR_LAYOUT_MODEL=/path/to/PP-DocLayoutV3.onnx
#   - a `recognize` binary built against mistral.rs (examples/recognize.rs) on PATH or set RECOGNIZE_BIN
#   - the PaddleOCR-VL checkpoint dir; export PADDLEOCR_VL_WEIGHTS=/path/to/PaddleOCR-VL-1.5
#
# Usage: examples/pdf_to_markdown.sh <input.pdf> <out_dir>
set -euo pipefail

PDF="${1:?usage: pdf_to_markdown.sh <input.pdf> <out_dir>}"
OUT="${2:?usage: pdf_to_markdown.sh <input.pdf> <out_dir>}"
LAYOUT_BIN="${LAYOUT_BIN:-./target/release/paddleocr-layout}"
RECOGNIZE_BIN="${RECOGNIZE_BIN:-recognize}"

mkdir -p "$OUT/pages"
echo ">> [1/4] rendering PDF pages -> PNG"
pdftoppm -png -r 200 "$PDF" "$OUT/pages/page"

DOC_MD="$OUT/document.md"
: > "$DOC_MD"

for page_png in "$OUT"/pages/page*.png; do
  name="$(basename "$page_png" .png)"
  page_dir="$OUT/$name"
  mkdir -p "$page_dir"

  echo ">> [2/4] layout: $name"
  "$LAYOUT_BIN" "$page_png" "$page_dir"

  echo ">> [3/4] recognize (PaddleOCR-VL via mistral.rs): $name"
  "$RECOGNIZE_BIN" "$page_dir"   # writes $page_dir/results.json

  echo ">> [4/4] assemble markdown: $name"
  "$LAYOUT_BIN" assemble "$page_dir/results.json" >> "$DOC_MD"
  printf '\n\n' >> "$DOC_MD"
done

echo ">> done -> $DOC_MD"
