#!/usr/bin/env bash
# Drive the 3-stage Rust pipeline over a list of OmniDocBench GT images, producing one
# scorer-ready markdown per page at <preds>/<image_stem>.md, where <image_stem> is the GT
# image filename with its last 4 chars (".png"/".jpg") stripped -- byte-for-byte what the
# official scorer looks up (end2end_dataset.py: `img_name[:-4] + '.md'`).
#
# Idempotent + resumable: a page whose <preds>/<stem>.md already exists (non-empty) is skipped,
# so a crash/kill can be re-run without redoing finished pages.
#
# ponytail: recognize reloads the ~1.9GB checkpoint per page invocation (one model load per page).
# Fine for the 5/subset smoke runs; a load-once page-iterating recognize mode is the upgrade path
# before the full 1651-page run (tracked in FUTURE_WORK / §2.4).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

STEMS_FILE="${1:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"
PREDS="${2:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"
WORK="${3:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"

IMAGES="${IMAGES:-$HERE/data/images}"
LAYOUT_BIN="${LAYOUT_BIN:-$ROOT/target/release/paddleocr-layout}"
RECOGNIZE_BIN="${RECOGNIZE_BIN:-/home/sb/mistral-paddle/mistralrs/target/release/examples/paddleocr_vl_recognize}"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-/home/sb/mistral-paddle/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0}"
export PADDLEOCR_LAYOUT_MODEL="${PADDLEOCR_LAYOUT_MODEL:-/home/sb/mistral-paddle/layout/models/PP-DocLayoutV3.onnx}"
export PADDLEOCR_VL_WEIGHTS="${PADDLEOCR_VL_WEIGHTS:-/home/sb/mistral-paddle/ref/weights}"
export PADDLEOCR_VL_GPU="${PADDLEOCR_VL_GPU:-1}"

mkdir -p "$PREDS" "$WORK"
n=0; done=0
while IFS= read -r img; do
  [ -z "$img" ] && continue
  n=$((n+1))
  stem="${img:0:${#img}-4}"          # strip exactly 4 chars -> match scorer's img_name[:-4]
  out_md="$PREDS/$stem.md"
  if [ -s "$out_md" ]; then echo "[$n] skip (exists): $stem"; done=$((done+1)); continue; fi
  src="$IMAGES/$img"
  [ -f "$src" ] || { echo "[$n] MISSING IMAGE: $src" >&2; continue; }
  page_dir="$WORK/$stem"
  mkdir -p "$page_dir"
  t0=$(date +%s)
  echo "[$n] == $stem =="
  echo "[$n] stage1 layout"
  # Fault-isolate the page like stage2 does: a layout failure (unsupported/corrupt image) skips
  # just this page instead of aborting the whole run under `set -e`.
  if ! "$LAYOUT_BIN" "$src" "$page_dir"; then
    echo "[$n] LAYOUT FAILED -> skip page" >&2
    continue
  fi
  echo "[$n] stage2 recognize (GPU=$PADDLEOCR_VL_GPU)"
  # The recognize binary writes results.json ONCE after all crops are done, then can
  # segfault on CUDA/mistral.rs teardown (exit 139) -- a cosmetic crash AFTER the output
  # is already on disk. Trust the output file, not the exit code: proceed iff results.json
  # has one entry per manifest task. A genuine mid-recognition failure (missing/short/
  # truncated results.json) skips just this one page instead of aborting the whole run.
  # ponytail: skip-on-failure, no retry -- add a single retry if real (pre-write) crashes appear.
  rc=0; "$RECOGNIZE_BIN" "$page_dir" || rc=$?
  if ! python3 -c "import json,sys; m=json.load(open('$page_dir/manifest.json')); r=json.load(open('$page_dir/results.json')); sys.exit(0 if len(m)>0 and len(r)==len(m) else 1)" 2>/dev/null; then
    echo "[$n] RECOGNIZE FAILED (rc=$rc, results.json missing/incomplete) -> skip page" >&2
    continue
  fi
  [ "$rc" -ne 0 ] && echo "[$n] note: recognize exited $rc but results.json complete (teardown crash tolerated)"
  echo "[$n] stage3 assemble"
  "$LAYOUT_BIN" assemble "$page_dir/results.json" > "$out_md"
  t1=$(date +%s)
  echo "[$n] wrote $out_md ($(wc -c <"$out_md") bytes) in $((t1-t0))s"
  done=$((done+1))
done < "$STEMS_FILE"
echo "DONE: $done/$n pages have predictions in $PREDS"
