#!/usr/bin/env bash
# Drive the 3-stage Rust pipeline over a list of OmniDocBench GT images, producing one
# scorer-ready markdown per page at <preds>/<image_stem>.md, where <image_stem> is the GT
# image filename with its last 4 chars (".png"/".jpg") stripped -- byte-for-byte what the
# official scorer looks up (end2end_dataset.py: `img_name[:-4] + '.md'`).
#
# Load-once: recognition runs as ONE process over every pending page (`recognize --list`), so the
# ~1.9GB checkpoint is loaded once per RUN, not once per page (the old shape paid a measured
# 1.76s/page of spawn+load -- a harness artifact, ~48 min over the 1651-page set). Layout (ONNX) and
# assembly stay per-page: they load nothing heavy. Byte-identical output to the old per-page shape is
# enforced by `loadonce_parity.sh`, not assumed.
#
# Idempotent + resumable: a page whose <preds>/<stem>.md already exists (non-empty) is skipped, and
# every pass re-derives what is still pending, so a crash/kill/watchdog-kill is just re-run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

STEMS_FILE="${1:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"
PREDS="${2:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"
WORK="${3:?usage: run_pipeline.sh <image-list-file> <preds_dir> <work_dir>}"

WS="${WS:-$(cd "$ROOT/.." && pwd)}"   # workspace holding the out-of-tree deps (mistral.rs, weights,
                                      # the ONNX layout model, onnxruntime). Override if they differ.
IMAGES="${IMAGES:-$HERE/data/images}"
LAYOUT_BIN="${LAYOUT_BIN:-$ROOT/target/release/paddleocr-layout}"
RECOGNIZE_BIN="${RECOGNIZE_BIN:-$WS/mistralrs/target/release/examples/paddleocr_vl_recognize}"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$WS/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0}"
export PADDLEOCR_LAYOUT_MODEL="${PADDLEOCR_LAYOUT_MODEL:-$WS/layout/models/PP-DocLayoutV3.onnx}"
export PADDLEOCR_VL_WEIGHTS="${PADDLEOCR_VL_WEIGHTS:-$WS/ref/weights}"
export PADDLEOCR_VL_GPU="${PADDLEOCR_VL_GPU:-1}"
# Runaway guard, load-once shape. Inside the binary: the per-region tokio timeout still records empty
# text and moves on, and its hard backstop (an OS-thread watchdog at 2x that budget, for a wedged
# engine that blocks the tokio worker -- the case the old outer `timeout` existed for) kills the
# process and drops a TIMEOUT_SKIP marker in the offending page dir. Out here: that page is excluded,
# and the next pass resumes everything else. One hung crop costs one page, not the run. MAX_PASSES
# bounds the retry loop; a pass that recognizes nothing new also stops it.
MAX_PASSES="${MAX_PASSES:-5}"

mkdir -p "$PREDS" "$WORK"
complete() {  # page dir holds one result per manifest task
  python3 -c "import json,sys; m=json.load(open('$1/manifest.json')); r=json.load(open('$1/results.json')); sys.exit(0 if len(m)>0 and len(r)==len(m) else 1)" 2>/dev/null
}

total=0; skipped=0
for pass in $(seq 1 "$MAX_PASSES"); do
  pending=()   # page dirs still needing recognition on this pass
  n=0
  while IFS= read -r img; do
    [ -z "$img" ] && continue
    [ "$pass" -eq 1 ] && total=$((total+1))
    stem="${img:0:${#img}-4}"          # strip exactly 4 chars -> match scorer's img_name[:-4]
    [ -s "$PREDS/$stem.md" ] && continue
    page_dir="$WORK/$stem"
    if [ -f "$page_dir/TIMEOUT_SKIP" ]; then
      [ "$pass" -eq 1 ] && { echo "skip (watchdog TIMEOUT_SKIP): $stem" >&2; skipped=$((skipped+1)); }
      continue
    fi
    src="$IMAGES/$img"
    [ -f "$src" ] || { echo "MISSING IMAGE: $src" >&2; continue; }
    n=$((n+1))
    # stage1 layout (ONNX): idempotent, cheap. Fault-isolated -- a corrupt image skips just its page.
    if [ ! -s "$page_dir/manifest.json" ]; then
      mkdir -p "$page_dir"
      echo "[$n] stage1 layout: $stem"
      "$LAYOUT_BIN" "$src" "$page_dir" || { echo "[$n] LAYOUT FAILED -> skip page: $stem" >&2; continue; }
    fi
    complete "$page_dir" || pending+=("$page_dir")
  done < "$STEMS_FILE"

  if [ "${#pending[@]}" -gt 0 ]; then
    echo "== pass $pass: stage2 recognize, load-once over ${#pending[@]} page(s) (GPU=$PADDLEOCR_VL_GPU) =="
    printf '%s\n' "${pending[@]}" > "$WORK/pending.list"
    # The binary writes each page's results.json as that page finishes, and can segfault on
    # CUDA/mistral.rs teardown (exit 139) AFTER every output is already on disk. Trust the files, not
    # the exit code: pages with a complete results.json are assembled below, the rest retry next pass.
    rc=0; "$RECOGNIZE_BIN" --list "$WORK/pending.list" || rc=$?
    [ "$rc" -ne 0 ] && echo "== pass $pass: recognize exited $rc (falling through to per-page outputs) ==" >&2
  fi

  # stage3 assemble: every page whose recognition landed but whose markdown is not written yet.
  new=0
  while IFS= read -r img; do
    [ -z "$img" ] && continue
    stem="${img:0:${#img}-4}"
    page_dir="$WORK/$stem"
    [ -s "$PREDS/$stem.md" ] && continue
    { [ -s "$page_dir/results.json" ] && complete "$page_dir"; } || continue
    "$LAYOUT_BIN" assemble "$page_dir/results.json" > "$PREDS/$stem.md"
    echo "wrote $PREDS/$stem.md ($(wc -c <"$PREDS/$stem.md") bytes)"
    new=$((new+1))
  done < "$STEMS_FILE"

  [ "${#pending[@]}" -eq 0 ] && break                    # nothing left to recognize -> done
  [ "$new" -eq 0 ] && { echo "pass $pass recognized 0 new pages -> stopping (no progress)" >&2; break; }
done

have=$(while IFS= read -r img; do [ -z "$img" ] && continue; s="${img:0:${#img}-4}"; [ -s "$PREDS/$s.md" ] && echo x; done < "$STEMS_FILE" | wc -l)
echo "DONE: $have/$total pages have predictions in $PREDS (watchdog-skipped: $skipped)"
