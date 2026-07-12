#!/usr/bin/env bash
# Correctness gate for load-once recognize mode: the SAME binary, over the SAME crops+prompts, must
# emit byte-identical results.json whether it reloads the checkpoint per page or loads it once and
# iterates. A usability mode that changes a single token is a bug, not a tradeoff.
#
# Both arms are run here and now with the CURRENT binary -- comparing against the stored results.json
# from the full run would confound a mode change with a rebuild.
#
#   ./loadonce_parity.sh [stems_file]     # default parity24.stems (24 pages, all 22 layout classes)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STEMS="${1:-$HERE/parity24.stems}"
OUT="$HERE/work_parity"
RECOGNIZE_BIN="${RECOGNIZE_BIN:-/home/sb/mistral-paddle/mistralrs/target/release/examples/paddleocr_vl_recognize}"
export PADDLEOCR_VL_WEIGHTS="${PADDLEOCR_VL_WEIGHTS:-/home/sb/mistral-paddle/ref/weights}"
export PADDLEOCR_VL_GPU="${PADDLEOCR_VL_GPU:-1}"

rm -rf "$OUT"; mkdir -p "$OUT/reload" "$OUT/loadonce"
# Seed both arms from the layout stage's output (manifest + crops only -- never the old results.json).
while IFS= read -r stem; do
  [ -z "$stem" ] && continue
  for arm in reload loadonce; do
    mkdir -p "$OUT/$arm/$stem"
    cp "$HERE/work/$stem/manifest.json" "$OUT/$arm/$stem/"
    cp "$HERE/work/$stem"/crop_*.png "$OUT/$arm/$stem/"
  done
  echo "$OUT/loadonce/$stem"
done < "$STEMS" > "$OUT/loadonce.list"
n=$(wc -l < "$OUT/loadonce.list")

echo "== arm A: per-page reload ($n invocations, one checkpoint load each) =="
ta=$(date +%s)
while IFS= read -r stem; do
  [ -z "$stem" ] && continue
  "$RECOGNIZE_BIN" "$OUT/reload/$stem" > "$OUT/reload/$stem/log.txt" 2>&1 || true
done < "$STEMS"
tb=$(date +%s)

echo "== arm B: load-once ($n page dirs, ONE checkpoint load) =="
tc=$(date +%s)
"$RECOGNIZE_BIN" --list "$OUT/loadonce.list" > "$OUT/loadonce.log" 2>&1 || true
td=$(date +%s)

echo "== diff =="
fail=0; ok=0
while IFS= read -r stem; do
  [ -z "$stem" ] && continue
  a="$OUT/reload/$stem/results.json"; b="$OUT/loadonce/$stem/results.json"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then echo "MISSING results.json: $stem" >&2; fail=$((fail+1)); continue; fi
  if cmp -s "$a" "$b"; then ok=$((ok+1)); else echo "DIFFERS: $stem" >&2; diff "$a" "$b" | head -20 >&2; fail=$((fail+1)); fi
done < "$STEMS"

echo "---"
echo "pages: $n  identical: $ok  differing/missing: $fail"
echo "arm A (per-page reload): $((tb-ta))s   arm B (load-once): $((td-tc))s"
[ "$fail" -eq 0 ] && echo "PARITY: PASS (byte-identical results.json on all $ok pages)" || { echo "PARITY: FAIL"; exit 1; }
