#!/usr/bin/env bash
# Wait for an OmniDocBench pipeline runner to finish, then score with the official
# scorer. Idempotent: safe to re-run; only scores a COMPLETE set (>= THRESHOLD preds).
# Never relaunches the pipeline. Refuses to score a partial/incomplete run.
# Args (all optional, defaults = subset150 for back-compat):
#   $1 RUNNER_PID  $2 PREDS_DIR  $3 CFG  $4 THRESHOLD  $5 LOG
set -u
cd "$(dirname "$0")/.."                      # -> bench/
RUNNER_PID="${1:-258369}"
PREDS="${2:-omnidocbench/preds/subset150}"
CFG="${3:-omnidocbench/data/subsets/subset150.end2end.yaml}"
THRESHOLD="${4:-150}"
LOG="${5:-omnidocbench/logs/score-subset150.log}"
: > "$LOG"

log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

# Wait for runner to exit. SLEEP=30, ITERS=1600 -> ~13.3h cap (covers the 6-8h full
# run; subset breaks early when the runner dies). Never hang the loop forever.
for _ in $(seq 1 1600); do
  kill -0 "$RUNNER_PID" 2>/dev/null || break
  sleep 30
done

CNT=$(ls $PREDS/*.md 2>/dev/null | wc -l)
if kill -0 "$RUNNER_PID" 2>/dev/null; then
  log "TIMEOUT: runner $RUNNER_PID still alive after cap; pred=$CNT/$THRESHOLD; NOT scoring"; exit 2
fi
log "runner $RUNNER_PID exited; pred count=$CNT/$THRESHOLD"
if [ "$CNT" -lt "$THRESHOLD" ]; then
  log "INCOMPLETE: $CNT/$THRESHOLD preds after runner exit; NOT scoring a partial set"; exit 3
fi

log "scoring: pdf_validation.py -c $CFG"
cd OmniDocBench
../omnidocbench/scorer-venv/bin/python pdf_validation.py -c "../$CFG" 2>&1 | tee -a "../$LOG"
rc=${PIPESTATUS[0]}
log "SCORER_EXIT=$rc  results under bench/OmniDocBench/result/ (prefix = preds basename '$(basename "$PREDS")')"
exit "$rc"
