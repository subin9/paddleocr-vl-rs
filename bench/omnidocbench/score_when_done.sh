#!/usr/bin/env bash
# Wait for the subset150 pipeline runner to finish, then score with the official
# OmniDocBench scorer. Idempotent: safe to re-run; only scores a COMPLETE 150/150 set.
# Never relaunches the pipeline. Refuses to score a partial/incomplete run.
set -u
cd "$(dirname "$0")/.."                      # -> bench/
RUNNER_PID="${1:-258369}"
PREDS=omnidocbench/preds/subset150
CFG=omnidocbench/data/subsets/subset150.end2end.yaml
LOG=omnidocbench/logs/score-subset150.log
: > "$LOG"

log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

# Wait for runner to exit (cap ~40 min so we never hang the loop forever).
for _ in $(seq 1 120); do
  kill -0 "$RUNNER_PID" 2>/dev/null || break
  sleep 20
done

CNT=$(ls $PREDS/*.md 2>/dev/null | wc -l)
if kill -0 "$RUNNER_PID" 2>/dev/null; then
  log "TIMEOUT: runner $RUNNER_PID still alive after ~40min; pred=$CNT/150; NOT scoring"; exit 2
fi
log "runner $RUNNER_PID exited; pred count=$CNT/150"
if [ "$CNT" -lt 150 ]; then
  log "INCOMPLETE: $CNT/150 preds after runner exit; NOT scoring a partial set"; exit 3
fi

log "scoring: pdf_validation.py -c $CFG"
cd OmniDocBench
../omnidocbench/scorer-venv/bin/python pdf_validation.py -c "../$CFG" 2>&1 | tee -a "../$LOG"
rc=${PIPESTATUS[0]}
log "SCORER_EXIT=$rc  result=bench/OmniDocBench/result/subset150_quick_match_metric_result.json"
exit "$rc"
