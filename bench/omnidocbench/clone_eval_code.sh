#!/usr/bin/env bash
# Clone the official OmniDocBench evaluation code, pinned to the v1_5 branch tip.
# Code is Apache-2.0; the clone is gitignored (bench/OmniDocBench/), not vendored.
set -euo pipefail

PIN="59b103c4b47d3a01fada83491585d6512a40c0bc"  # v1_5 branch @ 2026-04-10
DEST="$(cd "$(dirname "$0")/.." && pwd)/OmniDocBench"

git clone -b v1_5 https://github.com/opendatalab/OmniDocBench.git "$DEST"
git -C "$DEST" checkout "$PIN"

# CDM compatibility: scikit-image renamed ransac()'s `random_state` to `rng` (>= 0.23) and dropped the
# old name, so the pinned scorer raises TypeError inside CDM's box matcher. CDM catches *everything*
# (`except: return {"F1_score": 0}`), so the failure is SILENT: every formula scores 0.0, which reads
# exactly like "the model got every formula wrong". It is a seed-parameter rename -- the algorithm and
# the seed value are unchanged -- so renaming the keyword is behaviour-preserving.
#
# Run cdm_smoke.py after this. It exists because of exactly this class of failure: it asserts an
# identical gt/pred pair scores F1 = 1.0, and fails loudly if the environment is quietly returning 0.
sed -i 's/^\( *\)random_state=42$/\1rng=42/' "$DEST/metrics/cdm_metric.py"
grep -q "rng=42" "$DEST/metrics/cdm_metric.py" || { echo "CDM ransac patch did not apply" >&2; exit 1; }

echo "Eval code at $DEST @ $PIN (CDM ransac patched; now run cdm_smoke.py before trusting any CDM number)"
