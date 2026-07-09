#!/usr/bin/env bash
# Stand up an ISOLATED CPU-only venv for the OmniDocBench scorer.
# The scorer pins numpy 1.24.4 / pandas 2.0.3 / sklearn 1.1.2 etc. — these
# conflict with the inference venv's torch 2.12, so keep them apart. Python 3.10
# because those old pins have cp310 manylinux wheels (no source builds).
# Output (scorer-venv/) is gitignored, never committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/scorer-venv"
REQ="$HERE/../OmniDocBench/requirements.txt"   # from clone_eval_code.sh (gitignored)

[ -f "$REQ" ] || { echo "Missing $REQ — run clone_eval_code.sh first"; exit 1; }

uv venv --python 3.10 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$REQ"

echo "Scorer venv -> $VENV"
echo "Smoke-test:  (cd $HERE/../OmniDocBench && $VENV/bin/python pdf_validation.py -c configs/end2end.yaml)"
