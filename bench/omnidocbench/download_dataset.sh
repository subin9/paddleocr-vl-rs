#!/usr/bin/env bash
# Download OmniDocBench v1.5 dataset (images + GT JSON), pinned for reproducibility.
# Research-use-only / non-commercial license — output is gitignored, never committed.
# Idempotent: hf download resumes from the local-dir cache, so re-running skips
# already-fetched files. ~1.5 GB, 1651 images + OmniDocBench.json (~42 MB).
set -euo pipefail

# Pinned dataset revision (HF opendatalab/OmniDocBench, v1.5 tip when this was pinned).
REV="aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
DEST="$(cd "$(dirname "$0")" && pwd)/data"

hf download opendatalab/OmniDocBench \
  --repo-type dataset --revision "$REV" \
  --local-dir "$DEST"

echo "Done -> $DEST"
