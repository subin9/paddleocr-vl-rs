#!/usr/bin/env python3
"""ONNX spot-check: does TODAY's layout binary still emit the geometry the run logs recorded?

`tests/parity_nested.rs` proves the shipped `drop_nested()` keeps the same regions as the Python
filter GIVEN the same boxes. That leaves one assumption: the boxes the binary produces today are the
boxes the full run logged (i.e. ONNX is deterministic on this box, and the guard did not disturb the
decode path). Check it directly on the worst-nested pages -- the ones the fix actually changes.

Per page: run the binary with PADDLEOCR_VL_KEEP_NESTED=1 (ablation switch -> pre-drop regions) and
assert the printed boxes == the logged boxes; then run it with the guard ON and assert the printed
boxes == the boxes python's nested_indices() keeps. Both must hold for the scored A/B numbers to
transfer to the shipped pipeline.

Usage: spotcheck_layout_onnx.py [n_pages]   (needs LAYOUT_BIN, PADDLEOCR_LAYOUT_MODEL, ORT_DYLIB_PATH)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BIN = os.environ.get("LAYOUT_BIN", str(ROOT / "target/release/paddleocr-layout"))
IMAGES = Path(os.environ.get("IMAGES", HERE / "data/images"))
REG = re.compile(r"^\s+read_order=\s*(\d+)\s+(\S+)\s+score=([\d.]+)\s+bbox=\[([^\]]+)\]")


def run_layout(img, out_dir, keep_nested):
    env = {**os.environ}
    if keep_nested:
        env["PADDLEOCR_VL_KEEP_NESTED"] = "1"
    else:
        env.pop("PADDLEOCR_VL_KEEP_NESTED", None)
    p = subprocess.run([BIN, str(img), out_dir], capture_output=True, text=True, env=env, check=True)
    return [[float(x) for x in m.group(4).split(",")] for m in map(REG.match, p.stdout.splitlines()) if m]


def main():
    n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    parity = json.loads((HERE / "work/nested_parity.json").read_text())
    # Worst-nested pages first: they exercise the drop, so a silent no-op would show up here.
    stems = sorted(parity, key=lambda s: len(parity[s]["boxes"]) - len(parity[s]["keep"]), reverse=True)
    by_stem = {p.stem: p for p in IMAGES.iterdir()} if IMAGES.is_dir() else {}

    ok = mismatched = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for stem in stems[:n_pages]:
            img = by_stem.get(stem)
            if img is None:
                print(f"  SKIP (no image): {stem}")
                skipped += 1
                continue
            want_all = parity[stem]["boxes"]
            want_kept = [want_all[i] for i in parity[stem]["keep"]]
            got_all = run_layout(img, tmp, keep_nested=True)
            got_kept = run_layout(img, tmp, keep_nested=False)
            bad = []
            if got_all != want_all:
                bad.append(f"pre-drop {len(got_all)} boxes != logged {len(want_all)}")
            if got_kept != want_kept:
                bad.append(f"kept {len(got_kept)} != python-kept {len(want_kept)}")
            if bad:
                mismatched += 1
                print(f"  MISMATCH {stem}: {'; '.join(bad)}")
            else:
                ok += 1
                print(f"  ok {stem}: {len(got_all)} regions -> {len(got_kept)} kept "
                      f"({len(want_all) - len(want_kept)} dropped, matches logs + python)")
    print(f"\nspot-check: {ok} ok, {mismatched} mismatched, {skipped} skipped")
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
