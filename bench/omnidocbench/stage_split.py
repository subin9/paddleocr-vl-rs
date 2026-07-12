#!/usr/bin/env python3
"""What IS the Rust pipeline's 12.4s fixed per-page cost?

speed_stats.py fits `seconds ~= intercept + slope*crops` over the full run and gets a ~12s intercept.
FUTURE_WORK.md asserts the per-page model load is "only ~1.5s", so the intercept cannot be load
alone, and attributing it to load without measuring would be a guess. Time the two stages directly:

  layout    : paddleocr-layout <image> <dir>        (ONNX PP-DocLayoutV3, once per page)
  recognize : paddleocr_vl_recognize <page_dir>     (spawn + bf16 load + N crops)

Run recognize over pages of differing crop counts; the fit's intercept is spawn+load, the slope is
per-crop. Layout is timed standalone. spawn+load+layout should reconstruct the ~12s.
"""
import json, os, pathlib, statistics as st, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
WS = os.environ.get("WS") or str(HERE.parents[2])   # out-of-tree deps; see speed_loadonce.py
LAYOUT = HERE / "../../target/release/paddleocr-layout"
RECOG = pathlib.Path(os.environ.get(
    "RECOGNIZE_BIN", f"{WS}/mistralrs/target/release/examples/paddleocr_vl_recognize"))
WORK = HERE / "work_reflayout"


def timed(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, check=False)
    return time.perf_counter() - t0


pages = sorted(((len(json.load(open(m))), m.parent) for m in WORK.glob("*/manifest.json")),
               key=lambda p: p[0])
# a couple of near-empty pages (isolate spawn+load) and a couple of fat ones (get the slope)
sample = [p for p in pages if p[0] in (1, 2)][:3] + [p for p in pages if 18 <= p[0] <= 26][:3]

pts = []
for n, d in sample:
    s = timed([str(RECOG), str(d)])
    pts.append((n, s))
    print(f"recognize  crops={n:3d}  {s:6.2f}s  {d.name[:60]}")

mx = sum(p[0] for p in pts) / len(pts)
my = sum(p[1] for p in pts) / len(pts)
slope = (sum((p[0]-mx)*(p[1]-my) for p in pts) / sum((p[0]-mx)**2 for p in pts))
load = my - slope*mx
print(f"\nrecognize fit: {load:.2f}s spawn+bf16 load + {slope:.2f}s/crop")

imgs = sorted((HERE / "data/images").iterdir())[:5]
lay = [timed([str(LAYOUT), str(i), "/tmp/lay_probe"]) for i in imgs]
print(f"layout stage : median {st.median(lay):.2f}s over {len(lay)} pages "
      f"(min {min(lay):.2f} max {max(lay):.2f})")
print(f"\nreconstructed fixed per-page overhead = {load:.2f} (spawn+load) + {st.median(lay):.2f} "
      f"(layout) = {load + st.median(lay):.2f}s   [full-run regression said ~12.4s]")
