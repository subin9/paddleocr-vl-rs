#!/usr/bin/env python3
"""Speed, load-once: re-time the Rust pipeline on the SAME 118 pages as the clean-box run,
with recognition loading the checkpoint ONCE for the whole run instead of once per page.

Emits a per-page CSV (stem, crops, layout_s, recognize_s, assemble_s) that `speed_stats.py --rust-csv`
reads, so the llama.cpp side keeps its existing corrections (layout added back, pre-`-cram 0` pages
dropped) rather than being re-derived here.

Stage timing:
  layout    : wall-clock of each `paddleocr-layout <image> <dir>` (ONNX, once per page)
  recognize : ONE `paddleocr_vl_recognize --list` process; per-page cost is the gap between its
              `== page k/n ==` lines. The checkpoint load happens BEFORE the first such line, so it is
              measured once, on its own, and reported separately -- it is no longer a per-page cost.
  assemble  : wall-clock of each `paddleocr-layout assemble` (pure CPU, string work)

Run on a verified-clean box (no rust-analyzer, no swap pressure, llama-server stopped) or the number
is a box artifact, not a pipeline property -- that is the mistake speed_stats.py caught.
"""
import csv, json, os, pathlib, re, statistics as st, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
# WS: workspace holding this repo's out-of-tree deps (mistral.rs, weights, the ONNX layout model,
# onnxruntime). Defaults to the repo's parent dir; override any path below via its env var.
WS = os.environ.get("WS") or str(HERE.parents[2])
LAYOUT = (HERE / "../../target/release/paddleocr-layout").resolve()
RECOG = pathlib.Path(os.environ.get(
    "RECOGNIZE_BIN", f"{WS}/mistralrs/target/release/examples/paddleocr_vl_recognize"))
IMAGES = HERE / "data/images"
WORK = HERE / "work_speed_lo"
OUT_CSV = HERE / "logs/speed_loadonce.csv"
STEMS = HERE / (sys.argv[1] if len(sys.argv) > 1 else "speed120.stems")

env = dict(os.environ)
env.setdefault("ORT_DYLIB_PATH",
               f"{WS}/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.27.0")
env.setdefault("PADDLEOCR_LAYOUT_MODEL", f"{WS}/layout/models/PP-DocLayoutV3.onnx")
env.setdefault("PADDLEOCR_VL_WEIGHTS", f"{WS}/ref/weights")
env.setdefault("PADDLEOCR_VL_GPU", "1")

imgs = [l.strip() for l in open(STEMS) if l.strip()]
pages = []  # (stem, image_path, page_dir)
for img in imgs:
    stem = img[:-4]
    src = IMAGES / img
    if not src.exists():
        print(f"MISSING IMAGE: {src}", file=sys.stderr)
        continue
    pages.append((stem, src, WORK / stem))

# ---- stage 1: layout, per page ----------------------------------------------------------------
layout_s = {}
for i, (stem, src, d) in enumerate(pages, 1):
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = subprocess.run([str(LAYOUT), str(src), str(d)], capture_output=True, env=env)
    layout_s[stem] = time.perf_counter() - t0
    if r.returncode != 0 or not (d / "manifest.json").exists():
        print(f"[{i}] LAYOUT FAILED -> drop page: {stem}", file=sys.stderr)
        layout_s.pop(stem, None)
print(f"layout: {len(layout_s)} pages, median {st.median(layout_s.values()):.2f}s", flush=True)

pages = [p for p in pages if p[0] in layout_s]
crops = {stem: len(json.load(open(d / "manifest.json"))) for stem, _, d in pages}

# ---- stage 2: recognition, ONE process over every page ------------------------------------------
listfile = WORK / "speed.list"
listfile.write_text("\n".join(str(d) for _, _, d in pages) + "\n")
recognize_s, order = {}, [stem for stem, _, _ in pages]
t_start = time.perf_counter()
load_s, prev_t, prev_stem = None, None, None
proc = subprocess.Popen([str(RECOG), "--list", str(listfile)], stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
for line in proc.stdout:
    m = re.match(r"== page (\d+)/(\d+): (.+) ==", line.strip())
    if not m:
        continue
    now = time.perf_counter()
    if prev_t is None:
        load_s = now - t_start          # process spawn + checkpoint load, paid ONCE
        print(f"checkpoint load (once for the whole run): {load_s:.2f}s", flush=True)
    else:
        recognize_s[prev_stem] = now - prev_t
    prev_t, prev_stem = now, pathlib.Path(m.group(3)).name
    print(f"[{m.group(1)}/{m.group(2)}] {prev_stem} ({crops.get(prev_stem, 0)} crops)", flush=True)
proc.wait()
if prev_stem is not None:               # last page ends at process exit (teardown segfault included)
    recognize_s[prev_stem] = time.perf_counter() - prev_t

# ---- stage 3: assemble, per page -----------------------------------------------------------------
assemble_s = {}
preds = WORK / "preds"
preds.mkdir(exist_ok=True)
for stem, _, d in pages:
    if not (d / "results.json").exists():
        continue
    t0 = time.perf_counter()
    md = subprocess.run([str(LAYOUT), "assemble", str(d / "results.json")],
                        capture_output=True, env=env)
    assemble_s[stem] = time.perf_counter() - t0
    (preds / f"{stem}.md").write_bytes(md.stdout)

# ---- report --------------------------------------------------------------------------------------
OUT_CSV.parent.mkdir(exist_ok=True)
rows = [(s, crops[s], layout_s[s], recognize_s[s], assemble_s[s])
        for s in order if s in recognize_s and s in assemble_s]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stem", "crops", "layout_s", "recognize_s", "assemble_s"])
    for s, c, l, rr, a in rows:
        w.writerow([s, c, f"{l:.3f}", f"{rr:.3f}", f"{a:.3f}"])

tot = [l + rr + a for _, _, l, rr, a in rows]
print(f"pages: {len(rows)}   checkpoint load (once): {load_s:.2f}s")
print(f"median per page:  layout {st.median([r[2] for r in rows]):.2f}s   "
      f"recognize {st.median([r[3] for r in rows]):.2f}s   "
      f"assemble {st.median([r[4] for r in rows]):.3f}s   "
      f"end-to-end {st.median(tot):.2f}s")
per_crop = [r[3] / r[1] for r in rows if r[1]]
print(f"recognition per crop: median {st.median(per_crop):.2f}s")
print(f"wrote {OUT_CSV}")
