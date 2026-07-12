#!/usr/bin/env python3
"""Speed: Rust port vs llama.cpp, same box, same pages, same crops. Reproduces the table in
docs/BENCHMARKS.md "Speed, honestly".

Two things make the naive comparison a lie, and both are corrected here:

1. **The Rust full run was timed on a degraded box.** It executed while rust-analyzer held 6.4GB and
   swap was 100% full; llama.cpp was timed after the cleanup. Its median page
   read 17s, vs 10s on the clean box -- quoting that against llama.cpp's 2.2s would manufacture a
   7.7x speedup out of a memory leak. So we do NOT read Rust timings from the full run log: we read
   them from `logs/speed_rust.log`, a re-run of a 118-page stratified sample on the CLEAN box with
   llama-server stopped.

2. **llama.cpp never pays for layout.** It re-reads the crops the Rust run already wrote, so its
   per-page timer covers recognition + assembly only. The measured ONNX layout cost (stage_split.py:
   0.88s/page) is added back to every llama.cpp page, or it would be credited with work it skipped.

Pages are paired by stem and bucketed by crop count, because per-page cost is driven by crops, and an
unbucketed median would just compare the two runs' page-size mixes.
"""
import argparse, re, statistics as st

AP = argparse.ArgumentParser()
AP.add_argument("--rust-log", default="logs/speed_rust.log")
AP.add_argument("--rust-csv", default=None,
                help="speed_loadonce.py per-page CSV (load-once mode); replaces --rust-log. Rust's "
                     "end-to-end page cost becomes layout+recognize+assemble, with the checkpoint "
                     "load paid ONCE for the run instead of once per page")
AP.add_argument("--llama-log", default="logs/llamacpp_recognize.log")
AP.add_argument("--stems", default="speed120.stems")
AP.add_argument("--layout-secs", type=float, default=0.88,
                help="measured ONNX layout cost llama.cpp inherits for free (stage_split.py)")
AP.add_argument("--llama-after", type=int, default=211,
                help="llama.cpp pages before this index ran under the pre `-cram 0` server on a "
                     "thrashing box; their timings are box artifacts, not model outcomes")
A = AP.parse_args()

stems = {l.strip()[:-4] for l in open(A.stems) if l.strip()}

rust = {}
if A.rust_csv:
    import csv
    for r in csv.DictReader(open(A.rust_csv)):
        rust[r["stem"]] = float(r["layout_s"]) + float(r["recognize_s"]) + float(r["assemble_s"])
else:
    stem_of = {}
    for line in open(A.rust_log, errors="replace"):
        m = re.match(r"\[(\d+)\] == (.+) ==$", line.strip())
        if m:
            stem_of[m.group(1)] = m.group(2)
            continue
        m = re.match(r"\[(\d+)\] wrote .* in (\d+)s$", line.strip())
        if m and m.group(1) in stem_of:
            rust[stem_of[m.group(1)]] = float(m.group(2))

llama = {}
for line in open(A.llama_log, errors="replace"):
    m = re.match(r"\[(\d+)/\d+\] (.+): (\d+) crops, \d+ bytes, ([\d.]+)s$", line.strip())
    if m and int(m.group(1)) >= A.llama_after and m.group(2) in stems:
        llama[m.group(2)] = (int(m.group(3)), float(m.group(4)))

both = [(llama[s][0], rust[s], llama[s][1] + A.layout_secs) for s in rust if s in llama]
mode = "load-once" if A.rust_csv else "per-page reload"
print(f"paired pages: {len(both)}  (Rust: clean-box, {mode}; llama.cpp: pages >= {A.llama_after}; "
      f"+{A.layout_secs}s layout added to llama.cpp)\n")
print(f"{'crops':>10}  {'n':>3}  {'Rust end-to-end':>16}  {'llama.cpp +layout':>18}  {'speedup':>8}")
for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 25), (26, 40), (41, 200)]:
    v = [(r, l) for n, r, l in both if lo <= n <= hi]
    if not v:
        continue
    mr, ml = st.median([x[0] for x in v]), st.median([x[1] for x in v])
    print(f"{f'{lo}-{hi}':>10}  {len(v):>3}  {mr:>15.1f}s  {ml:>17.1f}s  {mr/ml:>7.1f}x")

ar, al = [x[1] for x in both], [x[2] for x in both]
print(f"\n{'ALL':>10}  {len(both):>3}  {st.median(ar):>15.1f}s  {st.median(al):>17.1f}s  "
      f"{st.median(ar)/st.median(al):>7.1f}x   <- per-page medians")
