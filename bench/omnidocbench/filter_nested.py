#!/usr/bin/env python3
"""Price the nested-sub-region duplication (§2.5) with a causal A/B -- zero GPU, zero re-recognition.

PP-DocLayoutV3 emits container->child region hierarchies: an `inline_formula` box sits INSIDE the
`text` box that contains it, a `reference_content` box inside its `reference` box. Our port flattens
them into independent crops, so both parent and child are recognized and BOTH are emitted -- the same
content lands in the markdown twice. Measured on the full run: 96.1% of inline_formula and 99.1% of
reference_content boxes are >=80% contained in a strictly larger region, and the parent's own OCR
already renders the math inline (verified: parent text OCR reads `The coefficients \\(a_{kj}\\) are
stored in a matrix A`, while the child is emitted again as a display block `\\[a_{kj}\\]`).

The fix belongs in the LAYOUT stage (drop nested sub-regions before cropping). But the effect can be
priced exactly without re-running layout or recognition: dropping a region at layout time and dropping
its row at assembly time produce byte-identical markdown (crops are recognized independently, so a
sibling's presence never changes another's text). So this script filters the recognized rows and
re-runs the REAL assembler -- never a re-implementation of it.

Region geometry comes from the run logs (the layout CLI prints `read_order/class/score/bbox` per
region); results.json rows are aligned to them by index, which is exactly how plan_tasks() emits them.

Usage: filter_nested.py <preds_out_dir> [--containment 0.8]
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ASSEMBLE_BIN = os.environ.get("LAYOUT_BIN", str(ROOT / "target/release/paddleocr-layout"))
# Chronological: a page re-run in a later log overwrote work/<stem>/, so last-wins matches results.json.
LOGS = ["results/subset150.run.log", "results/full1651.run.log", "logs/rerun8.104303.log"]

HDR = re.compile(r"^\[\d+\] == (.+) ==$")
REG = re.compile(r"^\s+read_order=\s*(\d+)\s+(\w+)\s+score=([\d.]+)\s+bbox=\[([^\]]+)\]")


def parse_logs():
    """{stem: [(read_order, class, bbox)]} -- last log to mention a page wins."""
    pages = {}
    for name in LOGS:
        path = HERE / name
        if not path.exists():
            continue
        cur = None
        for line in path.open(errors="replace"):
            line = line.rstrip("\n")
            m = HDR.match(line)
            if m:
                cur = m.group(1)
                pages[cur] = []  # reset: this run's layout supersedes any earlier one
                continue
            m = REG.match(line)
            if m and cur is not None:
                pages[cur].append(
                    (int(m.group(1)), m.group(2), [float(x) for x in m.group(4).split(",")])
                )
    return pages


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# Classes the assembler never emits (assemble.rs::VISUAL_ONLY_CLASSES). A parent of one of these
# classes absorbs NOTHING into the markdown, so a child nested inside it is not a duplicate -- it is
# the only copy of that text, and dropping it deletes the content outright.
VISUAL_ONLY = {"chart", "image", "header_image", "footer_image", "seal"}


def nested_indices(regions, thresh, absorber_aware=False):
    """Indices of regions >=thresh contained in a STRICTLY larger region (the parent keeps the child's
    content in its own OCR, so the child is a duplicate). Strictly-larger breaks the symmetry of two
    near-identical boxes -- without it a duplicate pair would delete both.

    `absorber_aware`: only a parent whose text actually REACHES the markdown can absorb a child, so
    containers in VISUAL_ONLY don't count as parents (see that constant)."""
    drop = set()
    for i, r in enumerate(regions):
        a = area(r[2])
        if a <= 0:
            continue
        for j, o in enumerate(regions):
            if i == j or area(o[2]) <= a:
                continue
            if absorber_aware and o[1] in VISUAL_ONLY:
                continue
            if intersection(r[2], o[2]) / a >= thresh:
                drop.add(i)
                break
    return drop


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "preds/full1651_nonest")
    thresh = 0.8
    if "--containment" in sys.argv:
        thresh = float(sys.argv[sys.argv.index("--containment") + 1])
    absorber_aware = "--absorber-aware" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = parse_logs()
    preds = sorted(p.stem for p in (HERE / "preds/full1651").glob("*.md"))
    print(f"layout logged: {len(pages)} pages; baseline preds: {len(preds)}; containment>={thresh}"
          f"; absorber_aware={absorber_aware}")

    n_pages = n_dropped = n_rows = misaligned = changed = 0
    from collections import Counter

    dropped_by_class = Counter()
    for stem in preds:
        rj = HERE / f"work/{stem}/results.json"
        regions = pages.get(stem)
        if regions is None or not rj.exists():
            print(f"  !! no layout/results for {stem} -- copying baseline pred unchanged")
            (out_dir / f"{stem}.md").write_text((HERE / f"preds/full1651/{stem}.md").read_text())
            continue
        rows = json.loads(rj.read_text())
        # Rows are emitted one-per-region in region order (plan_tasks enumerates the same slice), so
        # index IS the join key. Verify it rather than trust it -- a mismatch means the log and the
        # results came from different layout runs, and silently mis-dropping rows would corrupt the A/B.
        aligned = len(rows) == len(regions) and all(
            r["read_order"] == g[0] and r["class"] == g[1] for r, g in zip(rows, regions)
        )
        if not aligned:
            misaligned += 1
            print(f"  !! misaligned {stem} ({len(rows)} rows vs {len(regions)} regions) -- baseline copy")
            (out_dir / f"{stem}.md").write_text((HERE / f"preds/full1651/{stem}.md").read_text())
            continue

        drop = nested_indices(regions, thresh, absorber_aware)
        n_pages += 1
        n_rows += len(rows)
        n_dropped += len(drop)
        for i in drop:
            dropped_by_class[regions[i][1]] += 1
        kept = [r for i, r in enumerate(rows) if i not in drop]
        if drop:
            changed += 1
        tag = "nonest2" if absorber_aware else "nonest"
        filtered = HERE / f"work/{stem}/results_{tag}.json"
        filtered.write_text(json.dumps(kept, ensure_ascii=False))
        md = subprocess.run(
            [ASSEMBLE_BIN, "assemble", str(filtered)], capture_output=True, text=True, check=True
        ).stdout
        (out_dir / f"{stem}.md").write_text(md)

    print(f"\npages assembled: {n_pages} ({changed} changed, {misaligned} misaligned->baseline copy)")
    print(f"regions: {n_rows} total, {n_dropped} dropped ({100 * n_dropped / max(n_rows, 1):.2f}%)")
    print("dropped by class:", dropped_by_class.most_common())


if __name__ == "__main__":
    main()
