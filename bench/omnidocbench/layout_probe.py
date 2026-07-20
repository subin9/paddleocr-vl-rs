#!/usr/bin/env python3
"""Layout probe: is LAYOUT_PARTIAL a PORT DEFECT or PP-DocLayoutV3's own difficulty profile?

`error_budget.py` attributes (a) LAYOUT 0.0353 of the 0.0662 text_block edit_whole, the largest single
cause being LAYOUT_PARTIAL (our boxes cover <70% of a GT text block's area -> the rest is never
cropped, never recognized, never emitted). That is a statement about OUR boxes. It does not say whose
fault it is. This script asks the reference.

THREE box sets on the same GT blocks, because "the official model" is ambiguous and the difference
between the two official runs is the whole answer:
  ours     -- what the scored pipeline emitted.
  raw      -- official PP-DocLayoutV3 at its OWN inference.yml defaults (score>0.5, no NMS, no merge).
              This is the thing our ONNX port reimplements.
  pipeline -- official PP-DocLayoutV3 as the SHIPPED reference pipeline runs it
              (`paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml`: threshold 0.3, layout_nms,
              per-class layout_merge_bboxes_mode) -- i.e. how the paper's number was produced.

Rules fixed BEFORE the numbers land (pre-registered, as with every other A/B in this bench):
  ours ~= raw                        => our port faithfully reproduces the raw detector.
  ours ~= raw, pipeline covers MORE  => PORT DEFECT, and it is precisely the OMITTED POST-PROCESSING
                                        (threshold/NMS/merge), not the box decode and not the VLM.
  ours << raw                        => PORT DEFECT in our decode/preprocess itself.
  pipeline ~= ours ~= raw            => PP-DocLayoutV3's own profile; the layout stage is faithful and
                                        the published number cannot come from this stage alone.

Coverage = |GT_block ∩ union(boxes)| / |GT_block|, exact via coordinate compression (union, so
overlapping boxes are not double-counted). The SAME function scores both sides -- the comparison is
about the boxes, not about the metric.

Our boxes: the ones the scored run actually emitted (run log -> parse_logs -> the shipped
absorber-aware nested guard) -- identical to the error budget's, and spot-checked against today's
binary by spotcheck_layout_onnx.py.

Usage: ./scorer-venv/bin/python layout_probe.py [n_pages] [tag]
       (run official_layout.py in paddle-venv first, or this script will tell you to)
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from error_budget import GT_JSON, RESULT, TEXT_CATS, classify, poly_box, union_box
from filter_nested import nested_indices, parse_logs

HERE = Path(__file__).parent
COVER_OK = 0.7  # same bar the error budget used to call a block LAYOUT_PARTIAL
OFFICIAL = HERE / "work/official_layout.json"           # raw detector defaults
OFFICIAL_PIPE = HERE / "work/official_layout_pipeline.json"  # the reference pipeline's config
PAGES_JSON = HERE / "work/probe_pages.json"


def union_cover(gt, boxes):
    """|gt ∩ union(boxes)| / |gt|, exact. Coordinate compression: ~30 boxes/page, so a tiny grid."""
    gx0, gy0, gx1, gy1 = gt
    ga = max(gx1 - gx0, 0) * max(gy1 - gy0, 0)
    if ga <= 0:
        return 0.0
    clipped = [
        (max(b[0], gx0), max(b[1], gy0), min(b[2], gx1), min(b[3], gy1))
        for b in boxes
    ]
    clipped = [c for c in clipped if c[2] > c[0] and c[3] > c[1]]
    if not clipped:
        return 0.0
    xs = sorted({c[0] for c in clipped} | {c[2] for c in clipped})
    ys = sorted({c[1] for c in clipped} | {c[3] for c in clipped})
    covered = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(c[0] <= cx <= c[2] and c[1] <= cy <= c[3] for c in clipped):
                covered += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return covered / ga


def our_boxes_by_stem():
    out = {}
    for stem, regions in parse_logs().items():
        drop = nested_indices(regions, 0.8, absorber_aware=True)  # the SHIPPED guard
        out[stem] = [r[2] for i, r in enumerate(regions) if i not in drop]
    return out


def rank_pages(tag, n_pages, ours):
    """The pages carrying the most LAYOUT_PARTIAL edits -- where the cause actually lives."""
    samples = json.loads((RESULT / f"{tag}_quick_match_text_block_result.json").read_text())
    gt_by_img = {p["page_info"]["image_path"]: p for p in json.loads(GT_JSON.read_text())}
    partial = Counter()
    for s in samples:
        if s["Edit_num"] == 0:
            continue
        stem = s["img_id"].rsplit(".", 1)[0]
        page, boxes = gt_by_img.get(s["img_id"]), ours.get(stem)
        if not page or not boxes:
            continue
        orders = set(s["gt_position"] if isinstance(s["gt_position"], list) else [s["gt_position"]])
        annos = [d for d in page["layout_dets"] if d.get("order") in orders
                 and d.get("category_type") in TEXT_CATS and not d.get("ignore")]
        if not annos:
            continue
        if classify(s, union_box([poly_box(a["poly"]) for a in annos]), boxes) == "LAYOUT_PARTIAL":
            partial[stem] += s["Edit_num"]
    return [stem for stem, _ in partial.most_common(n_pages)], partial


def main():
    n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tag = sys.argv[2] if len(sys.argv) > 2 else "full1651_nonest2"

    ours = our_boxes_by_stem()
    pages, partial = rank_pages(tag, n_pages, ours)
    PAGES_JSON.parent.mkdir(parents=True, exist_ok=True)
    PAGES_JSON.write_text(json.dumps(pages, indent=1))
    print(f"== {n_pages} worst LAYOUT_PARTIAL pages ({tag}), by edits attributed to that cause:")
    for p in pages:
        print(f"   {partial[p]:6d} edits  {p}")

    sets = {"ours": ours}
    for name, path in (("raw", OFFICIAL), ("pipeline", OFFICIAL_PIPE)):
        if not path.exists():
            print(f"\nMissing {name} boxes. Run:\n  paddle-venv/bin/python official_layout.py "
                  f"{PAGES_JSON} {path} {name}")
            return 2
        blob = json.loads(path.read_text())
        print(f"\n{name:9} official config: {json.dumps(blob['config'], default=str)[:400]}")
        sets[name] = {s: v["boxes"] for s, v in blob["pages"].items()}

    gt_by_img = {p["page_info"]["image_path"]: p for p in json.loads(GT_JSON.read_text())}
    by_stem = {img.rsplit(".", 1)[0]: img for img in gt_by_img}

    names = list(sets)
    rows = []  # (stem, {name: cover})
    for stem in pages:
        if any(stem not in sets[n] for n in names):
            continue
        page = gt_by_img[by_stem[stem]]
        for d in page["layout_dets"]:
            if d.get("category_type") not in TEXT_CATS or d.get("ignore"):
                continue
            gb = poly_box(d["poly"])
            rows.append((stem, {n: union_cover(gb, sets[n][stem]) for n in names}))

    n_blocks = len(rows)
    print(f"\n== GT text-block coverage on those pages ({n_blocks} blocks; "
          f"cover = |GT ∩ union(boxes)| / |GT|, same metric for all three)\n")
    print(f"   {'boxes':9} {'regions':>8} {'mean cover':>11} {'blocks < ' + str(COVER_OK):>14}")
    for nm in names:
        regions = sum(len(sets[nm].get(s, [])) for s in pages)
        mean = sum(r[1][nm] for r in rows) / n_blocks if n_blocks else 0
        below = sum(1 for r in rows if r[1][nm] < COVER_OK)
        print(f"   {nm:9} {regions:8d} {mean:11.3f} {below:14d}")

    def cross(a, b):
        return sum(1 for r in rows if r[1][a] < COVER_OK <= r[1][b])

    print(f"\n   ours FAIL / raw OK       (our decode/preprocess is at fault) : {cross('ours', 'raw')}")
    print(f"   raw FAIL / ours OK       (we are better than raw)            : {cross('raw', 'ours')}")
    print(f"   ours FAIL / pipeline OK  (omitted post-processing is at fault): {cross('ours', 'pipeline')}")
    print(f"   ALL THREE FAIL           (PP-DocLayoutV3's own profile)      : "
          f"{sum(1 for r in rows if all(r[1][n] < COVER_OK for n in names))}")

    print("\n   per page (blocks below the bar: ours / raw / pipeline, of total):")
    for stem in pages:
        pr = [r for r in rows if r[0] == stem]
        if pr:
            b = {n: sum(1 for r in pr if r[1][n] < COVER_OK) for n in names}
            print(f"     {b['ours']:3d} / {b['raw']:3d} / {b['pipeline']:3d}  of {len(pr):3d}   {stem}")

    faithful = abs(sum(r[1]['ours'] for r in rows) - sum(r[1]['raw'] for r in rows)) / max(n_blocks, 1) < 0.02
    helps = cross("ours", "pipeline") > 2 * cross("pipeline", "ours")
    print(f"\n   port faithful to RAW detector : {faithful}")
    print(f"   pipeline post-proc recovers    : {helps}")
    print(f"\n   VERDICT: {'PORT DEFECT = OMITTED PIPELINE POST-PROCESSING (our boxes == raw detector; the reference config frames what we miss)' if faithful and helps else 'PORT DEFECT IN OUR DECODE (we are worse than the raw detector itself)' if not faithful else 'MODEL PROFILE (even the reference config is partial here)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
