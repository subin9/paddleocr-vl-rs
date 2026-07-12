#!/usr/bin/env python3
"""Split the official text_block Edit_dist into an error budget and attribute every edit to
LAYOUT (a) or RECOGNITION (b) by geometry. Zero GPU, zero re-scoring.

The scorer reports Edit_dist = sum(Edit_num) / sum(upper_len) over its own matched samples
(metrics/cal_metric.py:168, `all_total_avg`). That is a ratio of sums, so it decomposes EXACTLY: each
sample's Edit_num lands in one bucket and the buckets add back to the published number. We never
recompute a distance -- we read the scorer's own per-sample `Edit_num`/`upper_len` out of
result/<tag>_quick_match_text_block_result.json, and the scorer's own `gt_position` (= the GT `order`
values it matched, merged blocks included) to find each GT block's box in OmniDocBench.json.

Our side: the boxes the layout stage actually emitted (parsed from the run log by
filter_nested.parse_logs), filtered by the SHIPPED absorber-aware nested guard -- i.e. the exact
regions the scored pipeline cropped.

Attribution, per edit-bearing sample (edit-weighted, not sample-counted):
  (b) RECOG_SUBST    our box ~= the GT box (IoU>=0.7) and we emitted a similar-length string:
                     right crop, wrong characters -> the VLM misread it.
  (b) RECOG_SHORT    our box ~= the GT box, but our text is <0.9x its length: right crop, the VLM
                     did not transcribe all of it (early stop / truncation).
  (b) RECOG_LONG     our box ~= the GT box, but our text is >1.1x: right crop, the VLM over-generated.
  (b) RECOG_EMPTY    our box was there, no text reached the markdown at all.
  (a) LAYOUT_MISS    no box of ours overlaps the GT block: PP-DocLayoutV3 never framed that content.
  (a) LAYOUT_SPLIT   several of our boxes tile the GT block: we cut one GT block into pieces.
  (a) LAYOUT_MERGE   one of our boxes swallows the GT block and is much bigger: several GT blocks
                     landed in one crop.
  (a) LAYOUT_PARTIAL our boxes cover <70% of the GT block's area: part of it was never cropped.

Usage: error_budget.py [tag]      (default full1651_nonest2 = the shipped config)
Needs the scorer venv (pylatexenc): ./scorer-venv/bin/python error_budget.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from filter_nested import area, intersection, nested_indices, parse_logs

HERE = Path(__file__).parent
RESULT = HERE.parent / "OmniDocBench/result"
GT_JSON = HERE / "data/OmniDocBench.json"

# The GT categories the scorer folds into text_block (dataset/end2end_dataset.py:285,329).
TEXT_CATS = {
    "text_block", "title", "code_txt", "code_txt_caption", "reference", "equation_caption",
    "figure_caption", "figure_footnote", "table_caption", "table_footnote", "code_algorithm",
    "code_algorithm_caption", "header", "footer", "page_footnote", "page_number",
}
SAME_BOX_IOU = 0.7   # our box is "the same region" as the GT block
COVER_OK = 0.7       # our boxes cover this much of the GT block's area
LAYOUT = {"LAYOUT_MISS", "LAYOUT_SPLIT", "LAYOUT_MERGE", "LAYOUT_PARTIAL"}


def poly_box(poly):
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def bucket(s):
    gt, pred = s["norm_gt"], s["norm_pred"]
    if gt and not pred:
        return "MISS"
    if pred and not gt:
        return "SPURIOUS"
    return "EXACT" if s["Edit_num"] == 0 else "RECOG"


def classify(s, gb, boxes):
    """Attribute one edit-bearing sample to a layout or a recognition cause, from geometry."""
    best = max((iou(gb, b) for b in boxes), default=0.0)
    inside = [b for b in boxes if area(b) > 0 and intersection(gb, b) / area(b) >= 0.5]
    cover = sum(intersection(gb, b) for b in inside) / area(gb) if area(gb) > 0 else 0.0
    swallow = [b for b in boxes if area(gb) > 0 and intersection(gb, b) / area(gb) >= 0.8
               and area(b) >= 1.5 * area(gb)]

    if bucket(s) == "MISS":
        return "RECOG_EMPTY" if (best >= 0.5 or cover >= COVER_OK) else "LAYOUT_MISS"
    if best >= SAME_BOX_IOU:  # same region -> whatever went wrong, the crop was right
        r = len(s["norm_pred"]) / len(s["norm_gt"])
        return "RECOG_SHORT" if r < 0.9 else "RECOG_LONG" if r > 1.1 else "RECOG_SUBST"
    if cover >= COVER_OK and len(inside) >= 2:
        return "LAYOUT_SPLIT"
    if swallow:
        return "LAYOUT_MERGE"
    if cover < COVER_OK:
        return "LAYOUT_PARTIAL"
    return "OTHER"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "full1651_nonest2"
    samples = json.loads((RESULT / f"{tag}_quick_match_text_block_result.json").read_text())

    # --- 1. Error budget (bookkeeping over the scorer's own numbers) ------------------------
    tot_edit = sum(s["Edit_num"] for s in samples)
    tot_upper = sum(s["upper_len"] for s in samples)
    overall = tot_edit / tot_upper
    buckets = defaultdict(lambda: {"n": 0, "edit": 0, "upper": 0})
    for s in samples:
        b = buckets[bucket(s)]
        b["n"] += 1
        b["edit"] += s["Edit_num"]
        b["upper"] += s["upper_len"]

    print(f"== {tag}: text_block Edit_dist (edit_whole) = {overall:.4f}  "
          f"({tot_edit} edits / {tot_upper} chars over {len(samples)} matched samples)\n")
    print(f"{'bucket':9} {'samples':>7} {'edits':>8} {'chars':>9}   contribution")
    for name in ("MISS", "SPURIOUS", "RECOG", "EXACT"):
        b = buckets[name]
        print(f"{name:9} {b['n']:7d} {b['edit']:8d} {b['upper']:9d}   "
              f"{b['edit'] / tot_upper:.4f}  ({100 * b['edit'] / tot_edit:5.1f}% of all edits)")
    assert abs(sum(b["edit"] for b in buckets.values()) / tot_upper - overall) < 1e-12
    print("  NOTE SPURIOUS is structurally 0: quick_match emits no gt-empty sample, so a pred block it\n"
          "  cannot match is silently UNSCORED -- text_block never penalises over-generation directly.")

    # --- 2. Geometric attribution: layout (a) vs recognition (b) ----------------------------
    gt_pages = json.loads(GT_JSON.read_text())
    gt_by_img = {p["page_info"]["image_path"]: p for p in gt_pages}
    our_pages = parse_logs()
    our_boxes = {}
    for stem, regions in our_pages.items():
        drop = nested_indices(regions, 0.8, absorber_aware=True)  # the SHIPPED guard
        our_boxes[stem] = [r[2] for i, r in enumerate(regions) if i not in drop]

    kinds = defaultdict(lambda: {"n": 0, "edit": 0})
    unloc = {"n": 0, "edit": 0}
    pages_by_kind = defaultdict(Counter)
    cat_by_kind = defaultdict(Counter)
    for s in samples:
        if s["Edit_num"] == 0:
            continue
        img, stem = s["img_id"], s["img_id"].rsplit(".", 1)[0]
        page, boxes = gt_by_img.get(img), our_boxes.get(stem)
        orders = set(s["gt_position"] if isinstance(s["gt_position"], list) else [s["gt_position"]])
        annos = [d for d in page["layout_dets"]
                 if d.get("order") in orders and d.get("category_type") in TEXT_CATS
                 and not d.get("ignore")] if page else []
        if not annos or not boxes:
            unloc["n"] += 1
            unloc["edit"] += s["Edit_num"]
            continue
        gb = union_box([poly_box(a["poly"]) for a in annos])  # merged GT blocks -> their union
        k = classify(s, gb, boxes)
        kinds[k]["n"] += 1
        kinds[k]["edit"] += s["Edit_num"]
        pages_by_kind[k][stem] += s["Edit_num"]
        cat_by_kind[k][annos[0]["category_type"]] += 1

    lay = sum(v["edit"] for k, v in kinds.items() if k in LAYOUT)
    rec = sum(v["edit"] for k, v in kinds.items() if k.startswith("RECOG"))
    print(f"\n== Geometric attribution of the {tot_edit} edits (edit-weighted)\n")
    print(f"{'cause':16} {'samples':>7} {'edits':>8}   contribution   share")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]["edit"]):
        side = "(a) layout" if k in LAYOUT else "(b) recog" if k.startswith("RECOG") else ""
        print(f"{k:16} {v['n']:7d} {v['edit']:8d}   {v['edit'] / tot_upper:.4f}        "
              f"{100 * v['edit'] / tot_edit:5.1f}%  {side}")
    print(f"{'unlocatable':16} {unloc['n']:7d} {unloc['edit']:8d}   {unloc['edit'] / tot_upper:.4f}        "
          f"{100 * unloc['edit'] / tot_edit:5.1f}%  (GT order not resolvable -- not guessed)")
    print(f"\n  (a) LAYOUT      {lay:8d} edits   {lay / tot_upper:.4f} of {overall:.4f}   {100 * lay / tot_edit:5.1f}% of all edits")
    print(f"  (b) RECOGNITION {rec:8d} edits   {rec / tot_upper:.4f} of {overall:.4f}   {100 * rec / tot_edit:5.1f}% of all edits")
    if rec:
        print(f"  -> layout : recognition = {lay / rec:.2f} : 1")

    print("\n  top GT category per cause:")
    for k, c in sorted(cat_by_kind.items(), key=lambda kv: -kinds[kv[0]]["edit"]):
        print(f"    {k:16} {', '.join(f'{cat}:{n}' for cat, n in c.most_common(3))}")
    print("\n  worst pages, layout causes:")
    agg = Counter()
    for k in LAYOUT:
        agg.update(pages_by_kind[k])
    for stem, e in agg.most_common(8):
        print(f"    {e:6d} edits  {stem}")

    # --- 3. Is the uncredited text LOST, or just in a differently-split block? --------------
    # Because SPURIOUS is structurally 0, a GT block can be charged a full-length edit while the text
    # sits in our markdown one block over. Conservative test: is the GT block's normalized text a
    # literal substring of the page's whole normalized prediction? A hit means the pipeline DID read
    # those characters and the metric did not credit them. Substring = strict lower bound.
    sys.path.insert(0, str(HERE.parent / "OmniDocBench"))
    from utils.data_preprocess import clean_string

    pred_dir = HERE / f"preds/{tag}"
    if not pred_dir.exists():
        print(f"\n(skip: {pred_dir} missing)")
        return
    cache = {}

    def norm_page(img):
        stem = img.rsplit(".", 1)[0]
        if stem not in cache:
            f = pred_dir / f"{stem}.md"
            cache[stem] = clean_string(f.read_text()) if f.exists() else None
        return cache[stem]

    print("\n== Of the edits blamed on LAYOUT, how much text is nonetheless IN our markdown?")
    for k in ("LAYOUT_MISS", "LAYOUT_SPLIT", "LAYOUT_PARTIAL"):
        found = tot = f_e = t_e = 0
        for s in samples:
            if s["Edit_num"] == 0 or len(s["norm_gt"]) < 8:
                continue
            img, stem = s["img_id"], s["img_id"].rsplit(".", 1)[0]
            page, boxes = gt_by_img.get(img), our_boxes.get(stem)
            if not page or not boxes:
                continue
            orders = set(s["gt_position"] if isinstance(s["gt_position"], list) else [s["gt_position"]])
            annos = [d for d in page["layout_dets"] if d.get("order") in orders
                     and d.get("category_type") in TEXT_CATS and not d.get("ignore")]
            if not annos or classify(s, union_box([poly_box(a["poly"]) for a in annos]), boxes) != k:
                continue
            p = norm_page(img)
            if p is None:
                continue
            tot += 1
            t_e += s["Edit_num"]
            if s["norm_gt"] in p:
                found += 1
                f_e += s["Edit_num"]
        if tot:
            print(f"    {k:15} {found:4d}/{tot:4d} blocks ({100 * found / tot:4.1f}%) present verbatim -> "
                  f"{f_e:6d}/{t_e:6d} of their edits ({f_e / tot_upper:.4f} of {overall:.4f}) are "
                  f"RECOVERABLE by better block granularity, not misread")


if __name__ == "__main__":
    main()
