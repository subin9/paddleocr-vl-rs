#!/usr/bin/env python3
"""Divergence analysis: how much recognized text does the assembler's visual-only skip drop,
and do the pages it drops from score worse?

Reads the recognition stage's own results.json (work/<stem>/results.json) -- no GPU, no re-run --
and cross-references the official scorer's per-page edit distances. Correlational only; the causal
number comes from the PADDLEOCR_VL_KEEP_VISUAL=1 re-assembly + re-score A/B.

Usage: python3 analyze_visual_drop.py            (from bench/omnidocbench/)
"""
import json
import os
import statistics as st
from collections import Counter

VISUAL_ONLY = {"chart", "image", "header_image", "footer_image", "seal"}  # assemble.rs
EDIT = json.load(open("results/full1651_quick_match_text_block_per_page_edit.json"))
RO = json.load(open("results/full1651_quick_match_reading_order_per_page_edit.json"))


def page_key(stem):
    """work/ dir stem -> scorer's per-page key (the GT image filename)."""
    return next((stem + e for e in (".png", ".jpg", ".jpeg") if stem + e in EDIT), None)


rows, cls_n, cls_chars = [], Counter(), Counter()
for stem in os.listdir("work"):
    path = f"work/{stem}/results.json"
    if not os.path.exists(path):
        continue
    try:
        res = json.load(open(path))
    except json.JSONDecodeError:  # truncated results.json from a killed page
        continue
    dropped = [r for r in res if r["class"] in VISUAL_ONLY]
    for r in dropped:
        cls_n[r["class"]] += 1
        cls_chars[r["class"]] += len(r["text"].strip())
    rows.append(
        {
            "stem": stem,
            "n_dropped": len(dropped),
            "chars_dropped": sum(len(r["text"].strip()) for r in dropped),
            "chars_kept": sum(len(r["text"].strip()) for r in res if r["class"] not in VISUAL_ONLY),
        }
    )

lost = [r for r in rows if r["chars_dropped"] > 0]
none = [r for r in rows if r["chars_dropped"] == 0]
d = sum(r["chars_dropped"] for r in rows)
k = sum(r["chars_kept"] for r in rows)
print(f"pages with results.json           : {len(rows)}")
print(f"pages with >=1 visual-only region : {sum(r['n_dropped'] > 0 for r in rows)}")
print(f"pages that LOST text to the skip  : {len(lost)} ({100 * len(lost) / len(rows):.1f}%)")
print(f"chars dropped / kept              : {d:,} / {k:,}  ({100 * d / (d + k):.2f}% of recognized text)")
for c in cls_n:
    print(f"    {c:14s} regions={cls_n[c]:5d}  chars={cls_chars[c]:8,d}")

for name, metric in (("text_block", EDIT), ("reading_order", RO)):
    print(f"\n{name} Edit_dist        n   median     mean")
    for label, group in (("pages that lost text", lost), ("pages with no loss  ", none)):
        v = [metric[page_key(r["stem"])] for r in group if page_key(r["stem"]) in metric]
        print(f"  {label}  {len(v):5d}   {st.median(v):.4f}   {sum(v) / len(v):.4f}")

print("\ntop-10 pages by dropped chars (dropped / kept / text-edit):")
for r in sorted(lost, key=lambda r: -r["chars_dropped"])[:10]:
    e = EDIT.get(page_key(r["stem"]) or "", float("nan"))
    print(f"  {r['stem'][:52]:52s} {r['chars_dropped']:6d} / {r['chars_kept']:6d}  edit={e:.4f}")
