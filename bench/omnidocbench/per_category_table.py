#!/usr/bin/env python3
"""Pull the per-`data_source` edit-distance rows out of an official-scorer log and emit markdown.

The scorer prints its per-attribute breakdown as a tabulate block per metric; the machine-readable
`*_metric_result.json` carries an EMPTY `group` dict, so the log IS the only source for these.
Usage:  per_category_table.py <metric> <run.scorer.log> [<run2.scorer.log> ...]
        metric in {text_block, display_formula, table, reading_order}
Reused for every backend (baseline / reflayout / llama.cpp) so the table is never hand-typed.
"""
import re
import sys

# arXiv 2603.24326 Table 6, PaddleOCR-VL-L, **OCR-block** task (GT layout given, model only reads
# text). NOT our end-to-end task -- see docs/BENCHMARKS.md for the caveat. Reference only.
PUBLISHED_OCR_BLOCK = {
    "PPT2PDF": 0.049, "academic_literature": 0.021, "book": 0.047,
    "colorful_textbook": 0.082, "exam_paper": 0.115, "magazine": 0.020,
    "newspaper": 0.035, "note": 0.077, "research_report": 0.031,
}


def parse(path, metric):
    """-> (ALL, {category: edit_dist}) for the given metric's attribute block."""
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    # The attribute block is the SECOND `Edit_dist:` under the metric header (the first is the
    # ALL_page_avg summary); take everything from the header to the next metric header.
    block = re.split(r"【", text)
    block = next(b for b in block if b.startswith(metric + "】"))
    rows = dict(re.findall(r"^(?:data_source: )?(\S+) +([\d.]+)$", block, re.M))
    cats = {k: float(v) for k, v in rows.items() if k in PUBLISHED_OCR_BLOCK}
    return float(rows["ALL"]), cats


def main():
    metric, logs = sys.argv[1], sys.argv[2:]
    runs = [(l.split("/")[-1].replace(".scorer.log", ""), *parse(l, metric)) for l in logs]
    names = [r[0] for r in runs]
    # The two tasks sit on DIFFERENT scales (end-to-end page_avg vs OCR-block), so absolute
    # differences are meaningless. Normalising each column by its own mean compares the *shape* of
    # the difficulty profile, which IS comparable: a positive delta = a category where our
    # end-to-end pipeline is disproportionately worse than the model is at just reading the text.
    last = runs[-1]
    pub_mean = sum(PUBLISHED_OCR_BLOCK.values()) / len(PUBLISHED_OCR_BLOCK)
    our_mean = sum(last[2][c] for c in PUBLISHED_OCR_BLOCK) / len(PUBLISHED_OCR_BLOCK)
    print(f"| category | {' | '.join(names)} | published (OCR-block) | rel. delta |")
    print("|---|" + "---|" * (len(names) + 2))
    for cat, pub in sorted(PUBLISHED_OCR_BLOCK.items(), key=lambda kv: -(
            last[2][kv[0]] / our_mean - kv[1] / pub_mean)):
        vals = " | ".join(f"{r[2].get(cat, float('nan')):.4f}" for r in runs)
        rel = last[2][cat] / our_mean - pub / pub_mean
        print(f"| {cat} | {vals} | {pub:.3f} | {rel:+.2f} |")
    print(f"| **ALL (page_avg)** | {' | '.join(f'**{r[1]:.4f}**' for r in runs)} | (see note) | |")


if __name__ == "__main__":
    main()
