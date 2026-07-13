#!/usr/bin/env python3
"""Score CDM over the formula pairs the OmniDocBench scorer dumped, on `subset: v1.5`.

The scorer's `CDM_plain` metric does NOT compute CDM. It only writes the (gt, pred) formula pairs to
`result/<name>_display_formula_formula.json` for an external pass. This is that pass -- there is no CDM
column in the scorer's output, and its absence is easy to misread as "CDM ran and found nothing".

Reported both ways, because they are not the same number and the comparison target is the page one:
  * page-avg    -- mean over pages of the mean over that page's formulas. This is the scorer's `page`
                   reduction, the definition BENCHMARKS.md's 91.77 and the published 94.21 use.
  * formula-avg -- mean over formulas. Differs whenever pages carry unequal formula counts.

RUN `cdm_smoke.py` FIRST. CDM wraps its whole render-and-match path in a bare `except: return 0`, so a
broken environment fabricates a 0.0 per formula that is indistinguishable from a model failure.

    cdm_score.py <preds_tag> [<preds_tag> ...]      # tags name result/preds_<tag>_quick_match_*.json

Every rendered F1 is cached under the scratch dir: CDM is one xelatex run per formula.
"""
import json, os, sys, tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "OmniDocBench")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
SP = os.environ.get("CDM_CACHE", tempfile.gettempdir())

GT = json.load(open("../omnidocbench/data/OmniDocBench.json", encoding="utf-8"))
V15, LANG = set(), {}
for p in GT:
    stem = os.path.basename(p["page_info"]["image_path"])[:-4]
    attr = p["page_info"]["page_attribute"]
    if attr.get("subset") == "v1.5":
        V15.add(stem)
        LANG[stem] = attr.get("language")


def work(a):
    i, gt, pred = a
    from metrics.cdm_metric import CDM
    global _C
    try:
        _C
    except NameError:
        _C = CDM(output_root=tempfile.mkdtemp())
    try:
        return _C.evaluate(gt, pred, f"p{os.getpid()}_{i}")["F1_score"]
    except Exception:
        return 0.0


def score(tag):
    cache = f"{SP}/cdmcache_{tag}.json"
    rows = json.load(open(f"result/preds_{tag}_quick_match_display_formula_formula.json",
                          encoding="utf-8"))
    keep = [r for r in rows if r.get("image_name", "")[:-4] in V15]
    if os.path.exists(cache):
        f1 = json.load(open(cache))
    else:
        with ProcessPoolExecutor(max_workers=12) as ex:
            f1 = list(ex.map(work, [(i, r["gt"], r["pred"]) for i, r in enumerate(keep)], chunksize=8))
        json.dump(f1, open(cache, "w"))

    per_page, per_lang = defaultdict(list), defaultdict(list)
    for r, v in zip(keep, f1):
        stem = r["image_name"][:-4]
        per_page[stem].append(v)
    page_means = {p: sum(v) / len(v) for p, v in per_page.items()}
    for p, m in page_means.items():
        per_lang[LANG.get(p)].append(m)

    page_avg = sum(page_means.values()) / len(page_means)
    sample_avg = sum(f1) / len(f1)
    langs = {k: (sum(v) / len(v), len(v)) for k, v in sorted(per_lang.items(), key=lambda x: -len(x[1]))}
    return page_avg, sample_avg, len(page_means), len(f1), langs


if __name__ == "__main__":
    print(f"{'set':<16}{'CDM page-avg':>14}{'CDM formula-avg':>18}{'pages':>7}{'formulas':>10}")
    res = {}
    for tag in sys.argv[1:]:
        pa, sa, np_, nf, langs = score(tag)
        res[tag] = (pa, langs)
        print(f"{tag:<16}{pa:>14.4f}{sa:>18.4f}{np_:>7}{nf:>10}", flush=True)
    print("\nby language (page-avg), the split BENCHMARKS.md attributes the −2.44 to:")
    for tag, (pa, langs) in res.items():
        s = "  ".join(f"{k}: {v:.4f} ({n}pg)" for k, (v, n) in langs.items() if k)
        print(f"  {tag:<16}{s}")
