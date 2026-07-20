#!/usr/bin/env python3
"""Deterministic stratified subset sampler for OmniDocBench v1.5.

Usage: sample_stratified.py [N] > subsets/subsetN.txt   (default N=150)

Strategy (no RNG -> byte-for-byte reproducible, committable):
  * Proportional allocation by data_source (10 sources), largest-remainder rounding to N,
    with a floor of 1 per non-empty source so tiny strata (historical_document, n=5) appear.
  * Within each source, sort by (language, layout, image_path) and take an evenly-spaced
    *systematic* sample. Systematic sampling over the (language,layout)-sorted list spreads the
    picks across languages and layouts inside every source -> all three attributes get covered
    without a joint stratification blowup.

Prints one GT image_path per line (stdout) + a stratum summary to stderr. Feed the stdout to
make_subset.py. The committed reproducible artifact is this script + the resulting stem list.
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
GT_ALL = os.path.join(HERE, "data", "OmniDocBench.json")


def largest_remainder(counts, total):
    """Allocate `total` across strata proportional to `counts` (dict), floor 1 each, sum==total."""
    keys = list(counts)
    grand = sum(counts.values())
    raw = {k: counts[k] / grand * total for k in keys}
    alloc = {k: max(1, int(raw[k])) for k in keys}      # floor 1 so tiny strata survive
    # fix up to hit `total` exactly, trading on fractional remainder
    diff = total - sum(alloc.values())
    order = sorted(keys, key=lambda k: raw[k] - int(raw[k]), reverse=(diff > 0))
    i = 0
    while diff != 0 and order:
        k = order[i % len(order)]
        if diff > 0:
            alloc[k] += 1; diff -= 1
        elif alloc[k] > 1:                               # never drop a stratum below its floor
            alloc[k] -= 1; diff += 1
        i += 1
        if i > 10000:  # safety
            break
    return alloc


def systematic(items, k):
    """Pick k items evenly spaced across `items` (already sorted). k<=len(items)."""
    n = len(items)
    if k >= n:
        return list(items)
    # centered systematic: indices round((i+0.5)*n/k)
    return [items[min(n - 1, int((i + 0.5) * n / k))] for i in range(k)]


def main(N):
    gt = json.load(open(GT_ALL))
    by_source = collections.defaultdict(list)
    for e in gt:
        by_source[e["page_info"]["page_attribute"]["data_source"]].append(e)
    counts = {s: len(v) for s, v in by_source.items()}
    alloc = largest_remainder(counts, N)

    picked = []
    for src in sorted(by_source):
        entries = sorted(
            by_source[src],
            key=lambda e: (
                e["page_info"]["page_attribute"]["language"],
                e["page_info"]["page_attribute"]["layout"],
                e["page_info"]["image_path"],
            ),
        )
        picked.extend(systematic(entries, alloc[src]))

    for e in picked:
        print(e["page_info"]["image_path"])

    # summary to stderr (does not pollute the stem list on stdout)
    def dist(key):
        c = collections.Counter(e["page_info"]["page_attribute"][key] for e in picked)
        return "  ".join(f"{k}={v}" for k, v in c.most_common())
    print(f"\n[subset] picked {len(picked)} pages (target {N})", file=sys.stderr)
    print(f"[subset] by data_source: {dist('data_source')}", file=sys.stderr)
    print(f"[subset] by language:    {dist('language')}", file=sys.stderr)
    print(f"[subset] by layout:      {dist('layout')}", file=sys.stderr)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
