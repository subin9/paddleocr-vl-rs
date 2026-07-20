#!/usr/bin/env python3
"""Dump the corpus's real layout geometry + the Python filter's kept-set, for the Rust parity test.

The scored nonest A/B (text-edit 0.0797 -> 0.0725) was produced by `filter_nested.py`, which drops
rows AFTER recognition. The shipped fix drops regions BEFORE cropping, inside `run_layout`. Those two
only yield the same markdown if the Rust `drop_nested()` keeps exactly the same regions as the Python
`nested_indices()` on real data -- otherwise the scored numbers do not transfer to the pipeline.

So: reuse filter_nested's OWN parser and predicate (imported, not re-implemented) to emit, per page,
the region boxes plus the indices Python keeps. `tests/parity_nested.rs` feeds the same boxes to the
shipped Rust function and asserts the kept sets are identical. Zero ONNX, zero GPU.

Writes to work/ (gitignored -- derived from run logs, which echo dataset text).
Usage: dump_nested_parity.py [out.json]
"""
import json
import sys
from pathlib import Path

from filter_nested import nested_indices, parse_logs

HERE = Path(__file__).parent


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "work/nested_parity.json")
    pages = parse_logs()
    dump, n_regions, n_dropped = {}, 0, 0
    for stem, regions in sorted(pages.items()):
        # absorber_aware: matches the shipped Rust guard (a VISUAL_ONLY parent absorbs nothing, so it
        # cannot make a child a duplicate). The predicate is class-sensitive, so dump the class too.
        drop = nested_indices(regions, 0.8, absorber_aware=True)
        dump[stem] = {
            "boxes": [r[2] for r in regions],
            "classes": [r[1] for r in regions],
            "read_order": [r[0] for r in regions],
            "keep": [i for i in range(len(regions)) if i not in drop],
        }
        n_regions += len(regions)
        n_dropped += len(drop)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump))
    print(f"{len(dump)} pages, {n_regions} regions, {n_dropped} dropped by python -> {out}")


if __name__ == "__main__":
    main()
