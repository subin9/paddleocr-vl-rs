#!/usr/bin/env python3
"""Dump an OTSL->HTML parity fixture from the REFERENCE implementation.

A port of someone else's algorithm is only worth anything if it agrees with the original, so we
take every table OTSL string the full OmniDocBench run actually produced, run it through PaddleX's
own `convert_otsl_to_html`, and record (otsl, html). `tests/otsl_html_parity.rs` then demands our
Rust `otsl_to_html` return the SAME html, byte for byte, for every one of them.

Unlike the layout post-processor, there is no resampler difference to isolate here: the input is a
string we already have, so the assert can be exact.

  ./paddle-venv/bin/python dump_otsl_parity.py work_reflayout work/otsl_html_fixture.json

Output lands in gitignored `work/` -- the OTSL strings are model output over OmniDocBench pages
(dataset-derived content), so the fixture is never committed; the test skips when it is absent.
"""
import json
import sys
from pathlib import Path

from paddlex.inference.pipelines.paddleocr_vl.uilts import convert_otsl_to_html

work, out = Path(sys.argv[1]), Path(sys.argv[2])
cases, seen = [], set()
for results in sorted(work.glob("*/results.json")):
    for region in json.loads(results.read_text()):
        if "table" not in str(region.get("class", "")):
            continue
        otsl = region.get("text") or ""
        if not otsl or otsl in seen:
            continue
        seen.add(otsl)
        cases.append({"page": results.parent.name, "otsl": otsl, "html": convert_otsl_to_html(otsl)})

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cases, ensure_ascii=False, indent=1))
spans = sum(any(t in c["otsl"] for t in ("<lcel>", "<ucel>", "<xcel>")) for c in cases)
print(f"{len(cases)} distinct table OTSL strings -> {out} ({spans} carry a span token)")
