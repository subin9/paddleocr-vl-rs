#!/usr/bin/env python3
"""Prove the CDM environment actually renders, before trusting any CDM score.

CDM (metrics/cdm_metric.py:203-210) wraps its whole render+match path in a bare
`except: return {"recall":0,"precision":0,"F1_score":0}`. A missing xelatex or
ImageMagick therefore produces a SILENT 0.0 per formula -- indistinguishable
from "the model got every formula wrong". This smoke test is the guard:

  identical gt/pred  -> F1 must be 1.0   (env works AND matching works)
  different gt/pred  -> F1 must be < 1.0 (it is really comparing, not stubbing)

If the env is broken both come back 0.0 and we FAIL loudly instead of reporting
a fabricated 0.0 CDM. Run from the OmniDocBench clone root.
"""
import sys, tempfile
from metrics.cdm_metric import CDM

# Uses xeCJK/upgreek/amsmath features the real GT formulas hit.
SAME = r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u + \sum_{i=1}^{n} \beta_i x_i"
DIFF = r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u"

with tempfile.TemporaryDirectory() as out:
    cdm = CDM(output_root=out)
    same = cdm.evaluate(SAME, SAME, "same")["F1_score"]
    diff = cdm.evaluate(SAME, DIFF, "diff")["F1_score"]

print(f"identical formula  F1 = {same}   (expect 1.0)")
print(f"truncated  formula F1 = {diff}   (expect 0 < F1 < 1)")

if same == 0.0:
    sys.exit("FAIL: identical formulas scored 0.0 -> CDM env is BROKEN "
             "(xelatex/magick missing); any CDM number would be a fabricated zero.")
if same != 1.0:
    sys.exit(f"FAIL: identical formulas scored {same}, expected 1.0.")
if not (0.0 < diff < 1.0):
    sys.exit(f"FAIL: truncated formula scored {diff}, expected strictly between 0 and 1.")
print("PASS: CDM renders and discriminates -> CDM scores are trustworthy.")
