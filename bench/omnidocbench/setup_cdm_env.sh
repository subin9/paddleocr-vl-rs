#!/usr/bin/env bash
# Stand up the CDM (Character Detection Matching) environment for formula scoring,
# then PROVE it works before any score is trusted.
#
# WHY: the paper reports formulas with CDM (94.21). Our 0.2490 is edit-distance --
# a different metric on the same task, NOT comparable. The official scorer already
# implements CDM (METRIC_REGISTRY: [...'CDM','CDM_plain']) but ships it OFF:
# configs/end2end.yaml uses CDM_plain, commented "CDM can be calculated directly by
# calling CDM in config file if you have CDM environment". So this is a sanctioned
# config switch -- the scorer is NOT modified, we only build the toolchain it
# shells out to.
#
# CDM renders gt and pred LaTeX to PDF (xelatex), rasterises them (ImageMagick +
# ghostscript), and character-matches the two IMAGES. Every token is rendered in a
# unique RGB colour and recovered by exact pixel-colour lookup.
#
# READ THIS BEFORE CHANGING ANYTHING: metrics/cdm_metric.py:203-210 wraps the whole
# render+match path in a bare `except: return {"F1_score":0,...}`. A broken toolchain
# therefore scores 0.0 on EVERY formula and aggregates to a plausible-looking
# "CDM ~ 0" -- indistinguishable from "the model got every formula wrong". Step 5 is
# not optional; it is the only thing standing between us and a fabricated number.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# 1. Raster toolchain + ghostscript. (CJK font: OmniDocBench formulas contain CJK and
#    the CDM template loads xeCJK -- a missing font fails the render outright.)
sudo apt-get install -y imagemagick ghostscript fonts-noto-cjk

# 2. xcolor >= 3.0, into TEXMFHOME. CDM's token colouring emits `\mathcolor[RGB]{...}`
#    (latex_processor.token_add_color_RGB), which the distro's xcolor 2.x does not
#    define. The failure is SILENT: under -interaction=nonstopmode an undefined macro
#    does not fail the build -- xelatex drops the colour and emits a valid,
#    correct-looking BLACK pdf. Zero coloured pixels => zero bboxes => F1=0.0 on every
#    formula, with no error anywhere (see the bare `except` in the header).
#    The CTAN zip ships xcolor.sty (3.x) AND xcolor-2022-06-12.sty. BOTH are required:
#    the second is the dated rollback file the kernel demands when a package declares a
#    release newer than the running format. Copying only xcolor.sty is what makes this
#    look like it needs a full upstream TeX Live -- it does not.
#    (The distro's stable kernel is the one to use. Pairing xcolor 3.x with the
#    latex-dev kernel, /usr/bin/xelatex-dev, breaks instead: l3keys ".code is unknown".)
if ! kpsewhich xcolor-2022-06-12.sty >/dev/null 2>&1; then
  TMP=$(mktemp -d)
  curl -sSL -o "$TMP/xcolor.zip" https://mirrors.ctan.org/macros/latex/contrib/xcolor.zip
  ( cd "$TMP" && unzip -qo xcolor.zip && cd xcolor && latex -interaction=nonstopmode xcolor.ins >/dev/null )
  XCDIR="$(kpsewhich -var-value TEXMFHOME)/tex/latex/xcolor"
  mkdir -p "$XCDIR"
  cp "$TMP/xcolor/xcolor.sty" "$TMP/xcolor/xcolor-2022-06-12.sty" "$XCDIR/"
  rm -rf "$TMP"
fi

# 3. `magick` shim. latex2bbox_color.py:99 calls `magick`, the ImageMagick 7 CLI;
#    Ubuntu 22.04 ships ImageMagick 6, whose binary is `convert` (identical flags for
#    this invocation). CDM's README builds IM7 from source; the shim is enough -- what
#    CDM needs is exact RGB preservation, which step 5 verifies end-to-end.
if ! command -v magick >/dev/null; then
  printf '#!/bin/sh\nexec /usr/bin/convert "$@"\n' | sudo tee /usr/local/bin/magick >/dev/null
  sudo chmod +x /usr/local/bin/magick
fi

# 4. Allow ImageMagick to READ PDF. Debian/Ubuntu ship a policy.xml denying the PDF
#    coder outright (CVE-2016-3714 era hardening); with it in place every rasterisation
#    silently fails. We rasterise only PDFs we generated ourselves, seconds earlier.
POLICY=/etc/ImageMagick-6/policy.xml
if [ -f "$POLICY" ] && grep -q 'domain="coder" rights="none" pattern="PDF"' "$POLICY"; then
  sudo cp "$POLICY" "$POLICY.bak"
  sudo sed -i 's|<policy domain="coder" rights="none" pattern="PDF" />|<policy domain="coder" rights="read\|write" pattern="PDF" />|' "$POLICY"
fi

# 5. Python deps CDM needs on top of the scorer venv (metrics/cdm/requirements.txt).
uv pip install --python "$HERE/scorer-venv/bin/python" "scikit-image<=0.20.0" matplotlib

# 6. The scoring config: same GT, same preds, same quick_match as every other run --
#    the ONLY delta is CDM alongside the edit proxy on display_formula.
#    (Lives in the gitignored dataset dir like every other end2end config, so it is
#    written here rather than committed, keeping the repro self-contained.)
cat > "$HERE/data/subsets/cdm1651.end2end.yaml" <<YAML
{
  "end2end_eval": {
    "metrics": {
      "display_formula": { "metric": ["Edit_dist", "CDM"] }
    },
    "dataset": {
      "dataset_name": "end2end_dataset",
      "ground_truth": { "data_path": "$HERE/data/OmniDocBench.json" },
      "prediction":   { "data_path": "$HERE/preds/otslhtml1651" },
      "match_method": "quick_match"
    }
  }
}
YAML

# 7. THE GATE. Identical gt/pred must score F1=1.0; a truncated pred must score
#    strictly between 0 and 1 (proves it renders AND discriminates). Non-zero exit
#    here means any CDM number would be fabricated -- do not score, report BLOCKED.
cd "$REPO/bench/OmniDocBench"
PYTHONPATH=. "$HERE/scorer-venv/bin/python" "$HERE/cdm_smoke.py"

echo
echo "CDM env OK. Score with:"
echo "  cd $REPO/bench/OmniDocBench && \\"
echo "    PYTHONPATH=. ../omnidocbench/scorer-venv/bin/python pdf_validation.py -c ../omnidocbench/data/subsets/cdm1651.end2end.yaml"
