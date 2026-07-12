#!/usr/bin/env python3
"""Run the OFFICIAL PaddlePaddle PP-DocLayoutV3 on a list of pages and dump its boxes.

Layout probe: the error budget attributes 0.0353 of our 0.0662 text_block edit_whole to the LAYOUT
stage, of which LAYOUT_PARTIAL (our box covers <70% of the GT block) is the single largest cause. Two
explanations, and they demand opposite conclusions:
  - PORT DEFECT: the official model frames the whole GT block and our ONNX port's post-processing
    (score threshold / NMS / unclip / box decode) throws away or shrinks the coverage.
  - MODEL PROFILE: PP-DocLayoutV3 itself frames those blocks partially; the published number must
    then come from a different layout stage or extra merging, and our port is faithful.
The only way to tell them apart is to run the reference. This script IS the reference side: official
paddle weights, official pre/post-processing, official defaults -- nothing of ours in the path.

Two configs, because "the official model" is ambiguous and the difference between the two IS the
answer:
  raw      -- PP-DocLayoutV3 with its OWN inference.yml defaults (score>0.5, no NMS, no box merging).
              This is what our ONNX port reimplements. If our boxes match these, the port is a
              FAITHFUL copy of the raw detector and the fault is not in our decode.
  pipeline -- the SHIPPED reference pipeline `paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml` --
              the one that produces the paper's number: threshold 0.3 (not 0.5), layout_nms, and
              per-class layout_merge_bboxes_mode (union/large). Settings are READ FROM THAT FILE, not
              retyped here, so the config is provably the reference's.
Our port implements `raw` and skips every one of the pipeline's post-processing steps.

Runs in `paddle-venv` (paddlepaddle + paddleocr), NOT the scorer venv. Writes
{"config": ..., "pages": {stem: {"boxes": [[x0,y0,x1,y1], ...], "classes": [...], "scores": [...]}}}

Usage: paddle-venv/bin/python official_layout.py <pages.json> <out.json> [raw|pipeline]
"""
import json
import sys
from pathlib import Path

import yaml
from paddleocr import LayoutDetection

HERE = Path(__file__).parent
IMAGES = HERE / "data/images"
PIPELINE_YAML = (Path(LayoutDetection.__module__ and __import__("paddlex").__file__).parent
                 / "configs/pipelines/PaddleOCR-VL-1.5.yaml")
KNOBS = ("threshold", "layout_nms", "layout_unclip_ratio", "layout_merge_bboxes_mode")


def pipeline_kwargs():
    """The reference pipeline's LayoutDetection settings, read from the shipped yaml."""
    sub = yaml.safe_load(PIPELINE_YAML.read_text())["SubModules"]["LayoutDetection"]
    return {k: sub[k] for k in KNOBS if k in sub}


def spy_on_postprocess():
    """Capture the raw detection array the reference post-processor is HANDED, per page.

    `fixture` mode needs BOTH sides of the post-processing step: its input (the raw [cls, score,
    x0,y0,x1,y1, order] array, pre-round/threshold/NMS/merge) and its output (the reference's final
    boxes). Dumping the input lets the Rust port be tested on the post-processing ALONE -- feed it
    the official detector's own detections, demand the official pipeline's boxes back. That isolates
    the logic under test from the CatmullRom-vs-cv2.INTER_CUBIC resampler difference, which shifts a
    few boxes by a pixel or two and would otherwise make an exact-match assert impossible.
    """
    from paddlex.inference.models.layout_analysis import processors as lap

    seen = {}
    original = lap.LayoutAnalysisProcess.apply

    def spy(self, boxes, img_size, *a, **kw):
        # copy BEFORE apply(): its first act is an in-place np.round of the coordinates.
        seen["dets"] = [[float(v) for v in row] for row in boxes]
        seen["img_size"] = [int(v) for v in img_size]  # (w, h)
        return original(self, boxes, img_size, *a, **kw)

    lap.LayoutAnalysisProcess.apply = spy
    return seen


def main():
    pages = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "work/official_layout.json"
    mode = sys.argv[3] if len(sys.argv) > 3 else "raw"

    kwargs = pipeline_kwargs() if mode in ("pipeline", "fixture") else {}
    cfg = {"mode": mode, "source": str(PIPELINE_YAML) if kwargs else "PP-DocLayoutV3/inference.yml",
           **kwargs}
    # `fixture` pins layout_shape_mode=rect: the reference's default `auto` feeds the instance MASKS
    # into filter_boxes' polygon-overlap rescue, and our ONNX port decodes boxes only (masks are
    # `fetch_name_2`, unused). Testing the port against the mask-driven variant would assert a
    # behaviour it cannot have. rect is the same code path with masks off -- an honest target. The
    # cost of that choice is measured separately (see layout_probe.py), not assumed away.
    predict_kwargs = {"layout_shape_mode": "rect"} if mode == "fixture" else {}
    cfg.update(predict_kwargs)
    spy = spy_on_postprocess() if mode == "fixture" else None
    det = LayoutDetection(model_name="PP-DocLayoutV3", **kwargs)
    print("effective official config:", json.dumps(cfg, default=str), flush=True)

    by_stem = {p.stem: p for p in IMAGES.iterdir()}
    out = {}
    for stem in pages:
        img = by_stem.get(stem)
        if img is None:
            print(f"  SKIP (no image): {stem}", flush=True)
            continue
        (res,) = det.predict(str(img), batch_size=1, **predict_kwargs)
        boxes = [[float(x) for x in d["coordinate"]] for d in res["boxes"]]
        out[stem] = {
            "boxes": boxes,
            "classes": [d["label"] for d in res["boxes"]],
            "scores": [float(d["score"]) for d in res["boxes"]],
        }
        if spy is not None:
            out[stem]["raw_dets"] = spy["dets"]
            out[stem]["img_size"] = spy["img_size"]
        print(f"  {stem}: {len(boxes)} regions", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": cfg, "pages": out}, default=str, indent=1))
    print(f"wrote {out_path} ({len(out)} pages)")


if __name__ == "__main__":
    main()
