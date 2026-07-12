//! Reference-layout parity: the shipped `ref_postprocess()` vs PaddleX's own post-processing.
//!
//! The layout probe found the port defect: our ONNX stage reproduces PP-DocLayoutV3's RAW
//! detector faithfully, but the paper's number comes from the reference PIPELINE, which runs the
//! same weights through post-processing we had none of (threshold 0.3, `layout_nms`, per-class
//! `layout_merge_bboxes_mode`, `filter_overlap_boxes`). `ref_postprocess()` ports that. A port of
//! someone else's algorithm is only worth anything if it agrees with the original, so: run the
//! OFFICIAL model, capture the exact array its post-processor is handed, feed that same array to
//! ours, and demand the official's final boxes back -- class for class, pixel for pixel.
//!
//! Feeding it the OFFICIAL detections (not our ONNX ones) is what makes an exact assert possible:
//! our resampler is CatmullRom where cv2 uses INTER_CUBIC, which moves boxes a pixel or two. That
//! difference is real, measured (mean GT coverage 0.750 vs the raw detector's 0.739, symmetric), and
//! deliberately OUT of scope here -- this test is about the post-processing logic alone.
//!
//! Fixture: `bench/omnidocbench/official_layout.py <pages> <out> fixture` (needs `paddle-venv`),
//! written to gitignored `work/`, so this test SKIPS when it is absent. The unit tests in
//! `src/lib.rs` cover the predicate edge cases unconditionally.
//!
//! Caveat recorded, not hidden: the fixture pins `layout_shape_mode="rect"`. The reference's default
//! (`auto`) feeds the model's instance MASKS into `filter_boxes`, which can rescue a box the
//! rectangle test would drop. Our ONNX stage decodes boxes only (masks are `fetch_name_2`, unused),
//! so the mask-driven variant is a behaviour it cannot have; `rect` is the same code path with masks
//! off. The cost of that choice is measured separately by `layout_probe.py`, not assumed to be zero.

use paddleocr_vl_rs::{ref_postprocess, Region, LABEL_LIST};
use std::path::Path;

const FIXTURE: &str = "bench/omnidocbench/work/layout_postproc_fixture.json";

#[test]
fn ref_postprocess_matches_paddlex_on_official_detections() {
    if !Path::new(FIXTURE).exists() {
        eprintln!("SKIP: {FIXTURE} absent (run official_layout.py ... fixture)");
        return;
    }
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();
    let pages = fixture["pages"].as_object().unwrap();
    assert!(!pages.is_empty(), "fixture has no pages");

    let (mut n_raw, mut n_kept) = (0usize, 0usize);
    for (stem, page) in pages {
        let raw = page["raw_dets"].as_array().unwrap();
        let size = page["img_size"].as_array().unwrap();
        let (w, h) = (size[0].as_f64().unwrap() as f32, size[1].as_f64().unwrap() as f32);

        let mut regions: Vec<Region> = raw
            .iter()
            .map(|d| {
                let v: Vec<f32> = d.as_array().unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                let label = v[0].max(0.0) as usize;
                Region {
                    class: LABEL_LIST[label].to_string(),
                    label,
                    score: v[1],
                    bbox: [v[2], v[3], v[4], v[5]],
                    read_order: v[6] as i64,
                }
            })
            .collect();
        n_raw += regions.len();
        ref_postprocess(&mut regions, w, h);
        n_kept += regions.len();

        let want_boxes = page["boxes"].as_array().unwrap();
        let want_classes = page["classes"].as_array().unwrap();
        let got: Vec<(String, [i64; 4])> = regions
            .iter()
            .map(|r| (r.class.clone(), r.bbox.map(|v| v as i64)))
            .collect();
        let want: Vec<(String, [i64; 4])> = want_classes
            .iter()
            .zip(want_boxes)
            .map(|(c, b)| {
                let v = b.as_array().unwrap();
                let mut bb = [0i64; 4];
                for (k, slot) in bb.iter_mut().enumerate() {
                    *slot = v[k].as_f64().unwrap() as i64;
                }
                (c.as_str().unwrap().to_string(), bb)
            })
            .collect();
        assert_eq!(
            got, want,
            "{stem}: {} regions kept, reference kept {}",
            got.len(),
            want.len()
        );
    }
    eprintln!(
        "ref_postprocess == PaddleX on {} pages: {n_raw} raw detections -> {n_kept} regions",
        pages.len()
    );
}
