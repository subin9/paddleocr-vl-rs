//! Layout parity: Rust preprocess+run+decode vs an onnxruntime reference golden.
//!
//! The golden (`golden/doc/meta.json`) is dumped by a Python onnxruntime script with the
//! config.json preprocess (cv2.INTER_CUBIC 800^2 + /255). Rust resizes with CatmullRom (approx.
//! CUBIC), so boxes are NOT raw-tensor-exact -- we assert the same region count, the same class per
//! region in reading order, and per-box max-abs error within a few px. Measured error on the sample
//! page is ~0.05 px; the 2.0 px tolerance leaves margin for resampler drift on other images.
//!
//! The ONNX graph, the dlopened ONNX Runtime .so, and the golden are not shipped in this repo, so
//! this test SKIPS gracefully when they are absent (a fresh `cargo test` must not hard-fail). It
//! runs the real assertion where those artifacts exist locally.

use ort::session::Session;
use paddleocr_vl_rs::{run_layout, set_default_dylib};
use std::path::Path;

const BOX_TOL_PX: f32 = 2.0;

#[test]
fn rust_layout_matches_golden() {
    set_default_dylib();
    let model = std::env::var("PADDLEOCR_LAYOUT_MODEL")
        .unwrap_or_else(|_| "models/PP-DocLayoutV3.onnx".to_string());
    let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    if !Path::new(&model).exists() || !Path::new(&dylib).exists() || !Path::new("golden/doc/meta.json").exists() {
        eprintln!("SKIP: missing {model}, ORT dylib {dylib}, or golden/doc/meta.json");
        return;
    }

    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("golden/doc/meta.json").unwrap()).unwrap();
    let want = golden["regions"].as_array().unwrap();

    let mut session = Session::builder().unwrap().commit_from_file(&model).unwrap();
    let img = image::open("fixtures/doc.png").unwrap().to_rgb8();
    let got = run_layout(&mut session, &img).unwrap();

    assert_eq!(
        got.len(),
        want.len(),
        "region count: got {} want {}",
        got.len(),
        want.len()
    );

    let mut max_box_err = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.class, w["class"].as_str().unwrap(), "region {i} class");
        let wbox = w["bbox"].as_array().unwrap();
        for (k, (gv, wj)) in g.bbox.iter().zip(wbox).enumerate() {
            let wv = wj.as_f64().unwrap() as f32;
            let err = (gv - wv).abs();
            max_box_err = max_box_err.max(err);
            assert!(err <= BOX_TOL_PX, "region {i} bbox[{k}]: got {gv} want {wv} err {err}");
        }
    }
    eprintln!(
        "parity OK: {} regions, classes+order match, max bbox err {max_box_err:.3} px (tol {BOX_TOL_PX})",
        got.len()
    );
}
