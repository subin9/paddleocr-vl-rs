//! Nested-drop parity: the shipped `drop_nested()` vs the Python filter that produced the scored A/B.
//!
//! `bench/omnidocbench/filter_nested.py` priced the nested-sub-region duplication by dropping rows
//! after recognition (text-edit page_avg 0.0797 -> 0.0725, official scorer, 1649 pages). The fix
//! ships in `run_layout`, dropping regions before cropping. Those numbers only transfer to the
//! pipeline if both keep the SAME regions on real data -- so assert exactly that, over every region
//! of the full 1651-page run (~39k boxes), not a synthetic sample.
//!
//! The fixture is dumped by `bench/omnidocbench/dump_nested_parity.py` into gitignored `work/`
//! (it is derived from run logs, which echo dataset text), so this test SKIPS when it is absent.
//! The unit tests in `src/lib.rs` cover the predicate's edge cases unconditionally.

use paddleocr_vl_rs::{drop_nested, Region};
use std::path::Path;

const FIXTURE: &str = "bench/omnidocbench/work/nested_parity.json";

#[test]
fn drop_nested_matches_python_filter_on_full_corpus() {
    if !Path::new(FIXTURE).exists() {
        eprintln!("SKIP: {FIXTURE} absent (run bench/omnidocbench/dump_nested_parity.py)");
        return;
    }
    let pages: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).unwrap()).unwrap();

    let (mut n_regions, mut n_dropped) = (0usize, 0usize);
    for (stem, page) in &pages {
        let boxes = page["boxes"].as_array().unwrap();
        let orders = page["read_order"].as_array().unwrap();
        let mut regions: Vec<Region> = boxes
            .iter()
            .zip(orders)
            .map(|(b, o)| {
                let b = b.as_array().unwrap();
                Region {
                    class: "text".into(),
                    label: 0,
                    score: 0.9,
                    bbox: std::array::from_fn(|i| b[i].as_f64().unwrap() as f32),
                    read_order: o.as_i64().unwrap(),
                }
            })
            .collect();
        n_regions += regions.len();
        drop_nested(&mut regions);

        // Compare by original index: keep[] holds the indices python kept, in order.
        let want: Vec<usize> = page["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect();
        n_dropped += boxes.len() - want.len();
        let got: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox).collect();
        let want_boxes: Vec<[f32; 4]> = want
            .iter()
            .map(|&i| {
                let b = boxes[i].as_array().unwrap();
                std::array::from_fn(|k| b[k].as_f64().unwrap() as f32)
            })
            .collect();
        assert_eq!(got, want_boxes, "kept-set differs on page {stem}");
    }
    assert!(n_regions > 30_000, "fixture too small: {n_regions} regions");
    eprintln!("parity OK: {} pages, {n_regions} regions, {n_dropped} dropped", pages.len());
}
