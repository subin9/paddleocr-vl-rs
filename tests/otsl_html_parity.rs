//! OTSL->HTML parity: the shipped table renderer vs PaddleX's own `convert_otsl_to_html`.
//!
//! The §2.3 table-gap diagnosis found the defect: the VLM emits OTSL with span markers
//! (`<lcel>`/`<ucel>`/`<xcel>` on 34% of the run's tables), but the assembler flattened every one of
//! them to a plain cell and rendered a GitHub pipe-table -- a format that cannot express a merged
//! cell. The benchmark scores tables with TEDS, which compares exactly that cell tree.
//!
//! So the renderer now ports the reference's `convert_otsl_to_html` (PaddleX
//! `pipelines/paddleocr_vl/uilts.py`), and a port is only worth anything if it agrees with the
//! original: this feeds every table OTSL string the full run actually produced through BOTH and
//! demands the same HTML back, byte for byte. No resampler caveat here, unlike the layout parity
//! test -- the input is a string we already have, so the assert is exact.
//!
//! Fixture: `./paddle-venv/bin/python bench/omnidocbench/dump_otsl_parity.py work_reflayout \
//!   work/otsl_html_fixture.json` (needs `paddle-venv`), written to gitignored `work/` because the
//! OTSL is model output over dataset pages. This test SKIPS when it is absent; the unit tests in
//! `src/assemble.rs` cover the span/ragged/escape cases unconditionally.

use paddleocr_vl_rs::assemble::assemble_markdown;
use std::path::Path;

const FIXTURE: &str = "bench/omnidocbench/work/otsl_html_fixture.json";

#[test]
fn rust_otsl_to_html_matches_paddlex() {
    let Ok(raw) = std::fs::read_to_string(Path::new(FIXTURE)) else {
        eprintln!("SKIP: no fixture at {FIXTURE} (see the module docs to regenerate)");
        return;
    };
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("fixture is json");
    assert!(!cases.is_empty(), "fixture is empty");

    let (mut spans, mut mismatches) = (0usize, Vec::new());
    for case in &cases {
        let (page, otsl, want) = (
            case["page"].as_str().unwrap(),
            case["otsl"].as_str().unwrap(),
            case["html"].as_str().unwrap(),
        );
        if ["<lcel>", "<ucel>", "<xcel>"].iter().any(|t| otsl.contains(t)) {
            spans += 1;
        }
        // through the real assembler, not a private helper: a `table` block IS this rendering.
        let got = assemble_markdown(&[("table".to_string(), otsl.to_string())]);
        if got != want {
            mismatches.push(format!("{page}\n  want: {want}\n   got: {got}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{}/{} tables disagree with PaddleX:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
    eprintln!("{} tables byte-identical to PaddleX ({spans} carry a span)", cases.len());
}
