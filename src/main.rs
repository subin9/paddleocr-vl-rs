//! Layout stage-1 producer and stage-3 assembler.
//!
//! Stage 1 (default): load the PP-DocLayoutV3 ONNX graph through `ort`, preprocess the input image
//! with the config.json recipe, run it, decode the `Vec<Region>` in reading order, then write one
//! crop PNG per region plus `manifest.json` -- the data bridge the recognition stage consumes.
//! Layout is ONNX-only by design: no RT-DETR / deformable attention / backbone here.
//!
//! Stage 3: `assemble <results.json>` reads the recognition stage's output and prints the
//! reassembled reading-order markdown. No ONNX or engine touched -- pure data.
//!
//! Run: `paddleocr-layout <image> <out_dir>` (stage 1) or `paddleocr-layout assemble <results.json>`
//! (stage 3). Set `PADDLEOCR_LAYOUT_MODEL` to override the ONNX path (default `models/PP-DocLayoutV3.onnx`).

use ort::session::Session;
use paddleocr_vl_rs::assemble::{
    assemble_markdown, crop_region, manifest_json, plan_tasks, read_results,
};
use paddleocr_vl_rs::{run_layout, set_default_dylib};
use std::path::Path;

const DEFAULT_MODEL: &str = "models/PP-DocLayoutV3.onnx";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_else(|| "fixtures/doc.png".to_string());

    // Stage 3: `assemble <results.json>` reads the recognition stage's output and prints the
    // reassembled markdown doc.
    if first == "assemble" {
        let results = args.next().unwrap_or_else(|| "out/results.json".to_string());
        let blocks = read_results(&std::fs::read_to_string(&results)?)?;
        println!("{}", assemble_markdown(&blocks));
        return Ok(());
    }

    // Stage 1 (default): run layout on <image> and write crops + manifest to <out_dir>.
    let fixture = first;
    let out_dir = args.next().unwrap_or_else(|| "out".to_string());
    let model = std::env::var("PADDLEOCR_LAYOUT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    set_default_dylib();
    let mut session = Session::builder()?.commit_from_file(&model)?;
    let img = image::open(&fixture)?.to_rgb8();

    let regions = run_layout(&mut session, &img)?;
    println!("{} region(s) in reading order:", regions.len());
    for r in &regions {
        println!(
            "  read_order={:3}  {:16} score={:.3}  bbox={:?}",
            r.read_order, r.class, r.score, r.bbox
        );
    }

    // Stage-1 -> stage-2 handoff: one crop per region + a manifest carrying the resolved prompt.
    let out = Path::new(&out_dir);
    std::fs::create_dir_all(out)?;
    let tasks = plan_tasks(&regions);
    for (task, region) in tasks.iter().zip(&regions) {
        crop_region(&img, &region.bbox).save(out.join(&task.crop))?;
    }
    std::fs::write(out.join("manifest.json"), manifest_json(&tasks))?;
    println!("wrote {} crop(s) + manifest.json to {}/", tasks.len(), out_dir);
    Ok(())
}
