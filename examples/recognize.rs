//! PaddleOCR-VL-1.5 recognition stage -- the glue that links mistral.rs.
//!
//! Reads the layout stage's `manifest.json` + crop PNGs (produced by the `paddleocr-layout` bin in
//! this repo), builds ONE mistral.rs engine on the local checkpoint (it auto-detects
//! `PaddleOCRVLForConditionalGeneration` -> the PaddleOCR-VL loader), runs each crop through the
//! VLM with its manifest-resolved task `prompt` (greedy), and writes `results.json` =
//! `[{read_order, class, text}]` next to the manifest.
//!
//! This is the ONLY stage that links an inference engine: the layout and recognition builds stay
//! independent, and the two JSON files are the entire contract between them. Because it depends on
//! mistral.rs (which hosts the PaddleOCR-VL model, upstreamed as EricLBuehler/mistral.rs#2320), it
//! is NOT compiled by this crate (`autoexamples = false` in Cargo.toml). Build it against a
//! mistral.rs checkout that has the PaddleOCR-VL model: add `mistralrs`, `anyhow`, `image`,
//! `serde`, `serde_json`, and `tokio` to a small binary crate and drop this file in as `main.rs`,
//! or copy it into `mistralrs/examples/`. See README "Quick start".
//!
//! Run: `PADDLEOCR_VL_WEIGHTS=<checkpoint_dir> <this_binary> <manifest_dir>`

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use mistralrs::{
    ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use serde::{Deserialize, Serialize};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
/// Cap per-crop generation so a pathological region can't run away; real crops stop on EOS well
/// under this. Fixed cap: raise if a legitimately long table/formula ever truncates.
const MAX_NEW_TOKENS: usize = 2048;

/// One layout task, as emitted by this repo's `assemble::manifest_json`. The `prompt` is already
/// resolved by class in the layout stage, so recognition is a dumb crop+prompt -> text mapper.
#[derive(Debug, Deserialize)]
struct Task {
    read_order: i64,
    class: String,
    prompt: String,
    crop: String,
}

/// One recognized block, the contract `assemble::assemble_markdown` reads back.
#[derive(Debug, Serialize)]
struct Recognized {
    read_order: i64,
    class: String,
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let weights_dir = std::env::var(WEIGHTS_ENV)
        .with_context(|| format!("set {WEIGHTS_ENV} to the local PaddleOCR-VL checkpoint dir"))?;
    let manifest_dir = match std::env::args().nth(1) {
        Some(d) => PathBuf::from(d),
        None => bail!("usage: recognize <manifest_dir>  (dir holding manifest.json)"),
    };
    let manifest_path = manifest_dir.join("manifest.json");

    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let tasks: Vec<Task> =
        serde_json::from_str(&manifest).context("parsing manifest.json into tasks")?;
    println!("loaded {} task(s) from {}", tasks.len(), manifest_path.display());

    // Default: CPU/f32 -- the deterministic parity path. Set PADDLEOCR_VL_GPU=1 to run bf16 on the
    // default accelerator instead (needs a `--features cuda,flash-attn` build); much faster prefill
    // on dense regions. Env toggle, no CLI flag.
    let gpu = std::env::var("PADDLEOCR_VL_GPU").is_ok();
    let mut builder = MultimodalModelBuilder::new(&weights_dir)
        .with_dtype(if gpu { ModelDType::BF16 } else { ModelDType::F32 })
        .with_logging();
    if !gpu {
        builder = builder.with_force_cpu();
    }
    let model = builder.build().await?;

    let mut results = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let crop_path = manifest_dir.join(&task.crop);
        let image = image::open(&crop_path)
            .with_context(|| format!("opening crop {}", crop_path.display()))?;

        let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
            TextMessageRole::User,
            &task.prompt,
            vec![image],
        ))
        .set_sampler_max_len(MAX_NEW_TOKENS);

        let resp = model.send_chat_request(req).await?;
        let text = resp.choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default();

        println!(
            "[{}] {} ({}) -> {text:?}",
            task.read_order, task.crop, task.class
        );
        results.push(Recognized {
            read_order: task.read_order,
            class: task.class.clone(),
            text,
        });
    }

    // Keep reading order stable regardless of manifest ordering.
    results.sort_by_key(|r| r.read_order);

    let results_path = manifest_dir.join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&results_path, format!("{json}\n"))
        .with_context(|| format!("writing {}", results_path.display()))?;
    println!("wrote {} result(s) -> {}", results.len(), results_path.display());

    Ok(())
}
