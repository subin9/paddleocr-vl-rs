//! PaddleOCR-VL recognition stage (1.5 or 1.6) -- the glue that links mistral.rs.
//!
//! 1.6 is a weights-only release: byte-identical `config.json`, tokenizer and preprocessing, and an
//! identical 620-tensor safetensors signature, so it loads through this same path with no code
//! change. Point `PADDLEOCR_VL_WEIGHTS` at either checkout. See the README.
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
//! Run (one page):   `PADDLEOCR_VL_WEIGHTS=<checkpoint_dir> <this_binary> <manifest_dir>`
//! Run (load-once):  `... <this_binary> <dir1> <dir2> ...` or `... <this_binary> --list <file>`
//!
//! Load-once mode is the same engine, the same greedy sampler and the same per-crop loop -- it just
//! amortizes the ~1.9GB checkpoint load (measured 1.76s/page of spawn+load) across every page in
//! one process instead of paying it per page. Outputs are byte-identical to the per-page path; that
//! is enforced by `bench/omnidocbench/loadonce_parity.sh`, not assumed.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use mistralrs::{
    MultimodalMessages, Model, ModelDType, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use serde::{Deserialize, Serialize};

const WEIGHTS_ENV: &str = "PADDLEOCR_VL_WEIGHTS";
/// Cap per-crop generation so a pathological region can't run away; real crops stop on EOS well
/// under this. Fixed cap: raise if a legitimately long table/formula ever truncates.
const MAX_NEW_TOKENS: usize = 2048;

/// Per-region wall-clock guard on top of the token cap: a region that blows this budget records
/// empty text and the run continues instead of hanging. Override via PADDLEOCR_VL_REGION_TIMEOUT_SECS.
const REGION_TIMEOUT_SECS: u64 = 120;

/// The page dir + start time of the in-flight region, for the watchdog thread (`None` = idle).
static IN_FLIGHT: Mutex<Option<(PathBuf, Instant)>> = Mutex::new(None);

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

/// Page dirs to run: `<dir>...` or `--list <file>` (one dir per line, blanks/`#` ignored).
fn page_dirs() -> Result<Vec<PathBuf>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((flag, rest)) if flag == "--list" => {
            let list = rest
                .first()
                .context("--list needs a file of page dirs, one per line")?;
            let body = std::fs::read_to_string(list)
                .with_context(|| format!("reading page-dir list {list}"))?;
            Ok(body
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(PathBuf::from)
                .collect())
        }
        Some(_) => Ok(args.iter().map(PathBuf::from).collect()),
        None => bail!("usage: recognize <manifest_dir>... | --list <file>"),
    }
}

/// Hard backstop for a wedged engine. The per-region `tokio::time::timeout` below cannot fire if a
/// deadlock blocks the tokio worker itself -- which is why the old harness wrapped every page in an
/// outer `timeout(1)` process kill. Load-once has no per-page process to kill, so the guard moves in
/// here: a plain OS thread (nothing the runtime does can block it) kills the process if a region
/// overruns its soft budget by 2x, leaving a `TIMEOUT_SKIP` marker so the resumable runner steps
/// over that page instead of wedging on it again. Finished pages keep their `results.json`.
fn spawn_watchdog(hard: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(5));
        let in_flight = IN_FLIGHT.lock().unwrap();
        if let Some((dir, started)) = in_flight.as_ref() {
            if started.elapsed() > hard {
                let _ = std::fs::write(dir.join("TIMEOUT_SKIP"), "region exceeded hard watchdog\n");
                eprintln!(
                    "WATCHDOG: region in {} exceeded {}s (engine wedged) -> killing process, page marked TIMEOUT_SKIP",
                    dir.display(),
                    hard.as_secs()
                );
                std::process::exit(124);
            }
        }
    });
}

/// Recognize every crop in one page dir and write its `results.json`. Identical work per page
/// whether it is the only page in the process or one of 1651.
async fn recognize_page(model: &Model, dir: &Path, region_timeout: Duration) -> Result<()> {
    let manifest_path = dir.join("manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let tasks: Vec<Task> =
        serde_json::from_str(&manifest).context("parsing manifest.json into tasks")?;
    println!("loaded {} task(s) from {}", tasks.len(), manifest_path.display());

    let mut results = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let crop_path = dir.join(&task.crop);
        let image =
            image::open(&crop_path).with_context(|| format!("opening crop {}", crop_path.display()))?;

        let req = RequestBuilder::from(MultimodalMessages::new().add_image_message(
            TextMessageRole::User,
            &task.prompt,
            vec![image],
        ))
        .set_sampler_max_len(MAX_NEW_TOKENS);

        *IN_FLIGHT.lock().unwrap() = Some((dir.to_path_buf(), Instant::now()));
        let text = match tokio::time::timeout(region_timeout, model.send_chat_request(req)).await {
            Ok(resp) => resp?.choices[0].message.content.clone().unwrap_or_default(),
            Err(_) => {
                eprintln!(
                    "[{}] {} ({}) -> TIMEOUT after {}s, recording empty text",
                    task.read_order,
                    task.crop,
                    task.class,
                    region_timeout.as_secs()
                );
                String::new()
            }
        };
        *IN_FLIGHT.lock().unwrap() = None;

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

    let results_path = dir.join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&results_path, format!("{json}\n"))
        .with_context(|| format!("writing {}", results_path.display()))?;
    println!("wrote {} result(s) -> {}", results.len(), results_path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let weights_dir = std::env::var(WEIGHTS_ENV)
        .with_context(|| format!("set {WEIGHTS_ENV} to the local PaddleOCR-VL checkpoint dir"))?;
    let dirs = page_dirs()?;

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

    let region_timeout = Duration::from_secs(
        std::env::var("PADDLEOCR_VL_REGION_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(REGION_TIMEOUT_SECS),
    );
    spawn_watchdog(region_timeout * 2);

    // Fault-isolate the page: one unreadable manifest/crop skips that page, it does not abort the
    // other 1650. The runner re-derives what is missing, so a skipped page is simply retried.
    let mut failed = 0;
    for (i, dir) in dirs.iter().enumerate() {
        println!("== page {}/{}: {} ==", i + 1, dirs.len(), dir.display());
        if let Err(e) = recognize_page(&model, dir, region_timeout).await {
            eprintln!("PAGE FAILED -> skip: {} ({e:#})", dir.display());
            failed += 1;
        }
    }
    println!("done: {}/{} page(s) recognized", dirs.len() - failed, dirs.len());
    Ok(())
}
