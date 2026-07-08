//! Glue: turn layout `Region`s into recognition tasks and reassemble the per-region recognition
//! output into one markdown document.
//!
//! Pure functions only -- no VLM/engine dependency. A driver crops each region, picks its task
//! prompt via [`task_prompt`], runs the crop through the PaddleOCR-VL VLM (via mistral.rs; see the
//! repo's recognize example), then formats the collected results with [`assemble_markdown`].
//! Keeping this VLM-free keeps the crate independent of any inference engine: the two stages talk
//! only through the manifest.json / results.json contract.

use crate::Region;
use image::RgbImage;

/// Map a layout class to its PaddleOCR-VL task prompt (exact strings the model was trained on).
/// Text-like classes (and anything unrecognized) default to `OCR:`.
pub fn task_prompt(class: &str) -> &'static str {
    match class {
        "table" => "Table Recognition:",
        "display_formula" | "inline_formula" => "Formula Recognition:",
        "chart" => "Chart Recognition:",
        "seal" => "Seal Recognition:",
        // figure/image regions also fall through to OCR: for now -- a photo yields junk text, but
        // captions/labels are worth OCR'ing; add a skip-list when it measurably hurts.
        _ => "OCR:",
    }
}

/// Crop the original-pixel bbox `[x0, y0, x1, y1]` out of the source image, clamped to bounds.
/// Always returns at least a 1x1 image (degenerate/off-image boxes never panic).
pub fn crop_region(img: &RgbImage, bbox: &[f32; 4]) -> RgbImage {
    let (iw, ih) = (img.width(), img.height());
    let x0 = bbox[0].min(bbox[2]).clamp(0.0, iw as f32).round() as u32;
    let y0 = bbox[1].min(bbox[3]).clamp(0.0, ih as f32).round() as u32;
    let x1 = bbox[0].max(bbox[2]).clamp(0.0, iw as f32).round() as u32;
    let y1 = bbox[1].max(bbox[3]).clamp(0.0, ih as f32).round() as u32;
    let x = x0.min(iw.saturating_sub(1));
    let y = y0.min(ih.saturating_sub(1));
    let w = (x1 - x0).clamp(1, iw - x);
    let h = (y1 - y0).clamp(1, ih - y);
    image::imageops::crop_imm(img, x, y, w, h).to_image()
}

/// Reassemble `(class, recognized_text)` blocks (already in reading order) into one markdown doc.
/// Title classes get heading prefixes; table/formula/chart text is emitted verbatim (it is already
/// markup/LaTeX). Empty results are skipped. Blocks are separated by a blank line.
pub fn assemble_markdown(blocks: &[(String, String)]) -> String {
    let mut out = Vec::new();
    for (class, text) in blocks {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        out.push(match class.as_str() {
            "doc_title" => format!("# {text}"),
            "paragraph_title" | "figure_title" => format!("## {text}"),
            "table" => otsl_to_markdown(text),
            _ => text.to_string(),
        });
    }
    out.join("\n\n")
}

/// Render PaddleOCR-VL table output (OTSL: `<fcel>`=cell, `<nl>`=row; merge/empty markers
/// `<ecel>/<lcel>/<ucel>/<xcel>` collapse to a cell boundary) as a GitHub markdown table. Falls
/// back to the raw string if it doesn't parse as a grid.
fn otsl_to_markdown(otsl: &str) -> String {
    let norm = otsl
        .replace("<ecel>", "<fcel>")
        .replace("<lcel>", "<fcel>")
        .replace("<ucel>", "<fcel>")
        .replace("<xcel>", "<fcel>");
    let rows: Vec<Vec<String>> = norm
        .split("<nl>")
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .map(|row| {
            row.split("<fcel>")
                .skip(1) // drop the fragment before the first cell marker
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect();
    let ncol = rows.iter().map(Vec::len).max().unwrap_or(0);
    if ncol == 0 {
        return otsl.to_string();
    }
    let render = |r: &Vec<String>| {
        let mut c = r.clone();
        c.resize(ncol, String::new());
        format!("| {} |", c.join(" | "))
    };
    let mut out = vec![render(&rows[0]), format!("|{}", " --- |".repeat(ncol))];
    out.extend(rows[1..].iter().map(render));
    out.join("\n")
}

/// One recognition task the layout stage hands to the recognition stage: the crop lives at
/// `crop` (relative to the manifest), to be recognized with `prompt`; `class`/`read_order` carry
/// through so the assembly stage can rebuild the markdown. This is the ONLY contract between the
/// two stages -- the recognition stage never needs the class->prompt rules (already resolved here).
#[derive(Debug, Clone, PartialEq)]
pub struct RegionTask {
    pub read_order: i64,
    pub class: String,
    pub prompt: &'static str,
    pub crop: String,
}

/// Turn decoded regions (already in reading order -- [`crate::run_layout`] sorts them) into tasks,
/// assigning each a stable crop filename `crop_{i:03}_{class}.png`.
pub fn plan_tasks(regions: &[Region]) -> Vec<RegionTask> {
    regions
        .iter()
        .enumerate()
        .map(|(i, r)| RegionTask {
            read_order: r.read_order,
            class: r.class.clone(),
            prompt: task_prompt(&r.class),
            crop: format!("crop_{i:03}_{}.png", r.class),
        })
        .collect()
}

/// Serialize tasks to a JSON array (the manifest the recognition stage reads).
// Hand-written JSON, no serde dep: every value is layout-generated and known-safe (class is in
// LABEL_LIST so `[a-z_]`, prompt is a fixed literal, crop is a generated filename), so none can
// contain a `"` or `\`. If a future field ever carries untrusted text, switch to serde_json.
pub fn manifest_json(tasks: &[RegionTask]) -> String {
    if tasks.is_empty() {
        return "[]\n".to_string();
    }
    let rows: Vec<String> = tasks
        .iter()
        .map(|t| {
            format!(
                "  {{\"read_order\": {}, \"class\": \"{}\", \"prompt\": \"{}\", \"crop\": \"{}\"}}",
                t.read_order, t.class, t.prompt, t.crop
            )
        })
        .collect();
    format!("[\n{}\n]\n", rows.join(",\n"))
}

/// Parse the recognition stage's `results.json` (shape `[{read_order, class, text}]`) into
/// `(class, text)` blocks sorted by `read_order` -- the input [`assemble_markdown`] expects.
/// `text` carries arbitrary VLM output (escaped `\n`/`"`/`\`), so this uses a real JSON parser.
/// Rows missing `class`/`text` are skipped; a missing `read_order` sorts first.
pub fn read_results(json: &str) -> serde_json::Result<Vec<(String, String)>> {
    let mut rows: Vec<(i64, String, String)> =
        serde_json::from_str::<Vec<serde_json::Value>>(json)?
            .into_iter()
            .filter_map(|v| {
                let class = v.get("class")?.as_str()?.to_string();
                let text = v.get("text")?.as_str()?.to_string();
                let read_order = v.get("read_order").and_then(|o| o.as_i64()).unwrap_or(0);
                Some((read_order, class, text))
            })
            .collect();
    rows.sort_by_key(|(order, _, _)| *order);
    Ok(rows.into_iter().map(|(_, class, text)| (class, text)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_map_by_class() {
        assert_eq!(task_prompt("text"), "OCR:");
        assert_eq!(task_prompt("paragraph_title"), "OCR:");
        assert_eq!(task_prompt("doc_title"), "OCR:");
        assert_eq!(task_prompt("table"), "Table Recognition:");
        assert_eq!(task_prompt("display_formula"), "Formula Recognition:");
        assert_eq!(task_prompt("inline_formula"), "Formula Recognition:");
        assert_eq!(task_prompt("chart"), "Chart Recognition:");
        assert_eq!(task_prompt("seal"), "Seal Recognition:");
        assert_eq!(task_prompt("image"), "OCR:"); // default fallthrough
    }

    #[test]
    fn otsl_table_becomes_markdown() {
        // real PaddleOCR-VL table output (greedy, `</s>` already stripped).
        let md = otsl_to_markdown("<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>");
        assert_eq!(md, "| A | B |\n| --- | --- |\n| 1 | 2 |");
        // a "table" block routes through the converter in assemble_markdown.
        let doc = assemble_markdown(&[("table".to_string(), "<fcel>A<fcel>B<nl>".to_string())]);
        assert_eq!(doc, "| A | B |\n| --- | --- |");
        // non-OTSL text falls back unchanged (never panics / eats content).
        assert_eq!(otsl_to_markdown("plain text"), "plain text");
    }

    #[test]
    fn crop_clamps_to_bounds() {
        let mut img = RgbImage::new(10, 10);
        img.put_pixel(3, 4, image::Rgb([1, 2, 3]));

        let c = crop_region(&img, &[2.0, 3.0, 6.0, 8.0]);
        assert_eq!(c.dimensions(), (4, 5));
        assert_eq!(c.get_pixel(1, 1).0, [1, 2, 3]); // src (3,4) -> crop origin (2,3) -> local (1,1)

        // bbox exceeding bounds is clamped, never panics.
        assert_eq!(crop_region(&img, &[-5.0, -5.0, 100.0, 100.0]).dimensions(), (10, 10));
        // zero-area bbox yields at least 1x1.
        assert_eq!(crop_region(&img, &[5.0, 5.0, 5.0, 5.0]).dimensions(), (1, 1));
    }

    #[test]
    fn markdown_headings_and_order() {
        let blocks = vec![
            ("doc_title".to_string(), "Quarterly Report".to_string()),
            ("text".to_string(), "Body one.".to_string()),
            ("paragraph_title".to_string(), "Section".to_string()),
            ("text".to_string(), "  ".to_string()), // blank -> skipped
            ("table".to_string(), "<fcel>A".to_string()),
        ];
        // the trailing "table" block (`<fcel>A`) routes through otsl_to_markdown: one cell -> a
        // 1-column markdown table.
        assert_eq!(
            assemble_markdown(&blocks),
            "# Quarterly Report\n\nBody one.\n\n## Section\n\n| A |\n| --- |"
        );
    }

    fn region(class: &str, read_order: i64) -> Region {
        Region {
            class: class.to_string(),
            label: 0,
            score: 0.9,
            bbox: [0.0, 0.0, 1.0, 1.0],
            read_order,
        }
    }

    #[test]
    fn plan_tasks_names_crops_and_resolves_prompts() {
        let regions = vec![region("paragraph_title", 10), region("table", 20)];
        let tasks = plan_tasks(&regions);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].crop, "crop_000_paragraph_title.png");
        assert_eq!(tasks[0].prompt, "OCR:");
        assert_eq!(tasks[1].crop, "crop_001_table.png");
        assert_eq!(tasks[1].prompt, "Table Recognition:");
        assert_eq!(tasks[1].read_order, 20);
    }

    #[test]
    fn read_results_sorts_and_unescapes() {
        // out-of-order rows with escaped text (embedded newline + quote) prove real JSON unescaping
        // and the read_order sort -- the exact shape the recognition stage writes.
        let json = r#"[
          {"read_order": 20, "class": "text", "text": "a said \"hi\"\nnext line"},
          {"read_order": 10, "class": "doc_title", "text": "Title"}
        ]"#;
        let blocks = read_results(json).unwrap();
        assert_eq!(
            blocks,
            vec![
                ("doc_title".to_string(), "Title".to_string()),
                ("text".to_string(), "a said \"hi\"\nnext line".to_string()),
            ]
        );
        // and it feeds straight into the assembler.
        assert_eq!(
            assemble_markdown(&blocks),
            "# Title\n\na said \"hi\"\nnext line"
        );
    }

    #[test]
    fn manifest_json_shape() {
        assert_eq!(manifest_json(&[]), "[]\n");
        let tasks = plan_tasks(&[region("text", 5)]);
        assert_eq!(
            manifest_json(&tasks),
            "[\n  {\"read_order\": 5, \"class\": \"text\", \"prompt\": \"OCR:\", \"crop\": \"crop_000_text.png\"}\n]\n"
        );
    }
}
