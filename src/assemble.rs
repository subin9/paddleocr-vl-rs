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

/// Graphics-only layout classes: pure images/charts/seals with no counterpart in any OmniDocBench
/// text/table/formula GT category. Recognizing them (`OCR:` / `Chart Recognition:`) yields junk --
/// a photo OCR's to gibberish, a chart to a long `col | val` numeric dump -- and that junk pollutes
/// the scored markdown, so we drop it from assembly. Measured, in-session A/B on the §2.2 smoke5
/// slice (assemble the SAME results.json with vs without this skip, score both back-to-back with the
/// official scorer): dropping `chart` on the academic page moves text_block edit 0.9953 -> 0.0000,
/// table TEDS 0.6883 -> 0.9969, reading_order 0.1333 -> 0.0000 (the chart's pipe-rows were being
/// parsed as a table AND as text). Non-chart pages are unaffected. `image`/`header_image`/
/// `footer_image`/`seal` share the identical mechanism (visual-only, unmatched pred -> pollution).
pub(crate) const VISUAL_ONLY_CLASSES: [&str; 5] =
    ["chart", "image", "header_image", "footer_image", "seal"];

/// Reassemble `(class, recognized_text)` blocks (already in reading order) into one markdown doc.
/// Title classes get heading prefixes; table/formula text is emitted verbatim (it is already
/// markup/LaTeX). Empty results and [`VISUAL_ONLY_CLASSES`] are skipped. Blocks are separated by a
/// blank line.
pub fn assemble_markdown(blocks: &[(String, String)]) -> String {
    // Ablation knob for the §2.5 divergence analysis: `PADDLEOCR_VL_KEEP_VISUAL=1` keeps the
    // visual-only blocks so the same results.json can be scored with and without the skip.
    let keep_visual = std::env::var("PADDLEOCR_VL_KEEP_VISUAL").is_ok_and(|v| v == "1");
    let mut out = Vec::new();
    for (class, text) in blocks {
        if !keep_visual && VISUAL_ONLY_CLASSES.contains(&class.as_str()) {
            continue;
        }
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        out.push(match class.as_str() {
            "doc_title" => format!("# {text}"),
            "paragraph_title" | "figure_title" => format!("## {text}"),
            "table" => otsl_to_html(text),
            _ => text.to_string(),
        });
    }
    out.join("\n\n")
}

/// The six OTSL v1.0 tags PaddleOCR-VL emits for a table: `<fcel>` = cell with content, `<ecel>` =
/// empty cell, `<lcel>`/`<ucel>`/`<xcel>` = this grid slot is a continuation of the span reaching
/// left / up / both, `<nl>` = end of row.
const OTSL_CELL_TAGS: [&str; 5] = ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"];

/// Split one OTSL row into its `(tag, text)` cells -- each tag plus the text up to the next tag.
/// Text before the first tag is dropped, as the reference's `OTSL_FIND_PATTERN` does.
fn otsl_cells(row: &str) -> Vec<(&'static str, &str)> {
    let next_tag = |s: &str| {
        OTSL_CELL_TAGS
            .iter()
            .filter_map(|t| s.find(t).map(|p| (p, *t)))
            .min()
    };
    let mut cells = Vec::new();
    let mut rest = row;
    while let Some((pos, tag)) = next_tag(rest) {
        let after = &rest[pos + tag.len()..];
        let end = next_tag(after).map_or(after.len(), |(p, _)| p);
        cells.push((tag, after[..end].trim()));
        rest = &after[end..];
    }
    cells
}

/// Pad the rows out to one common width, porting the reference's `otsl_pad_to_sqr_v2`: pick the
/// width minimising the total number of cells added or dropped, but never narrower than the last
/// `<fcel>` of any row (so padding can truncate a run of trailing spans, never real content).
fn otsl_pad_to_rect(rows: &mut Vec<Vec<(&'static str, &str)>>) {
    rows.retain(|r| !r.is_empty());
    let min_width = rows
        .iter()
        .map(|r| r.iter().rposition(|(t, _)| *t == "<fcel>").map_or(0, |i| i + 1))
        .max()
        .unwrap_or(0);
    let max_width = rows.iter().map(Vec::len).max().unwrap_or(0);
    let cost = |w: usize| -> usize { rows.iter().map(|r| r.len().abs_diff(w)).sum() };
    let width = (min_width..=max_width.max(min_width))
        .min_by_key(|&w| cost(w))
        .unwrap_or(0);
    for row in rows.iter_mut() {
        row.resize(width, ("<ecel>", ""));
    }
}

/// Render PaddleOCR-VL table output (OTSL) as an HTML table, porting the reference's
/// `convert_otsl_to_html` (PaddleX `pipelines/paddleocr_vl/uilts.py`: `otsl_pad_to_sqr_v2` ->
/// `otsl_parse_texts` -> `export_to_html`). Spans MUST survive as `rowspan`/`colspan`: the
/// benchmark scores tables with TEDS, which compares the cell tree, and a GitHub pipe-table -- what
/// this used to emit -- cannot express a merged cell at all. Falls back to the raw string when the
/// text holds no grid.
fn otsl_to_html(otsl: &str) -> String {
    let mut rows: Vec<Vec<(&str, &str)>> = otsl.split("<nl>").map(otsl_cells).collect();
    otsl_pad_to_rect(&mut rows);
    let (nrows, ncols) = (rows.len(), rows.first().map_or(0, Vec::len));
    if ncols == 0 {
        // DELIBERATE divergence: the reference returns "" here, dropping the region's content. A
        // mis-classified region keeps its text instead, so the text metric can still see it. Never
        // fires on real data (0 of the 739 tables in the full run).
        return otsl.trim().to_string();
    }

    // The reference walks an INTERLEAVED list -- each cell's tag, then its text when it has one --
    // rather than the grid, and reads a cell's text as "the next element in that list". Rebuilt
    // here because that detail is load-bearing: see the quirk below.
    let mut flat: Vec<&str> = Vec::new();
    for row in &rows {
        for (tag, text) in row {
            flat.push(tag);
            if !text.is_empty() {
                flat.push(text);
            }
        }
        flat.push("<nl>");
    }

    // (start_row, start_col, row_span, col_span, text) per origin cell. A span reaches right over
    // `<lcel>`/`<xcel>` and down over `<ucel>`/`<xcel>`, exactly as the reference counts it.
    let span_right = |r: usize, mut c: usize| {
        let mut n = 0;
        while matches!(rows[r].get(c), Some(("<lcel>" | "<xcel>", _))) {
            (n, c) = (n + 1, c + 1);
        }
        n
    };
    let span_down = |mut r: usize, c: usize| {
        let mut n = 0;
        while matches!(rows.get(r).and_then(|row| row.get(c)), Some(("<ucel>" | "<xcel>", _))) {
            (n, r) = (n + 1, r + 1);
        }
        n
    };

    let mut cells = Vec::new();
    let (mut r, mut c) = (0usize, 0usize);
    for (i, &tag) in flat.iter().enumerate() {
        if tag == "<fcel>" || tag == "<ecel>" {
            // QUIRK, reproduced on purpose: the reference takes a `<fcel>`'s text to be the next
            // element of the interleaved list -- so an `<fcel>` carrying NO text swallows the next
            // TAG, and that tag's literal string ("<ecel>", "<nl>") ends up escaped into the cell.
            // Its colspan probe is knocked one element out of step for the same reason. Measured:
            // fires on 5 of the run's 739 tables, always an empty `<fcel>`, and it can only ADD
            // junk to a cell -- reproducing it costs us a little TEDS rather than winning any, and
            // it keeps `tests/otsl_html_parity.rs` an exact pin against PaddleX.
            let (text, right_offset) = match tag {
                "<ecel>" => ("", 1),
                _ => (*flat.get(i + 1).unwrap_or(&""), 2),
            };
            let mut col_span = 1;
            if matches!(flat.get(i + right_offset), Some(&("<lcel>" | "<xcel>"))) {
                col_span += span_right(r, c + 1);
            }
            let mut row_span = 1;
            if matches!(rows.get(r + 1).and_then(|row| row.get(c)), Some(("<ucel>" | "<xcel>", _))) {
                row_span += span_down(r + 1, c);
            }
            cells.push((r, c, row_span, col_span, text));
        }
        if OTSL_CELL_TAGS.contains(&tag) {
            c += 1;
        } else if tag == "<nl>" {
            (r, c) = (r + 1, 0);
        }
    }

    // Paint each cell over the slots it spans, last writer winning -- the reference builds its grid
    // the same way, so a malformed table where a later cell overlaps an earlier one's span degrades
    // identically. A slot no cell ever claims stays empty and still renders, as an empty `<td>`.
    let mut grid = vec![vec![usize::MAX; ncols]; nrows];
    for (k, &(r, c, row_span, col_span, _)) in cells.iter().enumerate() {
        for row in grid.iter_mut().skip(r).take(row_span) {
            for slot in row.iter_mut().skip(c).take(col_span) {
                *slot = k;
            }
        }
    }

    let mut html = String::from("<table>");
    for (i, row) in grid.iter().enumerate() {
        html.push_str("<tr>");
        for (j, &k) in row.iter().enumerate() {
            let Some(&(r, c, row_span, col_span, text)) = cells.get(k) else {
                html.push_str("<td></td>");
                continue;
            };
            if (r, c) != (i, j) {
                continue; // a slot this cell spans into, not its origin
            }
            html.push_str("<td");
            if row_span > 1 {
                html.push_str(&format!(" rowspan=\"{row_span}\""));
            }
            if col_span > 1 {
                html.push_str(&format!(" colspan=\"{col_span}\""));
            }
            html.push_str(&format!(">{}</td>", escape_html(text)));
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

/// Escape cell text for HTML, matching Python's `html.escape` (which the reference calls): `&`
/// first, then `<`, `>`, and both quote forms. Real tables carry these -- `S&P 500`, `<0.05`.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
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

    // Every expectation below is the REFERENCE's own output for that input, captured by running
    // PaddleX's `convert_otsl_to_html` on it -- not hand-derived from the OTSL spec.
    #[test]
    fn otsl_table_becomes_html() {
        // real PaddleOCR-VL table output (greedy, `</s>` already stripped).
        assert_eq!(
            otsl_to_html("<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>"),
            "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        );
        // a "table" block routes through the converter in assemble_markdown.
        assert_eq!(
            assemble_markdown(&[("table".to_string(), "<fcel>A<fcel>B<nl>".to_string())]),
            "<table><tr><td>A</td><td>B</td></tr></table>"
        );
        // a row with no trailing `<nl>` still closes.
        assert_eq!(otsl_to_html("<fcel>A"), "<table><tr><td>A</td></tr></table>");
    }

    #[test]
    fn otsl_spans_become_rowspan_colspan() {
        // The whole point of emitting HTML: TEDS compares the cell tree, and 34% of the tables in
        // the full OmniDocBench run carry a span token. A pipe-table cannot express either of these.
        assert_eq!(
            otsl_to_html("<fcel>Y<lcel><nl><fcel>a<fcel>b<nl>"),
            "<table><tr><td colspan=\"2\">Y</td></tr><tr><td>a</td><td>b</td></tr></table>"
        );
        assert_eq!(
            otsl_to_html("<fcel>H<fcel>I<nl><ucel><fcel>d<nl>"),
            "<table><tr><td rowspan=\"2\">H</td><td>I</td></tr><tr><td>d</td></tr></table>"
        );
        // `<xcel>` spans both ways at once.
        assert_eq!(
            otsl_to_html("<fcel>A<lcel><fcel>B<nl><ucel><xcel><fcel>C<nl>"),
            "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td><td>B</td></tr>\
             <tr><td>C</td></tr></table>"
        );
    }

    #[test]
    fn otsl_ragged_and_hostile_input_survives() {
        // Short row is padded to the table's width with empty cells (reference `otsl_pad_to_sqr_v2`).
        assert_eq!(
            otsl_to_html("<fcel>A<fcel>B<fcel>C<nl><fcel>1<nl>"),
            "<table><tr><td>A</td><td>B</td><td>C</td></tr>\
             <tr><td>1</td><td></td><td></td></tr></table>"
        );
        // Cell text carrying HTML metacharacters is escaped, not left to corrupt the parse tree
        // (real pages: `S&P 500`, `p<0.05`).
        assert_eq!(
            otsl_to_html("<fcel>S&P<fcel>x<0.05<nl>"),
            "<table><tr><td>S&amp;P</td><td>x&lt;0.05</td></tr></table>"
        );
        // DELIBERATE divergence from the reference, which returns "" here: text with no grid in it
        // falls back to itself, so a mis-classified region keeps its content for the text metric
        // instead of vanishing.
        assert_eq!(otsl_to_html("plain text"), "plain text");
    }

    #[test]
    fn visual_only_classes_are_dropped() {
        // Graphics-only regions OCR to junk; they must not reach the scored markdown, but real
        // text/table blocks around them survive in order. (See VISUAL_ONLY_CLASSES: measured to
        // move the academic smoke5 page text_block 0.9953 -> 0.0000.)
        let blocks = vec![
            ("text".to_string(), "before".to_string()),
            ("chart".to_string(), "col A | col B\n1 | 2\n3 | 4".to_string()),
            ("image".to_string(), "logo gibberish".to_string()),
            ("seal".to_string(), "OFFICIAL STAMP".to_string()),
            ("text".to_string(), "after".to_string()),
        ];
        assert_eq!(assemble_markdown(&blocks), "before\n\nafter");
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
        // the trailing "table" block (`<fcel>A`) routes through otsl_to_html: one cell -> a
        // 1x1 HTML table.
        assert_eq!(
            assemble_markdown(&blocks),
            "# Quarterly Report\n\nBody one.\n\n## Section\n\n<table><tr><td>A</td></tr></table>"
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
