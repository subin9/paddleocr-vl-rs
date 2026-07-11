//! PP-DocLayoutV3 layout stage: preprocess / run / decode the ONNX graph via ONNX Runtime.
//!
//! Layout is ONNX-only by design: RT-DETR / deformable attention / the CNN backbone are NOT
//! reimplemented. This crate preprocesses one image with the model's config.json recipe, runs the
//! graph through `ort`, and decodes its detections into `Vec<Region>` in ORIGINAL-image pixels
//! (the model divides boxes by `scale_factor`).

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

/// Region -> task prompt, crop, and reading-order markdown assembly.
/// VLM-free (pure data) so this crate never links the recognition engine: the layout and
/// recognition stages talk only through the manifest.json / results.json contract.
pub mod assemble;

/// 25 layout classes; index == the ONNX `label` column. Order matches PP-DocLayoutV3's label map.
pub const LABEL_LIST: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// Detections below this confidence are dropped (matches PaddleX `draw_threshold`).
pub const SCORE_THRESH: f32 = 0.5;
/// A region this fraction contained in a strictly larger one is a nested sub-region: its parent's
/// own OCR already renders that content, so cropping it too emits the text twice. See [`drop_nested`].
pub const NEST_CONTAINMENT: f32 = 0.8;
/// Fixed square the graph expects (`image` input is [N,3,800,800]).
const SIDE: u32 = 800;

/// Default ONNX Runtime shared library to dlopen. The `ort` crate uses `load-dynamic`, so it needs
/// a path to the runtime at startup. Override with `ORT_DYLIB_PATH` to point at your
/// libonnxruntime.so (e.g. the one inside a Python `onnxruntime` install, or a system install).
pub const DEFAULT_ORT_DYLIB: &str = "libonnxruntime.so";

/// Point `ort` at [`DEFAULT_ORT_DYLIB`] unless the caller already set `ORT_DYLIB_PATH`.
pub fn set_default_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        std::env::set_var("ORT_DYLIB_PATH", DEFAULT_ORT_DYLIB);
    }
}

/// One decoded layout region; `bbox` = [x0, y0, x1, y1] in ORIGINAL-image pixels.
#[derive(Debug, Clone)]
pub struct Region {
    pub class: String,
    pub label: usize,
    pub score: f32,
    pub bbox: [f32; 4],
    pub read_order: i64,
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

fn intersection(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x1, y1) = (a[2].min(b[2]), a[3].min(b[3]));
    (x1 - x0).max(0.0) * (y1 - y0).max(0.0)
}

/// Drop regions >=[`NEST_CONTAINMENT`] contained in a STRICTLY larger region.
///
/// PP-DocLayoutV3 emits container->child hierarchies (an `inline_formula` box inside the `text` box
/// around it, a `reference_content` inside its `reference`). Cropping each box independently
/// recognizes the child twice: once inline inside the parent's own OCR, once as its own block. On
/// OmniDocBench v1.5 that duplicated 236k chars across 449 pages; dropping the children measured
/// text-edit page_avg 0.0797 -> 0.0725 and TEDS 83.31 -> 83.36 under the official scorer.
///
/// Class-agnostic on purpose: the same containment shape is the bug for every class pair, so one
/// geometric guard beats a per-class allowlist. STRICTLY larger breaks the symmetry of two
/// near-identical boxes -- without it a duplicate pair would delete BOTH.
pub fn drop_nested(regions: &mut Vec<Region>) {
    let boxes: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox).collect();
    let mut keep = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let a = area(b);
        let nested = a > 0.0
            && boxes
                .iter()
                .enumerate()
                .any(|(j, o)| i != j && area(o) > a && intersection(b, o) / a >= NEST_CONTAINMENT);
        keep.push(!nested);
    }
    let mut it = keep.iter();
    regions.retain(|_| *it.next().unwrap_or(&true));
}

/// config.json preprocess recipe: resize to 800x800 (CatmullRom approximates cv2.INTER_CUBIC, so
/// the resampler differs and boxes match a cv2 reference only to a few px), `/255` only
/// (mean=0, std=1), CHW, BGR channel order (the model was trained on cv2 BGR). Returns the flat
/// [1,3,800,800] blob.
pub fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(img, SIDE, SIDE, FilterType::CatmullRom);
    let (w, h) = (SIDE as usize, SIDE as usize);
    let plane = w * h;
    let mut blob = vec![0f32; 3 * plane];
    // BGR channel order: dst channel 0 <- blue(src 2), 1 <- green(src 1), 2 <- red(src 0).
    let src = [2usize, 1, 0];
    for y in 0..h {
        for x in 0..w {
            let px = resized.get_pixel(x as u32, y as u32).0; // [r, g, b]
            for c in 0..3 {
                blob[c * plane + y * w + x] = px[src[c]] as f32 / 255.0;
            }
        }
    }
    blob
}

/// Run the layout graph on one image and decode `fetch_name_0` -> `Vec<Region>`:
/// keep the first `bbox_num` detections with score > [`SCORE_THRESH`], sorted by reading order.
/// (Instance masks `fetch_name_2` are unused: boxes suffice for the crop-and-recognize glue.)
pub fn run_layout(session: &mut Session, img: &image::RgbImage) -> ort::Result<Vec<Region>> {
    let (ow, oh) = (img.width() as f32, img.height() as f32);
    let blob = preprocess(img);

    let image = Tensor::from_array((vec![1i64, 3, SIDE as i64, SIDE as i64], blob))?;
    let im_shape = Tensor::from_array((vec![1i64, 2], vec![oh, ow]))?; // original [H, W]

    // scale_factor is IDENTITY: PP-DocLayoutV3 already denormalizes boxes to original pixels via
    // `im_shape`, so any non-1 scale_factor over-scales every box by (orig/800), clipping edge
    // glyphs and misplacing boxes on non-square pages. Verified vs the onnxruntime reference:
    // boxes land exactly on content for square and portrait inputs alike.
    let scale = Tensor::from_array((vec![1i64, 2], vec![1.0f32, 1.0]))?;

    let outputs = session.run(ort::inputs![
        "image" => image,
        "im_shape" => im_shape,
        "scale_factor" => scale,
    ])?;

    let (_, dets) = outputs["fetch_name_0"].try_extract_tensor::<f32>()?;
    let (_, bbox_num) = outputs["fetch_name_1"].try_extract_tensor::<i32>()?;
    let n = bbox_num.first().copied().unwrap_or(0).max(0) as usize;

    let mut regions: Vec<Region> = dets
        .chunks_exact(7)
        .take(n)
        .filter(|r| r[1] > SCORE_THRESH)
        .map(|r| {
            let label = r[0].max(0.0) as usize;
            Region {
                class: LABEL_LIST
                    .get(label)
                    .map_or_else(|| format!("?{label}"), |s| (*s).to_string()),
                label,
                score: r[1],
                bbox: [r[2], r[3], r[4], r[5]],
                read_order: r[6] as i64,
            }
        })
        .collect();
    regions.sort_by_key(|reg| reg.read_order);
    // Ablation switch (mirrors `PADDLEOCR_VL_KEEP_VISUAL`): keeps the duplicated nested crops so the
    // A/B that priced this guard stays reproducible.
    if std::env::var_os("PADDLEOCR_VL_KEEP_NESTED").is_none() {
        drop_nested(&mut regions);
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(class: &str, bbox: [f32; 4], read_order: i64) -> Region {
        Region {
            class: class.into(),
            label: 0,
            score: 0.9,
            bbox,
            read_order,
        }
    }

    #[test]
    fn drops_child_keeps_parent_and_siblings() {
        // The real shape: an `inline_formula` fully inside the `text` block that already OCRs it,
        // next to a disjoint `table`. Only the child goes.
        let mut regions = vec![
            reg("text", [0.0, 0.0, 100.0, 100.0], 0),
            reg("inline_formula", [10.0, 10.0, 30.0, 30.0], 1),
            reg("table", [200.0, 200.0, 300.0, 300.0], 2),
        ];
        drop_nested(&mut regions);
        let kept: Vec<&str> = regions.iter().map(|r| r.class.as_str()).collect();
        assert_eq!(kept, ["text", "table"]);
    }

    #[test]
    fn partial_overlap_below_threshold_survives() {
        // 25% of the small box lies in the big one -- a real neighbour, not a nested child.
        let mut regions = vec![
            reg("text", [0.0, 0.0, 100.0, 100.0], 0),
            reg("text", [90.0, 90.0, 110.0, 110.0], 1),
        ];
        drop_nested(&mut regions);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn identical_boxes_keep_one() {
        // Strictly-larger guard: neither box is larger than the other, so neither is dropped.
        // Without it, a duplicate pair would delete BOTH and lose the content entirely.
        let mut regions = vec![
            reg("text", [0.0, 0.0, 100.0, 100.0], 0),
            reg("text", [0.0, 0.0, 100.0, 100.0], 1),
        ];
        drop_nested(&mut regions);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn chained_containment_drops_both_descendants() {
        // formula inside text inside a page-sized `region`: every child is a duplicate of some
        // strictly larger ancestor, so only the outermost box survives.
        let mut regions = vec![
            reg("text", [0.0, 0.0, 100.0, 100.0], 1),
            reg("inline_formula", [10.0, 10.0, 30.0, 30.0], 2),
            reg("abstract", [0.0, 0.0, 200.0, 200.0], 0),
        ];
        drop_nested(&mut regions);
        let kept: Vec<&str> = regions.iter().map(|r| r.class.as_str()).collect();
        assert_eq!(kept, ["abstract"]);
    }
}
