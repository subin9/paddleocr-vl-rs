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

/// Detections below this confidence are dropped by the RAW detector (PP-DocLayoutV3's own
/// `inference.yml` `draw_threshold`). The reference PIPELINE overrides it -- see [`REF_THRESH`].
pub const SCORE_THRESH: f32 = 0.5;
/// A region this fraction contained in a strictly larger one is a nested sub-region: its parent's
/// own OCR already renders that content, so cropping it too emits the text twice. See [`drop_nested`].
pub const NEST_CONTAINMENT: f32 = 0.8;

/// Reference-pipeline score threshold (`threshold: 0.3` in `PaddleOCR-VL-1.5.yaml`), NOT the raw
/// detector's 0.5: the published number keeps every detection down to 0.3. See [`ref_postprocess`].
pub const REF_THRESH: f32 = 0.3;
/// `layout_nms` IoU thresholds: same class suppresses hard, different classes almost never.
const IOU_SAME: f32 = 0.6;
const IOU_DIFF: f32 = 0.98;
/// Classes whose `layout_merge_bboxes_mode` is `large` (all others are `union` = no-op). A box
/// >=90% inside one of these is absorbed by it. Indices 3/5/6/15/17 of [`LABEL_LIST`] in the yaml.
const MERGE_LARGE: [&str; 5] = [
    "chart",
    "display_formula",
    "doc_title",
    "inline_formula",
    "paragraph_title",
];
/// `filter_boxes` never drops a pair straddling these: a caption over a figure is not a duplicate.
const PICTORIAL: [&str; 4] = ["image", "table", "seal", "chart"];
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
/// STRICTLY larger breaks the symmetry of two near-identical boxes -- without it a duplicate pair
/// would delete BOTH.
///
/// Only a parent whose own text REACHES the markdown may absorb a child. A container in
/// [`assemble::VISUAL_ONLY_CLASSES`] is never emitted, so it absorbs nothing: its "child" is not a
/// duplicate, it is the only copy, and dropping it deletes the content outright. The class-agnostic
/// version of this guard did exactly that on 15 corpus regions (`text` inside `image`) -- an
/// interaction of two independently-correct policies, worst page `newspaper_Daily Mirror…page_029`
/// text-edit 0.186 -> 0.506. Absorber-awareness restores it to 0.186. Aggregate effect is neutral
/// (page_avg 0.0725 -> 0.0722, inside the pre-registered +-0.0005 noise band); it ships because
/// deleting a page's only copy of its text is a bug regardless of score.
pub fn drop_nested(regions: &mut Vec<Region>) {
    let boxes: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox).collect();
    let absorbs: Vec<bool> = regions
        .iter()
        .map(|r| !assemble::VISUAL_ONLY_CLASSES.contains(&r.class.as_str()))
        .collect();
    let mut keep = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let a = area(b);
        let nested = a > 0.0
            && boxes.iter().enumerate().any(|(j, o)| {
                i != j && absorbs[j] && area(o) > a && intersection(b, o) / a >= NEST_CONTAINMENT
            });
        keep.push(!nested);
    }
    let mut it = keep.iter();
    regions.retain(|_| *it.next().unwrap_or(&true));
}

/// Area with the +1 padding PaddleX's `iou()` uses (and only there -- its containment/overlap
/// helpers use the unpadded area, so the two must not be unified).
fn area_p1(b: &[f32; 4]) -> f32 {
    (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
}

fn iou_p1(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x1, y1) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x1 - x0 + 1.0).max(0.0) * (y1 - y0 + 1.0).max(0.0);
    inter / (area_p1(a) + area_p1(b) - inter)
}

/// `is_contained`: >=90% of `inner`'s own area lies in `outer`.
fn is_contained(inner: &[f32; 4], outer: &[f32; 4]) -> bool {
    let a = area(inner);
    a > 0.0 && intersection(inner, outer) / a >= 0.9
}

/// `calculate_overlap_ratio(mode="small")`: intersection over the SMALLER box's area.
fn overlap_small(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let r = area(a).min(area(b));
    if r == 0.0 {
        0.0
    } else {
        intersection(a, b) / r
    }
}

/// The reference pipeline's layout post-processing, ported from PaddleX
/// `layout_analysis/processors.py` (`LayoutAnalysisProcess.apply` + `filter_boxes`) as the shipped
/// `paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml` configures it.
///
/// PP-DocLayoutV3's *raw* detector (its own `inference.yml`: score>0.5, no NMS, no merging) is what
/// this crate shipped, and a probe against the official model established the port reproduces that
/// detector faithfully (mean GT-block coverage 0.750 vs its 0.739 on the 10 worst pages). But the
/// PAPER's number is not produced by the raw detector: the reference runs the same weights through
/// the post-processing below, which lifts coverage to 0.860 and recovers 102 of our failed blocks.
/// Omitting it was the port defect; this is the fix.
///
/// Steps, in the reference's order (their order is load-bearing -- NMS before merging, both before
/// the pairwise overlap filter, which is index-sensitive and so runs on the reading-order list):
/// 1. round coordinates (`np.round` = ties-to-EVEN, not Rust's ties-away-from-zero)
/// 2. score > [`REF_THRESH`] (0.3, not our 0.5)
/// 3. `layout_nms`: greedy, descending score, [`IOU_SAME`] / [`IOU_DIFF`] by class agreement
/// 4. drop an `image` box that covers ~the whole page (a full-page scan detected as a figure)
/// 5. `layout_merge_bboxes_mode`: the [`MERGE_LARGE`] classes absorb what they contain
/// 6. reading order, then clip to the page and drop degenerate boxes
/// 7. `filter_boxes`: drop `reference` containers, sub-6px slivers, and overlapping duplicates
///
/// (`layout_unclip_ratio: [1.0, 1.0]` is the identity, so there is nothing to port for it.)
///
/// This SUPERSEDES [`drop_nested`], which hand-rolled step 7's job: `filter_boxes` drops an
/// `inline_formula` overlapping any other box, and the `reference` container outright. It even
/// carries the same guard we had to add by hand -- a pair straddling [`PICTORIAL`] is never merged,
/// so a `text` inside an `image` keeps its only copy.
pub fn ref_postprocess(regions: &mut Vec<Region>, img_w: f32, img_h: f32) {
    for r in regions.iter_mut() {
        for v in r.bbox.iter_mut() {
            *v = v.round_ties_even();
        }
    }
    regions.retain(|r| r.score > REF_THRESH);
    layout_nms(regions);
    filter_page_sized_image(regions, img_w, img_h);
    merge_large(regions);
    regions.sort_by_key(|r| r.read_order);
    clip_to_page(regions, img_w, img_h);
    filter_boxes(regions);
}

/// Greedy NMS, highest score first. A survivor suppresses a later box at IoU >= 0.6 if they share a
/// class, >= 0.98 if they do not (i.e. only near-exact duplicates cross class lines).
fn layout_nms(regions: &mut Vec<Region>) {
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| {
        regions[b].score
            .partial_cmp(&regions[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = vec![false; regions.len()];
    let mut pool: Vec<usize> = order;
    while let Some(&cur) = pool.first() {
        keep[cur] = true;
        let (cbox, clabel) = (regions[cur].bbox, regions[cur].label);
        pool = pool[1..]
            .iter()
            .copied()
            .filter(|&i| {
                let thresh = if regions[i].label == clabel { IOU_SAME } else { IOU_DIFF };
                iou_p1(&regions[i].bbox, &cbox) < thresh
            })
            .collect();
    }
    let mut it = keep.iter();
    regions.retain(|_| *it.next().unwrap_or(&true));
}

/// An `image` detection covering >=82% (landscape) / 93% (portrait) of the page is the page itself,
/// not a figure in it. Dropped -- unless it is all we have, in which case the reference keeps
/// everything rather than return nothing.
fn filter_page_sized_image(regions: &mut Vec<Region>, img_w: f32, img_h: f32) {
    if regions.len() <= 1 {
        return;
    }
    let thresh = if img_w > img_h { 0.82 } else { 0.93 };
    let limit = thresh * img_w * img_h;
    let keep: Vec<bool> = regions
        .iter()
        .map(|r| {
            r.class != "image" || {
                let b = [
                    r.bbox[0].max(0.0),
                    r.bbox[1].max(0.0),
                    r.bbox[2].min(img_w),
                    r.bbox[3].min(img_h),
                ];
                (b[2] - b[0]) * (b[3] - b[1]) <= limit
            }
        })
        .collect();
    if keep.iter().any(|&k| k) {
        let mut it = keep.iter();
        regions.retain(|_| *it.next().unwrap_or(&true));
    }
}

/// `layout_merge_bboxes_mode: large` -- drop any box >=90% inside a box of a [`MERGE_LARGE`] class.
/// Every class's containment is computed against the SAME pre-merge set (the reference ANDs the
/// per-class masks), so a drop never cascades.
fn merge_large(regions: &mut Vec<Region>) {
    let absorbers: Vec<bool> = regions
        .iter()
        .map(|r| MERGE_LARGE.contains(&r.class.as_str()))
        .collect();
    let boxes: Vec<[f32; 4]> = regions.iter().map(|r| r.bbox).collect();
    let keep: Vec<bool> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            !boxes
                .iter()
                .enumerate()
                .any(|(j, o)| i != j && absorbers[j] && is_contained(b, o))
        })
        .collect();
    let mut it = keep.iter();
    regions.retain(|_| *it.next().unwrap_or(&true));
}

/// `restructured_boxes`: clamp to the page, truncate to integer pixels, drop what collapses.
fn clip_to_page(regions: &mut Vec<Region>, img_w: f32, img_h: f32) {
    for r in regions.iter_mut() {
        r.bbox = [
            r.bbox[0].max(0.0).trunc(),
            r.bbox[1].max(0.0).trunc(),
            r.bbox[2].min(img_w).trunc(),
            r.bbox[3].min(img_h).trunc(),
        ];
    }
    regions.retain(|r| r.bbox[2] > r.bbox[0] && r.bbox[3] > r.bbox[1]);
}

/// `filter_boxes`: the reference's own de-duplication pass (predictor default
/// `filter_overlap_boxes=True`), ported for `layout_shape_mode="rect"`.
///
/// - `reference` containers are dropped outright -- their `reference_content` children carry the text.
/// - Boxes under 6px on a side are slivers.
/// - An `inline_formula` more than half-inside any other box is already in that box's OCR.
/// - Otherwise, of two boxes >70% overlapping (relative to the SMALLER), the smaller goes -- except
///   when the pair straddles [`PICTORIAL`], where the overlap is a caption on a figure, not a copy.
///
/// The index-sensitive quirks are the reference's, and are preserved deliberately: the sliver test
/// marks box `i` only when the outer loop reaches it, so a sliver can still evict a full-size box
/// that precedes it; and `dropped` is consulted, not compacted, so a dropped box stops participating
/// but does not shift the indices of the rest.
fn filter_boxes(regions: &mut Vec<Region>) {
    regions.retain(|r| r.class != "reference");
    let n = regions.len();
    let mut dropped = vec![false; n];
    for i in 0..n {
        let bi = regions[i].bbox;
        if bi[2] - bi[0] < 6.0 || bi[3] - bi[1] < 6.0 {
            dropped[i] = true;
        }
        for j in (i + 1)..n {
            if dropped[i] || dropped[j] {
                continue;
            }
            let bj = regions[j].bbox;
            let (ci, cj) = (regions[i].class.as_str(), regions[j].class.as_str());
            let ratio = overlap_small(&bi, &bj);
            if ci == "inline_formula" || cj == "inline_formula" {
                if ratio > 0.5 {
                    dropped[i] |= ci == "inline_formula";
                    dropped[j] |= cj == "inline_formula";
                }
                continue;
            }
            if ratio > 0.7 {
                let pictorial = PICTORIAL.contains(&ci) || PICTORIAL.contains(&cj);
                let table = ci == "table" || cj == "table";
                let both_pictorial = PICTORIAL.contains(&ci) && PICTORIAL.contains(&cj);
                // A mixed pair touching a figure/table/seal/chart is a caption, not a duplicate --
                // EXCEPT a table overlapping a non-pictorial box, which really is a duplicate.
                if pictorial && ci != cj && (!table || both_pictorial) {
                    continue;
                }
                if area(&bi) >= area(&bj) {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                }
            }
        }
    }
    let mut it = dropped.iter();
    regions.retain(|_| !*it.next().unwrap_or(&false));
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

    // Keep every detection here: the reference post-processing thresholds at 0.3, so filtering at
    // our 0.5 up front would throw away the 0.3-0.5 band before it ever gets a say.
    let mut regions: Vec<Region> = dets
        .chunks_exact(7)
        .take(n)
        .filter(|r| r[0] > -1.0)
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

    // DEFAULT: the reference pipeline's post-processing. Priced on the full 1651 pages under the
    // official scorer (`reflayout1651` vs the `nonest2` baseline): text-edit page_avg 0.0722 ->
    // 0.0428, reading-order 0.0917 -> 0.0500, TEDS 0.8336 -> 0.8364, formula 0.2564 -> 0.2495 --
    // better on every metric, so the pre-registered rule flips the default. See `ref_postprocess`.
    //
    // `PADDLEOCR_VL_RAW_LAYOUT=1` restores the raw detector's own defaults (threshold 0.5, no NMS,
    // no merge) plus the hand-rolled `drop_nested` it needed. Ablation only -- it is the SCORED
    // BASELINE the flip was measured against, so it stays runnable and bit-identical.
    if std::env::var_os("PADDLEOCR_VL_RAW_LAYOUT").is_none() {
        ref_postprocess(&mut regions, ow, oh);
        return Ok(regions);
    }
    regions.retain(|r| r.score > SCORE_THRESH);
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
    fn visual_only_parent_absorbs_nothing_so_child_survives() {
        // `text` inside `image`: the assembler never emits `image`, so dropping the child would
        // delete the only copy of that text. The child stays; a real absorber still eats its own.
        let mut regions = vec![
            reg("image", [0.0, 0.0, 100.0, 100.0], 0),
            reg("text", [10.0, 10.0, 30.0, 30.0], 1),
            reg("text", [200.0, 200.0, 300.0, 300.0], 2),
            reg("inline_formula", [210.0, 210.0, 230.0, 230.0], 3),
        ];
        drop_nested(&mut regions);
        let kept: Vec<&str> = regions.iter().map(|r| r.class.as_str()).collect();
        assert_eq!(kept, ["image", "text", "text"]);
    }

    // The corpus fixture (tests/ref_postproc_parity.rs) pins `ref_postprocess` against PaddleX on
    // 6600 real detections and kills a mutation of every step -- except these two, which no page in
    // it happens to trigger (0 sub-6px detections, 0 page-sized `image` boxes). Synthetic, so they
    // are covered anyway rather than left to chance.

    #[test]
    fn slivers_are_dropped() {
        // `filter_boxes`: under 6px on a side is a detector artefact, not a region.
        let mut regions = vec![
            reg("text", [0.0, 0.0, 100.0, 100.0], 0),
            reg("text", [200.0, 200.0, 260.0, 205.0], 1), // 5px tall
        ];
        ref_postprocess(&mut regions, 800.0, 800.0);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].bbox, [0.0, 0.0, 100.0, 100.0]);
    }

    #[test]
    fn page_sized_image_dropped_unless_it_is_all_there_is() {
        // A full-page scan detected as one `image` covers the text under it; the reference drops it
        // (>93% of a portrait page) -- but never returns nothing, so alone it survives.
        let page = [0.0, 0.0, 800.0, 1000.0];
        let mut regions = vec![
            reg("image", page, 0),
            reg("text", [10.0, 10.0, 700.0, 100.0], 1),
        ];
        ref_postprocess(&mut regions, 800.0, 1000.0);
        let kept: Vec<&str> = regions.iter().map(|r| r.class.as_str()).collect();
        assert_eq!(kept, ["text"]);

        let mut only = vec![reg("image", page, 0)];
        ref_postprocess(&mut only, 800.0, 1000.0);
        assert_eq!(only.len(), 1, "the reference keeps everything over returning nothing");
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
