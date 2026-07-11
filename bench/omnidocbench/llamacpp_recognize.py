#!/usr/bin/env python3
"""Cross-stack (§2.6): re-recognize the Rust pipeline's OWN crops with llama.cpp, then assemble
with the SAME assembler, so the only thing that differs between the two runs is the recognition
backend.

The layout stage is byte-identical *by construction*, not merely "the same code": we read the crop
PNGs and manifests the scored REF_LAYOUT run already wrote to work_reflayout/<stem>/, and never
re-run the detector. Per-class prompts come from the manifest verbatim ("OCR:" / "Table
Recognition:" / "Formula Recognition:").

Output contract is the recognize binary's: results_llamacpp.json = [{read_order, class, text}, ...],
which `paddleocr-layout assemble` consumes unchanged.

Idempotent + resumable: a page whose pred .md already exists is skipped, so a kill/restart costs
nothing. Timeout + retry per crop; a crop that still fails records empty text (mirroring the Rust
side's region-timeout policy) so results.json stays complete and the page still assembles.
"""
import argparse, base64, json, pathlib, subprocess, sys, time
import urllib.error, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("stems_file")
AP.add_argument("preds_dir")
AP.add_argument("--work", default="work_reflayout")
AP.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
AP.add_argument("--assemble-bin", default="../../target/release/paddleocr-layout")
AP.add_argument("--max-tokens", type=int, default=2048)  # mirrors the Rust guard's MAX_NEW_TOKENS
AP.add_argument("--timeout", type=int, default=180)
A = AP.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
work = (HERE / A.work) if not pathlib.Path(A.work).is_absolute() else pathlib.Path(A.work)
preds = pathlib.Path(A.preds_dir)
if not preds.is_absolute():
    preds = HERE / preds
preds.mkdir(parents=True, exist_ok=True)
assemble = pathlib.Path(A.assemble_bin)
if not assemble.is_absolute():
    assemble = (HERE / assemble).resolve()


def recognize(crop: pathlib.Path, prompt: str) -> str:
    """One crop -> text. Greedy (temp 0, top_k 1) to match the Rust port's decoding."""
    b64 = base64.b64encode(crop.read_bytes()).decode()
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0, "top_k": 1, "max_tokens": A.max_tokens,
    }).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(A.url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=A.timeout) as r:
                out = json.load(r)
            return out["choices"][0]["message"]["content"]
        except Exception as e:  # server hiccup / timeout / truncated read
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"    CROP FAILED after retries ({last}) -> empty text: {crop.name}", file=sys.stderr)
    return ""


stems = [l.strip() for l in open(A.stems_file) if l.strip()]
n_done = n_skip = n_miss = 0
for i, img in enumerate(stems, 1):
    stem = img[:-4]                       # scorer's img_name[:-4] + '.md'
    out_md = preds / f"{stem}.md"
    if out_md.exists() and out_md.stat().st_size > 0:
        n_skip += 1
        continue
    page = work / stem
    man = page / "manifest.json"
    if not man.exists():                  # empty-layout page: the Rust run has none either
        print(f"[{i}] NO MANIFEST (empty layout) -> skip: {stem}", file=sys.stderr)
        n_miss += 1
        continue
    tasks = json.load(open(man))
    t0 = time.time()
    results = [{"read_order": t["read_order"], "class": t["class"],
                "text": recognize(page / t["crop"], t["prompt"])} for t in tasks]
    res_path = page / "results_llamacpp.json"
    res_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    md = subprocess.run([str(assemble), "assemble", str(res_path)],
                        capture_output=True, text=True, check=True).stdout
    out_md.write_text(md)
    n_done += 1
    print(f"[{i}/{len(stems)}] {stem}: {len(tasks)} crops, {len(md)} bytes, {time.time()-t0:.1f}s",
          flush=True)

print(f"DONE: {n_done} newly written, {n_skip} already present, {n_miss} no-manifest, "
      f"of {len(stems)} pages -> {preds}")
