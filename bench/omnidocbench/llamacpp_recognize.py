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
nothing.

Two failure modes, deliberately NOT conflated (they were, and it silently poisoned a run):
  * a crop that exceeds the per-region time budget is a *model* outcome -> record empty text,
    mirroring the Rust side's REGION_TIMEOUT policy, and count it. The page still assembles.
  * an unreachable server (the kernel OOM-killed llama-server mid-run on 2026-07-12) is an
    *infrastructure* outcome and says nothing about the model -> block on /health until the
    supervised restart lands, then carry on. NEVER write a prediction for a dead server: empty
    output that looks like a real one is the single worst thing a benchmark harness can produce.
"""
import argparse, base64, json, pathlib, socket, subprocess, sys, time
import urllib.error, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("stems_file")
AP.add_argument("preds_dir")
AP.add_argument("--work", default="work_reflayout")
AP.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
AP.add_argument("--assemble-bin", default="../../target/release/paddleocr-layout")
AP.add_argument("--max-tokens", type=int, default=2048)  # mirrors the Rust guard's MAX_NEW_TOKENS
AP.add_argument("--timeout", type=int, default=120)      # mirrors REGION_TIMEOUT_SECS
AP.add_argument("--server-wait", type=int, default=900)  # how long to wait out a restart
A = AP.parse_args()

HEALTH = A.url.split("/v1/")[0] + "/health"
n_region_timeout = 0
n_server_restart = 0


def wait_for_server() -> bool:
    """Block until llama-server answers /health again (reload after an OOM kill is ~30s)."""
    global n_server_restart
    n_server_restart += 1
    t0 = time.time()
    while time.time() - t0 < A.server_wait:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if r.status == 200:
                    print(f"    server back after {time.time()-t0:.0f}s", file=sys.stderr)
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False

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
    global n_region_timeout
    b64 = base64.b64encode(crop.read_bytes()).decode()
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0, "top_k": 1, "max_tokens": A.max_tokens,
    }).encode()
    attempts = 0
    while True:
        try:
            req = urllib.request.Request(A.url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=A.timeout) as r:
                out = json.load(r)
            return out["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            # NB: HTTPError subclasses URLError -- it must be caught FIRST or a 500 would be read as
            # "server down", and we'd poll a perfectly healthy /health forever. The server answered;
            # it just answered badly. Bounded retry, then fail loud: a crop that reliably 500s is a
            # real anomaly to look at, not something to paper over with an empty prediction.
            attempts += 1
            if attempts >= 3:
                raise SystemExit(f"HTTP {e.code} on {crop.name} after {attempts} attempts: "
                                 f"{e.read()[:300]!r}")
            time.sleep(2 * attempts)

        except (TimeoutError, socket.timeout):
            # Model outcome: this region blew the time budget. Same policy as the Rust guard.
            n_region_timeout += 1
            print(f"    REGION TIMEOUT (>{A.timeout}s) -> empty text: {crop.name}", file=sys.stderr)
            return ""

        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                n_region_timeout += 1
                print(f"    REGION TIMEOUT (>{A.timeout}s) -> empty text: {crop.name}",
                      file=sys.stderr)
                return ""
            # Infrastructure outcome: server is gone (OOM kill / restart). Wait it out and retry the
            # same crop -- never fabricate a prediction the model never made.
            print(f"    server unreachable ({e.reason}) -- waiting for restart", file=sys.stderr)
            if not wait_for_server():
                raise SystemExit(f"llama-server did not return within {A.server_wait}s -- aborting "
                                 f"rather than writing empty predictions")

        except Exception as e:  # truncated read / connection reset / malformed JSON
            attempts += 1
            if attempts >= 3:
                raise SystemExit(f"{type(e).__name__} on {crop.name} after {attempts} attempts: {e}")
            time.sleep(2 * attempts)


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
print(f"GUARD: {n_region_timeout} crops hit the {A.timeout}s region timeout (empty text, as the "
      f"Rust guard does); server restarts waited out: {n_server_restart}")
