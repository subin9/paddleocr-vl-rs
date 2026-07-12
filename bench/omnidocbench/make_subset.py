#!/usr/bin/env python3
"""Build an OmniDocBench end2end scoring subset from a list of GT image filenames.

Usage: make_subset.py <stems_file> <subset_name>

<stems_file>  one GT image filename per line (e.g. `foo.png`), matching OmniDocBench.json's
              page_info.image_path values.
Writes (paths relative to this script's dir):
  data/subsets/<name>.gt.json        filtered GT (dataset-derived -> gitignored)
  data/subsets/<name>.end2end.yaml   scorer config, committed

The scorer opens `data_path` relative to its OWN cwd, and it is always run from the eval clone --
so the config's paths are written relative to that dir, making it machine-independent (an absolute
path would pin every score to one home dir). Run it exactly like this:
  cd ../OmniDocBench && ../omnidocbench/scorer-venv/bin/python pdf_validation.py \
      -c ../omnidocbench/data/subsets/<name>.end2end.yaml
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GT_ALL = os.path.join(HERE, "data", "OmniDocBench.json")
EVAL_DIR = os.path.join(HERE, "..", "OmniDocBench")   # the scorer's cwd; need not exist yet

def main(stems_file, name):
    wanted = [l.strip() for l in open(stems_file) if l.strip()]
    gt = json.load(open(GT_ALL))
    by_img = {e["page_info"]["image_path"]: e for e in gt}
    missing = [w for w in wanted if w not in by_img]
    if missing:
        sys.exit(f"ERROR: {len(missing)} requested images not in GT: {missing[:5]}")
    subset = [by_img[w] for w in wanted]

    outdir = os.path.join(HERE, "data", "subsets")
    os.makedirs(outdir, exist_ok=True)
    gt_path = os.path.join(outdir, f"{name}.gt.json")
    json.dump(subset, open(gt_path, "w"), ensure_ascii=False)

    preds_dir = os.path.join(HERE, "preds", name)
    cfg = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": os.path.relpath(gt_path, EVAL_DIR)},
                "prediction": {"data_path": os.path.relpath(preds_dir, EVAL_DIR)},
                "match_method": "quick_match",
            },
        }
    }
    # minimal YAML writer (avoid a pyyaml dep in this repo's python; the scorer venv has it but
    # this script may run under any python). ponytail: json is valid yaml, so dump json.
    cfg_path = os.path.join(outdir, f"{name}.end2end.yaml")
    json.dump(cfg, open(cfg_path, "w"), ensure_ascii=False, indent=2)
    print(f"subset '{name}': {len(subset)} pages")
    print(f"  GT     -> {gt_path}")
    print(f"  config -> {cfg_path}")
    print(f"  preds  -> {preds_dir}  (create via run_pipeline.sh)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
