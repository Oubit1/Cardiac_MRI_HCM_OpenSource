"""Entry points for the released revision experiments with user-provided inputs."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "2026.11.06"
BENCH = CORE / "benchmark_1059"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["init", "tabular", "train", "fusion", "roi-fusion", "roi-frozen", "metrics", "recalibration"])
    parser.add_argument("--gpu", default="0", help="CUDA index or GPU UUID")
    args = parser.parse_args()
    os.environ["JMRI_GPU_UUID"] = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for folder in [CORE, BENCH, CORE/"nested_validation", CORE/"recalibration"]:
        for name in ["inputs", "plans", "runs", "logs", "audit", "results", "tables", "figures", "response"]:
            (folder/name).mkdir(parents=True, exist_ok=True)
    if args.task == "init":
        print("Created work directories. Supply inputs as described in revision/README.md.")
        return
    os.chdir(ROOT)
    if args.task == "tabular":
        sys.path.insert(0, str(CORE/"nested_validation/code"))
        import tabular_suite as tab
        tab.prepare()
        for variant in ["all_features", "no_shape"]:
            directory = tab.BASE/variant
            for fold in range(1, 6):
                tab.outer(directory, fold)
            tab.finalize(directory)
        return
    if args.task == "train":
        source = CORE/"nested_validation/tabular_full/all_features/folds.json"
        target = BENCH/"plans/outer_folds.json"
        if target.exists():
            if json.loads(target.read_text()) != json.loads(source.read_text()):
                raise ValueError("Training and tabular folds differ")
        else:
            target.write_bytes(source.read_bytes())
        checkpoint = BENCH/"inputs/pretrained.ckpt"
        if not checkpoint.exists():
            checkpoint.symlink_to(CORE/"inputs/pretrained.ckpt")
    scripts = {
        "train": BENCH/"code/run.py",
        "fusion": BENCH/"code/fusion1059.py",
        "roi-fusion": BENCH/"code/roi_fusion1059.py",
        "roi-frozen": BENCH/"code/run_roi_frozen.py",
        "metrics": BENCH/"code/finish1059.py",
        "recalibration": CORE/"recalibration/code/run.py",
    }
    script = scripts[args.task]
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__":
    main()
