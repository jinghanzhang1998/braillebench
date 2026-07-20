"""
Run full-scale baseline evaluation on original (non-Braille) datasets.

6 models x 5 datasets. Saves results incrementally.
"""

import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    "claude-opus-4.8",
    "claude-haiku-4.5",
    "llama3.3-70b",
    "llama3.1-8b",
    "qwen3-32b",
    # "qwen3-1.5b",  # local GPU model, run separately
]

DATASETS = {
    "gsm8k": None,            # full test set (1319)
    "aime24": None,           # full (30)
    "commonsenseqa": None,    # full dev (1221)
    "hotpotqa": 1500,         # dev first 1500
    "2wikimultihopqa": 1500,  # dev first 1500
}

SCRIPT = Path(__file__).parent / "evaluate.py"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "baseline_full"


def run_eval(model: str, dataset: str, limit: int = None):
    """Run evaluation for one model-dataset pair."""
    output_dir = RESULTS_DIR / model
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_file = output_dir / f"{model}_{dataset}_details.jsonl"
    if detail_file.exists():
        print(f"  SKIP {model} x {dataset} (already exists)")
        return True

    cmd = [
        sys.executable, str(SCRIPT),
        "--models", model,
        "--datasets", dataset,
        "--output-dir", str(output_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n>>> {model} x {dataset} (limit={limit or 'full'})")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(MODELS) * len(DATASETS)
    done = 0
    failed = []
    consecutive_failures = 0

    start = time.time()
    for model in MODELS:
        for dataset, limit in DATASETS.items():
            done += 1
            print(f"\n{'='*70}")
            print(f"[{done}/{total}] {model} x {dataset}")
            print(f"{'='*70}")

            success = run_eval(model, dataset, limit)
            if not success:
                failed.append(f"{model} x {dataset}")
                consecutive_failures += 1
                print(f"  FAILED!")
                if consecutive_failures >= 3:
                    elapsed = time.time() - start
                    print(f"\n{'='*70}")
                    print(f"ABORTED: 3 consecutive failures (likely credential expiry)")
                    print(f"Completed: {done-len(failed)}/{total} in {elapsed/60:.1f} min")
                    print(f"Re-run after refreshing credentials.")
                    sys.exit(1)
            else:
                consecutive_failures = 0

    elapsed = time.time() - start
    print(f"\n\n{'='*70}")
    print(f"ALL DONE in {elapsed/60:.1f} minutes")
    print(f"Failed: {len(failed)}/{total}")
    if failed:
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
