"""
Run braille evaluation for specified configs.
Wraps evaluate_braille.py with skip-on-exist logic.
"""

import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    # Fast models first to maximize progress within credential window
    "claude-haiku-4.5",
    "llama3.3-70b",
    "llama3.1-8b",
    "qwen3-32b",
    "claude-opus-4.8",  # slowest last
    # "qwen3-1.5b",  # local GPU model, run separately with model_client_local.py
]

DATASETS = ["gsm8k", "aime24", "commonsenseqa", "hotpotqa", "2wikimultihopqa"]

DATASET_LIMITS = {
    "hotpotqa": 1500,
    "2wikimultihopqa": 1500,
}

SCRIPT = Path(__file__).parent / "evaluate_braille.py"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "braille_eval"


def run_eval(model, dataset, config, fmt):
    """Run one evaluation (supports resume from checkpoint)."""
    tag = f"{model}_{dataset}_{config}_{fmt}"
    detail_file = RESULTS_DIR / f"{tag}_details.jsonl"

    # Check if already complete
    limit = DATASET_LIMITS.get(dataset)
    expected_n = limit if limit else {
        "gsm8k": 1319, "aime24": 30, "commonsenseqa": 1221,
        "hotpotqa": 7405, "2wikimultihopqa": 12576,
    }.get(dataset, 0)
    if limit:
        expected_n = limit

    if detail_file.exists():
        with open(detail_file) as f:
            existing = sum(1 for _ in f)
        if existing >= expected_n:
            print(f"  SKIP {tag} (complete: {existing})")
            return True
        else:
            print(f"\n>>> {tag} (RESUME from {existing}/{expected_n})")
    else:
        print(f"\n>>> {tag}")

    cmd = [
        sys.executable, str(SCRIPT),
        "--models", model,
        "--datasets", dataset,
        "--configs", config,
        "--formats", fmt,
        "--output-dir", str(RESULTS_DIR),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="e.g. G1-EN G2-EN")
    parser.add_argument("--formats", nargs="+", default=["ascii"])
    parser.add_argument("--models", nargs="+", default=None, help="Override model list")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override dataset list")
    args = parser.parse_args()

    models = args.models or MODELS
    datasets = args.datasets or DATASETS
    formats = args.formats

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(models) * len(datasets) * len(args.configs) * len(formats)
    done = 0
    failed = []
    consecutive_failures = 0

    start = time.time()
    for config in args.configs:
        for fmt in formats:
            for model in models:
                for dataset in datasets:
                    done += 1
                    success = run_eval(model, dataset, config, fmt)
                    if not success:
                        failed.append(f"{model}/{dataset}/{config}/{fmt}")
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            elapsed = time.time() - start
                            print(f"\n{'='*60}")
                            print(f"ABORTED: 3 consecutive failures (likely credential expiry)")
                            print(f"Completed: {done-len(failed)}/{total} in {elapsed/60:.1f} min")
                            print(f"Re-run after refreshing credentials.")
                            sys.exit(1)
                    else:
                        consecutive_failures = 0

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    print(f"Failed: {len(failed)}/{total}")
    if failed:
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
