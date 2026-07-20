#!/usr/bin/env python3
"""Shard Qwen3-1.5B BrailleBench jobs without changing evaluation logic."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_local_eval import run_evaluation

DATASETS = ["gsm8k", "aime24", "commonsenseqa", "hotpotqa", "2wikimultihopqa"]
BRAILLE_CONFIGS = ["EN-G1", "EN-G2", "G1-EN", "G2-EN", "G1-G1", "G2-G2"]


def build_jobs():
    jobs = [(dataset, "EN-EN", None) for dataset in DATASETS]
    for config in BRAILLE_CONFIGS:
        for dataset in DATASETS:
            jobs.append((dataset, config, "ascii"))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-1.5b")
    parser.add_argument("--output-dir", default="data/results/local")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        parser.error("--worker-index must satisfy 0 <= worker-index < num-workers")

    jobs = build_jobs()
    selected = [
        job for idx, job in enumerate(jobs)
        if idx % args.num_workers == args.worker_index
    ]
    print(f"Worker {args.worker_index}/{args.num_workers}: {len(selected)} of {len(jobs)} jobs")

    for dataset, config, braille_format in selected:
        run_evaluation(args.model, dataset, config, braille_format, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
