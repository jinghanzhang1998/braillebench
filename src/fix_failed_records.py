"""
Re-run failed records (ERROR responses) in existing result files.

Reads the result file, identifies records with ERROR responses,
re-runs them, and rewrites the file with fixed results.

Usage:
    python src/fix_failed_records.py data/results/braille_eval/claude-opus-4.8_commonsenseqa_G2-EN_ascii_details.jsonl
    python src/fix_failed_records.py --all  # fix all files with errors
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOUIS_TABLEPATH", "/Users/jinghanz/local/share/liblouis/tables")

sys.path.insert(0, str(Path(__file__).parent))
from model_client import invoke_with_retry
from evaluate_braille import (
    CONFIGS, BRAILLE_DIR, DATA_RAW, RESULTS_DIR,
    load_dataset_records, get_gold_answers, get_braille_gold_answers,
    make_braille_prompt, extract_answer, compute_metrics,
)


def fix_file(filepath: str):
    """Re-run ERROR records in a result file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    with open(filepath) as f:
        records = [json.loads(l) for l in f if l.strip()]

    errors = [i for i, r in enumerate(records) if r.get("full_response", "").startswith("ERROR")]
    if not errors:
        print(f"  No errors in {filepath.name}")
        return

    # Parse filename to get config
    # Format: model_dataset_config_format_details.jsonl
    name = filepath.stem.replace("_details", "")
    parts = name.split("_")

    # Find config (G1-EN, G2-EN, etc.)
    config_name = None
    for i, p in enumerate(parts):
        if p in CONFIGS:
            config_name = p
            break
        # Handle hyphenated configs like G1-EN
        if i < len(parts) - 1 and f"{p}-{parts[i+1]}" in CONFIGS:
            config_name = f"{p}-{parts[i+1]}"
            break

    if not config_name:
        print(f"  Cannot parse config from {filepath.name}")
        return

    # Extract model, dataset, format from filename
    config_idx = name.index(config_name)
    model_dataset = name[:config_idx].rstrip("_")
    braille_format = name[config_idx + len(config_name):].lstrip("_") or "ascii"

    # Split model from dataset
    datasets_list = ["gsm8k", "aime24", "commonsenseqa", "hotpotqa", "2wikimultihopqa"]
    dataset_name = None
    model_name = None
    for ds in datasets_list:
        if model_dataset.endswith(ds):
            dataset_name = ds
            model_name = model_dataset[: -(len(ds) + 1)]
            break

    if not model_name or not dataset_name:
        print(f"  Cannot parse model/dataset from {filepath.name}")
        return

    config = CONFIGS[config_name]
    input_type = config["input"]
    output_type = config["output"]
    is_braille_input = input_type in ("grade1", "grade2")

    # Load source data
    if is_braille_input:
        source_records = load_dataset_records(dataset_name, input_type, braille_format)
    else:
        source_records = load_dataset_records(dataset_name)

    print(f"  Fixing {filepath.name}: {len(errors)} errors to retry (model={model_name}, ds={dataset_name}, cfg={config_name})")

    fixed = 0
    for idx in errors:
        if idx >= len(source_records):
            continue
        rec = source_records[idx]
        gold_answers = get_gold_answers(rec, dataset_name)
        prompt = make_braille_prompt(dataset_name, rec, config, braille_format)
        max_tok = 4096 if dataset_name == "aime24" else 1024

        try:
            response = invoke_with_retry(model_name, prompt, max_tokens=max_tok, temperature=0.0)
            predicted = extract_answer(response)

            if output_type in ("grade1", "grade2"):
                braille_gold = get_braille_gold_answers(rec, dataset_name, output_type, braille_format)
                metrics = compute_metrics(predicted, braille_gold, dataset_name)
            else:
                metrics = compute_metrics(predicted, gold_answers, dataset_name)

            records[idx] = {
                "id": rec.get("id", idx),
                "gold_answer": gold_answers,
                "predicted": predicted,
                "metrics": metrics,
                "full_response": response,
            }
            fixed += 1
        except Exception as e:
            error_str = str(e)
            if "credential" in error_str.lower() or "ExpiredToken" in error_str:
                print(f"    Credential error at record {idx}. Stopping fix.")
                break
            # Still failed - leave as is

        if (fixed) % 20 == 0 and fixed > 0:
            print(f"    Fixed {fixed}/{len(errors)}...")

    # Write back
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    remaining = sum(1 for r in records if r.get("full_response", "").startswith("ERROR"))
    print(f"    Done: fixed {fixed}, remaining errors: {remaining}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to fix")
    parser.add_argument("--all", action="store_true", help="Fix all files with errors")
    args = parser.parse_args()

    if args.all:
        for f in sorted(RESULTS_DIR.rglob("*_details.jsonl")):
            with open(f) as fh:
                first_lines = [json.loads(l) for l in fh.readlines()[:10]]
            if any(r.get("full_response", "").startswith("ERROR") for r in first_lines):
                fix_file(str(f))
    else:
        for f in args.files:
            fix_file(f)


if __name__ == "__main__":
    main()
