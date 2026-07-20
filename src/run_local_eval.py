"""
Run evaluation using local GPU models (Qwen3-1.5B, etc.)

This script replaces the Bedrock API calls with local model inference.
Run on a machine with GPU.

Usage:
    # Baseline (EN-EN)
    python src/run_local_eval.py --model qwen3-1.5b --mode baseline

    # Braille evaluation
    python src/run_local_eval.py --model qwen3-1.5b --mode braille --configs G1-EN G2-EN --formats ascii
"""

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

sys.path.insert(0, str(Path(__file__).parent))
from model_client_local import invoke_local_model, invoke_local_batch, LOCAL_MODELS

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
BRAILLE_DIR = Path(__file__).parent.parent / "data" / "braille"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

DATASETS = {
    "gsm8k": {"path": DATA_RAW / "gsm8k" / "test.jsonl", "question_field": "problem_or_question", "limit": None},
    "aime24": {"path": DATA_RAW / "aime24" / "aime2024.jsonl", "question_field": "problem", "limit": None},
    "commonsenseqa": {"path": DATA_RAW / "commonsenseqa" / "dev.jsonl", "question_field": "question", "limit": None},
    "hotpotqa": {"path": DATA_RAW / "hotpotqa" / "dev.jsonl", "question_field": "question", "limit": 1500},
    "2wikimultihopqa": {"path": DATA_RAW / "2wikimultihopqa" / "dev.jsonl", "question_field": "question", "limit": 1500},
}

BRAILLE_CONFIGS = {
    "EN-EN": {"input": "english", "output": "english"},
    "G1-EN": {"input": "grade1", "output": "english"},
    "G2-EN": {"input": "grade2", "output": "english"},
    "EN-G1": {"input": "english", "output": "grade1"},
    "EN-G2": {"input": "english", "output": "grade2"},
    "G1-G1": {"input": "grade1", "output": "grade1"},
    "G2-G2": {"input": "grade2", "output": "grade2"},
}


# --- Metrics (same as evaluate.py / evaluate_braille.py) ---

def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(predicted, gold_answers):
    norm_pred = normalize_answer(predicted)
    return 1.0 if any(normalize_answer(g) == norm_pred for g in gold_answers) else 0.0


def token_f1(predicted, gold_answers):
    norm_pred = normalize_answer(predicted)
    pred_tokens = norm_pred.split()
    best_f1 = 0.0
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        gold_tokens = norm_gold.split()
        if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gold: continue
        if norm_gold in ["yes", "no", "noanswer"] and norm_pred != norm_gold: continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0: continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def math_verify_match(predicted, gold):
    try:
        gold_parsed = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred_parsed = parse(predicted, extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=0),
            ExprExtractionConfig()])
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        return 0.0


def clean_response(response):
    response = re.sub(r"<\|[^|]+\|>.*", "", response)
    return response.strip()


def extract_answer(response):
    response = clean_response(response)
    patterns = [
        r"[Tt]he answer is:?\s*\*?\*?(.+?)\*?\*?(?:\.|$)",
        r"[Ff]inal answer:?\s*\*?\*?(.+?)\*?\*?(?:\.|$)",
        r"\\boxed\{([^}]+)\}",
        r"\*\*(\d+)\*\*\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        if matches:
            return matches[-1].strip().strip(".").strip("$")
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1].strip(".").strip("$") if lines else ""


# --- Data Loading ---

def load_records(dataset_name, braille_grade=None, braille_format=None, limit=None):
    if braille_grade and braille_format:
        path = BRAILLE_DIR / dataset_name / braille_grade / f"{braille_format}.jsonl"
    else:
        path = DATASETS[dataset_name]["path"]

    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    ds_limit = limit or DATASETS[dataset_name]["limit"]
    if ds_limit:
        records = records[:ds_limit]
    return records


def get_gold_answers(record, dataset_name):
    if "golden_answers" in record:
        return record["golden_answers"]
    if "answer" in record:
        ans = record["answer"]
        if dataset_name == "gsm8k" and "####" in str(ans):
            return [str(ans).split("####")[-1].strip()]
        return [str(ans)] if not isinstance(ans, list) else ans
    return [""]


def get_braille_gold(record, dataset_name, grade, fmt):
    import louis
    if "braille_golden_answers" in record:
        return record["braille_golden_answers"]
    gold_en = get_gold_answers(record, dataset_name)
    table = "en-ueb-g1.ctb" if "1" in grade else "en-ueb-g2.ctb"
    return [louis.translateString([table], a) for a in gold_en]


# --- Prompt Construction ---

def make_prompt(dataset_name, record, config, braille_format=None):
    input_type = config["input"]
    output_type = config["output"]
    is_braille_input = input_type in ("grade1", "grade2")

    if is_braille_input:
        q = record["braille_question"]
    else:
        q = record.get("problem", record.get("question", ""))

    if output_type == "english":
        output_instr = "Answer in plain English."
    else:
        grade_name = "Grade 1 (uncontracted)" if "1" in output_type else "Grade 2 (contracted)"
        output_instr = f"Answer in {grade_name} Braille ASCII notation."

    input_note = ""
    if is_braille_input:
        grade_name = "Grade 1 (uncontracted)" if "1" in input_type else "Grade 2 (contracted)"
        input_note = f"The following question is written in {grade_name} Braille ASCII notation. "

    if dataset_name in ("gsm8k", "aime24"):
        return f"{input_note}Solve the following math problem step by step. {output_instr} Put your final numerical answer within \\boxed{{}}.\n\nProblem: {q}"
    elif dataset_name in ("hotpotqa", "2wikimultihopqa"):
        return f"{input_note}Answer the following question with a short phrase or name. {output_instr} Write your answer after 'The answer is: '.\n\nQuestion: {q}"
    elif dataset_name == "commonsenseqa":
        choices = ""
        if is_braille_input and "braille_choices" in record:
            choices = " ".join(record["braille_choices"])
        elif "choices" in record:
            choices = " ".join(record["choices"]) if isinstance(record["choices"], list) else record["choices"]
        else:
            meta = record.get("metadata", {})
            if isinstance(meta, dict) and "choices" in meta:
                choices = " ".join(f"{c['label']}) {c['text']}" for c in meta["choices"])
        return f"{input_note}Answer the following multiple-choice question. {output_instr} Write only the answer text after 'The answer is: '.\n\nQuestion: {q}\nChoices: {choices}"
    return f"{input_note}Question: {q}\n{output_instr}"


# --- Main ---

def run_evaluation(model_name, dataset_name, config_name, braille_format=None, output_dir=None):
    config = BRAILLE_CONFIGS[config_name]
    input_type = config["input"]
    output_type = config["output"]
    is_braille_input = input_type in ("grade1", "grade2")

    records = load_records(dataset_name,
                          braille_grade=input_type if is_braille_input else None,
                          braille_format=braille_format if is_braille_input else None)

    tag = f"{model_name}_{dataset_name}_{config_name}"
    if braille_format and config_name != "EN-EN":
        tag += f"_{braille_format}"

    out_dir = Path(output_dir) if output_dir else RESULTS_DIR / "local"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{tag}_details.jsonl"

    if detail_path.exists():
        print(f"SKIP {tag}")
        return

    print(f"\n{'='*60}")
    print(f"{tag} ({len(records)} records)")
    print(f"{'='*60}")

    results = []
    metric_accum = {}
    start = time.time()

    for i, rec in enumerate(records):
        gold_answers = get_gold_answers(rec, dataset_name)
        prompt = make_prompt(dataset_name, rec, config, braille_format)
        max_tok = 4096 if dataset_name == "aime24" else 1024

        response = invoke_local_model(model_name, prompt, max_tokens=max_tok, temperature=0.0)
        predicted = extract_answer(response)

        if output_type in ("grade1", "grade2"):
            braille_gold = get_braille_gold(rec, dataset_name, output_type, braille_format)
            metrics = {"em": exact_match(predicted, braille_gold), "f1": token_f1(predicted, braille_gold)}
        else:
            metrics = {"em": exact_match(predicted, gold_answers), "f1": token_f1(predicted, gold_answers)}
            if dataset_name in ("gsm8k", "aime24"):
                metrics["math_verify"] = math_verify_match(predicted, gold_answers[0])

        for k, v in metrics.items():
            metric_accum.setdefault(k, []).append(v)

        results.append({"id": rec.get("id", i), "predicted": predicted, "metrics": metrics, "full_response": response})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            primary = "math_verify" if (dataset_name in ("gsm8k", "aime24") and output_type == "english") else "em"
            score = sum(metric_accum.get(primary, [0])) / max(len(metric_accum.get(primary, [1])), 1) * 100
            print(f"  [{i+1}/{len(records)}] {primary}={score:.1f}% ({elapsed:.0f}s)")

    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    summary = {k: sum(v)/len(v) for k, v in metric_accum.items()}
    print(f"  DONE: EM={summary['em']*100:.1f}% F1={summary['f1']*100:.1f}% ({elapsed:.0f}s) -> {detail_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-1.5b")
    parser.add_argument("--mode", choices=["baseline", "braille"], required=True)
    parser.add_argument("--configs", nargs="+", default=["G1-EN", "G2-EN"])
    parser.add_argument("--formats", nargs="+", default=["ascii"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    datasets = args.datasets or list(DATASETS.keys())

    if args.mode == "baseline":
        for ds in datasets:
            run_evaluation(args.model, ds, "EN-EN", output_dir=args.output_dir)
    else:
        for cfg in args.configs:
            for fmt in args.formats:
                for ds in datasets:
                    run_evaluation(args.model, ds, cfg, fmt, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
