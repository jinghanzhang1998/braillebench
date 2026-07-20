"""
Evaluate models on Braille benchmark across all input-output configurations.

Configurations:
  - EN→EN: English input, English output (baseline)
  - EN→G1/G2: English input, Braille output (Grade 1/2)
  - G1/G2→EN: Braille input, English output
  - G1→G1, G2→G2: Braille input, Braille output

Each Braille config tested with 3 formats: ascii, unicode, dots.

Usage:
    python src/evaluate_braille.py --models claude-haiku-4.5 --datasets gsm8k --configs G2-EN --formats ascii --limit 10
    python src/evaluate_braille.py --models all --datasets all --configs all --formats all --limit 50
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

sys.path.insert(0, str(Path(__file__).parent))
from model_client import MODELS, invoke_with_retry

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results" / "braille_eval"
BRAILLE_DIR = Path(__file__).parent.parent / "data" / "braille"

CONFIGS = {
    "EN-EN": {"input": "english", "output": "english"},
    "EN-G1": {"input": "english", "output": "grade1"},
    "EN-G2": {"input": "english", "output": "grade2"},
    "G1-EN": {"input": "grade1", "output": "english"},
    "G2-EN": {"input": "grade2", "output": "english"},
    "G1-G1": {"input": "grade1", "output": "grade1"},
    "G2-G2": {"input": "grade2", "output": "grade2"},
}

FORMATS = ["ascii", "unicode", "dots"]

DATASETS = ["gsm8k", "aime24", "commonsenseqa", "hotpotqa", "2wikimultihopqa"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def load_dataset_records(dataset_name: str, braille_grade: str = None, braille_format: str = None):
    """Load dataset records. If braille params given, load from braille dir."""
    if braille_grade and braille_format:
        path = BRAILLE_DIR / dataset_name / braille_grade / f"{braille_format}.jsonl"
    else:
        raw_paths = {
            "gsm8k": DATA_RAW / "gsm8k" / "test.jsonl",
            "aime24": DATA_RAW / "aime24" / "aime2024.jsonl",
            "commonsenseqa": DATA_RAW / "commonsenseqa" / "dev.jsonl",
            "hotpotqa": DATA_RAW / "hotpotqa" / "dev.jsonl",
            "2wikimultihopqa": DATA_RAW / "2wikimultihopqa" / "dev.jsonl",
        }
        path = raw_paths[dataset_name]

    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def get_question_text(record: dict, dataset_name: str, is_braille_input: bool):
    """Get the question text from a record."""
    if is_braille_input:
        return record["braille_question"]
    else:
        if dataset_name == "aime24":
            return record.get("problem", record.get("question", ""))
        return record["question"]


def get_choices_text(record: dict, is_braille_input: bool):
    """Get choices text if available."""
    if is_braille_input and "braille_choices" in record:
        return " ".join(record["braille_choices"])
    elif "choices" in record:
        if isinstance(record["choices"], list):
            return " ".join(record["choices"])
        return record["choices"]
    # FlashRAG metadata format
    meta = record.get("metadata", {})
    if isinstance(meta, dict) and "choices" in meta:
        choices = meta["choices"]
        if isinstance(choices, list) and choices:
            if isinstance(choices[0], dict):
                return " ".join(f"{c['label']}) {c['text']}" for c in choices)
    return ""


def get_gold_answers(record: dict, dataset_name: str):
    """Get gold answer(s) as a list (English)."""
    if "golden_answers" in record:
        return record["golden_answers"]
    if "answer" in record:
        ans = record["answer"]
        if dataset_name == "gsm8k" and "####" in str(ans):
            return [str(ans).split("####")[-1].strip()]
        return [str(ans)] if not isinstance(ans, list) else ans
    return [""]


def get_braille_gold_answers(record: dict, dataset_name: str, grade: str, braille_format: str):
    """Get gold answer(s) in Braille for braille-output evaluation.

    Translates the short final answer to Braille on the fly.
    """
    import louis
    table = "en-ueb-g1.ctb" if "1" in grade else "en-ueb-g2.ctb"

    # If the braille data file already has braille_golden_answers, use them
    if "braille_golden_answers" in record:
        return record["braille_golden_answers"]

    # Otherwise translate the English gold answer to braille
    english_gold = get_gold_answers(record, dataset_name)
    braille_gold = []
    for ans in english_gold:
        # For GSM8K, the gold is just a number like "18"
        # For AIME24, it's an integer like "204"
        braille_gold.append(louis.translateString([table], ans))
    return braille_gold


def make_braille_prompt(dataset_name: str, record: dict, config: dict, braille_format: str = None):
    """Create evaluation prompt for any input-output configuration."""
    input_type = config["input"]
    output_type = config["output"]

    is_braille_input = input_type in ("grade1", "grade2")
    q = get_question_text(record, dataset_name, is_braille_input)
    choices = get_choices_text(record, is_braille_input)

    # Output instruction
    if output_type == "english":
        output_instr = "Answer in plain English."
    else:
        grade_name = "Grade 1 (uncontracted)" if "1" in output_type else "Grade 2 (contracted)"
        fmt_name = braille_format or "ascii"
        if fmt_name == "ascii":
            output_instr = f"Answer in {grade_name} Braille ASCII notation."
        elif fmt_name == "unicode":
            output_instr = f"Answer in {grade_name} Unicode Braille characters."
        else:
            output_instr = f"Answer in {grade_name} Braille dot notation (e.g., 1-2-5 for dots 1,2,5)."

    # Input context
    if is_braille_input:
        grade_name = "Grade 1 (uncontracted)" if "1" in input_type else "Grade 2 (contracted)"
        input_note = f"The following question is written in {grade_name} Braille ASCII notation. "
    else:
        input_note = ""

    # Dataset-specific prompts
    if dataset_name == "gsm8k":
        return (
            f"{input_note}Solve the following math problem step by step. "
            f"{output_instr} Put your final numerical answer within \\boxed{{}}.\n\n"
            f"Problem: {q}"
        )
    elif dataset_name == "aime24":
        return (
            f"{input_note}Solve the following competition math problem. "
            f"The answer is a non-negative integer between 0 and 999. "
            f"Think step by step. {output_instr} Put your final integer answer within \\boxed{{}}.\n\n"
            f"Problem: {q}"
        )
    elif dataset_name in ("2wikimultihopqa", "hotpotqa"):
        return (
            f"{input_note}Answer the following question with a short phrase or name. "
            f"{output_instr} Write your answer after 'The answer is: '.\n\n"
            f"Question: {q}"
        )
    elif dataset_name == "commonsenseqa":
        return (
            f"{input_note}Answer the following multiple-choice question. "
            f"{output_instr} Write only the answer text after 'The answer is: '.\n\n"
            f"Question: {q}\nChoices: {choices}"
        )
    else:
        return f"{input_note}Question: {q}\n{output_instr}"


# ---------------------------------------------------------------------------
# Answer extraction and metrics (reused from evaluate.py)
# ---------------------------------------------------------------------------


def clean_response(response: str) -> str:
    response = re.sub(r"<\|eot_id\|>.*", "", response)
    response = re.sub(r"<\|end_of_text\|>.*", "", response)
    response = re.sub(r"</s>.*", "", response)
    return response.strip()


def extract_answer(response: str) -> str:
    response = clean_response(response)
    patterns = [
        r"[Tt]he answer is:?\s*\*?\*?(.+?)\*?\*?(?:\.|$)",
        r"[Ff]inal answer:?\s*\*?\*?(.+?)\*?\*?(?:\.|$)",
        r"\\boxed\{([^}]+)\}",
        r"\$\\boxed\{([^}]+)\}\$",
        r"\*\*(\d+)\*\*\s*$",
        r"\*\*(.+?)\*\*\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        if matches:
            ans = matches[-1].strip().strip(".")
            ans = ans.strip("$").strip()
            return ans
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    ans = lines[-1].strip().strip(".") if lines else ""
    ans = ans.strip("$").strip()
    return ans


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(predicted: str, gold_answers: list) -> float:
    norm_pred = normalize_answer(predicted)
    for gold in gold_answers:
        if normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def sub_exact_match(predicted: str, gold_answers: list) -> float:
    norm_pred = normalize_answer(predicted)
    for gold in gold_answers:
        if normalize_answer(gold) in norm_pred:
            return 1.0
    return 0.0


def token_f1(predicted: str, gold_answers: list) -> float:
    norm_pred = normalize_answer(predicted)
    pred_tokens = norm_pred.split()
    best_f1 = 0.0
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        gold_tokens = norm_gold.split()
        if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
            continue
        if norm_gold in ["yes", "no", "noanswer"] and norm_pred != norm_gold:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def math_verify_match(predicted: str, gold: str) -> float:
    try:
        gold_parsed = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred_parsed = parse(predicted, extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=0),
            ExprExtractionConfig(),
        ])
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        pred_nums = re.findall(r"-?\d[\d,]*\.?\d*", predicted)
        gold_nums = re.findall(r"-?\d[\d,]*\.?\d*", gold)
        if pred_nums and gold_nums:
            try:
                return 1.0 if abs(float(pred_nums[-1].replace(",","")) - float(gold_nums[-1].replace(",",""))) < 1e-6 else 0.0
            except ValueError:
                pass
        return 0.0


def compute_metrics(predicted: str, gold_answers: list, dataset_name: str) -> dict:
    metrics = {
        "em": exact_match(predicted, gold_answers),
        "sub_em": sub_exact_match(predicted, gold_answers),
        "f1": token_f1(predicted, gold_answers),
    }
    if dataset_name in ("gsm8k", "aime24"):
        metrics["math_verify"] = math_verify_match(predicted, gold_answers[0])
    return metrics


def back_translate(braille_text: str, grade: str) -> str:
    """Back-translate Braille ASCII to English for reliable answer comparison."""
    import louis
    table = "en-ueb-g1.ctb" if "1" in grade else "en-ueb-g2.ctb"
    try:
        return louis.backTranslateString([table], braille_text)
    except Exception:
        return braille_text


def is_english_not_braille(text: str) -> bool:
    """Detect if output is English (or English-mixed) rather than valid Braille ASCII.

    Braille ASCII (liblouis) uses lowercase letters, `,` for capitals, and `#a-j`
    for digits. Uppercase A-Z or literal digits 0-9 signal the model wrote English
    instead of following the Braille-output instruction.
    """
    if re.search(r"[A-Z]", text):   # uppercase letter → English
        return True
    if re.search(r"[0-9]", text):   # literal digit → English (braille uses #a-j)
        return True
    return False


def compute_braille_output_metrics(predicted_braille: str, gold_english: list,
                                   grade: str, dataset_name: str) -> dict:
    """Metrics for braille-output configs.

    Two-stage validation:
    1. If output contains English signals (uppercase A-Z or literal digits 0-9),
       the model failed to write Braille → score 0 (records wrote_english=True).
    2. Otherwise back-translate the braille to English and compare to English gold
       (avoids information loss from normalizing away braille symbols).
    """
    wrote_english = is_english_not_braille(predicted_braille)
    if wrote_english:
        metrics = {"em": 0.0, "sub_em": 0.0, "f1": 0.0, "wrote_english": True,
                   "back_translated": ""}
        if dataset_name in ("gsm8k", "aime24"):
            metrics["math_verify"] = 0.0
        return metrics

    pred_en = back_translate(predicted_braille, grade)
    metrics = {
        "em": exact_match(pred_en, gold_english),
        "sub_em": sub_exact_match(pred_en, gold_english),
        "f1": token_f1(pred_en, gold_english),
        "wrote_english": False,
        "back_translated": pred_en,
    }
    if dataset_name in ("gsm8k", "aime24"):
        metrics["math_verify"] = math_verify_match(pred_en, gold_english[0])
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(model_name: str, dataset_name: str, config_name: str,
             braille_format: str = None, limit: int = None) -> dict:
    """Run evaluation with checkpoint/resume support.

    Results are written incrementally to the output file. If the file already
    has N lines, evaluation resumes from record N.
    """
    config = CONFIGS[config_name]
    input_type = config["input"]
    output_type = config["output"]
    is_braille_input = input_type in ("grade1", "grade2")

    # Load data
    if is_braille_input:
        records = load_dataset_records(dataset_name, input_type, braille_format or "ascii")
    else:
        records = load_dataset_records(dataset_name)

    if limit:
        records = records[:limit]

    # Determine output path
    tag = f"{model_name}_{dataset_name}_{config_name}"
    if braille_format:
        tag += f"_{braille_format}"
    detail_path = RESULTS_DIR / f"{tag}_details.jsonl"

    # Resume: count existing lines
    start_idx = 0
    if detail_path.exists():
        with open(detail_path) as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(records):
            print(f"\n  [{model_name}] {dataset_name} | {config_name} | COMPLETE ({start_idx} records)")
            return {"summary": {}, "results": []}
        print(f"\n  [{model_name}] {dataset_name} | {config_name} | RESUME from {start_idx}/{len(records)}")
    else:
        print(f"\n  [{model_name}] {dataset_name} | {config_name} | fmt={braille_format or 'N/A'} | n={len(records)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_accum = {}
    start_time = time.time()

    with open(detail_path, "a") as out_f:
        for i in range(start_idx, len(records)):
            rec = records[i]
            gold_answers = get_gold_answers(rec, dataset_name)
            prompt = make_braille_prompt(dataset_name, rec, config, braille_format)
            max_tok = 4096 if dataset_name == "aime24" else 1024

            try:
                response = invoke_with_retry(model_name, prompt, max_tokens=max_tok, temperature=0.0)
                predicted = extract_answer(response)

                if output_type in ("grade1", "grade2"):
                    # Braille output: back-translate to English then compare to English gold.
                    # This avoids information loss from normalize stripping braille symbols
                    # (critical for Grade 2 contractions).
                    metrics = compute_braille_output_metrics(predicted, gold_answers, output_type, dataset_name)
                else:
                    metrics = compute_metrics(predicted, gold_answers, dataset_name)
            except Exception as e:
                error_str = str(e)
                if "credential" in error_str.lower() or "ExpiredToken" in error_str or "ReadTimeout" in error_str:
                    print(f"\n  PAUSED at record {i}/{len(records)} (credential error). Resume later.")
                    raise RuntimeError(f"Credential error at record {i}/{len(records)}: {error_str[:100]}")
                response = f"ERROR: {e}"
                predicted = ""
                metrics = {"em": 0.0, "sub_em": 0.0, "f1": 0.0}
                if dataset_name in ("gsm8k", "aime24"):
                    metrics["math_verify"] = 0.0

            for k, v in metrics.items():
                metric_accum.setdefault(k, []).append(v)

            result_line = {
                "id": rec.get("id", i),
                "gold_answer": gold_answers,
                "predicted": predicted,
                "metrics": metrics,
                "full_response": response,
            }
            out_f.write(json.dumps(result_line, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                primary = "math_verify" if (dataset_name in ("gsm8k", "aime24") and output_type == "english") else "em"
                if primary in metric_accum:
                    score = sum(metric_accum[primary]) / len(metric_accum[primary]) * 100
                    print(f"    [{i+1}/{len(records)}] {primary}={score:.1f}% ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "config": config_name,
        "braille_format": braille_format,
        "total": len(records),
        "elapsed_seconds": elapsed,
    }
    for k, vals in metric_accum.items():
        summary[k] = sum(vals) / len(vals)

    if output_type in ("grade1", "grade2"):
        primary = "em"
    else:
        primary = "math_verify" if dataset_name in ("gsm8k", "aime24") else "em"
    if primary in summary:
        print(f"    DONE: {primary}={summary[primary]*100:.1f}% | F1={summary['f1']*100:.1f}% ({elapsed:.0f}s)")

    return {"summary": summary, "results": []}


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on Braille benchmark")
    parser.add_argument("--models", nargs="+", default=["claude-haiku-4.5"])
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"])
    parser.add_argument("--configs", nargs="+", default=["G2-EN"],
                        help="Configurations: EN-EN, EN-G1, EN-G2, G1-EN, G2-EN, G1-G1, G2-G2, or 'all'")
    parser.add_argument("--formats", nargs="+", default=["ascii"],
                        help="Braille formats: ascii, unicode, dots, or 'all'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.models == ["all"]:
        model_list = list(MODELS.keys())
    else:
        model_list = args.models

    if args.datasets == ["all"]:
        dataset_list = DATASETS
    else:
        dataset_list = args.datasets

    if args.configs == ["all"]:
        config_list = list(CONFIGS.keys())
    else:
        config_list = args.configs

    if args.formats == ["all"]:
        format_list = FORMATS
    else:
        format_list = args.formats

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in model_list:
        for dataset_name in dataset_list:
            for config_name in config_list:
                config = CONFIGS[config_name]
                needs_format = config["input"] != "english" or config["output"] != "english"

                if needs_format:
                    fmts = format_list
                else:
                    fmts = [None]

                for fmt in fmts:
                    # Check if already done
                    tag = f"{model_name}_{dataset_name}_{config_name}"
                    if fmt:
                        tag += f"_{fmt}"
                    detail_path = output_dir / f"{tag}_details.jsonl"

                    result = evaluate(model_name, dataset_name, config_name, fmt, args.limit)
                    all_summaries.append(result["summary"])

    # Save all summaries
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Print table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Dataset':<14} {'Config':<8} {'Fmt':<8} {'EM':>6} {'F1':>6} {'MathV':>6}")
    print("-" * 90)
    for s in all_summaries:
        em = f"{s['em']*100:.1f}%"
        f1 = f"{s['f1']*100:.1f}%"
        mv = f"{s.get('math_verify',0)*100:.1f}%" if "math_verify" in s else "  -"
        fmt = s.get("braille_format") or "-"
        print(f"{s['model']:<20} {s['dataset']:<14} {s['config']:<8} {fmt:<8} {em:>6} {f1:>6} {mv:>6}")


if __name__ == "__main__":
    main()
