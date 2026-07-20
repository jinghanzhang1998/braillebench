"""
Evaluate models on reasoning datasets (original text, no Braille).

Runs each model on the test split of each dataset, extracts answers,
and computes accuracy. Results saved to data/results/.

Usage:
    python src/evaluate.py --models claude-haiku-4.5 llama4-maverick --datasets gsm8k commonsenseqa
    python src/evaluate.py --models all --datasets all --limit 50
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

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

# ---------------------------------------------------------------------------
# Dataset loaders: return list of {id, question, answer, choices?}
# ---------------------------------------------------------------------------

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def load_gsm8k():
    records = []
    with open(DATA_RAW / "gsm8k" / "test.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            answer = r["answer"].split("####")[-1].strip()
            records.append({"id": f"gsm8k_{i}", "question": r["question"], "answer": [answer]})
    return records


def load_aime24():
    records = []
    with open(DATA_RAW / "aime24" / "aime2024.jsonl") as f:
        for line in f:
            r = json.loads(line)
            records.append({"id": f"aime24_{r['id']}", "question": r["problem"], "answer": [str(r["answer"])]})
    return records


def load_2wikimultihopqa():
    records = []
    with open(DATA_RAW / "2wikimultihopqa" / "dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["golden_answers"] if r["golden_answers"] else [""],
            })
    return records


def load_hotpotqa():
    records = []
    with open(DATA_RAW / "hotpotqa" / "dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["golden_answers"] if r["golden_answers"] else [""],
            })
    return records


def load_commonsenseqa():
    records = []
    with open(DATA_RAW / "commonsenseqa" / "dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("metadata", {})
            choices = meta.get("choices", [])
            choice_text = " ".join(f"{c['label']}) {c['text']}" for c in choices) if choices else ""
            records.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["golden_answers"] if r["golden_answers"] else [""],
                "choices": choice_text,
            })
    return records


DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "aime24": load_aime24,
    "2wikimultihopqa": load_2wikimultihopqa,
    "hotpotqa": load_hotpotqa,
    "commonsenseqa": load_commonsenseqa,
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def make_prompt(dataset_name: str, record: dict) -> str:
    """Create evaluation prompt based on dataset type."""
    q = record["question"]

    if dataset_name == "gsm8k":
        return (
            f"Solve the following math problem step by step. "
            f"Put your final numerical answer within \\boxed{{}}.\n\n"
            f"Problem: {q}"
        )
    elif dataset_name == "aime24":
        return (
            f"Solve the following competition math problem. The answer is a non-negative integer between 0 and 999. "
            f"Think step by step. Put your final integer answer within \\boxed{{}}.\n\n"
            f"Problem: {q}"
        )
    elif dataset_name in ("2wikimultihopqa", "hotpotqa"):
        return (
            f"Answer the following question with a short phrase or name. "
            f"Write your answer after 'The answer is: '.\n\n"
            f"Question: {q}"
        )
    elif dataset_name == "commonsenseqa":
        choices = record.get("choices", "")
        return (
            f"Answer the following multiple-choice question. "
            f"Write only the answer text after 'The answer is: '.\n\n"
            f"Question: {q}\nChoices: {choices}"
        )
    else:
        return f"Question: {q}\nAnswer:"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def clean_response(response: str) -> str:
    """Remove model-specific artifacts from response."""
    response = re.sub(r"<\|eot_id\|>.*", "", response)
    response = re.sub(r"<\|end_of_text\|>.*", "", response)
    response = re.sub(r"</s>.*", "", response)
    return response.strip()


def extract_answer(response: str) -> str:
    """Extract the answer from model response."""
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
            # Strip choice letter prefix like "A) ", "B) ", etc.
            ans = re.sub(r"^[A-E]\)\s*", "", ans)
            return ans
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    ans = lines[-1].strip().strip(".") if lines else ""
    ans = ans.strip("$").strip()
    # Strip choice letter prefix
    ans = re.sub(r"^[A-E]\)\s*", "", ans)
    return ans


# ---------------------------------------------------------------------------
# Evaluation metrics (following FlashRAG standard)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (FlashRAG standard).

    Pipeline: lowercase -> remove punctuation -> remove articles -> whitespace fix.
    """
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


def exact_match(predicted: str, gold_answers: list[str]) -> float:
    """Exact match: 1.0 if normalized prediction == any gold answer."""
    norm_pred = normalize_answer(predicted)
    for gold in gold_answers:
        if normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def sub_exact_match(predicted: str, gold_answers: list[str]) -> float:
    """Substring match: 1.0 if any normalized gold is contained in prediction."""
    norm_pred = normalize_answer(predicted)
    for gold in gold_answers:
        if normalize_answer(gold) in norm_pred:
            return 1.0
    return 0.0


def token_f1(predicted: str, gold_answers: list[str]) -> float:
    """Token-level F1 score (max over all gold answers)."""
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


def numeric_match(predicted: str, gold: str) -> float:
    """For math datasets: extract last number and compare with tolerance."""
    pred_nums = re.findall(r"-?\d[\d,]*\.?\d*", predicted)
    gold_nums = re.findall(r"-?\d[\d,]*\.?\d*", gold)
    if pred_nums and gold_nums:
        try:
            pred_val = float(pred_nums[-1].replace(",", ""))
            gold_val = float(gold_nums[-1].replace(",", ""))
            return 1.0 if abs(pred_val - gold_val) < 1e-6 else 0.0
        except ValueError:
            pass
    return exact_match(predicted, [gold])


def math_verify_match(predicted: str, gold: str) -> float:
    """Use math-verify (SymPy-based) for symbolic math equivalence checking.

    Follows OpenCompass MATHVerifyEvaluator approach.
    """
    try:
        gold_parsed = parse(
            f"${gold}$",
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        )
        pred_parsed = parse(
            predicted,
            extraction_config=[
                LatexExtractionConfig(
                    try_extract_without_anchor=True,
                    boxed_match_priority=0,
                ),
                ExprExtractionConfig(),
            ],
        )
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        return numeric_match(predicted, gold)


def compute_metrics(predicted: str, gold_answers: list[str], dataset_name: str) -> dict:
    """Compute all relevant metrics for a prediction.

    Math datasets: numeric match + math-verify (SymPy symbolic equivalence).
    QA datasets: EM + Sub_EM + F1 (FlashRAG standard).
    """
    metrics = {
        "em": exact_match(predicted, gold_answers),
        "sub_em": sub_exact_match(predicted, gold_answers),
        "f1": token_f1(predicted, gold_answers),
    }
    if dataset_name in ("gsm8k", "aime24"):
        metrics["numeric"] = numeric_match(predicted, gold_answers[0])
        metrics["math_verify"] = math_verify_match(predicted, gold_answers[0])
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model_on_dataset(model_name: str, dataset_name: str, limit: int = None) -> dict:
    """Run a model on a dataset and return results."""
    loader = DATASET_LOADERS[dataset_name]
    records = loader()
    if limit:
        records = records[:limit]

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} on {dataset_name} ({len(records)} samples)")
    print(f"{'='*60}")

    results = []
    metric_accum = {}
    start_time = time.time()

    for i, rec in enumerate(records):
        prompt = make_prompt(dataset_name, rec)
        max_tok = 4096 if dataset_name == "aime24" else 1024
        gold_answers = rec["answer"] if isinstance(rec["answer"], list) else [rec["answer"]]

        try:
            response = invoke_with_retry(model_name, prompt, max_tokens=max_tok, temperature=0.0)
            predicted = extract_answer(response)
            metrics = compute_metrics(predicted, gold_answers, dataset_name)
        except Exception as e:
            error_str = str(e)
            if "credential" in error_str.lower() or "ExpiredToken" in error_str or "ReadTimeout" in error_str:
                print(f"\n  FATAL: Credential/connection error at record {i}. Aborting (no partial save).")
                raise RuntimeError(f"Credential error at record {i}/{len(records)}: {error_str[:100]}")
            response = f"ERROR: {e}"
            predicted = ""
            metrics = {"em": 0.0, "sub_em": 0.0, "f1": 0.0}
            if dataset_name in ("gsm8k", "aime24"):
                metrics["numeric"] = 0.0

        for k, v in metrics.items():
            metric_accum.setdefault(k, []).append(v)

        results.append({
            "id": rec["id"],
            "question": rec["question"],
            "gold_answer": gold_answers,
            "predicted": predicted,
            "metrics": metrics,
            "full_response": response,
        })

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            em_so_far = sum(metric_accum["em"]) / len(metric_accum["em"]) * 100
            print(f"  [{i+1}/{len(records)}] EM={em_so_far:.1f}% elapsed={elapsed:.0f}s")

    elapsed = time.time() - start_time
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "total": len(records),
        "elapsed_seconds": elapsed,
    }
    for k, vals in metric_accum.items():
        summary[k] = sum(vals) / len(vals)

    primary = "math_verify" if (dataset_name in ("gsm8k", "aime24") and "math_verify" in summary) else "em"
    score = summary.get(primary, 0) * 100
    extra = f" | numeric={summary.get('numeric',0)*100:.1f}%" if "numeric" in summary else ""
    print(f"\n  RESULT: {primary}={score:.1f}%{extra} | EM={summary.get('em',0)*100:.1f}% | "
          f"F1={summary.get('f1',0)*100:.1f}% | sub_EM={summary.get('sub_em',0)*100:.1f}% | {elapsed:.0f}s")
    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--models", nargs="+", default=["claude-haiku-4.5"],
                        help="Model names to evaluate (or 'all')")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k"],
                        help="Dataset names to evaluate (or 'all')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per dataset (for testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/results/)")
    args = parser.parse_args()

    if args.models == ["all"]:
        model_list = list(MODELS.keys())
    else:
        model_list = args.models

    if args.datasets == ["all"]:
        dataset_list = list(DATASET_LOADERS.keys())
    else:
        dataset_list = args.datasets

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in model_list:
        for dataset_name in dataset_list:
            result = evaluate_model_on_dataset(model_name, dataset_name, args.limit)

            # Save detailed results
            detail_path = output_dir / f"{model_name}_{dataset_name}_details.jsonl"
            with open(detail_path, "w") as f:
                for r in result["results"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            all_summaries.append(result["summary"])

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Model':<22} {'Dataset':<16} {'EM':>6} {'Sub_EM':>7} {'F1':>6} {'Numeric':>8} {'MathVfy':>8} {'N':>5}")
    print("-" * 90)
    for s in all_summaries:
        em = f"{s.get('em', 0)*100:.1f}%"
        sub_em = f"{s.get('sub_em', 0)*100:.1f}%"
        f1 = f"{s.get('f1', 0)*100:.1f}%"
        numeric = f"{s.get('numeric', 0)*100:.1f}%" if "numeric" in s else "   -"
        math_v = f"{s.get('math_verify', 0)*100:.1f}%" if "math_verify" in s else "   -"
        print(f"{s['model']:<22} {s['dataset']:<16} {em:>6} {sub_em:>7} {f1:>6} {numeric:>8} {math_v:>8} {s['total']:>5}")


if __name__ == "__main__":
    main()
