"""
Translate all datasets to Braille (Grade 1 & Grade 2, 3 output formats each).

Output structure:
  data/braille/<dataset>/<grade>/<format>.jsonl

Each record contains original fields plus braille_question (and braille_choices if applicable).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("LOUIS_TABLEPATH", "/Users/jinghanz/local/share/liblouis/tables")

from translator import translate_text
from latex_preprocess import preprocess_text

OUTPUT_BASE = Path(__file__).parent.parent / "data" / "braille"

DATASETS = {
    "gsm8k": {
        "path": "data/raw/gsm8k/test.jsonl",
        "question_field": "question",
        "has_choices": False,
    },
    "aime24": {
        "path": "data/raw/aime24/aime2024.jsonl",
        "question_field": "problem",
        "has_choices": False,
    },
    "commonsenseqa": {
        "path": "data/raw/commonsenseqa/dev.jsonl",
        "question_field": "question",
        "has_choices": True,
    },
    "hotpotqa": {
        "path": "data/raw/hotpotqa/dev.jsonl",
        "question_field": "question",
        "has_choices": False,
    },
    "2wikimultihopqa": {
        "path": "data/raw/2wikimultihopqa/dev.jsonl",
        "question_field": "question",
        "has_choices": False,
    },
}

GRADES = ["grade1", "grade2"]
FORMATS = ["ascii", "unicode", "dots"]


def unicode_to_dots(s: str) -> str:
    """Convert Unicode Braille string to dot notation."""
    cells = []
    for ch in s:
        if ch == " " or ord(ch) < 0x2800 or ord(ch) > 0x28FF:
            cells.append(ch)
            continue
        offset = ord(ch) - 0x2800
        if offset == 0:
            cells.append("0")
            continue
        dots = []
        for i in range(8):
            if offset & (1 << i):
                dots.append(str(i + 1))
        cells.append("-".join(dots))
    return " ".join(cells)


def get_choices(record: dict) -> list[str] | None:
    """Extract choices from FlashRAG metadata format."""
    meta = record.get("metadata", {})
    if isinstance(meta, dict) and "choices" in meta:
        choices = meta["choices"]
        if isinstance(choices, list) and choices:
            if isinstance(choices[0], dict):
                return [f"{c['label']}) {c['text']}" for c in choices]
            return choices
    return None


def translate_to_format(text: str, grade: str, fmt: str) -> str:
    """Translate text to specified braille format."""
    if fmt == "ascii":
        return translate_text(text, grade, "ascii")
    elif fmt == "unicode":
        return translate_text(text, grade, "unicode")
    elif fmt == "dots":
        unicode_out = translate_text(text, grade, "unicode")
        return unicode_to_dots(unicode_out)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def translate_dataset(dataset_name: str, config: dict):
    """Translate one dataset into all grades and formats."""
    project_root = Path(__file__).parent.parent
    input_path = project_root / config["path"]
    question_field = config["question_field"]
    has_choices = config["has_choices"]

    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(records)} records)")
    print(f"{'='*60}")

    for grade in GRADES:
        for fmt in FORMATS:
            out_dir = OUTPUT_BASE / dataset_name / grade
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{fmt}.jsonl"

            if out_path.exists():
                print(f"  SKIP {grade}/{fmt} (exists)")
                continue

            print(f"  Translating {grade}/{fmt}...", end="", flush=True)
            start = time.time()
            output_records = []

            for rec in records:
                question = rec[question_field]
                clean_q = preprocess_text(question, use_opus_fallback=False)
                braille_q = translate_to_format(clean_q, grade, fmt)

                out_rec = {
                    "id": rec.get("id", ""),
                    "question": question,
                    "clean_question": clean_q,
                    "braille_question": braille_q,
                }

                if "answer" in rec:
                    answer_text = rec["answer"]
                    clean_a = preprocess_text(answer_text, use_opus_fallback=False)
                    out_rec["answer"] = answer_text
                    out_rec["clean_answer"] = clean_a
                    out_rec["braille_answer"] = translate_to_format(clean_a, grade, fmt)

                if "golden_answers" in rec:
                    golden = rec["golden_answers"]
                    out_rec["golden_answers"] = golden
                    clean_golden = [preprocess_text(a, use_opus_fallback=False) for a in golden]
                    out_rec["braille_golden_answers"] = [translate_to_format(a, grade, fmt) for a in clean_golden]

                if "solution" in rec:
                    sol = rec["solution"]
                    clean_sol = preprocess_text(sol, use_opus_fallback=False)
                    out_rec["solution"] = sol
                    out_rec["clean_solution"] = clean_sol
                    out_rec["braille_solution"] = translate_to_format(clean_sol, grade, fmt)

                if has_choices:
                    choices = get_choices(rec)
                    if choices:
                        clean_choices = [preprocess_text(c, use_opus_fallback=False) for c in choices]
                        braille_choices = [translate_to_format(c, grade, fmt) for c in clean_choices]
                        out_rec["choices"] = choices
                        out_rec["braille_choices"] = braille_choices

                output_records.append(out_rec)

            with open(out_path, "w") as f:
                for rec in output_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elapsed = time.time() - start
            print(f" {len(output_records)} records in {elapsed:.1f}s -> {out_path.name}")


def main():
    total_start = time.time()
    for name, config in DATASETS.items():
        translate_dataset(name, config)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.0f}s")
    print(f"Output: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
