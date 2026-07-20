"""
Convert reasoning datasets to Braille benchmark format.

Pipeline: raw text -> LaTeX preprocess -> liblouis translation -> 3 output formats.

Supports 3 Braille output formats:
  - ascii: Braille ASCII (BRF) e.g. ",hello _w"
  - unicode: Unicode Braille patterns e.g. "⠠⠓⠑⠇⠇⠕ ⠸⠺"
  - dots: Dot notation e.g. "6 1-2-5 1-5 1-2-3 1-2-3 1-3-5"

Input: JSONL with at minimum a "question" or "problem" field.
Output: JSONL with braille_* fields added for each format.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from translator import translate_text
from latex_preprocess import preprocess_text

OUTPUT_FORMATS = ["ascii", "unicode", "dots"]


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


def translate_all_formats(text: str, grade: str) -> dict:
    """Translate text to all 3 Braille formats."""
    ascii_out = translate_text(text, grade, "ascii")
    unicode_out = translate_text(text, grade, "unicode")
    dots_out = unicode_to_dots(unicode_out)
    return {
        "ascii": ascii_out,
        "unicode": unicode_out,
        "dots": dots_out,
    }


def get_question_field(record: dict) -> str:
    """Extract the question text from various dataset formats."""
    if "question" in record:
        return record["question"]
    if "problem" in record:
        return record["problem"]
    raise KeyError("Record has no 'question' or 'problem' field")


def get_choices(record: dict) -> list[str] | None:
    """Extract choices if present (handles FlashRAG metadata format)."""
    if "choices" in record and record["choices"]:
        return record["choices"]
    meta = record.get("metadata", {})
    if isinstance(meta, dict) and "choices" in meta:
        choices = meta["choices"]
        if isinstance(choices, list) and choices:
            if isinstance(choices[0], dict):
                return [f"{c['label']}) {c['text']}" for c in choices]
            return choices
    return None


def convert_record(record: dict, grade: str) -> dict:
    """Convert a single record, adding braille fields in all 3 formats."""
    out = dict(record)

    question = get_question_field(record)
    clean_question = preprocess_text(question, use_opus_fallback=False)
    braille = translate_all_formats(clean_question, grade)

    out["clean_question"] = clean_question
    out["braille_question_ascii"] = braille["ascii"]
    out["braille_question_unicode"] = braille["unicode"]
    out["braille_question_dots"] = braille["dots"]

    choices = get_choices(record)
    if choices:
        clean_choices = [preprocess_text(c, use_opus_fallback=False) for c in choices]
        out["clean_choices"] = clean_choices
        out["braille_choices_ascii"] = [translate_text(c, grade, "ascii") for c in clean_choices]
        out["braille_choices_unicode"] = [translate_text(c, grade, "unicode") for c in clean_choices]
        out["braille_choices_dots"] = [unicode_to_dots(translate_text(c, grade, "unicode")) for c in clean_choices]

    return out


def convert_file(input_path: str, output_path: str, grade: str):
    """Convert an entire JSONL file to Braille (all 3 formats in one output)."""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    converted = []
    failed = 0
    for i, rec in enumerate(records):
        try:
            converted.append(convert_record(rec, grade))
        except Exception as e:
            print(f"Warning: failed record {i} (id={rec.get('id', '?')}): {e}")
            converted.append(rec)
            failed += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} records ({failed} failed) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert reasoning dataset to Braille (all 3 formats)")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("--grade", choices=["grade1", "grade2"], required=True)
    parser.add_argument("--output", help="Output path (default: auto-generated under data/<grade>/)")
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base = Path(args.input).stem
        output_dir = Path(__file__).parent.parent / "data" / args.grade
        output_path = str(output_dir / f"{base}.jsonl")

    convert_file(args.input, output_path, args.grade)


if __name__ == "__main__":
    main()
