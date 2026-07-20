"""
Use Opus 4.8 to clean math-heavy text that rule-based preprocessing missed.

Targets: AIME24 and any text with residual LaTeX artifacts after rule-based cleaning.
Processes in batches to reduce API calls.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_client import invoke_with_retry

BATCH_SIZE = 1  # texts per API call (1 = sequential to avoid timeouts)

SYSTEM_PROMPT = """You are a LaTeX-to-plain-text converter. Convert mathematical text into clean plain text suitable for Braille translation.

Rules:
- Remove ALL LaTeX commands and convert to readable plain math notation
- \\frac{a}{b}, \\tfrac{a}{b}, \\dfrac{a}{b} -> (a)/(b)
- \\sqrt{x} -> sqrt(x), \\sqrt[n]{x} -> root(n, x)
- x^{2} -> x^(2), x_{i} -> x_(i)
- \\left( and \\right) -> just ( and )
- \\[ and \\] -> remove (display math delimiters)
- \\( and \\) -> remove (inline math delimiters)
- \\begin{...} and \\end{...} -> remove
- \\cdot -> *, \\times -> *, \\div -> /
- \\ldots, \\dots, \\cdots -> ...
- \\leq -> <=, \\geq -> >=, \\neq -> !=
- \\log, \\sin, \\cos, \\tan -> log, sin, cos, tan
- \\infty -> infinity
- Greek letters: \\pi -> pi, \\theta -> theta, etc.
- \\overline{AB} -> AB_bar, \\vec{v} -> vec(v)
- \\binom{n}{k} -> C(n,k)
- \\pmod{n} -> (mod n)
- \\text{...} -> just the text content
- \\quad, \\qquad, \\, -> single space
- Keep ALL original English text unchanged
- Keep variable names as-is
- Use parentheses for clarity in grouping
- Do NOT add explanations, just output the cleaned text
- Handle each text independently, separated by the delimiter"""


CLEAN_MODEL = "claude-sonnet-4.6"


def clean_batch_with_opus(texts: list[str]) -> list[str]:
    """Send a batch of texts to Claude for cleaning."""
    delimiter = "\n---NEXT---\n"
    combined = delimiter.join(texts)

    prompt = f"""{SYSTEM_PROMPT}

Convert each of the following texts (separated by ---NEXT---). Output them in the same order, separated by ---NEXT---.

{combined}"""

    response = invoke_with_retry(CLEAN_MODEL, prompt, max_tokens=4096 * 2, temperature=0.0)

    parts = response.split("---NEXT---")
    parts = [p.strip() for p in parts]

    if len(parts) != len(texts):
        # Fallback: process one by one
        results = []
        for text in texts:
            single_prompt = f"{SYSTEM_PROMPT}\n\nConvert this text:\n{text}"
            r = invoke_with_retry(CLEAN_MODEL, single_prompt, max_tokens=4096, temperature=0.0)
            results.append(r.strip())
        return results

    return parts


def needs_opus_cleaning(text: str) -> bool:
    """Check if text has residual LaTeX that rules didn't handle."""
    indicators = [
        "tfrac", "dfrac", "cfrac",
        "\\[", "\\]", "\\(", "\\)",
        "begin{", "end{",
        "<=ft", "right)",
        "*s,", "*s.",  # \ldots artifact
        "log_", "lim_",
        "pmod",
        "\\\\",
    ]
    for ind in indicators:
        if ind in text:
            return True
    return False


def clean_dataset_with_opus(input_path: str, output_path: str):
    """Clean a dataset's questions using Opus where needed."""
    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Identify which records need cleaning
    to_clean_idx = []
    to_clean_texts = []
    for i, rec in enumerate(records):
        q = rec.get("clean_question", rec.get("question", ""))
        if needs_opus_cleaning(q):
            to_clean_idx.append(i)
            to_clean_texts.append(q)

    print(f"  {len(to_clean_texts)}/{len(records)} records need Opus cleaning")

    if not to_clean_texts:
        return records

    # Process in batches
    cleaned_texts = []
    for batch_start in range(0, len(to_clean_texts), BATCH_SIZE):
        batch = to_clean_texts[batch_start:batch_start + BATCH_SIZE]
        print(f"    Batch {batch_start//BATCH_SIZE + 1}/{(len(to_clean_texts)-1)//BATCH_SIZE + 1} ({len(batch)} texts)...")
        results = clean_batch_with_opus(batch)
        cleaned_texts.extend(results)
        time.sleep(1)  # Rate limiting

    # Update records
    for idx, cleaned in zip(to_clean_idx, cleaned_texts):
        records[idx]["clean_question"] = cleaned

    # Also clean solutions if present
    sol_to_clean_idx = []
    sol_to_clean_texts = []
    for i, rec in enumerate(records):
        sol = rec.get("clean_solution", "")
        if sol and needs_opus_cleaning(sol):
            sol_to_clean_idx.append(i)
            sol_to_clean_texts.append(sol)

    if sol_to_clean_texts:
        print(f"  {len(sol_to_clean_texts)} solutions also need cleaning")
        cleaned_sols = []
        for batch_start in range(0, len(sol_to_clean_texts), BATCH_SIZE):
            batch = sol_to_clean_texts[batch_start:batch_start + BATCH_SIZE]
            print(f"    Sol batch {batch_start//BATCH_SIZE + 1}...")
            results = clean_batch_with_opus(batch)
            cleaned_sols.extend(results)
            time.sleep(1)

        for idx, cleaned in zip(sol_to_clean_idx, cleaned_sols):
            records[idx]["clean_solution"] = cleaned

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Saved -> {output_path}")
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aime24", help="Dataset to clean")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    # Use the raw data and re-apply latex_preprocess + opus
    from latex_preprocess import apply_rules

    raw_paths = {
        "aime24": "data/raw/aime24/aime2024.jsonl",
        "gsm8k": "data/raw/gsm8k/test.jsonl",
    }

    if args.dataset not in raw_paths:
        print(f"Dataset {args.dataset} doesn't need Opus cleaning")
        sys.exit(0)

    raw_path = project_root / raw_paths[args.dataset]
    question_field = "problem" if args.dataset == "aime24" else "question"

    with open(raw_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # First pass: rule-based cleaning
    for rec in records:
        rec["clean_question"] = apply_rules(rec[question_field])
        if "solution" in rec:
            rec["clean_solution"] = apply_rules(rec["solution"])

    # Save intermediate
    intermediate_path = project_root / "data" / "braille" / f"{args.dataset}_opus_cleaned.jsonl"
    with open(intermediate_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Opus cleaning pass
    print(f"Cleaning {args.dataset} with Opus 4.8...")
    clean_dataset_with_opus(str(intermediate_path), str(intermediate_path))

    print("Done! Use run_translate_all.py with the cleaned data to re-translate.")
