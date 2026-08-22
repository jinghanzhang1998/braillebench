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
    # Pure-Braille configs: the ENTIRE prompt (instruction + problem) is in Braille.
    # Tests whether models can operate with no English scaffolding at all.
    "FULLBR-G1": {"input": "grade1", "output": "english", "full_braille": True},
    "FULLBR-G2": {"input": "grade2", "output": "english", "full_braille": True},
    # Cheat-sheet ablation: same as G1-EN/G2-EN but with a Braille-ASCII reference
    # table prepended. Tests "can't read braille" vs "wasn't told the mapping".
    "G1-EN-CS": {"input": "grade1", "output": "english", "cheatsheet": True},
    "G2-EN-CS": {"input": "grade2", "output": "english", "cheatsheet": True},
}

FORMATS = ["ascii", "unicode", "dots"]

DATASETS = ["gsm8k", "aime24", "commonsenseqa", "hotpotqa", "2wikimultihopqa"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def load_dataset_records(dataset_name: str, braille_grade: str = None, braille_format: str = None):
    """Load records from raw data or the self-contained parallel Braille data.

    Public releases do not need to redistribute a second copy under ``data/raw``: every
    Braille record retains the English question, choices, and gold answer. Internal research
    checkouts still prefer the original raw file when it is present.
    """
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
        if not path.exists():
            path = BRAILLE_DIR / dataset_name / "grade1" / "ascii.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"No data file for {dataset_name!r}: expected {path}. "
            "Run src/validate_release.py to check the benchmark package."
        )

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


CHEATSHEET_G1 = """Braille ASCII reference (Grade 1, uncontracted):
- Letters: a-z map to the same lowercase letters (a=a, b=b, ..., z=z).
- Capital letter: a comma prefix. ",a"=A, ",cat"=Cat.
- Numbers: "#" prefix, then letters a-j for digits 1-9,0:
  #a=1 #b=2 #c=3 #d=4 #e=5 #f=6 #g=7 #h=8 #i=9 #j=0
  So "#af"=16, "#ac"=13, "#bjj"=200, "#bjbd"=2024.
- Punctuation: 1=comma, 4=period, 8=question mark, 6=exclamation,
  2=colon, 3=semicolon, 0=quote, -=hyphen, "< =open paren, >" =close paren.
"""

CHEATSHEET_G2 = CHEATSHEET_G1 + """
Grade 2 (contracted) additional signs:
Single-letter whole words (a letter alone = a word):
  b=but c=can d=do e=every f=from g=go h=have j=just k=knowledge
  l=like m=more n=not p=people q=quite r=rather s=so t=that
  u=us v=very w=will x=it y=you z=as
Other whole words / signs:
  *=child %=shall ?=this :=which |=out /=still 2=be 5=enough 7=were
  8=his 9=in 0=was !=the &=and ==for (=of )=with
Letter-group signs inside words:
  ch=* sh=% ed=$ er=} ing=+ ou=| ar=> st=/ con=3(prefix)
Examples: "! answ}"=the answer, "go+"=going, "ov}"=over, "c>d"=card, "f/"=first.
"""


def make_braille_prompt(dataset_name: str, record: dict, config: dict, braille_format: str = None):
    """Create evaluation prompt for any input-output configuration."""
    input_type = config["input"]
    output_type = config["output"]
    cheatsheet_prefix = ""
    if config.get("cheatsheet"):
        cheatsheet_prefix = (CHEATSHEET_G1 if "1" in input_type else CHEATSHEET_G2) + "\n"

    # Full-braille configs: build a plain-English prompt from the ORIGINAL english
    # question, then translate the entire prompt (instruction + problem) into braille.
    if config.get("full_braille"):
        import louis
        table = "en-ueb-g1.ctb" if "1" in input_type else "en-ueb-g2.ctb"
        # Build the English baseline prompt (english instruction + english question)
        en_config = {"input": "english", "output": "english"}
        english_prompt = make_braille_prompt(dataset_name, record, en_config, "ascii")
        fmt = braille_format or "ascii"
        if fmt == "ascii":
            return louis.translateString([table], english_prompt)
        unicode_prompt, *_ = louis.translate(
            [table], english_prompt, mode=louis.dotsIO | louis.ucBrl
        )
        if fmt == "unicode":
            return unicode_prompt
        if fmt == "dots":
            cells = []
            for char in unicode_prompt:
                if char == " " or not (0x2800 <= ord(char) <= 0x28FF):
                    cells.append(char)
                    continue
                offset = ord(char) - 0x2800
                cells.append(
                    "0" if offset == 0 else "-".join(
                        str(dot + 1) for dot in range(8) if offset & (1 << dot)
                    )
                )
            return " ".join(cells)
        raise ValueError(f"Unsupported Braille format: {fmt}")

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
        fmt_in = braille_format or "ascii"
        if fmt_in == "unicode":
            fmt_desc = "Unicode Braille characters"
        elif fmt_in == "dots":
            fmt_desc = "Braille dot notation (each cell as dot numbers, e.g. 1-2-5; space-separated)"
        else:
            fmt_desc = "Braille ASCII notation"
        input_note = f"The following question is written in {grade_name} {fmt_desc}. "
    else:
        input_note = ""
    # Prepend cheat-sheet reference (ablation) if enabled
    input_note = cheatsheet_prefix + input_note

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


# `\boxed{...}` in English and its braille-rendered forms. Models writing in braille render
# `\boxed{...}` as `_*boxed_<...>` / `box$_<..._>` / `_*boxed<...>*_`
# (`_*`=backslash, `_<`/`_>`=braces). Boxed answers are authoritative for math.
_BOXED_PATTERNS = [
    r"\\boxed\{([^}]+)\}",
    r"_?\*?box(?:ed|\$)_?<([^>_]+?)_?>",
    r"_?\*?box(?:ed|\$)<([^>*]+?)>\*?_?",
]

# Explicit answer marker. `(?:\s+is)?` lets "final answer is:" / "the answer is" match.
_MARKER_RE = re.compile(r"(?:final answer|the answer)(?:\s+is)?\s*:?\s*", re.IGNORECASE)


def _trim_answer_span(tail: str, braille_output: bool = False) -> str:
    """Given the text right after an answer marker, isolate the answer span.

    ENGLISH output — order of precedence, so `**Donald Trump Jr.**` and multi-sentence tails
    are handled:
    1. the FIRST bold span `**...**` on the first line (the primary answer; a trailing
       "(or **alt**)" parenthetical is ignored);
    2. else the first line, cut at the first sentence-final period (a period followed by
       whitespace/end — decimals like 1.5 are preserved).

    BRAILLE output — return the whole first line verbatim. In Braille ASCII (BRF) both `.` and
    `*` are LEGITIMATE cells (`.` = dot-46, `*` = dot-16 contraction indicator), so neither
    English sentence-boundary splitting nor Markdown-bold extraction is sound here: it would
    silently truncate a valid cell sequence before back-translation. Cf. code review
    2026-08-07 P0-1/P0-2.
    """
    first_line = tail.split("\n", 1)[0]
    if braille_output:
        return first_line
    bolds = re.findall(r"\*\*(.+?)\*\*", first_line)
    if bolds:
        return bolds[0]
    return re.split(r"\.(?=\s|$)", first_line, 1)[0]


def _clean_extracted(ans: str, braille_output: bool = False) -> str:
    """Trim an extracted answer.

    For ENGLISH/math output we strip stray Markdown/LaTeX markers (`*`, `$`) and a trailing
    sentence period. For BRAILLE output every one of those characters is a LEGITIMATE Braille
    ASCII cell, so the ONLY things we remove are surrounding whitespace and a print-English
    multiple-choice prefix (`A) `), which cannot occur in valid braille. Anything else would
    corrupt the cell sequence before back-translation.
    """
    ans = ans.strip()
    # Multiple-choice letter prefix is stripped in BOTH modes: it appears as print-English "A) "
    # when a model answers a CSQA braille-output item in English. A literal uppercase A-E can
    # never occur in valid Braille ASCII (capitals use a `,` prefix and the text stays lowercase),
    # so this is a zero-false-positive strip — the same argument as `is_english_not_braille`.
    ans = re.sub(r"^[A-E]\)\s*", "", ans)
    if braille_output:
        # Nothing else. `*` (dot-16 contraction), `$` (dot-1246), `.` (dot-46) and the digits used
        # as punctuation are all LEGITIMATE Braille ASCII cells. Symmetry alone (`*...*`, `$...$`)
        # cannot distinguish a Markdown/LaTeX wrapper from a genuine cell sequence, so stripping
        # them would corrupt valid answers. Cf. code review 2026-08-07 P0-2.
        return ans
    ans = re.sub(r"\.\s*$", "", ans)          # drop a single sentence-final period
    ans = ans.strip("*").strip("$").strip()   # strip stray bold/math markers
    return ans.strip()


def extract_answer(response: str, braille_output: bool = False) -> str:
    """Extract the final answer from a model response.

    braille_output=True preserves Braille ASCII cells that the English cleaner would delete
    (`*`, `$`, trailing period); pass it whenever the requested OUTPUT is braille (EN-G1/EN-G2/
    G1-G1/G2-G2), so the cell sequence reaches back-translation intact.
    """
    response = clean_response(response)

    # 1) A boxed answer is authoritative — take the box that appears LAST by POSITION in the
    #    response (not last within a per-pattern list), so a prompt echo / mid-reasoning marker
    #    cannot win, and a later English \boxed beats an earlier braille box or vice versa.
    boxed_matches = []
    for pattern in _BOXED_PATTERNS:
        for m in re.finditer(pattern, response, re.MULTILINE):
            boxed_matches.append((m.start(), m.group(1)))
    if boxed_matches:
        boxed_matches.sort(key=lambda t: t[0])
        return _clean_extracted(boxed_matches[-1][1], braille_output)

    # 2) Otherwise, the span after the LAST explicit answer marker, trimmed to one answer.
    markers = list(_MARKER_RE.finditer(response))
    if markers:
        span = _trim_answer_span(response[markers[-1].end():], braille_output)
        return _clean_extracted(span, braille_output)

    # 3) Bold final answer. Skipped for braille output: `*` is a valid cell, so a trailing
    #    `**...**` there is not reliably Markdown (code review 2026-08-07 P0-2).
    if not braille_output:
        for pattern in (r"\*\*(\d[\d,]*\.?\d*)\*\*\s*$", r"\*\*(.+?)\*\*\s*$"):
            m = re.findall(pattern, response, re.MULTILINE)
            if m:
                return _clean_extracted(m[-1], braille_output)

    # 4) Fall back to the last non-empty line.
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return _clean_extracted(lines[-1], braille_output) if lines else ""


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
        norm_gold = normalize_answer(gold)
        # An empty gold is a substring of everything — never let it score a spurious 1.0.
        if norm_gold and norm_gold in norm_pred:
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
    # Clean LaTeX/currency artifacts that break math-verify parsing:
    #   \$125 (escaped dollar), stray \text{...} units, commas in numbers.
    cleaned = predicted.replace("\\$", "").replace("$", "")
    cleaned = re.sub(r"\\text\{[^}]*\}", "", cleaned)
    try:
        gold_parsed = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred_parsed = parse(cleaned, extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=0),
            ExprExtractionConfig(),
        ])
        if verify(gold_parsed, pred_parsed):
            return 1.0
    except Exception:
        pass
    # Numeric fallback: compare last number (always run if symbolic verify didn't confirm).
    pred_nums = re.findall(r"-?\d[\d,]*\.?\d*", cleaned)
    gold_nums = re.findall(r"-?\d[\d,]*\.?\d*", gold)
    if pred_nums and gold_nums:
        try:
            return 1.0 if abs(float(pred_nums[-1].replace(",", "")) - float(gold_nums[-1].replace(",", ""))) < 1e-6 else 0.0
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


def strip_latex_escapes(text: str) -> str:
    """Remove presentation-layer LaTeX escapes that would corrupt liblouis back-translation.

    Models sometimes wrap a braille number in `\\boxed{\\#ah}`; after extraction the `\\#`
    escape survives and `\\#ah` fed to liblouis produces garbage like `\\12567/18`. We strip
    only the backslash of the LaTeX-escaped forms of characters that are meaningful in Braille
    ASCII (`#`, `$`, `%`, `&`, `_`), turning `\\#ah` into `#ah`. Additionally, a LaTeX math
    wrapper around a braille number renders as `\\$\\#..` -> after unescaping `$#..`; that leading
    `$` is the LaTeX math delimiter, not a braille cell, so drop a `$` that immediately precedes
    the braille number sign `#`. A bare `$` elsewhere is a valid braille cell and is left
    untouched (e.g. `,~$arhus` = Århus).
    """
    text = re.sub(r"\\([#$%&_])", r"\1", text)
    text = re.sub(r"\$(?=#)", "", text)
    return text


class BackTranslateError(RuntimeError):
    """Raised when liblouis back-translation fails — must not be silently swallowed."""


def back_translate(braille_text: str, grade: str) -> str:
    """Back-translate Braille ASCII to English for reliable answer comparison.

    Fail-CLOSED: if liblouis errors we raise rather than returning the input unchanged.
    Returning the input (the old behaviour) let a mis-configured LOUIS_TABLEPATH silently
    produce plausible-looking scores, because Grade-1 lowercase braille equals its English.
    """
    import louis
    table = "en-ueb-g1.ctb" if "1" in grade else "en-ueb-g2.ctb"
    try:
        return louis.backTranslateString([table], strip_latex_escapes(braille_text))
    except Exception as e:
        raise BackTranslateError(f"liblouis back-translation failed for {grade}: {e}") from e


def liblouis_selftest() -> None:
    """Verify liblouis + tables are correctly loaded via a known round-trip. Call at startup.

    Raises if the environment is misconfigured, so we fail loudly instead of scoring garbage.
    """
    import louis
    # grade1: exact forward check (near-1:1 cipher, stable). grade2: round-trip check.
    g1 = louis.translateString(["en-ueb-g1.ctb"], "Hello, world.")
    if g1 != ",hello1 world4":
        raise RuntimeError(
            f"liblouis self-test failed (grade1): got {g1!r}, expected ',hello1 world4'. "
            f"Check LOUIS_TABLEPATH and the UEB tables.")
    for grade, table in (("grade1", "en-ueb-g1.ctb"), ("grade2", "en-ueb-g2.ctb")):
        english = "the answer is forty two"
        back = louis.backTranslateString([table], louis.translateString([table], english))
        if english.lower() not in back.lower():
            raise RuntimeError(
                f"liblouis round-trip self-test failed ({grade}): {english!r} -> {back!r}. "
                f"Check LOUIS_TABLEPATH and the UEB tables.")


def is_english_not_braille(text: str) -> bool:
    """Detect if output is English rather than valid Braille ASCII.

    In liblouis Braille ASCII (BRF), a capital is written with a `,` prefix and the
    text stays LOWERCASE, so any literal uppercase A-Z is a reliable signal that the
    model wrote English instead of Braille — this check has zero false positives on
    the 147k ground-truth Braille strings in our datasets.

    NOTE: we intentionally do NOT flag literal digits 0-9 here. In Braille ASCII the
    characters 0-9 are legitimate PUNCTUATION (`1`=comma, `4`=period, `8`=question
    mark, ...), while Braille NUMBERS are written `#` + letters a-j and contain no
    literal digits. A blanket `[0-9]` check therefore mis-flags the majority of valid
    Braille (any sentence with a comma/period) as English. English *numeric answers*
    (e.g. writing "4" instead of Braille "#d") are still rejected correctly by the
    back-translation comparison downstream — and, for math, flagged explicitly via
    `wrote_english_math` below.
    """
    return bool(re.search(r"[A-Z]", text))


def wrote_english_math(predicted: str, gold_number: str, grade: str) -> bool:
    """Math-specific detector: did the model answer with an ENGLISH numeral instead of Braille?

    Reads the prediction two ways: as English (does it literally contain the gold number
    as a standalone token?) and as Braille (back-translate, then look for the gold number).
    If it matches when read as English but NOT as Braille, the model wrote an English digit.
    This distinguishes "4" (English, wrong) from "#d" (Braille "4", correct) without relying
    on the fragile literal-digit heuristic.
    """
    if not gold_number:
        return False
    import louis
    table = "en-ueb-g1.ctb" if "1" in grade else "en-ueb-g2.ctb"
    eng_match = bool(re.search(rf"(?<!\d){re.escape(gold_number)}(?!\d)", predicted))
    try:
        bt = louis.backTranslateString([table], predicted)
    except Exception:
        bt = predicted
    braille_match = gold_number in re.findall(r"-?\d+\.?\d*", bt)
    return eng_match and not braille_match


# Unicode Braille (U+2800 block) -> North American Braille ASCII (BRF) character map.
# Derived from the project's own aligned ascii/unicode data (same liblouis that produced both),
# and matches the canonical BRF table. Used to normalize unicode/dots model output to ASCII
# cells so the single ASCII validator (gate + back-translate) applies to every format.
_UNICODE_TO_BRF = {0x2800: ' ', 0x2801: 'a', 0x2802: '1', 0x2803: 'b', 0x2804: "'", 0x2805: 'k', 0x2806: '2', 0x2807: 'l', 0x2808: '`', 0x2809: 'c', 0x280A: 'i', 0x280B: 'f', 0x280C: '/', 0x280D: 'm', 0x280E: 's', 0x280F: 'p', 0x2810: '"', 0x2811: 'e', 0x2812: '3', 0x2813: 'h', 0x2814: '9', 0x2815: 'o', 0x2816: '6', 0x2817: 'r', 0x2818: '~', 0x2819: 'd', 0x281A: 'j', 0x281B: 'g', 0x281C: '>', 0x281D: 'n', 0x281E: 't', 0x281F: 'q', 0x2820: ',', 0x2821: '*', 0x2822: '5', 0x2823: '<', 0x2824: '-', 0x2825: 'u', 0x2826: '8', 0x2827: 'v', 0x2828: '.', 0x2829: '%', 0x282A: '{', 0x282B: '$', 0x282C: '+', 0x282D: 'x', 0x282E: '!', 0x282F: '&', 0x2830: ';', 0x2831: ':', 0x2832: '4', 0x2833: '|', 0x2834: '0', 0x2835: 'z', 0x2836: '7', 0x2837: '(', 0x2838: '_', 0x2839: '?', 0x283A: 'w', 0x283B: '}', 0x283C: '#', 0x283D: 'y', 0x283E: ')', 0x283F: '=', 0x2873: '\\'}


def _unicode_braille_to_ascii(text: str) -> str:
    return "".join(_UNICODE_TO_BRF.get(ord(ch), ch) for ch in text)


def _dots_to_ascii(text: str) -> str:
    """dots notation -> ASCII BRF. Cells are space-separated; each cell is dot numbers joined
    by '-' (e.g. '2-4-5'), and a lone '0' cell denotes an empty cell = space."""
    out = []
    for cell in text.split(" "):
        cell = cell.strip()
        if not cell or cell == "0":
            out.append(" ")            # empty/blank cell = space between words
            continue
        bits, ok = 0, True
        for d in cell.split("-"):
            if d.isdigit() and "1" <= d <= "8":
                bits |= 1 << (int(d) - 1)
            else:
                ok = False
                break
        out.append(_UNICODE_TO_BRF.get(0x2800 + bits, cell) if ok else cell)
    return "".join(out)


def normalize_braille_output(text: str, braille_format: str) -> str:
    """Normalize a model's braille output to ASCII BRF cells so the ASCII validator applies.

    ascii -> unchanged. unicode -> map U+28xx cells to BRF ASCII. dots -> parse dot numbers.
    """
    if not braille_format or braille_format == "ascii":
        return text
    if braille_format == "unicode":
        return _unicode_braille_to_ascii(text)
    if braille_format == "dots":
        return _dots_to_ascii(text)
    return text


# Trailing print-sentence-period normalization for braille output.
#
# WHY THIS EXISTS. For English output, EM's `normalize_answer` deletes all punctuation, so a model
# that ends its answer with a sentence period ("Washington.") is scored exactly like one that does
# not. For braille output there is no such forgiveness: a trailing print `.` is not the braille
# period (which is the cell `4`), so liblouis back-translates it to an unknown-cell escape
# (`washington\46/`) whose digits survive `normalize_answer` as the token "46" and destroy the
# match. The identical stylistic habit was therefore free in English and fatal in braille, which
# biases every braille-vs-English comparison against braille.
#
# WHY IT IS SAFE. Applied ONLY to a trailing run at the very end of the cell sequence, and only to
# the literal `.` character. A literal `.` IS a valid Braille ASCII cell mid-word (it appears in
# accented forms, e.g. `,stra.!burg`), and those are untouched. Measured against this project's own
# ground truth: of 87,392 gold braille answer strings, 5,431 contain a literal `.` somewhere but
# **0** end with one — so no valid answer can be altered by this rule.
#
# This is a deliberate, separately-reported normalization layer, not part of extraction: extraction
# stays verbatim, and both forms are recorded on every record (`predicted_raw` vs `back_translated`,
# plus the `trailing_period_normalized` flag) so the layer's effect stays auditable.
_TRAILING_PRINT_PERIOD_RE = re.compile(r"\.+\s*$")


def strip_trailing_print_period(braille_text: str) -> tuple:
    """Return (normalized_text, was_changed). Removes only a trailing literal `.` run."""
    stripped = _TRAILING_PRINT_PERIOD_RE.sub("", braille_text)
    return stripped, stripped != braille_text


def compute_braille_output_metrics(predicted_braille: str, gold_english: list,
                                   grade: str, dataset_name: str,
                                   braille_format: str = "ascii") -> dict:
    """Metrics for braille-output configs.

    Two-stage validation:
    1. If output contains a reliable English signal, the model failed to write Braille
       → score 0 (records wrote_english=True). The signal is uppercase A-Z (never present
       in valid Braille ASCII), plus, for math, an English numeral answer (see
       `wrote_english_math`). Literal digits 0-9 are NOT treated as an English signal
       because they are legitimate Braille punctuation.
    2. Otherwise back-translate the braille to English and compare to English gold
       (avoids information loss from normalizing away braille symbols). This step also
       rejects English numeric answers: read as Braille, "4" decodes to "." != gold.

    `braille_format` (ascii/unicode/dots) selects how the model output is normalized to ASCII
    BRF cells before validation, so unicode/dots output configs are scored correctly too.
    """
    # normalize non-ascii braille formats to ASCII BRF cells for validation
    raw_braille = predicted_braille
    predicted_braille = normalize_braille_output(predicted_braille, braille_format)
    # Trailing print-period normalization (see the note above `strip_trailing_print_period`):
    # makes a trailing sentence period as harmless for braille output as it already is for English.
    predicted_braille, period_normalized = strip_trailing_print_period(predicted_braille)
    is_math = dataset_name in ("gsm8k", "aime24")
    wrote_english = is_english_not_braille(predicted_braille)
    if not wrote_english and is_math:
        wrote_english = wrote_english_math(predicted_braille, gold_english[0], grade)
    if wrote_english:
        metrics = {"em": 0.0, "sub_em": 0.0, "f1": 0.0, "wrote_english": True,
                   "back_translated": "", "predicted_raw": raw_braille,
                   "trailing_period_normalized": period_normalized}
        if is_math:
            metrics["math_verify"] = 0.0
        return metrics

    pred_en = back_translate(predicted_braille, grade)
    metrics = {
        "em": exact_match(pred_en, gold_english),
        "sub_em": sub_exact_match(pred_en, gold_english),
        "f1": token_f1(pred_en, gold_english),
        "wrote_english": False,
        "back_translated": pred_en,
        # audit trail: the verbatim extraction before any normalization layer, and whether the
        # trailing-period rule fired on this record.
        "predicted_raw": raw_braille,
        "trailing_period_normalized": period_normalized,
    }
    if is_math:
        metrics["math_verify"] = math_verify_match(pred_en, gold_english[0])
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def _summarize(metric_accum: dict, model_name, dataset_name, config_name,
               braille_format, output_type, total, elapsed) -> dict:
    """Build a summary dict from accumulated per-record metric lists."""
    summary = {
        "model": model_name, "dataset": dataset_name, "config": config_name,
        "braille_format": braille_format, "total": total, "elapsed_seconds": elapsed,
    }
    for k, vals in metric_accum.items():
        numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric_vals:
            summary[k] = sum(numeric_vals) / len(numeric_vals)
        elif vals and all(isinstance(v, bool) for v in vals):  # wrote_english rate
            summary[k] = sum(1 for v in vals if v) / len(vals)
    return summary


def _accumulate_existing(detail_path) -> dict:
    """Read already-completed records so a resumed run's summary covers the WHOLE file,
    not just the newly-run tail. Returns metric_accum seeded from stored records."""
    acc = {}
    with open(detail_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in (r.get("metrics") or {}).items():
                acc.setdefault(k, []).append(v)
    return acc


def evaluate(model_name: str, dataset_name: str, config_name: str,
             braille_format: str = None, limit: int = None,
             output_dir: Path = None, max_tokens: int = 1024,
             aime_max_tokens: int = 4096) -> dict:
    """Run evaluation with checkpoint/resume support.

    Results are written incrementally to the output file. If the file already
    has N lines, evaluation resumes from record N.
    """
    config = CONFIGS[config_name]
    input_type = config["input"]
    output_type = config["output"]
    is_braille_input = input_type in ("grade1", "grade2")
    results_dir = Path(output_dir) if output_dir else RESULTS_DIR

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
    detail_path = results_dir / f"{tag}_details.jsonl"

    # Resume: count existing lines and seed the metric accumulator from them
    start_idx = 0
    metric_accum = {}
    if detail_path.exists():
        with open(detail_path) as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(records):
            print(f"\n  [{model_name}] {dataset_name} | {config_name} | COMPLETE ({start_idx} records)")
            # Build a real summary from the completed file so downstream code never sees {}.
            metric_accum = _accumulate_existing(detail_path)
            summary = _summarize(metric_accum, model_name, dataset_name, config_name,
                                 braille_format, output_type, len(records), 0.0)
            return {"summary": summary, "results": []}
        metric_accum = _accumulate_existing(detail_path)
        print(f"\n  [{model_name}] {dataset_name} | {config_name} | RESUME from {start_idx}/{len(records)}")
    else:
        print(f"\n  [{model_name}] {dataset_name} | {config_name} | fmt={braille_format or 'N/A'} | n={len(records)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    with open(detail_path, "a") as out_f:
        for i in range(start_idx, len(records)):
            rec = records[i]
            gold_answers = get_gold_answers(rec, dataset_name)
            # Reject records with no usable gold — scoring them yields meaningless 0/spurious 1.
            if not gold_answers or all(not str(g).strip() for g in gold_answers):
                raise ValueError(f"Record {i} ({rec.get('id')!r}) in {dataset_name} has empty gold; "
                                 f"fix the source data before evaluating.")
            prompt = make_braille_prompt(dataset_name, rec, config, braille_format)
            max_tok = aime_max_tokens if dataset_name == "aime24" else max_tokens

            try:
                response = invoke_with_retry(model_name, prompt, max_tokens=max_tok, temperature=0.0)
                is_braille_out = output_type in ("grade1", "grade2")
                # For braille output, extract WITHOUT the English cleaner so legitimate cells
                # (`*`, `$`, trailing period) survive to back-translation.
                predicted = extract_answer(response, braille_output=is_braille_out)

                if is_braille_out:
                    # Braille output: back-translate to English then compare to English gold.
                    # This avoids information loss from normalize stripping braille symbols
                    # (critical for Grade 2 contractions).
                    metrics = compute_braille_output_metrics(predicted, gold_answers, output_type, dataset_name, braille_format or "ascii")
                else:
                    metrics = compute_metrics(predicted, gold_answers, dataset_name)
            except NotImplementedError:
                # The public adapter has not been connected. Stop immediately instead of
                # serializing an entire benchmark of artificial zero-score ERROR rows.
                raise
            except Exception as e:
                error_str = str(e)
                # Infrastructure errors (credentials, network, timeouts, endpoint-connect) must
                # PAUSE, not score 0 — otherwise a transient outage is permanently recorded as a
                # model failure and the line-count checkpoint skips it on resume. Classification is
                # centralized in infra_errors.is_infra_error (the old inline list missed "Could not
                # connect to the endpoint URL" because it only matched "connection", not "connect").
                from infra_errors import is_infra_error
                if is_infra_error(error_str):
                    print(f"\n  PAUSED at record {i}/{len(records)} (infrastructure error). Resume later.")
                    raise RuntimeError(f"Infrastructure error at record {i}/{len(records)}: {error_str[:120]}")
                # A genuine model/validator error on THIS response: record it as an error (score 0)
                # so the run continues; these are re-runnable via fix_failed_records.py.
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

            # Progress uses the SAME primary metric as the final table: math_verify for math
            # (both English- and Braille-output), else EM. Prevents "em=0" looking like failure
            # when math_verify is the real score.
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                primary = "math_verify" if dataset_name in ("gsm8k", "aime24") else "em"
                if primary in metric_accum:
                    score = sum(metric_accum[primary]) / len(metric_accum[primary]) * 100
                    print(f"    [{i+1}/{len(records)}] {primary}={score:.1f}% ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    summary = _summarize(metric_accum, model_name, dataset_name, config_name,
                         braille_format, output_type, len(records), elapsed)

    primary = "math_verify" if dataset_name in ("gsm8k", "aime24") else "em"
    if primary in summary:
        print(f"    DONE: {primary}={summary[primary]*100:.1f}% | F1={summary.get('f1',0)*100:.1f}% ({elapsed:.0f}s)")

    return {"summary": summary, "results": []}


def configs_require_liblouis(config_names) -> bool:
    """Whether configs translate prompts or validate Braille output at runtime."""
    return any(
        CONFIGS[name]["output"] in ("grade1", "grade2")
        or CONFIGS[name].get("full_braille", False)
        for name in config_names
    )


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
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Generation limit for all datasets except AIME24 (default: 1024)")
    parser.add_argument("--aime-max-tokens", type=int, default=4096,
                        help="Generation limit for AIME24 (default: 4096)")
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.aime_max_tokens <= 0:
        parser.error("generation token limits must be positive")

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

    # Fail CLOSED before any braille-output config runs: a mis-configured LOUIS_TABLEPATH would
    # otherwise let back-translation silently return plausible-looking garbage. English-output-only
    # runs do not need liblouis at all, so the check is conditional. (code review 2026-08-07 P0-3)
    if configs_require_liblouis(config_list):
        liblouis_selftest()

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

                    result = evaluate(
                        model_name, dataset_name, config_name, fmt, args.limit, output_dir,
                        args.max_tokens, args.aime_max_tokens,
                    )
                    if result["summary"]:
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
        em = f"{s.get('em',0)*100:.1f}%"
        f1 = f"{s.get('f1',0)*100:.1f}%"
        mv = f"{s.get('math_verify',0)*100:.1f}%" if "math_verify" in s else "  -"
        fmt = s.get("braille_format") or "-"
        print(f"{s['model']:<20} {s['dataset']:<14} {s['config']:<8} {fmt:<8} {em:>6} {f1:>6} {mv:>6}")


if __name__ == "__main__":
    main()
