"""
Regression tests for extract_answer (evaluate_braille.py) — the fixes for review items
#3.1 (final answer is:), #3.2 (last box beats prose marker / prompt echo), #3.3 (decimals not
truncated at period), and #8 (CommonsenseQA choice-letter prefix strip).

Run: python src/test_extract_answer.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOUIS_TABLEPATH", os.environ.get("LOUIS_TABLEPATH", ""))
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_braille import extract_answer

fails = []
def check(desc, resp, want):
    got = extract_answer(resp)
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}\n         resp={resp!r}\n         got={got!r} want={want!r}")
    if not ok:
        fails.append(desc)

print("=== #3.1 'The final answer is:' ===")
check("final answer is: boxed", "The final answer is: \\boxed{13}", "13")
check("the answer is (no 'is' word issue)", "The answer is 42.", "42")
check("final answer: 7", "Final answer: 7", "7")

print("\n=== #3.2 last box beats prompt-echo marker ===")
check("prose marker then final box",
      "the answer is a non-negative integer between 0 and 999. Therefore \\boxed{900}", "900")
check("two boxes -> last",
      "First \\boxed{10} then reconsider \\boxed{20}", "20")

print("\n=== #3.3 decimals not truncated ===")
check("decimal 1.5", "The answer is 1.5", "1.5")
check("decimal in box", "\\boxed{3.14}", "3.14")
check("sentence-final period after decimal", "The answer is 2.5.", "2.5")

print("\n=== #8 CommonsenseQA choice-letter prefix ===")
check("strip 'A) bank'", "The answer is A) bank", "bank")
check("bold choice", "**C) library**", "library")

print("\n=== regressions found on real data ===")
# _clean_extracted strips the sentence-final period; EM's normalize_answer removes punctuation
# anyway, so "Donald Trump Jr" matches gold "Donald Trump Jr.".
check("bold answer with trailing parenthetical (first bold, period trimmed)",
      "The answer is: **Donald Trump Jr.** (or **Don Jr.**)", "Donald Trump Jr")
check("double marker in one line -> last span",
      'The answer is: The question is asking if "Elvis"... The answer is: Yes.', "Yes")
check("bold with alt in parens (first bold)",
      "The answer is: **Warner Bros.** (or **WB Records**)", "Warner Bros")

print("\n=== sanity: plain trailing line ===")
check("bare number line", "Let me compute...\n72", "72")

print("\n=== boxed chosen by POSITION, not pattern-list order (code review 2026-08-07) ===")
check("later English box beats earlier braille box",
      r"Braille attempt: _*boxed_<#bd>. Actually the answer is \boxed{18}", "18")
check("later braille box beats earlier English box",
      r"First attempt \boxed{10}. Reconsider in braille: _*boxed_<#bj>", "#bj")
check("two English boxes -> last by position", r"\boxed{12} then \boxed{20}", "20")

print("\n=== braille_output=True preserves EVERY legitimate Braille ASCII cell ===")
def check_br(desc, resp, want):
    got = extract_answer(resp, braille_output=True)
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}\n         got={got!r} want={want!r}")
    if not ok:
        fails.append(desc)

# P0-1 (code review 2026-08-07): a LITERAL `.` is Braille ASCII dot-46, not a sentence boundary.
# These use `.` itself — an earlier version tested `d4c4`, where `4` never triggers the period
# regex, so the bug survived the test.
check_br("literal trailing dot cell", "The answer is: abc.", "abc.")
check_br("single dot cell", "The answer is: .", ".")
check_br("dot cell mid-sequence kept", "The answer is: ab.cd ef", "ab.cd ef")
# `$` (dot-1246) and `*` (dot-16) are cells too.
check_br("keep '$' accented cell inside word", "The answer is: ,~$arhus", ",~$arhus")
check_br("keep '4' period cell", "The answer is: d4c4", "d4c4")

# P0-2: symmetry alone does NOT prove a Markdown/LaTeX wrapper — these must survive intact.
check_br("balanced '*...*' is a cell sequence, not italics", "The answer is: *swan lake*", "*swan lake*")
check_br("balanced '$...$' is a cell sequence, not math", "The answer is: $los angeles$", "$los angeles$")
check_br("'**...**' kept verbatim", "The answer is: **,we/on ,park**", "**,we/on ,park**")
check_br("trailing '**' kept", "The answer is: ,yes**", ",yes**")
check_br("leading '$ ' kept", "The answer is: $ kitchen", "$ kitchen")
# The ONE exception: a print-English multiple-choice prefix. Uppercase A-E cannot occur in valid
# Braille ASCII (capitals use a `,` prefix), so this strip has no false positives.
check_br("print-English 'A) ' prefix removed", "The answer is: A) bank", "bank")
# English mode still strips these:
check("english mode still strips trailing period", "The answer is: Washington.", "Washington")
check("english mode still strips italics", "The answer is: *swan lake*", "swan lake")

print("\n=== trailing print-period normalization (separate, auditable scoring layer) ===")
from evaluate_braille import strip_trailing_print_period as _strip


def check_norm(desc, text, want_text, want_changed):
    got = _strip(text)
    ok = got == (want_text, want_changed)
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}\n         got={got!r} want={(want_text, want_changed)!r}")
    if not ok:
        fails.append(desc)


# Fires only on a trailing run of literal '.' — the case English scoring already forgives.
check_norm("trailing period removed", "washington.", "washington", True)
check_norm("trailing period + space removed", "yes. ", "yes", True)
check_norm("repeated trailing periods removed", "yes...", "yes", True)
# A literal '.' is a VALID cell mid-word (accented forms) and must never be touched.
check_norm("mid-word '.' cell preserved (Straßburg)", ",stra.!burg", ",stra.!burg", False)
check_norm("mid-word '.' cell preserved (Ishqabad)", ".*,ishqabad", ".*,ishqabad", False)
check_norm("leading '.' cell preserved", ".*abc", ".*abc", False)
check_norm("no trailing period -> unchanged", ",we/on ,park", ",we/on ,park", False)
check_norm("braille period cell '4' untouched", "d4c4", "d4c4", False)

# End-to-end: the same stylistic habit must cost the same in both channels (it costs nothing).
from evaluate_braille import compute_metrics as _cm, compute_braille_output_metrics as _cbm
en = _cm("Washington.", ["Washington"], "hotpotqa")["em"]
br = _cbm("washington.", ["Washington"], "grade1", "hotpotqa")
print(f"  [{'PASS' if en == 1.0 else 'FAIL'}] english output forgives a trailing period"
      f"\n         em={en} want=1.0")
if en != 1.0:
    fails.append("english trailing-period baseline")
check_norm_flag = br.get("trailing_period_normalized")
print(f"  [{'PASS' if br['em'] == 1.0 else 'FAIL'}] braille output now forgives it too"
      f"\n         em={br['em']} want=1.0")
if br["em"] != 1.0:
    fails.append("braille trailing-period parity")
print(f"  [{'PASS' if check_norm_flag else 'FAIL'}] normalization is recorded on the record"
      f"\n         trailing_period_normalized={check_norm_flag} want=True")
if not check_norm_flag:
    fails.append("trailing_period_normalized flag")
print(f"  [{'PASS' if 'predicted_raw' in br else 'FAIL'}] verbatim extraction retained as predicted_raw"
      f"\n         predicted_raw={br.get('predicted_raw')!r}")
if "predicted_raw" not in br:
    fails.append("predicted_raw audit field")

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL EXTRACT TESTS PASSED")
