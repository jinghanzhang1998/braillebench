"""
Regression test for the braille-output English-detection gate (is_english_not_braille /
wrote_english_math / compute_braille_output_metrics).

Verifies the fix for the false-positive bug where literal digits 0-9 (legitimate Braille
punctuation: 1=comma, 4=period, 8=question mark) caused valid Braille to be scored 0.

Run: LOUIS_TABLEPATH=<tables> python src/test_english_gate.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOUIS_TABLEPATH", os.environ.get("LOUIS_TABLEPATH", ""))
sys.path.insert(0, str(Path(__file__).parent))

import louis
from evaluate_braille import (is_english_not_braille, wrote_english_math,
                              compute_braille_output_metrics)

G1 = ["en-ueb-g1.ctb"]

def tr(en):
    return louis.translateString(G1, en)

fails = []

def check(desc, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: got={got} want={want}")
    if not ok:
        fails.append(desc)

print("=== is_english_not_braille: valid braille with punctuation must NOT be flagged ===")
# these are valid braille (contain 0-9 as punctuation) — must be False (tr stays lowercase,
# capitals become ',' prefix), even though the source English had uppercase + periods.
check("braille 'Washington d.c.' (,washington d4c4)", is_english_not_braille(tr("Washington d.c.")), False)
check("braille sentence w/ comma+period", is_english_not_braille(tr("Hello, world.")), False)
check("braille question mark", is_english_not_braille(tr("Why?")), False)
check("braille number #bjd (=204)", is_english_not_braille(tr("204")), False)
check("braille decimal #b4d (=2.4)", is_english_not_braille(tr("2.4")), False)

print("\n=== is_english_not_braille: real English (uppercase) IS flagged ===")
check("literal 'The answer is 204'", is_english_not_braille("The answer is 204"), True)
check("literal 'Yes'", is_english_not_braille("Yes"), True)

print("\n=== wrote_english_math: english numeral vs braille numeral ===")
check("english '204' (gold 204)", wrote_english_math("204", "204", "grade1"), True)
check("braille '#bjd' (gold 204)", wrote_english_math(tr("204"), "204", "grade1"), False)
check("english '4' (gold 4)", wrote_english_math("4", "4", "grade1"), True)
check("braille '#d' (gold 4)", wrote_english_math(tr("4"), "4", "grade1"), False)
check("braille full answer (gold 204)", wrote_english_math(tr("The answer is 204."), "204", "grade1"), False)

print("\n=== end-to-end compute_braille_output_metrics (math) ===")
# correct braille answer -> math_verify 1, not flagged
m = compute_braille_output_metrics(tr("204"), ["204"], "grade1", "gsm8k")
check("correct braille '#bjd' math_verify", m["math_verify"], 1.0)
check("correct braille '#bjd' wrote_english", m["wrote_english"], False)
# english digit answer -> flagged, score 0
m = compute_braille_output_metrics("204", ["204"], "grade1", "gsm8k")
check("english '204' wrote_english", m["wrote_english"], True)
check("english '204' math_verify", m["math_verify"], 0.0)

print("\n=== end-to-end (QA): valid braille answer with punctuation scores correctly ===")
# gold 'yes' — correct braille 'yes' should match (g1 near-identity), not flagged
m = compute_braille_output_metrics(tr("yes"), ["yes"], "grade1", "hotpotqa")
check("braille 'yes' em", m["em"], 1.0)
check("braille 'yes' wrote_english", m["wrote_english"], False)
# a braille answer containing a comma (digit '1') must not be spuriously flagged
m = compute_braille_output_metrics(tr("bank, really"), ["bank"], "grade1", "commonsenseqa")
check("braille 'bank, really' wrote_english (comma=1)", m["wrote_english"], False)

print()
if fails:
    print(f"FAILED {len(fails)} checks: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
