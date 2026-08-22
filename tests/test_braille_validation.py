import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_braille import (  # noqa: E402
    compute_braille_output_metrics,
    extract_answer,
    is_english_not_braille,
)


class BrailleValidationTests(unittest.TestCase):
    def test_extracts_standard_and_braille_boxed_answers(self):
        self.assertEqual(extract_answer(r"Result: \boxed{#ah}"), "#ah")
        self.assertEqual(extract_answer("Result: _*boxed_<#ah_>"), "#ah")

    def test_last_box_is_selected_by_text_position(self):
        response = r"First _*boxed_<#a_>, then corrected to \boxed{#b}"
        self.assertEqual(extract_answer(response, braille_output=True), "#b")

    def test_braille_ascii_edge_cells_are_not_stripped(self):
        self.assertEqual(
            extract_answer("The answer is: *.$", braille_output=True),
            "*.$",
        )

    def test_english_signals_are_detected(self):
        self.assertTrue(is_english_not_braille("Answer"))
        self.assertFalse(is_english_not_braille("18"))
        self.assertFalse(is_english_not_braille("#ah"))

    def test_braille_number_scores_after_back_translation(self):
        metrics = compute_braille_output_metrics(
            "#ah", ["18"], "grade1", "gsm8k"
        )
        self.assertFalse(metrics["wrote_english"])
        self.assertEqual(metrics["back_translated"], "18")
        self.assertEqual(metrics["math_verify"], 1.0)

    def test_unprefixed_english_number_is_rejected(self):
        metrics = compute_braille_output_metrics("18", ["18"], "grade1", "gsm8k")
        self.assertTrue(metrics["wrote_english"])
        self.assertEqual(metrics["math_verify"], 0.0)

    def test_uppercase_english_output_is_rejected(self):
        metrics = compute_braille_output_metrics(
            "Bank", ["bank"], "grade1", "commonsenseqa"
        )
        self.assertTrue(metrics["wrote_english"])
        self.assertEqual(metrics["em"], 0.0)


if __name__ == "__main__":
    unittest.main()
