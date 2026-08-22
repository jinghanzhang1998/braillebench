import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# These tests exercise packaging/data behavior and do not call a model or symbolic parser.
# Lightweight modules keep this test runnable before optional native dependencies are installed.
math_verify = types.ModuleType("math_verify")
math_verify.parse = lambda *args, **kwargs: []
math_verify.verify = lambda *args, **kwargs: False
math_verify.LatexExtractionConfig = object
math_verify.ExprExtractionConfig = object
sys.modules.setdefault("math_verify", math_verify)

model_client = types.ModuleType("model_client")
model_client.MODELS = {"test-model": {}}
model_client.invoke_with_retry = lambda *args, **kwargs: ""
sys.modules["model_client"] = model_client

import evaluate_braille as evaluator  # noqa: E402
try:  # source checkout name
    import latex_preprocess_public as public_latex  # noqa: E402
except ModuleNotFoundError:  # packaged release name
    import latex_preprocess as public_latex  # noqa: E402
import validate_release  # noqa: E402


class PublicReleaseTests(unittest.TestCase):
    def test_all_data_variants_are_complete_and_aligned(self):
        manifest = validate_release.validate_data(ROOT / "data" / "braille")
        self.assertEqual(manifest["logical_records"], 22551)
        self.assertEqual(manifest["variant_rows"], 135306)
        self.assertEqual(len(manifest["datasets"]), 5)

    def test_english_input_falls_back_to_parallel_release_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evaluator, "DATA_RAW", Path(temporary) / "missing"):
                with mock.patch.object(evaluator, "BRAILLE_DIR", ROOT / "data" / "braille"):
                    for dataset, expected in validate_release.DATASETS.items():
                        records = evaluator.load_dataset_records(dataset)
                        self.assertEqual(len(records), expected)
                        self.assertTrue(records[0]["question"])
                        self.assertTrue(evaluator.get_gold_answers(records[0], dataset))

    def test_liblouis_gate_includes_full_braille_prompts(self):
        self.assertFalse(evaluator.configs_require_liblouis(["EN-EN", "G1-EN"]))
        self.assertTrue(evaluator.configs_require_liblouis(["EN-G1"]))
        self.assertTrue(evaluator.configs_require_liblouis(["FULLBR-G1"]))
        self.assertTrue(evaluator.configs_require_liblouis(["EN-EN", "FULLBR-G2"]))

    def test_unimplemented_model_adapter_stops_instead_of_scoring_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evaluator, "DATA_RAW", Path(temporary) / "missing"):
                with mock.patch.object(evaluator, "BRAILLE_DIR", ROOT / "data" / "braille"):
                    with mock.patch.object(
                        evaluator,
                        "invoke_with_retry",
                        side_effect=NotImplementedError("adapter missing"),
                    ):
                        with self.assertRaises(NotImplementedError):
                            evaluator.evaluate(
                                "missing-adapter",
                                "gsm8k",
                                "EN-EN",
                                limit=1,
                                output_dir=Path(temporary) / "results",
                            )

    def test_public_latex_preprocessor_is_deterministic_and_provider_free(self):
        source_path = SRC / "latex_preprocess_public.py"
        if not source_path.exists():
            source_path = SRC / "latex_preprocess.py"
        source = source_path.read_text(encoding="utf-8").lower()
        self.assertNotIn("boto", source)
        self.assertNotIn("bed" + "rock", source)
        self.assertEqual(
            public_latex.preprocess_text(r"Find $\frac{3}{4} + \sqrt{9}$.").strip(),
            "Find (3)/(4) + sqrt(9).",
        )


if __name__ == "__main__":
    unittest.main()
