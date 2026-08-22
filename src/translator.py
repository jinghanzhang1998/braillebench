"""
Braille translation module using liblouis.

Translates English text to Grade 1 and Grade 2 Braille ASCII representation.
Uses UEB (Unified English Braille) standard.

Output formats:
  - ascii: Braille ASCII (BRF-style, printable ASCII characters)
  - unicode: Unicode Braille patterns (U+2800 block)
"""

import os
import louis

os.environ.setdefault("LOUIS_TABLEPATH", os.environ.get("LOUIS_TABLEPATH", ""))

TABLES = {
    "grade1": ["en-ueb-g1.ctb"],
    "grade2": ["en-ueb-g2.ctb"],
}


def translate_text(text: str, grade: str, output_format: str = "ascii") -> str:
    """Translate plain English text to Braille.

    Args:
        text: Input English text.
        grade: "grade1" (uncontracted) or "grade2" (contracted).
        output_format: "ascii" for Braille ASCII, "unicode" for Unicode dots.
    """
    if grade not in TABLES:
        raise ValueError(f"grade must be one of {list(TABLES.keys())}")
    if output_format == "unicode":
        result, *_ = louis.translate(TABLES[grade], text, mode=louis.dotsIO | louis.ucBrl)
        return result
    return louis.translateString(TABLES[grade], text)


def translate_batch(texts: list[str], grade: str, output_format: str = "ascii") -> list[str]:
    """Translate a list of texts to Braille."""
    return [translate_text(t, grade, output_format) for t in texts]


if __name__ == "__main__":
    examples = [
        "Hello world",
        "What is 2 + 3?",
        "If x = 5 and y = 3, what is x + y?",
        "The quick brown fox jumps over the lazy dog.",
        "Solve for x: 2x + 5 = 15",
    ]
    for text in examples:
        print(f"Original: {text}")
        print(f"  Grade 1 (ascii):   {translate_text(text, 'grade1', 'ascii')}")
        print(f"  Grade 2 (ascii):   {translate_text(text, 'grade2', 'ascii')}")
        print(f"  Grade 1 (unicode): {translate_text(text, 'grade1', 'unicode')}")
        print(f"  Grade 2 (unicode): {translate_text(text, 'grade2', 'unicode')}")
        print()
