"""
LaTeX to plain math preprocessing for Braille translation.

Combines rule-based substitutions for common LaTeX patterns with
Opus 4.8 API calls for complex/ambiguous expressions.

Usage:
    from latex_preprocess import preprocess_text
    clean = preprocess_text("Solve $\\frac{3}{4} + \\frac{1}{2}$.")
    # -> "Solve 3/4 + 1/2."
"""

import re
import json
import boto3

BEDROCK_PROFILE = "h100"
BEDROCK_REGION = "us-east-1"
OPUS_MODEL_ID = "us.anthropic.claude-opus-4-8"

# ---------------------------------------------------------------------------
# Rule-based LaTeX cleaning (handles common patterns without API calls)
# ---------------------------------------------------------------------------

LATEX_RULES = [
    # Remove dollar-sign delimiters
    (r'\$\$(.+?)\$\$', r'\1'),
    (r'\$(.+?)\$', r'\1'),
    # \frac{a}{b} -> (a)/(b)
    (r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)'),
    # \sqrt{x} -> sqrt(x)
    (r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)'),
    # \sqrt[n]{x} -> nthroot(n, x)
    (r'\\sqrt\[([^]]+)\]\{([^{}]+)\}', r'root(\1, \2)'),
    # superscript: x^{abc} -> x^(abc), x^2 stays x^2
    (r'\^\{([^{}]+)\}', r'^(\1)'),
    # subscript: x_{abc} -> x_(abc)
    (r'_\{([^{}]+)\}', r'_(\1)'),
    # \cdot -> *
    (r'\\cdot', '*'),
    # \times -> *
    (r'\\times', '*'),
    # \div -> /
    (r'\\div', '/'),
    # \pm -> +-
    (r'\\pm', '+-'),
    # \mp -> -+
    (r'\\mp', '-+'),
    # \leq, \geq, \neq
    (r'\\leq', '<='),
    (r'\\geq', '>='),
    (r'\\neq', '!='),
    (r'\\le', '<='),
    (r'\\ge', '>='),
    # \left, \right (just remove)
    (r'\\left', ''),
    (r'\\right', ''),
    # \text{...} -> ...
    (r'\\text\{([^{}]+)\}', r'\1'),
    (r'\\textbf\{([^{}]+)\}', r'\1'),
    (r'\\textit\{([^{}]+)\}', r'\1'),
    # \mathbf, \mathrm, etc -> just content
    (r'\\math[a-z]+\{([^{}]+)\}', r'\1'),
    # \log, \sin, \cos, \tan, \ln, \exp -> log, sin, cos, tan, ln, exp
    (r'\\(log|sin|cos|tan|ln|exp|lim|inf|sup|min|max|gcd|lcm)', r'\1'),
    # \sum -> sum, \prod -> prod, \int -> integral
    (r'\\sum', 'sum'),
    (r'\\prod', 'prod'),
    (r'\\int', 'integral'),
    # \infty -> infinity
    (r'\\infty', 'infinity'),
    # \pi -> pi, \theta -> theta, etc.
    (r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)', r'\1'),
    (r'\\(Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)', r'\1'),
    # \binom{n}{k} -> C(n,k)
    (r'\\binom\{([^{}]+)\}\{([^{}]+)\}', r'C(\1,\2)'),
    # \overline{x} -> x_bar
    (r'\\overline\{([^{}]+)\}', r'\1_bar'),
    # \vec{x} -> vec(x)
    (r'\\vec\{([^{}]+)\}', r'vec(\1)'),
    # \hat{x} -> x_hat
    (r'\\hat\{([^{}]+)\}', r'\1_hat'),
    # Braces: \{ \} -> ( )
    (r'\\{', '('),
    (r'\\}', ')'),
    # \quad, \qquad, \, \; \! -> space
    (r'\\(quad|qquad)', ' '),
    (r'\\[,;!]', ' '),
    # \\ (newline) -> ; or space
    (r'\\\\', '; '),
    # \hspace, \vspace -> space
    (r'\\[hv]space\{[^{}]*\}', ' '),
    # Remaining \commandname without braces -> remove
    (r'\\([a-zA-Z]+)', r'\1'),
    # Clean up multiple spaces
    (r'  +', ' '),
]


def apply_rules(text: str) -> str:
    """Apply regex-based LaTeX to plain math conversion."""
    result = text
    for pattern, replacement in LATEX_RULES:
        result = re.sub(pattern, replacement, result)
    return result.strip()


def has_remaining_latex(text: str) -> bool:
    """Check if text still has LaTeX-like content that rules didn't handle."""
    indicators = [
        r'\\[a-zA-Z]+',        # remaining commands
        r'\\frac',             # nested fracs
        r'\{[^{}]*\{',        # nested braces
    ]
    for pattern in indicators:
        if re.search(pattern, text):
            return True
    return False


# ---------------------------------------------------------------------------
# Opus API for complex LaTeX
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        session = boto3.Session(profile_name=BEDROCK_PROFILE, region_name=BEDROCK_REGION)
        _client = session.client("bedrock-runtime")
    return _client


OPUS_PROMPT = """Convert the following text containing LaTeX math notation into plain text suitable for Braille translation. Rules:
- Replace LaTeX commands with plain ASCII math: \\frac{a}{b} -> (a)/(b), \\sqrt{x} -> sqrt(x), x^{2} -> x^(2)
- Keep the surrounding English text unchanged
- Keep variable names as-is (x, y, n, etc.)
- Use standard operators: +, -, *, /, =, <, >, <=, >=
- For complex expressions, use parentheses to clarify grouping
- Do NOT add explanations, just return the converted text

Input: {text}

Output:"""


def opus_convert(text: str) -> str:
    """Call Opus 4.8 to convert complex LaTeX to plain math."""
    client = _get_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0,
        "messages": [{"role": "user", "content": OPUS_PROMPT.format(text=text)}],
    })
    resp = client.invoke_model(
        modelId=OPUS_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_text(text: str, use_opus_fallback: bool = True) -> str:
    """Preprocess text for Braille translation.

    1. Apply rule-based LaTeX substitutions
    2. If complex LaTeX remains and use_opus_fallback=True, call Opus
    """
    cleaned = apply_rules(text)

    if use_opus_fallback and has_remaining_latex(cleaned):
        cleaned = opus_convert(text)
        cleaned = apply_rules(cleaned)

    return cleaned


def preprocess_batch(texts: list[str], use_opus_fallback: bool = True) -> list[str]:
    """Preprocess a batch of texts."""
    results = []
    opus_count = 0
    for text in texts:
        cleaned = apply_rules(text)
        if use_opus_fallback and has_remaining_latex(cleaned):
            cleaned = opus_convert(text)
            cleaned = apply_rules(cleaned)
            opus_count += 1
        results.append(cleaned)
    if opus_count > 0:
        print(f"  [latex_preprocess] {opus_count}/{len(texts)} texts required Opus fallback")
    return results


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Simple - rules only
        ("Simple fraction", r"Find $\frac{3}{4} + \frac{1}{2}$."),
        ("Quadratic", r"If $x^2 + 2x - 15 = 0$, find $x$."),
        ("Square root", r"What is $\sqrt{144}$?"),
        # Medium - rules should handle
        ("Binomial", r"Compute $\binom{10}{3}$."),
        ("Trig", r"Find $\sin(\theta) + \cos(\theta)$."),
        ("Inequality", r"Solve $2x + 3 \geq 7$."),
        # Complex - may need Opus
        ("Nested frac", r"Simplify $\frac{\frac{1}{2} + \frac{1}{3}}{\frac{1}{4}}$."),
        ("Sum notation", r"Evaluate $\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}$."),
        # Real AIME-style
        ("AIME style", r"Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards."),
    ]

    print("=" * 70)
    print("LaTeX Preprocessing Demo")
    print("=" * 70)
    print()

    for label, text in test_cases:
        rules_only = apply_rules(text)
        needs_opus = has_remaining_latex(rules_only)
        print(f"[{label}]")
        print(f"  Input:      {text}")
        print(f"  Rules only: {rules_only}")
        print(f"  Needs Opus: {needs_opus}")
        if needs_opus:
            try:
                full = preprocess_text(text, use_opus_fallback=True)
                print(f"  With Opus:  {full}")
            except Exception as e:
                print(f"  Opus error: {e}")
        print()
