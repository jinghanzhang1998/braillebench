"""Provider-neutral LaTeX cleanup used by the public translation utilities."""

import re


LATEX_RULES = [
    (r"\$\$(.+?)\$\$", r"\1"),
    (r"\$(.+?)\$", r"\1"),
    (r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)"),
    (r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)"),
    (r"\\sqrt\[([^]]+)\]\{([^{}]+)\}", r"root(\1, \2)"),
    (r"\^\{([^{}]+)\}", r"^(\1)"),
    (r"_\{([^{}]+)\}", r"_(\1)"),
    (r"\\cdot|\\times", "*"),
    (r"\\div", "/"),
    (r"\\pm", "+-"),
    (r"\\mp", "-+"),
    (r"\\leq|\\le", "<="),
    (r"\\geq|\\ge", ">="),
    (r"\\neq", "!="),
    (r"\\left|\\right", ""),
    (r"\\text\{([^{}]+)\}", r"\1"),
    (r"\\textbf\{([^{}]+)\}", r"\1"),
    (r"\\textit\{([^{}]+)\}", r"\1"),
    (r"\\math[a-z]+\{([^{}]+)\}", r"\1"),
    (r"\\(log|sin|cos|tan|ln|exp|lim|inf|sup|min|max|gcd|lcm)", r"\1"),
    (r"\\sum", "sum"),
    (r"\\prod", "prod"),
    (r"\\int", "integral"),
    (r"\\infty", "infinity"),
    (
        r"\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)",
        r"\1",
    ),
    (
        r"\\(Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)",
        r"\1",
    ),
    (r"\\binom\{([^{}]+)\}\{([^{}]+)\}", r"C(\1,\2)"),
    (r"\\overline\{([^{}]+)\}", r"\1_bar"),
    (r"\\vec\{([^{}]+)\}", r"vec(\1)"),
    (r"\\hat\{([^{}]+)\}", r"\1_hat"),
    (r"\\\{", "("),
    (r"\\\}", ")"),
    (r"\\(quad|qquad)", " "),
    (r"\\[,;!]", " "),
    (r"\\\\", "; "),
    (r"\\[hv]space\{[^{}]*\}", " "),
    (r"\\([a-zA-Z]+)", r"\1"),
    (r"  +", " "),
]


def apply_rules(text: str) -> str:
    result = text
    for pattern, replacement in LATEX_RULES:
        result = re.sub(pattern, replacement, result)
    return result.strip()


def has_remaining_latex(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in (r"\\[a-zA-Z]+", r"\\frac", r"\{[^{}]*\{")
    )


def preprocess_text(text: str, use_opus_fallback: bool = False) -> str:
    """Apply deterministic cleanup without calling an external model.

    ``use_opus_fallback`` is accepted for compatibility with the research code.
    The public utility fails explicitly if a caller requests that private fallback.
    """
    cleaned = apply_rules(text)
    if use_opus_fallback and has_remaining_latex(cleaned):
        raise RuntimeError(
            "Complex LaTeX remains. Supply your own preprocessing step; the public "
            "release does not call an LLM during dataset conversion."
        )
    return cleaned


def preprocess_batch(texts: list[str], use_opus_fallback: bool = False) -> list[str]:
    return [preprocess_text(text, use_opus_fallback=use_opus_fallback) for text in texts]
