"""
categorize_train.py
-------------------
Reads inputs/data/train.csv (id, prompt, answer) and writes
inputs/data/train.parquet with four columns:

  problem_id       — original puzzle id
  prompt           — unchanged prompt string
  answer           — unchanged answer string
  problem_category — one of:
      Bit Manipulation
      Gravity
      Unit conversion
      Cipher
      Numeral
      Equation Numeric
      Cryptarithm (Symbolic Equations)

problem_category is derived from the prompt text.  Five categories are
identified by a unique substring in the intro line.  The sixth intro
("a secret set of transformation rules is applied to equations") is shared
by Equation Numeric and Cryptarithm; those two are separated by computing
the digit-fraction of the example-line operands: Cryptarithm rows score 0.0
(all symbolic glyphs), Equation Numeric rows score ≥ 0.7 (mostly digits).
Threshold: digit_frac < 0.3 → Cryptarithm (Symbolic Equations).

Run:  uv run python src/pre_processing/categorize_train.py
"""

import re
from pathlib import Path

import pandas as pd

SRC = Path("inputs/data/train.csv")
OUT = Path("inputs/data/train.parquet")

# Ordered pairs: (unique intro substring, category label).
# The equations intro maps to the sentinel "_equations"; the helper below
# disambiguates it into Equation Numeric vs Cryptarithm (Symbolic Equations).
INTRO_CATEGORIES: list[tuple[str, str]] = [
    ("a secret bit manipulation rule transforms 8-bit binary numbers", "Bit Manipulation"),
    ("the gravitational constant has been secretly changed", "Gravity"),
    ("a secret unit conversion is applied to measurements", "Unit conversion"),
    ("secret encryption rules are used on text", "Cipher"),
    ("numbers are secretly converted into a different numeral system", "Numeral"),
    ("a secret set of transformation rules is applied to equations", "_equations"),
]


def _digit_frac(prompt: str) -> float:
    """Return fraction of non-whitespace, non-'=' chars that are digits,
    computed over example lines only (lines containing '=' that don't start
    with 'now')."""
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    example_lines = [
        l for l in lines if "=" in l and not l.lower().startswith("now")
    ]
    body = re.sub(r"[=\s]", "", "".join(example_lines))
    if not body:
        return 0.0
    return sum(c.isdigit() for c in body) / len(body)


def _classify(prompt: str) -> str:
    for substring, label in INTRO_CATEGORIES:
        if substring in prompt:
            if label != "_equations":
                return label
            # Disambiguate Equation Numeric vs Cryptarithm
            return (
                "Cryptarithm (Symbolic Equations)"
                if _digit_frac(prompt) < 0.3
                else "Equation Numeric"
            )
    raise ValueError(f"Unrecognised prompt intro:\n{prompt[:120]!r}")


def main() -> None:
    df = pd.read_csv(SRC)

    out = pd.DataFrame(
        {
            "problem_id": df["id"],
            "prompt": df["prompt"],
            "answer": df["answer"],
            "problem_category": df["prompt"].map(_classify),
        }
    )

    out.to_parquet(OUT, index=False)

    print(f"Written {len(out):,} rows → {OUT}\n")
    print(out["problem_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
