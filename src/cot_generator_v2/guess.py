"""Augmentation 4 — Guess samples via chain truncation.

For rows with k≥2 perturbation rounds, generates a truncated companion CoT
that ends with a "budget exhausted" conclusion instead of the full solution.

Only rows where the (k-1)-round answer matches the gold answer are kept.
This condition holds when none of the symbols swapped in the final perturbation
appear in the result variable — a minority of rows.

Public API:
  try_build_guess_sample(res, row, cfg, cot_column)
      → dict (new row) or None
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.render import CotResult, render_guess_conclusion


_TAG_RE = re.compile(r"<digit_mapping_analysis_follow_up_(\d+)>.*?</digit_mapping_analysis_follow_up_\1>",
                     re.DOTALL)


def _truncate_cot(text: str, keep_through_n: int) -> str | None:
    """Remove all blocks numbered > keep_through_n and the final conclusion.

    Returns the truncated text or None if truncation is not possible.
    """
    # Find the last follow-up block to keep
    last_match = None
    for m in _TAG_RE.finditer(text):
        n = int(m.group(1))
        if n == keep_through_n:
            last_match = m
    if last_match is None:
        return None

    # Truncate after the last kept follow-up block
    return text[:last_match.end()].rstrip()


def try_build_guess_sample(
    res: CotResult,
    row: dict,
    cot_column: str,
) -> dict | None:
    """Try to build a truncated guess-sample companion for a full-solution CoT.

    Conditions:
    - The CoT has k≥2 perturbations (num_perturbations ≥ 2).
    - The truncated (k-1)-round answer computed from the follow-up's \\boxed{}
      matches the gold answer.

    Returns a new row dict with:
      cot_column: the truncated CoT text
      cot_num_perturbations: k-1
      cot_variant_source: 'guess_truncated'
    or None if the conditions are not met.
    """
    k = res.num_perturbations
    if k < 2:
        return None

    gold = row.get("solver_answer", "")
    truncate_at = k - 1  # keep follow-ups 1..k-1, drop solution step k

    truncated = _truncate_cot(res.text, truncate_at)
    if truncated is None:
        return None

    # Extract the \boxed{} answer from follow-up k-1 (the last kept one)
    # The running \boxed{} is appended at the end of each follow-up block (Fix 9b)
    box_m = re.findall(r"\\boxed\{([^}]*)\}", truncated)
    if not box_m:
        return None
    candidate_answer = box_m[-1]

    # Filter: keep only if candidate matches gold (safe for SFT)
    if candidate_answer != gold:
        return None

    # Append budget-exhausted conclusion
    guess_conc = render_guess_conclusion({}, gold, truncate_at)
    truncated_with_conc = truncated + f"\n\n<conclusion>\n{guess_conc}\n</conclusion>"

    new_row = dict(row)
    new_row[cot_column] = truncated_with_conc
    new_row["cot_num_perturbations"] = truncate_at
    new_row["cot_variant_source"] = "guess_truncated"
    new_row["cot_variant_index"] = -1  # distinguish from order variants
    return new_row
