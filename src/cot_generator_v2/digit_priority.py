"""Digit-assignment priority for the <digit_priority> block.

Two strategies, selectable via config (digit_priority_mode):

  most_constrained — weighted, mirrors the template:
      Score(sym) = sum_op occ_op(sym) * W_op,   W_op = 1 + #candidate_families(op)
      occ_op(sym) = occurrences of sym across operand+rhs slots of op's equations
      Higher score first; ties broken by first appearance.

  occurrences — plain total occurrences across all example equations, highest
      first; ties broken by first appearance.

  both — choose one of the two per sample with a seeded RNG.

Returns the symbol order plus the rendered block text.

Run (self-test):
    uv run python src/cot_generator/digit_priority.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.equations import Equation, occurrences
from src.cot_generator_v2.factual_observations import OperatorAnalysis
from src.cot_generator_v2.families import derive_seed


@dataclass
class DigitPriority:
    mode: str
    order: list[str]            # symbols, highest priority first
    text: str                   # rendered <digit_priority> inner text


def _appearance_index(content_syms: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(content_syms)}


def _most_constrained(
    analyses: list[OperatorAnalysis], content_syms: list[str],
) -> DigitPriority:
    weights = {a.op: 1 + len(a.survivors) for a in analyses}
    appear = _appearance_index(content_syms)

    occ: dict[str, dict[str, int]] = {}
    score: dict[str, int] = {}
    for s in content_syms:
        occ[s] = {a.op: occurrences(s, a.eqs) for a in analyses}
        score[s] = sum(occ[s][a.op] * weights[a.op] for a in analyses)

    order = sorted(content_syms, key=lambda s: (-score[s], appear[s]))

    # Render explanation + table.
    lines = [
        "We assign digits most-constrained-first: symbols that recur in the tightest "
        "operator groups are hardest to satisfy, so we place them first. Score formula:",
        "",
    ]
    lines.append("Score(sym) = " + " + ".join(f"occ({a.op}) x W({a.op})" for a in analyses))
    wdesc = "; ".join(
        f"`{a.op}` has {len(a.survivors)} candidates -> W({a.op}) = {weights[a.op]}"
        for a in analyses
    )
    lines.append(f"  where W(op) = 1 + #candidates for that operator group ({wdesc}).")
    lines.append("")
    header = "sym  " + "  ".join(f"occ({a.op})" for a in analyses) + "   score"
    lines.append(header)
    for s in order:
        terms = "  ".join(f"{occ[s][a.op]:>{len('occ('+a.op+')')}}" for a in analyses)
        formula = " + ".join(f"{occ[s][a.op]}x{weights[a.op]}" for a in analyses)
        lines.append(f"{s:<4} {terms}   {formula} = {score[s]}")
    return DigitPriority("most_constrained", order, "\n".join(lines))


def _occurrences(
    examples: list[Equation], content_syms: list[str],
) -> DigitPriority:
    appear = _appearance_index(content_syms)
    total = {s: occurrences(s, examples) for s in content_syms}
    order = sorted(content_syms, key=lambda s: (-total[s], appear[s]))

    lines = [
        "We assign digits by raw occurrence count: symbols appearing in more "
        "operand and result slots are assigned first.",
        "",
        "sym   occurrences",
    ]
    for s in order:
        lines.append(f"{s:<4}  {total[s]}")
    return DigitPriority("occurrences", order, "\n".join(lines))


def compute_digit_priority(
    analyses: list[OperatorAnalysis],
    examples: list[Equation],
    content_syms: list[str],
    mode: str,
    seed: int,
    problem_id: str,
) -> DigitPriority:
    """Compute the digit-priority order + block text for the chosen mode."""
    if mode == "both":
        rng = random.Random(derive_seed(seed, problem_id, "digit_priority_mode"))
        mode = rng.choice(["most_constrained", "occurrences"])
    if mode == "most_constrained":
        return _most_constrained(analyses, content_syms)
    return _occurrences(examples, content_syms)


def _selftest() -> None:
    from src.cot_generator_v2.equations import parse_prompt, content_symbols
    from src.cot_generator_v2.factual_observations import analyze_operators
    from src.cot_generator_v2.redundancy_observations import apply_redundancies

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "[[-!' = @&\n`!*\\& = '@'{\n\\'-!` = \\\\\n\n"
        "Now, determine the result for: `!*[{"
    )
    examples, q = parse_prompt(prompt, "'\"[`")
    syms = content_symbols(examples)
    analyses = analyze_operators(examples, q.op, base=10)
    apply_redundancies(analyses)
    mc = compute_digit_priority(analyses, examples, syms, "most_constrained", 0, "00457d26")
    assert mc.mode == "most_constrained" and len(mc.order) == len(syms)
    oc = compute_digit_priority(analyses, examples, syms, "occurrences", 0, "00457d26")
    assert set(oc.order) == set(syms)
    # 'both' is deterministic for a fixed seed+id
    b1 = compute_digit_priority(analyses, examples, syms, "both", 3, "00457d26")
    b2 = compute_digit_priority(analyses, examples, syms, "both", 3, "00457d26")
    assert b1.mode == b2.mode
    print("digit_priority self-test PASS")
    print(mc.text)


if __name__ == "__main__":
    _selftest()
