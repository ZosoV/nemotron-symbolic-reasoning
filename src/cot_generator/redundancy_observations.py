"""Script #2 - post / cross-operator redundancy observations.

Runs *after* Script #1 (factual_observations). Where Script #1 rules out
families that are structurally impossible, Script #2 removes families that,
although individually possible, are *redundant* given the survivors — or that a
sibling operator already reasoned about identically. It mutates each operator's
surviving-family set and appends `category="redundancy"` observations.

Rules implemented:
  1. absdiff dominance — when the rhs is unsigned and `absdiff` survives, the
     signed families `sub_signed` and `rsub_signed` only ever need to match a
     non-negative result, and `|a-b|` already covers both orientations, so they
     are dropped as redundant (the template's "absdiff already covers both
     positive cases").
  2. identical-output collapse — when EVERY equation of an operator applies it to
     equal operands (L == R), `gcd` and `lcm` both equal x and `mod`/`rmod` both
     equal 0 on every equation, so each pair is indistinguishable; one is kept
     (the protected/solver family if present) and the twin is dropped.

Run (self-test):
    uv run python src/cot_generator/redundancy_observations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.factual_observations import OperatorAnalysis
from src.cot_generator.feasibility import Observation


def apply_redundancies(
    analyses: list[OperatorAnalysis],
    protect: dict[str, str] | None = None,
) -> None:
    """Mutate `analyses` in place, applying cross-family dominance rules.

    `protect` maps an operator glyph to a family name that must NOT be dropped
    (used to keep the question operator's solver-chosen family so the conclusion
    can reproduce the labelled answer).
    """
    protect = protect or {}
    for a in analyses:
        if not a.constrained:
            continue  # no equations -> nothing to reduce
        all_unsigned = not any(eq.has_sign for eq in a.eqs)
        keep_family = protect.get(a.op)
        # Rule 1: absdiff dominates sub_signed / rsub_signed for unsigned rhs.
        if all_unsigned and "absdiff" in a.survivors:
            dominated = [
                n for n in ("sub_signed", "rsub_signed")
                if n in a.survivors and n != keep_family
            ]
            if dominated:
                for n in dominated:
                    del a.survivors[n]
                a.observations.append(Observation(
                    "sub_signed and rsub_signed would each have to land on a strictly "
                    "positive result to match the unsigned rhs; absdiff already covers "
                    f"both orientations, so they are redundant and dropped: "
                    f"{', '.join(dominated)}.",
                    tuple(dominated), category="redundancy",
                ))

        # Rule 2: identical-output collapse on all-equal-operand groups.
        if a.eqs and all(eq.self_equal() for eq in a.eqs):
            for twins in (("gcd", "lcm"), ("mod", "rmod")):
                present = [n for n in twins if n in a.survivors]
                if len(present) < 2:
                    continue
                keep = keep_family if keep_family in present else present[0]
                dropped_twin = [n for n in present if n != keep]
                for n in dropped_twin:
                    del a.survivors[n]
                verb = "is" if len(dropped_twin) == 1 else "are"
                a.observations.append(Observation(
                    f"every equation applies `{a.op}` to equal operands, so {present[0]} and "
                    f"{present[1]} yield the same value on each one and cannot be told apart; "
                    f"{', '.join(dropped_twin)} {verb} dropped as redundant (keeping {keep}).",
                    tuple(dropped_twin), category="redundancy",
                ))


def _selftest() -> None:
    from src.cot_generator.equations import parse_prompt
    from src.cot_generator.factual_observations import analyze_operators

    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "[[-!' = @&\n`!*\\& = '@'{\n\\'-!` = \\\\\n\n"
        "Now, determine the result for: `!*[{"
    )
    examples, q = parse_prompt(prompt, "'\"[`")
    analyses = analyze_operators(examples, q.op, base=10)
    apply_redundancies(analyses)
    minus = next(a for a in analyses if a.op == "-")
    assert "sub_signed" not in minus.survivors and "rsub_signed" not in minus.survivors
    # Template's `-` group: add, mul, absdiff, gcd, lcm, fdiv, mod, rmod -> 8.
    assert set(minus.survivors) == {"add", "mul", "absdiff", "gcd", "lcm", "fdiv", "mod", "rmod"}, minus.survivors
    assert any(o.category == "redundancy" for o in minus.observations)

    # Rule 2: an all-equal-operand group (`AA+AA = AA`) keeps gcd/lcm (rhs equals
    # the operand) and mod/rmod (rhs may be 0); the redundancy collapse then drops
    # one of each twin pair.
    prompt2 = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "AA+AA = AA\n\n"
        "Now, determine the result for: BB+CC"
    )
    ex2, q2 = parse_prompt(prompt2, "DD")
    an2 = analyze_operators(ex2, q2.op, base=10)
    plus = next(a for a in an2 if a.op == "+")
    assert {"gcd", "lcm"} <= set(plus.survivors) and {"mod", "rmod"} <= set(plus.survivors), plus.survivors
    apply_redundancies(an2)
    assert len({"gcd", "lcm"} & set(plus.survivors)) == 1, plus.survivors
    assert len({"mod", "rmod"} & set(plus.survivors)) == 1, plus.survivors
    assert any(o.category == "redundancy" for o in plus.observations)
    print("redundancy_observations Rule 2 (identical-output collapse) PASS")

    print("redundancy_observations self-test PASS")


if __name__ == "__main__":
    _selftest()
