"""Script #2 - post / cross-operator redundancy observations.

Runs *after* Script #1 (factual_observations). Where Script #1 rules out
families that are structurally impossible, Script #2 removes families that,
although individually possible, are *redundant* given the survivors.
It mutates each operator's surviving-family set and appends
`category="redundancy"` observations.

Rules implemented:
  1. identical-output collapse — when EVERY equation of an operator applies it to
     equal operands (L == R), `gcd` and `lcm` both equal x and `mod`/`rmod` both
     equal 0 on every equation, so each pair is indistinguishable; one is kept
     (the protected/solver family if present) and the twin is dropped.

Note: sub_signed / rsub_signed elimination for unsigned rhs is now handled
upstream in feasibility.py's sign check (Script #1), not here.

Run (self-test):
    uv run python src/cot_generator/redundancy_observations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.factual_observations import OperatorAnalysis
from src.cot_generator_v2.feasibility import Observation


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
        keep_family = protect.get(a.op)
        def _drop_from_all(name: str) -> None:
            """Remove *name* from survivors and every mode view."""
            a.survivors.pop(name, None)
            for mode_view in a.survivors_by_mode.values():
                mode_view.pop(name, None)

        # Rule 1: identical-output collapse on all-equal-operand groups.
        if a.eqs and all(eq.self_equal() for eq in a.eqs):
            for twins in (("gcd", "lcm"), ("mod", "rmod")):
                present = [n for n in twins if n in a.survivors]
                if len(present) < 2:
                    continue
                keep = keep_family if keep_family in present else present[0]
                dropped_twin = [n for n in present if n != keep]
                for n in dropped_twin:
                    _drop_from_all(n)
                verb = "is" if len(dropped_twin) == 1 else "are"
                a.observations.append(Observation(
                    f"Every equation has equal operands so {present[0]} and {present[1]} "
                    f"always yield the same value and cannot be distinguished -> "
                    f"{', '.join(dropped_twin)} {verb} dropped (keeping {keep}).",
                    tuple(dropped_twin), category="redundancy",
                ))


def _selftest() -> None:
    from src.cot_generator_v2.equations import parse_prompt
    from src.cot_generator_v2.factual_observations import analyze_operators

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
    # sub_signed / rsub_signed already dropped at the sign stage (Script #1);
    # apply_redundancies no longer produces a redundancy bullet for them.
    assert "sub_signed" not in minus.survivors and "rsub_signed" not in minus.survivors
    # Template's `-` group: add, mul, absdiff, gcd, lcm, fdiv, mod, rmod -> 8.
    assert set(minus.survivors) == {"add", "mul", "absdiff", "gcd", "lcm", "fdiv", "mod", "rmod"}, minus.survivors
    # No redundancy observation expected: Rule 1 removed; Rule 2 needs L==R equations.
    assert not any(o.category == "redundancy" for o in minus.observations)

    # Rule 2: an all-equal-operand group (`AA+AA = AA`).
    # Rule C (structural) drops mod and rmod: mod(x,x) = 0 != x, rmod(x,x) = 0 != x.
    # gcd/lcm survive (gcd(x,x)=lcm(x,x)=x can match rhs=AA).
    # apply_redundancies then collapses the gcd/lcm twin (they are identical on L==R).
    prompt2 = (
        "In Alice's Wonderland, a secret set of transformation rules is applied "
        "to equations. Below are a few examples:\n"
        "AA+AA = AA\n\n"
        "Now, determine the result for: BB+CC"
    )
    ex2, q2 = parse_prompt(prompt2, "DD")
    an2 = analyze_operators(ex2, q2.op, base=10)
    plus = next(a for a in an2 if a.op == "+")
    # mod/rmod already eliminated by structural Rule C before redundancy runs.
    assert {"gcd", "lcm"} <= set(plus.survivors), plus.survivors
    assert "mod" not in plus.survivors and "rmod" not in plus.survivors, plus.survivors
    apply_redundancies(an2)
    assert len({"gcd", "lcm"} & set(plus.survivors)) == 1, plus.survivors
    assert any(o.category == "redundancy" for o in plus.observations)
    print("redundancy_observations Rule 2 (identical-output collapse) PASS")

    print("redundancy_observations self-test PASS")


if __name__ == "__main__":
    _selftest()
