"""Follow-up "tells": derive the next fix from a failing attempt (Fix 1 — honest).

Fix 1 rewrites this module to eliminate oracle values from the generated text.
The generated text now only describes what the BROKEN MAPPING's forward checks
produce — decoded operands, computed value, rhs target, gap — and frames the
fix as a suggestion ("one viable option is to swap A ↔ B") rather than a
certainty derived from the known solution.

The solution is used ONLY as a private selector (to find which equations newly
broke after the perturbation) — it never appears in the generated CoT text.

Run (self-test):
    uv run python src/cot_generator_v2/tells.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.decode import (
    LITTLE_ENDIAN,
    decode_operand,
    evaluate_equation,
    family_matches,
)
from src.cot_generator_v2.equations import Equation
from src.cot_generator_v2.families import apply_family
from src.cot_generator_v2.perturbation import ChainStep, Perturb


def _fix_change_phrase(fix: Perturb) -> str:
    """Suggest the fix without exposing oracle digit values."""
    if fix.kind == "swap":
        a, b = fix.symbols
        return f"swap {a} ↔ {b}"
    s = fix.symbols[0]
    return f"reassign {s} to an unused digit (it currently holds {fix.broken[s]})"


def _isolating_equation(
    fix: Perturb, other_broken: set[str], examples: list[Equation],
    mapping: dict[str, int], solution: dict[str, int],
    base: int, mode: str, program: dict[str, tuple[str, int]],
) -> Equation | None:
    """Find an example equation isolating the fix.

    Conditions:
    - Contains at least one fix symbol.
    - Does NOT contain any other-broken symbol (so it isolates the fix).
    - Fails under the broken mapping (so there is something to explain).
    - Passes under the solution (private check — selector only; not in text).
    """
    fix_syms = set(fix.symbols)
    for eq in examples:
        syms = set(eq.symbols)
        if not (syms & fix_syms):
            continue
        if syms & other_broken:
            continue
        fam, off = program[eq.op]
        if family_matches(fam, off, eq, mapping, base, mode).ok:
            continue
        if family_matches(fam, off, eq, solution, base, mode).ok:
            return eq
    return None


def _algebraic_tell(
    eq: Equation, fix: Perturb, mapping: dict[str, int],
    base: int, mode: str, program: dict[str, tuple[str, int]],
) -> str:
    """Build an honest algebraic tell from an isolating equation (Fix 1).

    Only uses the BROKEN MAPPING's forward-check results. Oracle values never
    appear in the generated text. The fix is framed as a suggestion.
    """
    fam, off = program[eq.op]
    L = decode_operand(eq.l0, eq.l1, mapping, base, mode)
    R = decode_operand(eq.r0, eq.r1, mapping, base, mode)
    ev = evaluate_equation(eq, mapping, base, mode)
    val = apply_family(fam, off, L, R, base)

    operand_syms = {eq.l0, eq.l1, eq.r0, eq.r1}
    result_syms = set(eq.rhs)
    fix_syms = set(fix.symbols)

    off_txt = f" (offset {off:+d})" if off != 0 else ""
    head = f"eq[{eq.index}] ({eq.raw}) fails: with {fam}{off_txt}"
    fix_phrase = _fix_change_phrase(fix)
    gap = (ev.target - val) if val is not None else None
    gap_txt = f" (gap of {gap})" if gap is not None else ""

    if fix_syms <= result_syms and not (fix_syms & operand_syms):
        # Result-position tell: both operands decode correctly; result symbol is off.
        syms_str = ", ".join(sorted(fix_syms))
        return (
            f"{head}, {eq.l0}{eq.l1} decodes to {L} and {eq.r0}{eq.r1} to {R}, "
            f"giving {val}. The rhs target is {ev.target}{gap_txt}. "
            f"Result symbol(s) {syms_str} sit in the rhs position and appear to carry "
            f"the wrong digit(s). One viable option is to {fix_phrase}."
        )

    # Operand-gap tell: the result is near the target; the discrepancy is in an operand.
    left_syms, right_syms = {eq.l0, eq.l1}, {eq.r0, eq.r1}
    in_left = bool(fix_syms & left_syms) and not (fix_syms & right_syms)
    in_right = bool(fix_syms & right_syms) and not (fix_syms & left_syms)

    if in_left or in_right:
        side = "left" if in_left else "right"
        side_val = L if in_left else R
        side_syms = fix_syms & (left_syms if in_left else right_syms)
        return (
            f"{head}, {eq.l0}{eq.l1} decodes to {L} and {eq.r0}{eq.r1} to {R}, "
            f"giving {val}. The rhs target is {ev.target}{gap_txt}. "
            f"The gap points at the {side} operand (currently {side_val}); "
            f"symbol(s) {', '.join(sorted(side_syms))} are there. "
            f"One viable option is to {fix_phrase}."
        )

    # Fallback: fix spans both operands or mixed position.
    return (
        f"{head}, operands decode to {L} and {R} giving {val}{gap_txt}. "
        f"The rhs target is {ev.target}. "
        f"Symbol(s) {', '.join(sorted(fix_syms))} appear to be causing the mismatch. "
        f"One viable option is to {fix_phrase}."
    )


def make_tell(
    step: ChainStep,
    examples: list[Equation],
    solution: dict[str, int],
    program: dict[str, tuple[str, int]],
    base: int,
    mode: str,
    tell_mode: str,
) -> str:
    """Render the follow-up text that derives `step`'s fix(es).

    Fix 1: all text is grounded in the broken mapping's forward checks only.
    The solution is used privately to (a) find isolating equations and
    (b) determine which equations newly broke — it never enters the CoT text.
    """
    if not step.fixes:
        return "All equations are satisfied and a feasible answer can be deduced. No follow-up required."

    # Private: which symbols are broken in other active perturbations (not this fix).
    other_broken = {s for p in step.active for s in p.symbols} - {
        s for f in step.fixes for s in f.symbols
    }

    bullets: list[str] = []
    for fix in step.fixes:
        eq = None
        if tell_mode == "algebraic":
            eq = _isolating_equation(fix, other_broken, examples, step.mapping,
                                     solution, base, mode, program)
        if eq is not None:
            # Fix 1: pass only mapping (no solution) to _algebraic_tell
            bullets.append("- " + _algebraic_tell(eq, fix, step.mapping, base, mode, program))
        else:
            # Simple fallback: name a failing equation touching the fix symbols.
            touched = set(fix.symbols)
            fail_eq = next(
                (e for e in examples
                 if set(e.symbols) & touched
                 and not family_matches(*program[e.op], e, step.mapping, base, mode).ok),
                None,
            )
            where = f"eq[{fail_eq.index}] ({fail_eq.raw})" if fail_eq else "the failing equation"
            bullets.append(
                f"- The clearest remaining discrepancy is in {where}; "
                f"one viable option is to {_fix_change_phrase(fix)}."
            )
    return "\n".join(bullets)


def _selftest() -> None:
    import json
    import pandas as pd
    from src.cot_generator_v2.equations import parse_prompt, content_symbols
    from src.cot_generator_v2.factual_observations import analyze_operators
    from src.cot_generator_v2.redundancy_observations import apply_redundancies
    from src.cot_generator_v2.families import resolve_registry_op
    from src.cot_generator_v2.perturbation import build_chain
    from src.cot_generator_v2.analysis import solution_program

    df = pd.read_parquet("inputs/data/v2/only_cryptharithm_solved_k_1.parquet")
    r = df.iloc[0]
    sol = json.loads(r["digits_mapping"])
    om = json.loads(r["ops_mapping"])
    opfam = {g: resolve_registry_op(n) for g, n in om.items()}
    base = int(round(float(r["solver_radix"])))
    mode = "standard" if "little" not in str(r["solver_strategy"]).lower() else "little_endian"
    examples, q = parse_prompt(r["prompt"], r["answer"])
    syms = content_symbols(examples + [q])
    analyses = analyze_operators(examples, q.op, base)
    apply_redundancies(analyses, protect={q.op: opfam[q.op][0]})
    program = solution_program(analyses, sol, base, mode, opfam)
    chain = build_chain(sol, syms, examples, program, base, 3, 0, "test", mode)

    for st in chain.steps[:-1]:
        t = make_tell(st, examples, sol, program, base, mode, "algebraic")
        assert t and t.startswith("-"), f"bad tell: {t!r}"
        # Fix 1 check: no oracle value should appear as "must be X (A=Y, B=Z)"
        assert "must be" not in t or "one viable option" in t, \
            f"possible oracle value leak in tell: {t!r}"
        print(f"follow_up_{st.index}:\n{t}\n")
    print("tells self-test PASS (Fix 1 — no oracle values)")


if __name__ == "__main__":
    _selftest()
