"""Follow-up "tells": derive the next fix from a failing attempt.

After an attempt fails, the <digit_mapping_analysis_follow_up_N> block explains
which single change to make next. Because the chain is built backward, we know
exactly which perturbation each follow-up must motivate (perturbation.Perturb).
The job here is to phrase that as a believable deduction:

  * result-position tell - the lhs operands decode correctly, so each result
    symbol must equal the computed result digit at its position; a mismatch
    names the wrong symbol.
  * operand-gap tell - the result is correct, so an op landing near the target
    reveals the wrong operand digit from the small gap.

A tell is cleanest when some equation isolates the fix (it contains the fix's
symbols and none of the other still-broken symbols). When none does, or when
tell_mode == 'simple', we emit a short, correct fallback that names the failing
equation and the change.

Run (self-test):
    uv run python src/cot_generator/tells.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.decode import (
    LITTLE_ENDIAN,
    decode_operand,
    evaluate_equation,
    family_matches,
)
from src.cot_generator.equations import Equation
from src.cot_generator.families import apply_family
from src.cot_generator.perturbation import ChainStep, Perturb


def _digits_str(mapping, symbols) -> str:
    return ", ".join(f"{s}={mapping[s]}" for s in symbols)


def _required_operand_digits(
    value: int, s0: str, s1: str, base: int, mode: str,
) -> dict[str, int] | None:
    """Per-symbol digit the 2-symbol operand `s0 s1` needs to decode to `value`.

    Inverts `decode_operand`: standard `s0 s1 = s0*base + s1`; little_endian swaps
    the two digits. Returns None when `value` does not fit two base-`base` digits.
    """
    if not (0 <= value < base * base):
        return None
    hi, lo = value // base, value % base
    if mode == LITTLE_ENDIAN:
        return {s0: lo, s1: hi}
    return {s0: hi, s1: lo}


def _isolating_equation(
    fix: Perturb, other_broken: set[str], examples: list[Equation],
    mapping, solution, base, mode, program,
) -> Equation | None:
    """An example eq that contains a fix symbol, no other broken symbol, fails
    now, and passes at the solution."""
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


def _fix_change_phrase(fix: Perturb) -> str:
    if fix.kind == "swap":
        a, b = fix.symbols
        return f"swap {a} <-> {b} ({a} = {fix.correct[a]}, {b} = {fix.correct[b]})"
    s = fix.symbols[0]
    return f"reassign {s} to the unused digit {fix.correct[s]} (it currently sits on {fix.broken[s]})"


def _algebraic_tell(
    eq: Equation, fix: Perturb, mapping, solution, base, mode, program,
) -> str:
    """Build an algebraic tell from an isolating equation."""
    fam, off = program[eq.op]
    L = decode_operand(eq.l0, eq.l1, mapping, base, mode)
    R = decode_operand(eq.r0, eq.r1, mapping, base, mode)
    ev = evaluate_equation(eq, mapping, base, mode)
    val = apply_family(fam, off, L, R, base)

    operand_syms = {eq.l0, eq.l1, eq.r0, eq.r1}
    result_syms = set(eq.rhs)
    fix_syms = set(fix.symbols)

    head = f"eq[{eq.index}] ({eq.raw}) still fails: with {fam} (offset {off:+d})"
    if fix_syms <= result_syms and not (fix_syms & operand_syms):
        # result-position tell: operands are right, the result symbol is wrong.
        Lc = decode_operand(eq.l0, eq.l1, solution, base, mode)
        Rc = decode_operand(eq.r0, eq.r1, solution, base, mode)
        target = apply_family(fam, off, Lc, Rc, base)
        return (
            f"{head}, the operands decode to {L} and {R} and the result should be "
            f"{target}. The result symbol(s) {', '.join(sorted(fix_syms))} disagree with that "
            f"digit, so they must change. Minimal fix: {_fix_change_phrase(fix)}; every other "
            "symbol keeps its value."
        )
    # operand-gap tell: the result is right, so an operand digit is off. When the
    # fix lies wholly inside ONE operand, that operand is fully determined: the
    # other operand and the result are already pinned by the equations that pass
    # (the isolating eq has no other broken symbol), so back-solve the broken
    # operand and name the exact digits it must take (4a back-solve + 4b narrowing).
    left_syms, right_syms = {eq.l0, eq.l1}, {eq.r0, eq.r1}
    in_left = bool(fix_syms) and fix_syms <= left_syms and not (fix_syms & right_syms)
    in_right = bool(fix_syms) and fix_syms <= right_syms and not (fix_syms & left_syms)
    if in_left or in_right:
        Lc = decode_operand(eq.l0, eq.l1, solution, base, mode)
        Rc = decode_operand(eq.r0, eq.r1, solution, base, mode)
        side, other = ("left", "right") if in_left else ("right", "left")
        known_val = R if in_left else L
        req = Lc if in_left else Rc
        s0, s1 = (eq.l0, eq.l1) if in_left else (eq.r0, eq.r1)
        reqd = _required_operand_digits(req, s0, s1, base, mode)
        if reqd is not None:
            need = ", ".join(f"{s}={reqd[s]}" for s in (s0, s1) if s in fix_syms)
            return (
                f"{head}, the operands decode to {L} and {R} giving {val}, but the rhs needs "
                f"{ev.target}. eq[{eq.index}]'s other symbols are already pinned by the "
                f"equations that pass, and with the {other} operand fixed at {known_val} the "
                f"{side} operand must be {req} ({need}). Minimal fix: {_fix_change_phrase(fix)}; "
                "every other symbol keeps its value."
            )
    # fallback: the fix spans both operands (or the operand can't be isolated).
    return (
        f"{head}, the operands decode to {L} and {R} giving {val}, but the rhs needs "
        f"{ev.target}. The gap points at the operand symbol(s) {', '.join(sorted(fix_syms))}; "
        f"correcting them lands on the target. Minimal fix: {_fix_change_phrase(fix)}; every "
        "other symbol keeps its value."
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
    """Render the follow-up text that derives `step`'s fix(es)."""
    if not step.fixes:
        return "All equations are satisfied and a feasible answer can be deduced. No follow-up required."

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
            bullets.append("- " + _algebraic_tell(eq, fix, step.mapping, solution, base, mode, program))
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
                f"- The clearest remaining discrepancy is in {where}; the minimal change that "
                f"resolves it is to {_fix_change_phrase(fix)}."
            )
    return "\n".join(bullets)


def _selftest() -> None:
    import json
    import pandas as pd
    from src.cot_generator.equations import parse_prompt, content_symbols
    from src.cot_generator.factual_observations import analyze_operators
    from src.cot_generator.redundancy_observations import apply_redundancies
    from src.cot_generator.families import resolve_registry_op
    from src.cot_generator.perturbation import build_chain
    from src.cot_generator.analysis import solution_program

    df = pd.read_parquet("inputs/data/result_examples_3_rand_k_6_max2.parquet")
    r = df[df["id"] == "00457d26_q0"].iloc[0]
    sol = json.loads(r["digits_mapping"])
    om = json.loads(r["ops_mapping"])
    opfam = {g: resolve_registry_op(n) for g, n in om.items()}
    base = int(round(float(r["solver_radix"])))
    mode = str(r["solver_strategy"]).strip().lower()
    examples, q = parse_prompt(r["prompt"], r["answer"])
    syms = content_symbols(examples)
    analyses = analyze_operators(examples, q.op, base)
    apply_redundancies(analyses, protect={q.op: opfam[q.op][0]})
    program = solution_program(analyses, sol, base, mode, opfam)
    chain = build_chain(sol, syms, examples, program, base, 4, 0, "00457d26", mode)

    # _required_operand_digits must invert decode_operand in both encodings.
    for mode_ in ("standard", "little_endian"):
        dm = {"x": 3, "y": 7}
        v = decode_operand("x", "y", dm, 10, mode_)
        assert _required_operand_digits(v, "x", "y", 10, mode_) == dm, (mode_, v)
    print("_required_operand_digits round-trip PASS")

    for st in chain.steps[:-1]:
        t = make_tell(st, examples, sol, program, base, mode, "algebraic")
        assert t and t.startswith("-")
        print(f"follow_up_{st.index}:\n{t}\n")
    print("tells self-test PASS")


if __name__ == "__main__":
    _selftest()
