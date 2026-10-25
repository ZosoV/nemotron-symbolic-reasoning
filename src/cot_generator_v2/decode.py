"""Operand decoding and result checking for the CoT generator.

Two encoding modes (base configurable, base 10 in practice):
  standard      AB = A x base + B   (result digits read most-significant first)
  little_endian AB = B x base + A   (result digit list reversed before reading)

This re-implements the decode/verify logic of src/encoding.py self-contained,
exposing exactly what the analysis renderer needs: decode a 2-symbol operand,
compute the integer a candidate op must equal, and test whether a family+offset
matches an equation under a mode.

Run (self-test):
    uv run python src/cot_generator/decode.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.equations import Equation
from src.cot_generator_v2.families import apply_family, int_to_base_digits

STANDARD = "standard"
LITTLE_ENDIAN = "little_endian"
MODES = (STANDARD, LITTLE_ENDIAN)


def decode_operand(s0: str, s1: str, digit_map: dict[str, int], base: int, mode: str) -> int:
    """Decode the two-symbol operand (s0, s1) to an integer under `mode`."""
    d0, d1 = digit_map[s0], digit_map[s1]
    if mode == LITTLE_ENDIAN:
        return d1 * base + d0
    return d0 * base + d1


@dataclass(frozen=True)
class EqEval:
    """Decoded view of one equation under one mode, ready for candidate checks."""
    L: int                       # decoded left operand
    R: int                       # decoded right operand
    target: int                  # integer the op result must equal (magnitude)
    target_digits: tuple[int, ...]  # rhs digits in compare order (for display)
    res_len: int
    has_sign: bool


def evaluate_equation(eq: Equation, digit_map: dict[str, int], base: int, mode: str) -> EqEval:
    """Decode an equation's operands and result target under `mode`."""
    L = decode_operand(eq.l0, eq.l1, digit_map, base, mode)
    R = decode_operand(eq.r0, eq.r1, digit_map, base, mode)
    digits = [digit_map[s] for s in eq.rhs]
    if mode == LITTLE_ENDIAN:
        digits = list(reversed(digits))
    target = 0
    for d in digits:
        target = target * base + d
    return EqEval(
        L=L, R=R, target=target, target_digits=tuple(digits),
        res_len=eq.res_len, has_sign=eq.has_sign,
    )


@dataclass(frozen=True)
class MatchResult:
    """Outcome of testing one family+offset against one equation."""
    ok: bool
    value: int | None            # raw (possibly signed/None) op value
    value_digits: tuple[int, ...] | None  # base digits of |value|, MSB first
    reason: str                  # short reason, '' when ok


def concat_pattern(family: str, eq: Equation) -> tuple[str, ...]:
    """Symbol sequence a concat family must reproduce, in written order.

    concat_fwd → left operand symbols then right; concat_rev → right then left.
    Concatenation is purely structural (mode-independent): the result symbols
    are exactly the operand symbols, so digit injectivity makes this a symbol
    identity check.
    """
    if family == "concat_fwd":
        return (eq.l0, eq.l1, eq.r0, eq.r1)
    return (eq.r0, eq.r1, eq.l0, eq.l1)


def detect_pure_concat(examples: list[Equation], q: Equation) -> str | None:
    """Re-derive whether the question operator is genuinely a concat fast-path.

    Implements the corrected evidence rule (memory/pure_concat_rule.md):
      (a) question rhs has no sign and exactly 4 symbols.
      (b) at least one EXAMPLE equation (not to_predict) shares the question op.
      (c) every such example is unanimously satisfied by one concat direction.

    Returns "concat_fwd" or "concat_rev" when all three conditions hold, else None.
    Never reads solver_strategy — the label may be stale.
    """
    # (a)
    if q.has_sign or q.res_len != 4:
        return None
    # (b) evidence must come from example equations only
    relevant = [e for e in examples if e.op == q.op]
    if not relevant:
        return None
    # (c) fwd-first matches the solver's tie-break when operands are identical
    for fam in ("concat_fwd", "concat_rev"):
        if all(e.rhs == concat_pattern(fam, e) for e in relevant):
            return fam
    return None


def family_matches(
    family: str, offset: int, eq: Equation, digit_map: dict[str, int],
    base: int, mode: str,
) -> MatchResult:
    """Test whether `family` at `offset` satisfies equation `eq` under `mode`."""
    if family in ("concat_fwd", "concat_rev"):
        expected = concat_pattern(family, eq)
        ok = eq.res_len == 4 and all(
            digit_map[a] == digit_map[b] for a, b in zip(expected, eq.rhs)
        ) and len(eq.rhs) == len(expected)
        vdigits = tuple(digit_map[s] for s in expected)
        reason = "" if ok else "operand symbols do not match the result pattern"
        return MatchResult(ok, None, vdigits, reason)

    ev = evaluate_equation(eq, digit_map, base, mode)
    value = apply_family(family, offset, ev.L, ev.R, base)
    if value is None:
        return MatchResult(False, None, None, "guard failed (no result)")

    magnitude = abs(value)
    vdigits = tuple(int_to_base_digits(magnitude, base))

    if ev.has_sign:
        if value >= 0:
            return MatchResult(False, value, vdigits, "result is non-negative but RHS is signed")
        ok = magnitude == ev.target
    else:
        if value < 0:
            return MatchResult(False, value, vdigits, "result is negative but RHS is unsigned")
        ok = value == ev.target

    if ok:
        return MatchResult(True, value, vdigits, "")
    if len(vdigits) > ev.res_len:
        return MatchResult(
            False, value, vdigits,
            f"{len(vdigits)}-digit result vs {ev.res_len}-digit target",
        )
    return MatchResult(False, value, vdigits, "value != target")


def predict_answer(
    q: Equation, family: str, offset: int, digit_map: dict[str, int],
    base: int, mode: str,
) -> str:
    """Encode the op result for the to_predict equation `q` into puzzle symbols.

    Concat families copy operand symbols structurally; arithmetic families
    zero-pad |value| to the gold answer width, reverse for little_endian, map
    digits to symbols, and attach the sign affix when value < 0.
    """
    inv = {d: s for s, d in digit_map.items()}
    width = q.res_len
    if family in ("concat_fwd", "concat_rev"):
        return "".join(inv[digit_map[s]] for s in concat_pattern(family, q))

    L = decode_operand(q.l0, q.l1, digit_map, base, mode)
    R = decode_operand(q.r0, q.r1, digit_map, base, mode)
    value = apply_family(family, offset, L, R, base)
    digits = int_to_base_digits(abs(value), base)
    if len(digits) < width:
        digits = [0] * (width - len(digits)) + digits
    if mode == LITTLE_ENDIAN:
        digits = list(reversed(digits))
    body = "".join(inv[d] for d in digits)
    # neg_absdiff always carries a sign; signed ops only when value < 0.
    emit_sign = (family == "neg_absdiff") or (family in ("sub_signed", "rsub_signed") and value < 0)
    if not emit_sign:
        return body
    # Sign position is read from the gold answer format, independent of mode.
    return f"{body}{q.op}" if q.sign_pos == "postfix" else f"{q.op}{body}"


def render_concat_check_block(examples: list[Equation], q: Equation, fam: str | None) -> str:
    """Render the concat-check text for either outcome.

    fam=None → Outcome B: concat ruled out (arithmetic puzzle).
    fam=str  → Outcome A: concat passes (concat puzzle).
    One failing equation is sufficient to rule out concat — stops immediately.
    """
    relevant = [e for e in examples if e.op == q.op]
    lines = ["Let's first check if the operator simply concatenates operand symbols."]

    if not relevant:
        lines.append(
            f"No example equation uses operator `{q.op}`, "
            "so concatenation cannot be verified from examples."
        )
        if fam is None:
            lines.append(
                "Proceeding to arithmetic analysis: a digit mapping, an operator mapping, "
                "and a magnitude encoding strategy."
            )
        return "\n".join(lines)

    eq_labels = ", ".join(f"eq[{e.index}]" for e in relevant)
    lines.append(f"Equations sharing operator `{q.op}` with the question: {eq_labels}.\n")

    if fam is not None:
        # Outcome A: all relevant equations satisfy the detected direction.
        direction_label = "forward (l0 l1 r0 r1)" if fam == "concat_fwd" else "reverse (r0 r1 l0 l1)"
        for e in relevant:
            lines.append(f"- eq[{e.index}]: {e.raw} → {fam} ✓")
        lines.append(
            f"\nAll relevant equations satisfy {direction_label} concatenation. "
            "The rule applies directly — no digit mapping needed."
        )
    else:
        # Outcome B: find first failing equation and stop.
        all_passed = True
        for e in relevant:
            rhs_str = "".join(e.rhs)
            pat_fwd = "".join(concat_pattern("concat_fwd", e))
            pat_rev = "".join(concat_pattern("concat_rev", e))
            if rhs_str == pat_fwd:
                lines.append(f"- eq[{e.index}]: {e.raw} → concat_fwd ✓")
            elif rhs_str == pat_rev:
                lines.append(f"- eq[{e.index}]: {e.raw} → concat_rev ✓")
            else:
                lines.append(
                    f"- eq[{e.index}]: {e.raw} → rhs {rhs_str!r} is neither "
                    f"{pat_fwd!r} (fwd) nor {pat_rev!r} (rev). Concat fails. ✗"
                )
                all_passed = False
                break
        if all_passed:
            lines.append(
                "Example equations are consistent with concat, but the question "
                "format is incompatible (e.g. rhs length or sign mismatch)."
            )
        lines.append(
            "\nConcat does not apply. The puzzle requires a hidden transformation rule: "
            "a digit mapping, an operator mapping, and a magnitude encoding strategy."
        )

    return "\n".join(lines)


def verify_program(
    examples: list[Equation], op_family: dict[str, tuple[str, int]],
    digit_map: dict[str, int], base: int, mode: str,
) -> bool:
    """True iff every example equation is satisfied by the (op, digit) program.

    op_family maps each operator glyph to a (family_name, offset) pair.
    """
    for eq in examples:
        family, offset = op_family[eq.op]
        if not family_matches(family, offset, eq, digit_map, base, mode).ok:
            return False
    return True


def _selftest() -> None:
    from src.cot_generator_v2.equations import parse_equation_str

    dm = {"\\": 3, "'": 2, "!": 6, "`": 5, "[": 4, "{": 9, '"': 7, "@": 1, "&": 8}
    # eq3: \'-!` = \\  -> L=32, R=65 standard; absdiff(32,65)=33 == \\=33
    e3 = parse_equation_str("\\'-!` = \\\\", "example", 3)
    ev = evaluate_equation(e3, dm, 10, STANDARD)
    assert ev.L == 32 and ev.R == 65 and ev.target == 33
    assert family_matches("absdiff", 0, e3, dm, 10, STANDARD).ok
    assert family_matches("rsub_signed", 0, e3, dm, 10, STANDARD).ok    # 65-32=33
    assert not family_matches("mul", 0, e3, dm, 10, STANDARD).ok        # 2080, too long
    assert "4-digit" in family_matches("mul", 0, e3, dm, 10, STANDARD).reason

    # eq1: `!*[{ = '"[`  -> L=56, R=49 standard; mul+1 = 2745 == '"[` = 2,7,4,5
    e1 = parse_equation_str("`!*[{ = '\"[`", "example", 1)
    ev1 = evaluate_equation(e1, dm, 10, STANDARD)
    assert ev1.L == 56 and ev1.R == 49 and ev1.target == 2745
    assert family_matches("mul", 1, e1, dm, 10, STANDARD).ok
    assert not family_matches("mul", 0, e1, dm, 10, STANDARD).ok

    # concat is a structural symbol-pattern match (mode-independent).
    ec = parse_equation_str("[!*`| = [!`|", "example", 1)
    dmc = {"[": 4, "!": 5, "`": 0, "|": 8}
    assert family_matches("concat_fwd", 0, ec, dmc, 10, LITTLE_ENDIAN).ok
    assert not family_matches("concat_rev", 0, ec, dmc, 10, STANDARD).ok

    # little_endian decode swaps operand digits
    ev_le = evaluate_equation(e3, dm, 10, LITTLE_ENDIAN)
    assert ev_le.L == 23 and ev_le.R == 56

    # --- detect_pure_concat ---
    from src.cot_generator_v2.equations import parse_prompt

    # Forward concat: examples A B # C D = A B C D, question also #
    prompt_fwd = (
        "AB#CD = ABCD\nEF#GH = EFGH\nNow, determine the result for: IJ#KL"
    )
    ex_fwd, q_fwd = parse_prompt(prompt_fwd, "IJKL")
    assert detect_pure_concat(ex_fwd, q_fwd) == "concat_fwd", "fwd expected"

    # Reverse concat: examples A B # C D = C D A B, question also #
    prompt_rev = (
        "AB#CD = CDAB\nEF#GH = GHEF\nNow, determine the result for: IJ#KL"
    )
    ex_rev, q_rev = parse_prompt(prompt_rev, "KLIJ")
    assert detect_pure_concat(ex_rev, q_rev) == "concat_rev", "rev expected"

    # No matching examples for question operator -> None
    prompt_no_ex = "AB*CD = ABCD\nNow, determine the result for: IJ#KL"
    ex_no, q_no = parse_prompt(prompt_no_ex, "IJKL")
    assert detect_pure_concat(ex_no, q_no) is None, "no matching example should be None"

    # Question rhs has a sign -> None (signed rhs indicates arithmetic, not concat)
    prompt_signed = "AB#CD = ABCD\nNow, determine the result for: IJ#KL"
    ex_sig, q_sig = parse_prompt(prompt_signed, "IJKL")
    # Patch has_sign to simulate signed rhs
    import dataclasses
    q_sig_signed = dataclasses.replace(q_sig, has_sign=True)
    assert detect_pure_concat(ex_sig, q_sig_signed) is None, "signed rhs should be None"

    # Question rhs not 4 symbols -> None
    prompt_short = "AB#CD = ABCD\nNow, determine the result for: IJ#KL"
    ex_sh, q_sh = parse_prompt(prompt_short, "IJKL")
    q_sh_short = dataclasses.replace(q_sh, rhs=("I", "J"))
    assert detect_pure_concat(ex_sh, q_sh_short) is None, "non-4-symbol rhs should be None"

    print("decode self-test PASS")


if __name__ == "__main__":
    _selftest()
