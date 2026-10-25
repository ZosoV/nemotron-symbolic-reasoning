"""Family-level structural feasibility filter (relocated from
src/heuristic_constraints/operation_feasibility.py and rewritten).

Given an operator glyph's group of encrypted equations, decide which op
*families* (and which offsets within a family) are structurally possible before
any digit value is known. Unlike the original — which worked on per-offset
registry names and only checked a lower magnitude bound — this version:

  * reasons about families with offset *sets* (offsets are narrowed, not whole
    instances dropped);
  * adds an **upper-bound digit-count check** (a family whose maximum result has
    fewer digits than the RHS field can never fill it — e.g. add/absdiff cannot
    produce a 4-digit result), reproducing the template's "add/absdiff can't make
    a 4-digit rhs" observations;
  * folds in the solver's group-level sign heuristic (src/heuristics.py
    sign_pattern_filter, #2) and concat opt-in (#4);
  * emits natural-language observations phrased like the CoT template's
    <factual_observations_operator_N> block.

An op is feasible for the operator iff it is feasible in **every** equation of
the group (one revealing equation rules a family out).

Run (smoke tests):
    uv run python src/cot_generator/feasibility.py --smoke
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.decode import concat_pattern
from src.cot_generator.equations import Equation
from src.cot_generator.families import (
    FAMILY_BY_NAME,
    FAMILY_NAMES,
    NEGATIVE_FAMILIES,
    max_result_digits,
    min_result_magnitude,
)

# Families that ALWAYS produce 0 when L == R (operands forced equal), at their
# only offset. absdiff is handled separately because only its offset-0 variant
# yields 0 (absdiff_p1 -> 1, etc.).
ZERO_ON_EQUAL: frozenset[str] = frozenset({
    "sub_signed", "rsub_signed", "neg_absdiff", "mod", "rmod",
})
# fdiv(x, x) == 1 when x != 0.
ONE_ON_EQUAL: frozenset[str] = frozenset({"fdiv"})


@dataclass
class Observation:
    """One factual-observation bullet plus the families it removes."""
    text: str
    dropped: tuple[str, ...] = ()
    # Offsets removed from still-surviving families: {family: (offsets...)}.
    narrowed: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # 'sign' | 'concat' | 'digit' | 'self' | 'redundancy' — lets the renderer
    # compress repeated sign/concat reasoning across operators.
    category: str = ""


@dataclass
class OperatorFeasibility:
    """Result of filtering one operator group."""
    op: str
    survivors: dict[str, tuple[int, ...]]   # family -> surviving offsets
    observations: list[Observation]
    discarded: dict[str, str]               # family -> aggregated reason


def _fmt_list(names) -> str:
    return ", ".join(names)


def _all_same(syms: tuple[str, ...]) -> bool:
    """True when every symbol is identical (vacuously true for 0/1 symbols)."""
    return len(set(syms)) <= 1


def _can_encode_one(rhs: tuple[str, ...]) -> bool:
    """Whether some encoding of `rhs` can represent the integer 1.

    Value 1 fills an L-symbol field as `0…0 1`: the L-1 leading positions are the
    single zero-symbol (so all identical) and the unit position is a *different*
    symbol (1 != 0, distinct digits => distinct symbols). Checked in both encodings
    (standard: MSB is rhs[0]; little_endian: MSB is rhs[-1]).
    """
    L = len(rhs)
    if L == 0:
        return False
    if L == 1:
        return True
    std = _all_same(rhs[:L - 1]) and rhs[L - 1] != rhs[0]
    le = _all_same(rhs[1:]) and rhs[0] != rhs[1]
    return std or le


def _can_encode_operand(rhs: tuple[str, ...], l0: str, l1: str) -> bool:
    """Whether some encoding of `rhs` can equal the operand value `l0 l1`.

    The value fills an L-symbol field as `0…0 l0 l1`: the leading positions share
    one (zero) symbol, then the operand's two symbols in order. By injectivity the
    result symbol at each significant position must be the operand symbol itself.
    """
    L = len(rhs)
    if L == 1:
        return rhs[0] == l1  # only realizable when the operand is a single digit
    std = rhs[L - 2] == l0 and rhs[L - 1] == l1 and _all_same(rhs[:L - 2])
    le = rhs[1] == l0 and rhs[0] == l1 and _all_same(rhs[2:])
    return std or le


def feasible_for_operator(op: str, eqs: list[Equation], base: int) -> OperatorFeasibility:
    """Structural feasibility for one operator glyph's equation group."""
    survivors: dict[str, list[int]] = {
        name: list(FAMILY_BY_NAME[name].offsets) for name in FAMILY_NAMES
    }
    observations: list[Observation] = []
    discarded: dict[str, str] = {}

    # Unconstrained operator: appears only in the question, no example equation.
    if not eqs:
        observations.append(Observation(
            f"No example equation contains `{op}`, so it cannot be narrowed from the "
            "examples; every operation remains possible and the choice is settled only by "
            "the question itself.",
        ))
        return OperatorFeasibility(
            op=op, survivors={n: tuple(o) for n, o in survivors.items()},
            observations=observations, discarded=discarded,
        )

    def drop(name: str, reason: str) -> None:
        if name in survivors:
            del survivors[name]
            discarded.setdefault(name, reason)

    signs = [eq.has_sign for eq in eqs]
    all_signed = all(signs)
    all_unsigned = not any(signs)

    # --- Sign check (folds in sign_pattern_filter, #2) ---------------------
    if all_signed:
        dropped = [n for n in list(survivors) if n not in NEGATIVE_FAMILIES]
        for n in dropped:
            drop(n, "RHS is signed (negative result); this family is always non-negative")
        observations.append(Observation(
            "The result carries a sign on the rhs (it is negative). Only families that "
            "can produce a negative value remain: sub_signed, rsub_signed, neg_absdiff. "
            f"All others are infeasible: {_fmt_list(dropped)}.",
            tuple(dropped), category="sign",
        ))
    elif any(signs):  # mixed
        keep = {"sub_signed", "rsub_signed"}
        dropped = [n for n in list(survivors) if n not in keep]
        for n in dropped:
            drop(n, "RHS sign is mixed across equations; only signed subtraction can do both")
        observations.append(Observation(
            "Some equations have a signed rhs and others do not. Only sub_signed and "
            "rsub_signed can yield both positive and negative results, so every other "
            f"family is infeasible: {_fmt_list(dropped)}.",
            tuple(dropped), category="sign",
        ))
    else:  # all unsigned
        # neg_absdiff is always <= 0; its only non-negative value (0) needs L == R.
        if not all(eq.self_equal() for eq in eqs):
            drop("neg_absdiff", "RHS is unsigned but neg_absdiff is always <= 0 (L != R here)")
            observations.append(Observation(
                f"The operator symbol `{op}` is not present on the rhs, so the result is "
                "non-negative. neg_absdiff always produces a negative result and is "
                "infeasible.",
                ("neg_absdiff",), category="sign",
            ))

    # --- Concat structural check (folds in concat_optin, #4) ---------------
    # Concatenation is purely structural: in the symbol domain the result must be
    # exactly the operand symbols in order (fwd: l0 l1 r0 r1; rev: r0 r1 l0 l1).
    # Injective digits make this a symbol-identity check, independent of values.
    # Each direction is judged independently across the whole operator group, so
    # an unambiguous equation pins the orientation; only identical operands (the
    # two patterns coincide) leave both directions feasible.
    concat_present = [n for n in ("concat_fwd", "concat_rev") if n in survivors]
    if concat_present:
        kept: list[str] = []
        dropped_fams: list[str] = []
        # (fam, want, bad_eq) — want/bad_eq are None for sign-based drops
        dropped_info: list[tuple[str, str | None, object]] = []
        for fam in concat_present:
            if not all_unsigned:
                drop(fam, "a concat result is always unsigned, but some rhs carries a sign")
                dropped_fams.append(fam)
                dropped_info.append((fam, None, None))
                continue
            bad = next((eq for eq in eqs if eq.rhs != concat_pattern(fam, eq)), None)
            if bad is None:
                kept.append(fam)
            else:
                want = "".join(concat_pattern(fam, bad))
                got = "".join(bad.rhs)
                drop(fam, f"{fam} needs rhs `{want}` at eq[{bad.index}], but it is `{got}`")
                dropped_fams.append(fam)
                dropped_info.append((fam, want, bad))
        if dropped_fams:
            if kept:
                di = dropped_info[0]
                if di[2] is not None:
                    fam_d, want_d, bad_d = di
                    text = (
                        "Concatenation must reproduce the operand symbols verbatim in the rhs. "
                        f"{', '.join(kept)} matches and stays feasible, but {fam_d} would "
                        f"need rhs `{want_d}` (eq[{bad_d.index}] has `{''.join(bad_d.rhs)}`), "
                        f"so it is infeasible."
                    )
                else:
                    text = (
                        "Concatenation must reproduce the operand symbols verbatim in the rhs. "
                        f"{', '.join(kept)} matches and stays feasible, while {dropped_fams[0]} is "
                        "infeasible (a concat result is always unsigned, but some rhs is signed)."
                    )
            else:
                struct = [(fam, want, bad) for fam, want, bad in dropped_info if bad is not None]
                sign_drops = [fam for fam, want, bad in dropped_info if bad is None]
                if len(struct) == 2 and not sign_drops and struct[0][2].index == struct[1][2].index:
                    # Both fail on the same equation — factor out the rhs.
                    fam0, want0, bad0 = struct[0]
                    fam1, want1, _ = struct[1]
                    text = (
                        "Concatenation must reproduce the operand symbols verbatim in the rhs, "
                        f"but neither direction matches eq[{bad0.index}] (`{''.join(bad0.rhs)}`): "
                        f"{fam0} needs `{want0}`, {fam1} needs `{want1}`. "
                        "concat_fwd and concat_rev are infeasible."
                    )
                else:
                    # Different failing equations or sign-based — list each separately.
                    parts: list[str] = []
                    for fam, want, bad in dropped_info:
                        if bad is None:
                            parts.append(f"{fam} (always unsigned, but some rhs is signed)")
                        else:
                            parts.append(
                                f"{fam} would need rhs `{want}` "
                                f"(eq[{bad.index}] has `{''.join(bad.rhs)}`)"
                            )
                    text = (
                        "Concatenation must reproduce the operand symbols verbatim in the rhs, "
                        f"but neither direction matches: {'; '.join(parts)}. "
                        "concat_fwd and concat_rev are infeasible."
                    )
            observations.append(Observation(text, tuple(dropped_fams), category="concat"))
        elif len(kept) == 2:
            observations.append(Observation(
                "The operands are identical in every equation, so forward and reverse "
                "concatenation coincide; concat_fwd and concat_rev both remain feasible "
                "(the direction is ambiguous).",
                (), category="concat",
            ))

    # --- Magnitude lower-bound check (per equation) ------------------------
    # Sound elimination only: a family is infeasible if the smallest magnitude
    # it can ever produce already overflows the rhs capacity. (An *upper* bound
    # on digit count is NOT sound here: results may carry leading zeros, so a
    # small value can still fill a wide rhs field, e.g. 71 -> "0071".)
    for eq in eqs:
        rhs_cap = base ** eq.res_len - 1
        too_big = [
            name for name in list(survivors)
            if min_result_magnitude(name, tuple(survivors[name]), base) > rhs_cap
        ]
        for n in too_big:
            drop(n, f"min result magnitude exceeds the {eq.res_len}-digit rhs capacity")
        if too_big:
            observations.append(Observation(
                f"eq[{eq.index}]: {_fmt_list(too_big)} each have a smallest possible result "
                f"that already exceeds the {eq.res_len}-digit rhs capacity, so "
                f"{'it is' if len(too_big) == 1 else 'they are'} infeasible.",
                tuple(too_big), category="digit",
            ))

    # --- Digit-length upper-bound check (per equation) ---------------------
    # The *upper* digit bound IS a sound elimination once combined with digit
    # injectivity: a family whose result has at most D significant digits can
    # only fill a wider L-symbol rhs by padding L-D leading zeros, and — since
    # exactly one symbol maps to 0 — those leading positions must be the *same*
    # symbol. We stay mode-agnostic: drop only when neither encoding can supply
    # an equal-symbol leading run (standard: rhs[:k]; little_endian: rhs[L-k:]).
    # This reframes the solver's magnitude pre-prune (heuristic #6) upper half;
    # the lower half is the magnitude check above.
    for eq in eqs:
        L = eq.res_len
        bad: list[str] = []
        for name in list(survivors):
            k = L - max_result_digits(name, tuple(survivors[name]), base)
            if k <= 0:
                continue
            if not _all_same(eq.rhs[:k]) and not _all_same(eq.rhs[L - k:]):
                bad.append(name)
        for n in bad:
            drop(n, f"result is too short for the {L}-symbol rhs; the required leading "
                    "zeros would need one repeated zero-symbol, which eq lacks")
        if bad:
            observations.append(Observation(
                f"{_fmt_list(bad)} always produce a result smaller than eq[{eq.index}]'s "
                f"{L}-symbol rhs; filling it would need leading zeros — and with one symbol "
                "per digit those zeros must be a single repeated symbol, but its leading "
                "symbols differ in either encoding. Infeasible.",
                tuple(bad), category="digit",
            ))

    # --- Self-application check (L == R forces a constant) -----------------
    for eq in eqs:
        if not eq.self_equal():
            continue
        distinct_rhs = len({*eq.rhs})
        forced0_invalid = eq.has_sign or distinct_rhs > 1  # 0 cannot match the rhs

        # absdiff: |x-x|+o = o. Drop offsets that go negative (guard fails) and
        # the offset-0 variant when a 0 result cannot match the rhs. Keep o > 0.
        if "absdiff" in survivors:
            removed = tuple(
                o for o in survivors["absdiff"]
                if o < 0 or (o == 0 and forced0_invalid)
            )
            if removed:
                kept = [o for o in survivors["absdiff"] if o not in removed]
                if kept:
                    survivors["absdiff"] = kept
                    observations.append(Observation(
                        f"eq[{eq.index}] applies `{op}` to equal operands (|x-x| = 0), so "
                        f"absdiff offsets {list(removed)} cannot match the rhs and are removed.",
                        (), {"absdiff": removed}, category="self",
                    ))
                else:
                    drop("absdiff", f"L == R: every surviving absdiff offset is ruled out at eq[{eq.index}]")
                    observations.append(Observation(
                        f"eq[{eq.index}] applies `{op}` to equal operands; no remaining "
                        "absdiff offset can match the rhs, so absdiff is infeasible.",
                        ("absdiff",), category="self",
                    ))

        # Pure zero-on-equal families: rhs would have to be a single zero digit.
        if forced0_invalid:
            forced0 = [n for n in list(survivors) if n in ZERO_ON_EQUAL]
            for n in forced0:
                drop(n, f"L == R forces a 0 result, but eq[{eq.index}] rhs cannot be 0")
            if forced0:
                observations.append(Observation(
                    f"eq[{eq.index}] applies `{op}` to equal operands, forcing a 0 result, "
                    f"which cannot match its rhs. Infeasible: {_fmt_list(forced0)}.",
                    tuple(forced0), category="self",
                ))

        # fdiv(x, x) == 1 (or guard-fails on x == 0); either way it can only match
        # by equalling 1. Drop it if the rhs cannot encode 1 in either encoding.
        for n in [f for f in ONE_ON_EQUAL if f in survivors]:
            if not _can_encode_one(eq.rhs):
                drop(n, f"L == R forces fdiv to 1, but eq[{eq.index}] rhs cannot encode 1")
                observations.append(Observation(
                    f"eq[{eq.index}] applies `{op}` to equal operands, so fdiv would have to "
                    "equal 1; its rhs cannot encode 1 in either encoding, so fdiv is infeasible.",
                    (n,), category="self",
                ))

        # gcd(x, x) == lcm(x, x) == x. Drop either family whose rhs cannot equal
        # the operand value `l0 l1` in either encoding.
        gl = [n for n in ("gcd", "lcm")
              if n in survivors and not _can_encode_operand(eq.rhs, eq.l0, eq.l1)]
        for n in gl:
            drop(n, f"L == R forces {n} to the operand value, but eq[{eq.index}] rhs cannot encode it")
        if gl:
            verb = "is" if len(gl) == 1 else "are"
            pronoun = "it" if len(gl) == 1 else "they"
            observations.append(Observation(
                f"eq[{eq.index}] applies `{op}` to equal operands, so {' and '.join(gl)} would "
                f"have to reproduce the operand `{eq.l0}{eq.l1}` in the rhs, which {pronoun} cannot; "
                f"{_fmt_list(gl)} {verb} infeasible.",
                tuple(gl), category="self",
            ))

    return OperatorFeasibility(
        op=op,
        survivors={n: tuple(offs) for n, offs in survivors.items()},
        observations=observations,
        discarded=discarded,
    )


# ---------------------------------------------------------------------------
# Smoke tests (family versions of the relocated module's canonical cases)
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    from src.cot_generator.equations import parse_equation_str

    # Case 1: unsigned, 2-digit rhs, L != R  -> neg_absdiff & concat dropped.
    e = parse_equation_str("`#-># = !>", "example", 1)
    r = feasible_for_operator("-", [e], 10)
    assert "neg_absdiff" not in r.survivors, "neg_absdiff must drop (unsigned, L!=R)"
    assert "concat_fwd" not in r.survivors and "concat_rev" not in r.survivors, "concat must drop (res_len 2)"
    assert "mul" in r.survivors and "add" in r.survivors and "sub_signed" in r.survivors
    print("Case 1 PASS  unsigned short rhs")

    # Case 2: signed rhs -> only sub_signed/rsub_signed/neg_absdiff survive.
    e2 = parse_equation_str("AB-CD = -EF", "example", 1)
    r2 = feasible_for_operator("-", [e2], 10)
    assert set(r2.survivors) <= {"sub_signed", "rsub_signed", "neg_absdiff"}, r2.survivors
    assert "mul" not in r2.survivors and "absdiff" not in r2.survivors
    print("Case 2 PASS  signed rhs")

    # Case 3: 4-digit unsigned rhs whose symbols do NOT match either concat
    # pattern (rhs '"[` != operands `![{ fwd or [{`! rev) -> both concat families
    # drop; neg_absdiff dropped (unsigned). Rule 1 (digit-length): gcd/fdiv/mod/
    # rmod top out at 2 digits, so a 4-symbol rhs needs two leading zeros = one
    # repeated zero-symbol; rhs `'"[`` has distinct leading pairs in BOTH encodings
    # (`'`!=`"` and `[`!=`` ` ``), so all four drop. lcm (4 digits) and add/mul stay.
    e3 = parse_equation_str("`!*[{ = '\"[`", "example", 1)
    r3 = feasible_for_operator("*", [e3], 10)
    assert "neg_absdiff" not in r3.survivors
    assert "concat_fwd" not in r3.survivors and "concat_rev" not in r3.survivors, r3.survivors
    assert {"gcd", "fdiv", "mod", "rmod"}.isdisjoint(r3.survivors), r3.survivors
    assert {"mul", "lcm", "add"} <= set(r3.survivors), r3.survivors
    assert r3.survivors["mul"] == (0, -1, 1)
    print("Case 3 PASS  4-digit rhs -> concat dropped + Rule 1 drops gcd/fdiv/mod/rmod")

    # Case 3d: same shape but the standard leading pair is equal (`}}<>`), so the
    # standard encoding CAN place the two leading zeros -> gcd/mod/... are kept.
    e3d = parse_equation_str("`!*[{ = }}<>", "example", 1)
    r3d = feasible_for_operator("*", [e3d], 10)
    assert {"gcd", "fdiv", "mod", "rmod"} <= set(r3d.survivors), r3d.survivors
    print("Case 3d PASS  4-digit rhs with equal leading pair -> Rule 1 keeps the five ops")

    # Case 3b: rhs equals the forward concat of the operand symbols -> concat_fwd
    # stays, concat_rev (which would need `|[!) drops. Direction is resolved.
    e3b = parse_equation_str("[!*`| = [!`|", "example", 1)
    r3b = feasible_for_operator("*", [e3b], 10)
    assert "concat_fwd" in r3b.survivors and "concat_rev" not in r3b.survivors, r3b.survivors
    print("Case 3b PASS  forward concat matches -> concat_fwd kept, concat_rev dropped")

    # Case 3c: identical operands -> the two concat patterns coincide, so the
    # direction is ambiguous and both concat families remain.
    e3c = parse_equation_str("[!*[! = [![!", "example", 1)
    r3c = feasible_for_operator("*", [e3c], 10)
    assert "concat_fwd" in r3c.survivors and "concat_rev" in r3c.survivors, r3c.survivors
    print("Case 3c PASS  identical operands -> both concat families kept (ambiguous)")

    # Case 4: L == R with distinct rhs digits. Zero-forced families drop;
    # absdiff keeps only offset +1 (|x-x|+1 = 1, e.g. rhs "01"), not offset 0.
    e4 = parse_equation_str("AA-AA = BC", "example", 1)
    r4 = feasible_for_operator("-", [e4], 10)
    assert r4.survivors.get("absdiff") == (1,), r4.survivors.get("absdiff")
    assert "sub_signed" not in r4.survivors and "rsub_signed" not in r4.survivors
    assert "mod" not in r4.survivors and "rmod" not in r4.survivors
    print("Case 4 PASS  L==R distinct rhs (absdiff keeps offset +1)")

    # Case 5: L == R with a *repeated* rhs symbol (`BB`, could be 00). Rule 2:
    #   fdiv(x,x)=1 needs a `0…0 1` rhs (two distinct symbols) -> `BB` cannot ->
    #   fdiv drops; gcd(x,x)=lcm(x,x)=x needs the rhs to reproduce operand `AA` ->
    #   `BB` (B != A) cannot -> gcd, lcm drop. mod/rmod survive (BB may be 00, so a
    #   0 result is still possible — forced0_invalid is False).
    e5 = parse_equation_str("AA*AA = BB", "example", 1)
    r5 = feasible_for_operator("*", [e5], 10)
    assert {"fdiv", "gcd", "lcm"}.isdisjoint(r5.survivors), r5.survivors
    assert "mod" in r5.survivors and "rmod" in r5.survivors, r5.survivors
    print("Case 5 PASS  L==R repeated rhs -> Rule 2 drops fdiv/gcd/lcm, keeps mod/rmod")

    # Case 5b: L == R where the rhs equals the operand (`AA`). gcd/lcm = x = AA
    # match -> kept; fdiv = 1 != AA -> dropped.
    e5b = parse_equation_str("AA*AA = AA", "example", 1)
    r5b = feasible_for_operator("*", [e5b], 10)
    assert "gcd" in r5b.survivors and "lcm" in r5b.survivors, r5b.survivors
    assert "fdiv" not in r5b.survivors, r5b.survivors
    print("Case 5b PASS  L==R rhs==operand -> gcd/lcm kept, fdiv dropped")

    print("\nAll feasibility smoke tests passed.")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _run_smoke_tests()
    else:
        _run_smoke_tests()
