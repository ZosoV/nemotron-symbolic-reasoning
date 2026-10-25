"""Per-equation structural refutations for operator family candidates.

Three digit-free rules that extend feasibility.py's checks:

  Rule B — zero-scarcity (GE_MAX): add / mul / lcm output >= max(operand);
            if the result field is shorter, excess leading symbols of both
            operands must all be 0, but only one digit can be 0 -> infeasible
            when >= 2 distinct symbols need to vanish.

  Rule C — result == operand identity: when the result token equals an
            operand token, certain ops are structurally impossible:
              mod:  a % b < b always  -> can never equal b (= result)
              rmod: b % a < a always  -> can never equal a (= result)
              add:  a + b = b forces a = 0, impossible if a has >= 2 symbols.
            Rule C is mode-agnostic (token equality is reading-independent).

  Rule D — fdiv squaring + shared-symbol: when result == right operand and
            both operands have the same length, fdiv needs B^2 <= A < B(B+1).
            For 2-digit operands this forces B's most-significant symbol to 0.
            If B's value-carrying symbol is also A's most-significant symbol
            then A >= base*B >= B(B+1), a contradiction. Encoding-specific.

Public entry:
    structural_drops(op, eqs, base, candidates) -> dict[str, dict[str, str]]
    Returns: family -> {mode -> reason} for each (family, mode) proven
    infeasible. A family absent from a mode's inner dict is still feasible.

Run smoke tests:
    uv run python src/cot_generator_v2/per_eq_structural_checks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.decode import LITTLE_ENDIAN, MODES, STANDARD
from src.cot_generator_v2.equations import Equation, parse_equation_str

# Families whose base output is >= max(a, b) when both operands >= 1.
_GE_MAX: frozenset[str] = frozenset({"add", "mul", "lcm"})

# All families that any rule here may drop.
_ALL_RULE_TARGETS: frozenset[str] = _GE_MAX | {"mod", "rmod", "fdiv"}


# ---------------------------------------------------------------------------
# Shape-fact helpers (digit-free)
# ---------------------------------------------------------------------------

def _repunit(n: int) -> int:
    """Integer with n repeated 1s in base 10: _repunit(3) == 111."""
    return int("1" * n) if n > 0 else 0


def _can_be_zero(tok: tuple[str, ...]) -> bool:
    """True only for all-same-symbol tokens — the only way a multi-digit
    symbolic number can represent 0 under a bijective digit map."""
    return len(set(tok)) <= 1


def _tok_lo(tok: tuple[str, ...]) -> int:
    """Structural lower bound on a token's integer value (digit-free)."""
    return 0 if _can_be_zero(tok) else 1


def _tok_hi(tok: tuple[str, ...], base: int) -> int:
    """Structural upper bound on a token's integer value (digit-free)."""
    n = len(tok)
    if _can_be_zero(tok):
        return (base - 1) * _repunit(n)
    return base ** n - 1


def _leading_excess(tok: tuple[str, ...], res_len: int, mode: str) -> set[str]:
    """Distinct symbols occupying the high-order positions of *tok* beyond the
    first *res_len* significant digits — these must all map to 0 for the
    token's value to fit in a *res_len*-digit result field.

    standard:       high-order symbols are at the LEFT  (tok[:extra])
    little_endian:  high-order symbols are at the RIGHT (tok[len-extra:])
    """
    extra = len(tok) - res_len
    if extra <= 0:
        return set()
    if mode == STANDARD:
        return set(tok[:extra])
    return set(tok[len(tok) - extra:])


def _msd(tok: tuple[str, ...], mode: str) -> str:
    """Most-significant-digit symbol of *tok* under *mode*."""
    return tok[0] if mode == STANDARD else tok[-1]


# ---------------------------------------------------------------------------
# Rule B — zero-scarcity for GE_MAX families
# ---------------------------------------------------------------------------

def _rule_b(
    eq: Equation,
    mode: str,
    live: set[str],
) -> dict[str, str]:
    """Return {family: reason} for GE_MAX families refuted in this eq+mode."""
    at = (eq.l0, eq.l1)
    bt = (eq.r0, eq.r1)
    rt = eq.rhs
    res_len = len(rt)

    # Both operands must be >= 1 for the GE_MAX property to hold.
    if _tok_lo(at) < 1 or _tok_lo(bt) < 1:
        return {}

    need_zero: set[str] = (
        _leading_excess(at, res_len, mode)
        | _leading_excess(bt, res_len, mode)
    )
    if len(need_zero) < 2:
        return {}

    syms_str = "{" + ", ".join(f"`{s}`" for s in sorted(need_zero)) + "}"
    dropped: dict[str, str] = {}
    _labels = {
        "add": "the sum a + b (plus any offset)",
        "mul": "the product a × b (plus any offset)",
        "lcm": "lcm(a, b)",
    }
    for fam in _GE_MAX:
        if fam not in live:
            continue
        dropped[fam] = (
            f"{_labels[fam]} ≥ max(a, b) when both operands ≥ 1; fitting in "
            f"{res_len} symbols forces {syms_str} ({len(need_zero)} distinct) "
            f"to all map to 0, but only one digit can be 0 -> `{fam}` is infeasible."
        )
    return dropped


# ---------------------------------------------------------------------------
# Rule C — result == operand identity  (mode-agnostic)
# ---------------------------------------------------------------------------

def _rule_c(
    eq: Equation,
    live: set[str],
) -> dict[str, str]:
    """Return {family: reason} for families refuted by result-equals-operand.

    Mode-agnostic: symbolic string equality is independent of read direction.
    """
    at = (eq.l0, eq.l1)
    bt = (eq.r0, eq.r1)
    rt = eq.rhs

    dropped: dict[str, str] = {}

    if rt == bt:
        # op(A, B) = B required.
        if "mod" in live:
            dropped["mod"] = (
                f"Result `{''.join(rt)}` equals rhs operand `{''.join(bt)}`, so `mod` "
                f"would need a % b = b; but a % b < b always (or b=0, undefined) "
                f"-> `mod` is infeasible."
            )
        if "add" in live and not _can_be_zero(at):
            dropped["add"] = (
                f"Result `{''.join(rt)}` equals rhs operand `{''.join(bt)}`, so `add` "
                f"would need a = 0; but `{''.join(at)}` uses {len(set(at))} distinct "
                f"symbols -> `add` is infeasible."
            )

    if rt == at:
        # op(A, B) = A required.
        if "rmod" in live:
            dropped["rmod"] = (
                f"Result `{''.join(rt)}` equals lhs operand `{''.join(at)}`, so `rmod` "
                f"would need b % a = a; but b % a < a always (or a=0, undefined) "
                f"-> `rmod` is infeasible."
            )
        if "add" in live and not _can_be_zero(bt) and "add" not in dropped:
            dropped["add"] = (
                f"Result `{''.join(rt)}` equals lhs operand `{''.join(at)}`, so `add` "
                f"would need b = 0; but `{''.join(bt)}` uses {len(set(bt))} distinct "
                f"symbols -> `add` is infeasible."
            )

    return dropped


# ---------------------------------------------------------------------------
# Rule D — fdiv squaring + shared-symbol  (encoding-specific)
# ---------------------------------------------------------------------------

def _rule_d(
    eq: Equation,
    mode: str,
    live: set[str],
    base: int,
) -> dict[str, str]:
    """Return {family: reason} for fdiv refuted by this eq+mode."""
    if "fdiv" not in live:
        return {}

    at = (eq.l0, eq.l1)
    bt = (eq.r0, eq.r1)
    rt = eq.rhs

    # Applies only when result == right operand.
    if rt != bt:
        return {}

    # Same-length operands make the squaring bound tight.
    if len(at) != len(bt):
        return {}

    a_hi = _tok_hi(at, base)
    max_b = math.isqrt(a_hi)          # largest B s.t. B² <= max(A)
    min_b_full = base ** (len(bt) - 1) # smallest B with non-zero leading digit

    # If a full-magnitude B already satisfies B² <= a_hi, the MSD need not be 0
    # and the squaring + shared-symbol argument does not apply.
    if min_b_full <= max_b:
        return {}

    # B's MSD symbol is forced to 0; the next symbol carries all of B's value.
    b_msd = _msd(bt, mode)
    # "next" symbol after MSD (value-carrier in the forced-zero-leading case):
    # standard: MSD=bt[0], value-carrier=bt[1]; little_endian: MSD=bt[-1], value-carrier=bt[-2]
    if mode == STANDARD:
        b_value_sym = bt[1] if len(bt) >= 2 else None
    else:
        b_value_sym = bt[-2] if len(bt) >= 2 else None

    if b_value_sym is None:
        return {}

    a_msd = _msd(at, mode)

    # If B's value-carrier is also A's MSD:
    #   A = base * d[b_value_sym] + ...  >= base * B  (since MSD of A = d[b_value_sym] = B)
    #   Need A < B(B+1) <= base*B  (for B <= base-1).  Contradiction.
    if b_value_sym != a_msd:
        return {}

    # Two sub-cases depending on whether b_msd and b_value_sym are the same symbol.
    if b_msd == b_value_sym:
        # All significant positions of bt share one symbol, so B is always 0
        # (the MSD is forced to 0 and the value-carrier is the same symbol).
        # fdiv(A, 0) is undefined — the equation always fails.
        return {
            "fdiv": (
                f"Result `{''.join(rt)}` equals rhs operand `{''.join(bt)}`, so fdiv "
                f"needs ⌊A/B⌋ = B; max A = {a_hi} forces `{b_msd}` = 0, but that "
                f"makes B = 0 (undefined) -> fdiv is infeasible."
            )
        }

    # b_msd != b_value_sym: they are distinct symbols.  The bijective digit mapping
    # forces d[b_msd] = 0 and d[b_value_sym] != 0, so B >= 1 (the guard never fires).
    # The squaring argument then closes the refutation on its own.
    return {
        "fdiv": (
            f"Result `{''.join(rt)}` equals rhs operand `{''.join(bt)}`, so fdiv "
            f"needs B² ≤ A < B(B+1); max A = {a_hi} forces `{b_msd}` = 0, making "
            f"`{b_value_sym}` carry B, but `{b_value_sym}` is also A's MSD so "
            f"A ≥ {base}·B ≥ B(B+1) — contradiction -> fdiv is infeasible."
        )
    }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def structural_drops(
    op: str,
    eqs: list[Equation],
    base: int = 10,
    candidates: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Compute per-equation structural refutations across both encoding modes.

    Parameters
    ----------
    op         : operator glyph (unused by the rules themselves; kept for API symmetry)
    eqs        : example equations for this operator
    base       : numeral base (default 10)
    candidates : families to check; defaults to all rule-relevant families.
                 Pass the current survivor set from feasibility to avoid
                 emitting observations for already-eliminated families.

    Returns
    -------
    dict[family, dict[mode, reason]]
        Inner dict contains only the modes for which the family is refuted.
        A family absent from the return value (or absent from a mode's inner
        dict) is still feasible for that mode.
    """
    if candidates is None:
        candidates = set(_ALL_RULE_TARGETS)

    # drops[family][mode] = reason (first refuting equation wins per mode).
    drops: dict[str, dict[str, str]] = {}

    def _already_dropped_in(fam: str, mode: str) -> bool:
        return mode in drops.get(fam, {})

    for eq in eqs:
        # --- Rule C: mode-agnostic — apply once, covers both modes. ---
        c_live = {f for f in candidates if not (
            _already_dropped_in(f, STANDARD) and _already_dropped_in(f, LITTLE_ENDIAN)
        )}
        for fam, reason in _rule_c(eq, c_live).items():
            entry = drops.setdefault(fam, {})
            for mode in MODES:
                if mode not in entry:
                    entry[mode] = reason

        # --- Rules B and D: per-mode. ---
        for mode in MODES:
            live = {f for f in candidates if not _already_dropped_in(f, mode)}

            for fam, reason in _rule_b(eq, mode, live).items():
                drops.setdefault(fam, {})[mode] = reason

            live = {f for f in candidates if not _already_dropped_in(f, mode)}
            for fam, reason in _rule_d(eq, mode, live, base).items():
                drops.setdefault(fam, {})[mode] = reason

    return drops


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke() -> None:
    eqs = [
        parse_equation_str("$:-[[ = (&", "example", 1),
        parse_equation_str(":?-)[ = $",  "example", 2),
        parse_equation_str("$(-&$ = &$", "example", 3),
    ]
    drops = structural_drops("-", eqs)

    # add / mul / lcm: Rule B on eq[2] — infeasible in BOTH modes.
    for fam in ("add", "mul", "lcm"):
        assert STANDARD in drops.get(fam, {}), \
            f"{fam} should be dropped in standard; got: {drops.get(fam)}"
        assert LITTLE_ENDIAN in drops.get(fam, {}), \
            f"{fam} should be dropped in little_endian; got: {drops.get(fam)}"
        print(f"  ✓ {fam} (both modes): {drops[fam][STANDARD][:90]}…")

    # mod: Rule C on eq[3] (rt == bt) — infeasible in BOTH modes.
    assert STANDARD in drops.get("mod", {}), \
        f"mod should be dropped in standard; got: {drops.get('mod')}"
    assert LITTLE_ENDIAN in drops.get("mod", {}), \
        f"mod should be dropped in little_endian; got: {drops.get('mod')}"
    print(f"  ✓ mod (both modes): {drops['mod'][STANDARD][:90]}…")

    # fdiv: Rule D on eq[3] — infeasible in STANDARD only.
    assert STANDARD in drops.get("fdiv", {}), \
        f"fdiv should be dropped in standard; got: {drops.get('fdiv')}"
    assert LITTLE_ENDIAN not in drops.get("fdiv", {}), \
        f"fdiv must NOT be dropped in little_endian; got: {drops.get('fdiv')}"
    print(f"  ✓ fdiv (standard only): {drops['fdiv'][STANDARD][:90]}…")

    # absdiff, gcd, rmod: no rule refutes them.
    for fam in ("absdiff", "gcd", "rmod"):
        assert fam not in drops, \
            f"{fam} should NOT be dropped but got: {drops.get(fam)}"
        print(f"  ✓ {fam}: correctly kept (not in drops)")

    print("\nper_eq_structural_checks smoke PASS")


if __name__ == "__main__":
    _run_smoke()
