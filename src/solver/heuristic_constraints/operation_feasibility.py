"""Structural feasibility filter for encrypted arithmetic equations.

Given an encrypted equation of the form `l0 l1 op r0 r1 = rhs`
(e.g. '`#+>@ = !>') and a set of candidate operation names, returns which
operations are structurally *impossible* before any digit value is known —
together with a human-readable reason for each discard.

Checks applied (in order, first-objection-wins per decode mode):
  check_sign            — sign affix on RHS vs. ops that may/must produce negative
  check_concat_length   — concat ops require res_len==4 and unsigned RHS
  check_magnitude_range — op result range vs. RHS representable range
  check_self_application — L≡R structurally forces a constant result

An op is discarded only when it fails under **all** decode modes (standard and
little_endian), matching the solver's own behaviour of trying both
(src/solver/solver_eq_symbolic.py:320). This gives zero false prunes.

Leading zeros are allowed (conservative): operand values range [0, base²−1].
The four checks are mode-invariant in v1; the mode parameter is reserved for
the Deferred coefficient-propagation checks described in the spec.

Spec: specs/operation-feasibility-constraints.md

Run:
    uv run python src/solver/heuristic_constraints/operation_feasibility.py \\
        --equation '`#+>@ = !>' --ops mul,add,sub_signed,neg_absdiff,concat_fwd

Smoke tests:
    uv run python src/solver/heuristic_constraints/operation_feasibility.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.op_index import CONCAT_OPS
from src.operations import OPERATIONS
from src.parsing import parse_equation


# ---------------------------------------------------------------------------
# Semantic op sets  (derived from src/operations.py semantics)
# ---------------------------------------------------------------------------

# Ops that CAN produce a negative result (encoded as a sign affix on the RHS).
# src/op_index.py:37 defines SIGNED_OPS = {sub_signed, rsub_signed}.
# neg_absdiff always ≤ 0 and uses the same sign-prefix convention when non-zero.
NEGATIVE_CAPABLE: frozenset[str] = frozenset({
    "sub_signed", "rsub_signed", "neg_absdiff",
})

# Ops whose result is always ≤ 0 (only non-negative value is 0, reached when L≡R).
ALWAYS_NONPOS: frozenset[str] = frozenset({"neg_absdiff"})

# When L≡R, these always produce 0 whenever the guard does not fail:
#   sub_signed(x,x)=0  rsub_signed(x,x)=0  absdiff(x,x)=0
#   neg_absdiff(x,x)=0  mod(x,x)=0 (x≠0)  rmod(x,x)=0 (x≠0)
ZERO_ON_EQUAL: frozenset[str] = frozenset({
    "sub_signed", "rsub_signed", "absdiff", "neg_absdiff", "mod", "rmod",
})

# When L≡R, the guard always fails (returns None) — result never exists:
#   absdiff_m1(x,x) = |x-x|-1 = -1 → None
#   absdiff_m2(x,x) = |x-x|-2 = -2 → None
NONE_ON_EQUAL: frozenset[str] = frozenset({"absdiff_m1", "absdiff_m2"})

# When L≡R and the guard succeeds, always produces 1:
#   fdiv(x,x) = x//x = 1 for x≠0
ONE_ON_EQUAL: frozenset[str] = frozenset({"fdiv"})

DECODE_MODES: tuple[str, str] = ("standard", "little_endian")

# The operand-symbol alphabet used in the Kaggle puzzle dataset.
DEFAULT_ALPHABET: tuple[str, ...] = (
    "#", "/", "|", "[", ">", "&", '"', "}", "-", "@",
    "?", ")", "{", ":", "!", "`", "'", "]", "$", "(",
    "\\", "<", "^", "%", "*", "+", ".",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EqStruct:
    """Normalised parse of a single encrypted equation.

    Fields map directly from parse_equation() (src/parsing.py:36):
      l0, l1   — left-operand symbols  (L = l0 l1)
      op_char  — operator glyph at LHS[2]
      r0, r1   — right-operand symbols (R = r0 r1)
      rhs_syms — magnitude symbols from actual_rhs (as a tuple)
      res_len  — len(rhs_syms)
      has_sign — True when RHS carries a sign affix (negative result)
      sign_pos — 'prefix' | 'postfix' | None
    """
    l0: str
    l1: str
    op_char: str
    r0: str
    r1: str
    rhs_syms: tuple[str, ...]
    res_len: int
    has_sign: bool
    sign_pos: str | None

    @classmethod
    def from_parsed(cls, eq: dict) -> EqStruct:
        """Build from a parse_equation() result dict."""
        lhs = eq["lhs"]
        rhs = eq["rhs"]
        actual = rhs["actual_rhs"]
        return cls(
            l0=lhs["l0"],
            l1=lhs["l1"],
            op_char=lhs["op"],
            r0=lhs["r0"],
            r1=lhs["r1"],
            rhs_syms=tuple(actual),
            res_len=len(actual),
            has_sign=rhs["op"] is not None,
            sign_pos=rhs.get("sign_pos"),
        )


@dataclass
class OpVerdict:
    """Feasibility verdict for one candidate operation."""
    op: str
    feasible: bool
    reason: str | None                     # None when kept; first discard reason
    per_mode: dict[str, str | None] = field(default_factory=dict)
    # per_mode[mode] = None (feasible) or a discard-reason string


@dataclass
class FeasibilityReport:
    """Full report returned by filter_feasible_operations()."""
    operator: str                          # op glyph from LHS[2]
    feasible: list[str]                    # op names that survive all checks
    discarded: dict[str, str]              # op name → aggregated discard reason
    verdicts: list[OpVerdict]              # one entry per candidate, input order


# ---------------------------------------------------------------------------
# Individual check functions
#
# Signature: (op, eq, base, mode) -> str | None
#   None  = no objection (op feasible under this check)
#   str   = discard reason (human-readable)
#
# v1 implementations are mode-invariant; the mode parameter is reserved for
# the Deferred coefficient-propagation extension.
# ---------------------------------------------------------------------------

def check_sign(op: str, eq: EqStruct, base: int, mode: str) -> str | None:
    """Discard based on sign mismatch between op output and RHS sign affix.

    Rules:
      has_sign=True  (negative result expected):
        Only NEGATIVE_CAPABLE ops can produce < 0; discard all others.

      has_sign=False (non-negative result expected):
        ALWAYS_NONPOS ops (neg_absdiff) produce ≤ 0.  Their only non-negative
        value is 0, reached only when L≡R.  Discard unless L≡R.
    """
    if eq.has_sign:
        if op not in NEGATIVE_CAPABLE:
            return (
                "RHS is signed (negative result) but this op always produces "
                "non-negative values"
            )
    else:
        if op in ALWAYS_NONPOS:
            self_equal = eq.l0 == eq.r0 and eq.l1 == eq.r1
            if not self_equal:
                return (
                    "RHS is unsigned but neg_absdiff always produces ≤ 0 "
                    "(its only non-negative value is 0, reached only when L≡R; "
                    "L≢R here)"
                )
    return None


def check_concat_length(op: str, eq: EqStruct, base: int, mode: str) -> str | None:
    """Discard concat ops when structural conditions are not satisfied.

    concat_fwd / concat_rev concatenate two 2-symbol operands, producing
    exactly 4 symbols.  Required: res_len == 4 AND unsigned RHS.
    """
    if op not in CONCAT_OPS:
        return None
    if eq.has_sign:
        return "concat result is always unsigned, but RHS carries a sign affix"
    if eq.res_len != 4:
        return (
            f"concat result is always 4 symbols (2 + 2), "
            f"but RHS magnitude has {eq.res_len} symbol(s)"
        )
    return None


def _op_magnitude_range(op: str, base: int) -> tuple[int, int]:
    """Achievable (min, max) result magnitude for operands in [0, base²−1].

    For signed ops the magnitude is |result|.
    Conservative: operand minimum is 0 (leading zeros allowed).
    """
    m = base * base - 1  # max operand value
    table: dict[str, tuple[int, int]] = {
        "mul":          (0, m * m),
        "add":          (0, 2 * m),
        "absdiff":      (0, m),
        "sub_signed":   (0, m),           # magnitude = |a-b|
        "rsub_signed":  (0, m),           # magnitude = |b-a|
        "neg_absdiff":  (0, m),           # magnitude = |a-b|
        "add_m1":       (0, max(0, 2 * m - 1)),
        "mul_m1":       (0, max(0, m * m - 1)),
        "mul_p1":       (1, m * m + 1),
        "add_p1":       (1, 2 * m + 1),
        "concat_fwd":   (0, base ** 4 - 1),
        "concat_rev":   (0, base ** 4 - 1),
        "mod":          (0, max(0, m - 1)),
        "gcd":          (0, m),
        "rmod":         (0, max(0, m - 1)),
        "lcm":          (0, m * m),
        "absdiff_p1":   (1, m + 1),
        "absdiff_m2":   (0, max(0, m - 2)),
        "absdiff_m1":   (0, max(0, m - 1)),
        "add_p2":       (2, 2 * m + 2),
        "fdiv":         (0, m),
        "a2_plus_b":    (0, m * m + m),
    }
    return table.get(op, (0, m * m + m))


def check_magnitude_range(op: str, eq: EqStruct, base: int, mode: str) -> str | None:
    """Discard when the op's minimum result magnitude exceeds the RHS capacity.

    Computes [lo, hi] for the op (operands ∈ [0, base²−1], leading zeros
    allowed) and checks whether lo ≤ base^res_len − 1.

    Under conservative assumptions this rarely fires, but correctly handles
    edge cases such as add_p2 (min magnitude 2) with very small bases or
    very short RHS fields.  Concat ops are skipped (handled by
    check_concat_length).
    """
    if op in CONCAT_OPS:
        return None
    lo, _hi = _op_magnitude_range(op, base)
    rhs_max = base ** eq.res_len - 1
    if lo > rhs_max:
        return (
            f"minimum achievable result magnitude is {lo}, but the RHS field "
            f"({eq.res_len} symbol(s) in base {base}) can represent at most {rhs_max}"
        )
    return None


def check_self_application(op: str, eq: EqStruct, base: int, mode: str) -> str | None:
    """Discard based on forced-constant behaviour when L≡R.

    When l0==r0 and l1==r1 (same operand glyphs), digit injectivity
    (distinct symbols → distinct digits, enforced by the Rust core) forces
    the two operands to be numerically equal.  Several ops collapse:

      NONE_ON_EQUAL   — guard always fails → op can never produce a result.
                        Always discard.

      ZERO_ON_EQUAL   — result always 0 (when guard succeeds).
                        Discard if has_sign=True (0 is not negative), or if
                        RHS magnitude has ≥ 2 distinct glyphs (they would
                        need to share the zero-digit, impossible by
                        injectivity).

      ONE_ON_EQUAL    — result always 1 (when L ≠ 0, i.e. guard succeeds).
        (fdiv)          Discard if has_sign=True (1 is not negative).
    """
    if not (eq.l0 == eq.r0 and eq.l1 == eq.r1):
        return None

    if op in NONE_ON_EQUAL:
        return (
            "L≡R: this op's guard always fails when operands are equal "
            "(result is always None — op can never succeed)"
        )

    if op in ZERO_ON_EQUAL:
        if eq.has_sign:
            return (
                "L≡R: this op always produces 0, but RHS is signed "
                "(0 is not negative)"
            )
        distinct_rhs = len(set(eq.rhs_syms))
        if distinct_rhs > 1:
            return (
                f"L≡R: this op always produces 0, so all {eq.res_len} RHS "
                f"magnitude symbol(s) must represent the zero-digit — but "
                f"they contain {distinct_rhs} distinct glyphs "
                f"(impossible by digit injectivity)"
            )

    if op in ONE_ON_EQUAL:
        if eq.has_sign:
            return (
                "L≡R: fdiv always produces 1 (when L ≠ 0), but RHS is signed "
                "(1 is not negative)"
            )

    return None


# ---------------------------------------------------------------------------
# Check pipeline
# ---------------------------------------------------------------------------

_CHECKS = (
    check_sign,
    check_concat_length,
    check_magnitude_range,
    check_self_application,
)


def _check_op_single_mode(
    op: str,
    eq: EqStruct,
    base: int,
    mode: str,
) -> str | None:
    """Run all checks for one op under one decode mode.

    Returns None if the op passes all checks, or the first discard reason.
    """
    for check in _CHECKS:
        reason = check(op, eq, base, mode)
        if reason is not None:
            return reason
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_feasible_operations(
    equation: str | dict,
    candidate_ops: list[str],
    base: int = 10,
    modes: tuple[str, ...] = DECODE_MODES,
) -> FeasibilityReport:
    """Return which candidate ops are structurally feasible for the equation.

    Args:
        equation:      Raw equation string (e.g. '`#+>@ = !>') or an
                       already-parsed dict from src.parsing.parse_equation().
        candidate_ops: Op names to evaluate.  Unknown names are discarded
                       immediately with reason "unknown operation".
        base:          Numeric base for range/magnitude checks (default 10).
                       Pass len(content_symbols(equations)) for per-puzzle base.
        modes:         Decode modes to try.  An op is discarded only when
                       infeasible under ALL modes (zero false prunes).

    Returns:
        FeasibilityReport with .feasible (kept ops) and .discarded (op → reason).
    """
    if isinstance(equation, str):
        parsed = parse_equation(equation, "to_predict")
    else:
        parsed = equation
    eq = EqStruct.from_parsed(parsed)

    verdicts: list[OpVerdict] = []

    for op in candidate_ops:
        if op not in OPERATIONS:
            verdicts.append(OpVerdict(
                op=op,
                feasible=False,
                reason="unknown operation",
                per_mode={m: "unknown operation" for m in modes},
            ))
            continue

        per_mode: dict[str, str | None] = {
            mode: _check_op_single_mode(op, eq, base, mode)
            for mode in modes
        }

        # Feasible iff at least one mode raises no objection.
        feasible = any(r is None for r in per_mode.values())

        if feasible:
            aggregated = None
        else:
            # All modes objected; aggregate distinct reasons (preserve order).
            seen: dict[str, None] = {}
            for r in per_mode.values():
                if r is not None:
                    seen[r] = None
            aggregated = " | ".join(seen)

        verdicts.append(OpVerdict(
            op=op,
            feasible=feasible,
            reason=aggregated,
            per_mode=per_mode,
        ))

    return FeasibilityReport(
        operator=eq.op_char,
        feasible=[v.op for v in verdicts if v.feasible],
        discarded={v.op: v.reason for v in verdicts if not v.feasible},
        verdicts=verdicts,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Structural feasibility filter for encrypted arithmetic equations.",
    )
    parser.add_argument(
        "--equation",
        required=True,
        help="Encrypted equation string, e.g. '`#+>@ = !>'",
    )
    parser.add_argument(
        "--ops",
        default=",".join(OPERATIONS.keys()),
        help="Comma-separated op names to evaluate (default: all 22 ops)",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10,
        help="Numeric base for range checks (default: 10)",
    )
    args = parser.parse_args(argv)

    candidate_ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    report = filter_feasible_operations(args.equation, candidate_ops, base=args.base)

    print(f"Equation : {args.equation}")
    print(f"Operator : {report.operator!r}")
    print(f"Base     : {args.base}")
    print()
    print(f"Feasible ({len(report.feasible)}): {', '.join(report.feasible) or '—'}")
    print()
    if report.discarded:
        print(f"Discarded ({len(report.discarded)}):")
        for op, reason in report.discarded.items():
            print(f"  {op:20s}  {reason}")
    else:
        print("No ops discarded.")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _run_smoke_tests() -> None:
    """Self-verifying assertions for the four cases documented in the spec."""

    # --- Case 1: unsigned RHS, short (res_len=2), L≢R ---
    # `#+>@ = !>  →  l0=`, l1=#, op=+, r0=>, r1=@; RHS=!> unsigned res_len=2
    r1 = filter_feasible_operations(
        "`#+>@ = !>",
        ["mul", "add", "sub_signed", "neg_absdiff", "concat_fwd"],
    )
    assert "neg_absdiff" in r1.discarded, (
        "neg_absdiff must be discarded (unsigned RHS, L≢R)"
    )
    assert "concat_fwd" in r1.discarded, (
        "concat_fwd must be discarded (res_len=2 ≠ 4)"
    )
    assert "mul" in r1.feasible, "mul must be kept"
    assert "add" in r1.feasible, "add must be kept"
    assert "sub_signed" in r1.feasible, "sub_signed must be kept"
    print("Case 1 PASS  unsigned, short RHS, L≢R")

    # --- Case 2: signed RHS (prefix) ---
    # AB-CD = -EF  →  has_sign=True; only {sub_signed, rsub_signed, neg_absdiff} kept
    r2 = filter_feasible_operations(
        "AB-CD = -EF",
        ["mul", "add", "sub_signed", "rsub_signed", "neg_absdiff", "absdiff"],
    )
    assert "mul" in r2.discarded, "mul must be discarded (signed RHS)"
    assert "add" in r2.discarded, "add must be discarded (signed RHS)"
    assert "absdiff" in r2.discarded, "absdiff must be discarded (signed RHS)"
    assert "sub_signed" in r2.feasible, "sub_signed must be kept"
    assert "rsub_signed" in r2.feasible, "rsub_signed must be kept"
    assert "neg_absdiff" in r2.feasible, "neg_absdiff must be kept"
    print("Case 2 PASS  signed RHS filters non-negative ops")

    # --- Case 3: L≡R with distinct RHS symbols ---
    # AA+AA = BC  →  l0=A, l1=A, r0=A, r1=A (L≡R); RHS='BC' has 2 distinct glyphs
    # sub_signed(AA, AA)=0 and absdiff(AA, AA)=0 but RHS needs 2 distinct digits → discard
    r3 = filter_feasible_operations(
        "AA+AA = BC",
        ["sub_signed", "rsub_signed", "absdiff", "add", "mul"],
    )
    assert "sub_signed" in r3.discarded, (
        "sub_signed must be discarded (L≡R → 0, RHS has 2 distinct glyphs)"
    )
    assert "rsub_signed" in r3.discarded, (
        "rsub_signed must be discarded (L≡R → 0, RHS has 2 distinct glyphs)"
    )
    assert "absdiff" in r3.discarded, (
        "absdiff must be discarded (L≡R → 0, RHS has 2 distinct glyphs)"
    )
    assert "add" in r3.feasible, "add(A,A)=2A — not forced to 0, keep"
    assert "mul" in r3.feasible, "mul(A,A)=A² — not forced to 0, keep"
    print("Case 3 PASS  L≡R with distinct RHS symbols")

    # --- Case 4: mode-invariance ---
    # v1 checks are mode-independent; verdicts must be identical for both modes.
    ops_all = list(OPERATIONS.keys())
    r_std = filter_feasible_operations("`#+>@ = !>", ops_all, modes=("standard",))
    r_le  = filter_feasible_operations("`#+>@ = !>", ops_all, modes=("little_endian",))
    assert r_std.feasible == r_le.feasible, (
        "v1 feasible lists must be identical across decode modes"
    )
    assert r_std.discarded == r_le.discarded, (
        "v1 discarded maps must be identical across decode modes"
    )
    print("Case 4 PASS  mode-invariant verdicts")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _run_smoke_tests()
    else:
        main()
