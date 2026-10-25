"""Standalone reference implementations of the 22 arithmetic operations
used by the symbolic equation solver, ordered by observed frequency in
the training data (see solver_results.parquet aggregation).

All numeric ops have signature (a: int, b: int) -> int | None.
  - None means the guard failed (e.g. division by zero, negative result).
  - This matches the convention in solver_eq_symbolic.py (OPERATIONS dict).

The two concat ops have signature (a, b) -> str. Operands are coerced via
str(), which assumes base-10 digits. If the puzzle uses a non-decimal base,
the caller must convert a and b to their base-N digit strings before calling;
see _int_to_base_digits() in solver_eq_symbolic.py:367 for that conversion.

For the full 47-op registry and Rust acceleration see:
  src/solver_eq_symbolic.py  — OPERATIONS dict (line 96), helpers (line 51)
  rust/alice_sovler_helper/src/lib.rs — apply() dispatcher (line 89)
"""

from math import gcd as _gcd
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _nonneg(v: int) -> Optional[int]:
    """Return v if non-negative, else None (guard for *_m1 / *_m2 ops)."""
    return v if v >= 0 else None


# ---------------------------------------------------------------------------
# rank 0 — mul  (count: 394, prob: 0.2092)
# ---------------------------------------------------------------------------

def mul(a: int, b: int) -> int:
    """a * b — standard integer multiplication."""
    return a * b


# ---------------------------------------------------------------------------
# rank 1 — add  (count: 383, prob: 0.2034)
# ---------------------------------------------------------------------------

def add(a: int, b: int) -> int:
    """a + b — standard integer addition."""
    return a + b


# ---------------------------------------------------------------------------
# rank 2 — absdiff  (count: 258, prob: 0.1370)
# ---------------------------------------------------------------------------

def absdiff(a: int, b: int) -> int:
    """abs(a - b) — absolute difference, always non-negative."""
    return abs(a - b)


# ---------------------------------------------------------------------------
# rank 3 — sub_signed  (count: 256, prob: 0.1360)
# ---------------------------------------------------------------------------

def sub_signed(a: int, b: int) -> int:
    """a - b — signed subtraction; result may be negative.

    The solver stores the sign separately via a prefix on the RHS symbol
    (see compute_v in solver_eq_symbolic.py:971). There is no guard here.
    """
    return a - b


# ---------------------------------------------------------------------------
# rank 4 — concat_fwd  (count: 110, prob: 0.0584)
# ---------------------------------------------------------------------------

def concat_fwd(a, b) -> str:
    """str(a) + str(b) — forward string concatenation of the two operands.

    Note: str() assumes base-10 digits. If operands live in a different base,
    convert them to the appropriate digit strings before calling this function.
    The solver handles base conversion in _int_to_base_digits()
    (solver_eq_symbolic.py:367) before entering _try_concat().
    """
    return str(a) + str(b)


# ---------------------------------------------------------------------------
# rank 5 — add_m1  (count: 85, prob: 0.0451)
# ---------------------------------------------------------------------------

def add_m1(a: int, b: int) -> Optional[int]:
    """a + b - 1 — sum minus one. Returns None if result is negative."""
    return _nonneg(a + b - 1)


# ---------------------------------------------------------------------------
# rank 6 — mul_m1  (count: 82, prob: 0.0435)
# ---------------------------------------------------------------------------

def mul_m1(a: int, b: int) -> Optional[int]:
    """a * b - 1 — product minus one. Returns None if result is negative.

    Fails when a == 0 or b == 0 (product is 0, result would be -1).
    """
    return _nonneg(a * b - 1)


# ---------------------------------------------------------------------------
# rank 7 — mul_p1  (count: 79, prob: 0.0420)
# ---------------------------------------------------------------------------

def mul_p1(a: int, b: int) -> int:
    """a * b + 1 — product plus one. Always non-negative for non-negative inputs."""
    return a * b + 1


# ---------------------------------------------------------------------------
# rank 8 — add_p1  (count: 78, prob: 0.0414)
# ---------------------------------------------------------------------------

def add_p1(a: int, b: int) -> int:
    """a + b + 1 — sum plus one. Always non-negative for non-negative inputs."""
    return a + b + 1


# ---------------------------------------------------------------------------
# rank 9 — rsub_signed  (count: 40, prob: 0.0212)
# ---------------------------------------------------------------------------

def rsub_signed(a: int, b: int) -> int:
    """b - a — reverse signed subtraction; result may be negative.

    Mirror of sub_signed with operands swapped. The sign is tracked by the
    solver separately (see compute_v in solver_eq_symbolic.py:971).
    """
    return b - a


# ---------------------------------------------------------------------------
# rank 10 — neg_absdiff  (count: 38, prob: 0.0202)
# ---------------------------------------------------------------------------

def neg_absdiff(a: int, b: int) -> int:
    """-abs(a - b) — negated absolute difference; result is always <= 0.

    Divergence from solver_eq_symbolic.py: the solver stores this op as
    abs(a-b) in OPERATIONS and applies the negation via the RHS sign prefix
    in compute_v() (line 971). Here the negation is folded in for clarity.
    The mathematical value is identical.
    """
    return -abs(a - b)


# ---------------------------------------------------------------------------
# rank 11 — concat_rev  (count: 28, prob: 0.0149)
# ---------------------------------------------------------------------------

def concat_rev(a, b) -> str:
    """str(b) + str(a) — reverse string concatenation (right operand first).

    Note: str() assumes base-10 digits. See concat_fwd for the base-N caveat.
    """
    return str(b) + str(a)


# ---------------------------------------------------------------------------
# rank 12 — mod  (count: 15, prob: 0.0080)
# ---------------------------------------------------------------------------

def mod(a: int, b: int) -> Optional[int]:
    """a % b — remainder of a divided by b. Returns None when b == 0."""
    return None if b == 0 else a % b


# ---------------------------------------------------------------------------
# rank 13 — gcd  (count: 14, prob: 0.0074)
# ---------------------------------------------------------------------------

def gcd(a: int, b: int) -> int:
    """math.gcd(a, b) — greatest common divisor. Always non-negative."""
    return _gcd(a, b)


# ---------------------------------------------------------------------------
# rank 14 — rmod  (count: 10, prob: 0.0053)
# ---------------------------------------------------------------------------

def rmod(a: int, b: int) -> Optional[int]:
    """b % a — reverse mod (b divided by a). Returns None when a == 0."""
    return None if a == 0 else b % a


# ---------------------------------------------------------------------------
# rank 15 — lcm  (count: 4, prob: 0.0021)
# ---------------------------------------------------------------------------

def lcm(a: int, b: int) -> int:
    """Least common multiple of a and b.

    Returns 0 when either operand is 0 (LCM is mathematically undefined /
    0 in that case), matching the Rust implementation in lib.rs.
    """
    if a == 0 or b == 0:
        return 0
    return a * b // _gcd(a, b)


# ---------------------------------------------------------------------------
# rank 16 — absdiff_p1  (count: 3, prob: 0.0016)
# ---------------------------------------------------------------------------

def absdiff_p1(a: int, b: int) -> int:
    """abs(a - b) + 1 — absolute difference plus one. Always >= 1."""
    return abs(a - b) + 1


# ---------------------------------------------------------------------------
# rank 17 — absdiff_m2  (count: 2, prob: 0.0011)
# ---------------------------------------------------------------------------

def absdiff_m2(a: int, b: int) -> Optional[int]:
    """abs(a - b) - 2 — absolute difference minus two.

    Returns None when result is negative (i.e. abs(a-b) < 2).
    """
    return _nonneg(abs(a - b) - 2)


# ---------------------------------------------------------------------------
# rank 18 — absdiff_m1  (count: 1, prob: 0.0005)
# ---------------------------------------------------------------------------

def absdiff_m1(a: int, b: int) -> Optional[int]:
    """abs(a - b) - 1 — absolute difference minus one.

    Returns None when result is negative (i.e. a == b).
    """
    return _nonneg(abs(a - b) - 1)


# ---------------------------------------------------------------------------
# rank 19 — add_p2  (count: 1, prob: 0.0005)
# ---------------------------------------------------------------------------

def add_p2(a: int, b: int) -> int:
    """a + b + 2 — sum plus two. Always non-negative for non-negative inputs."""
    return a + b + 2


# ---------------------------------------------------------------------------
# rank 20 — fdiv  (count: 1, prob: 0.0005)
# ---------------------------------------------------------------------------

def fdiv(a: int, b: int) -> Optional[int]:
    """a // b — floor division. Returns None when b == 0."""
    return None if b == 0 else a // b


# ---------------------------------------------------------------------------
# rank 21 — a2_plus_b  (count: 1, prob: 0.0005)
# ---------------------------------------------------------------------------

def a2_plus_b(a: int, b: int) -> int:
    """a*a + b — square of a plus b."""
    return a * a + b


# ---------------------------------------------------------------------------
# Registry — name → callable, mirroring solver_eq_symbolic.OPERATIONS
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, callable] = {
    "mul":          mul,
    "add":          add,
    "absdiff":      absdiff,
    "sub_signed":   sub_signed,
    "concat_fwd":   concat_fwd,
    "add_m1":       add_m1,
    "mul_m1":       mul_m1,
    "mul_p1":       mul_p1,
    "add_p1":       add_p1,
    "rsub_signed":  rsub_signed,
    "neg_absdiff":  neg_absdiff,
    "concat_rev":   concat_rev,
    "mod":          mod,
    "gcd":          gcd,
    "rmod":         rmod,
    "lcm":          lcm,
    "absdiff_p1":   absdiff_p1,
    "absdiff_m2":   absdiff_m2,
    "absdiff_m1":   absdiff_m1,
    "add_p2":       add_p2,
    "fdiv":         fdiv,
    "a2_plus_b":    a2_plus_b,
}

# ---------------------------------------------------------------------------
# Search tiers — filtered to the 22 ops in this module
#
# Mirrors the tiered strategy in solver_eq_symbolic.py:684 (_try_arithmetic).
# The solver tries cheaper/smaller op sets first to avoid spurious matches
# from overly-generic ops (e.g. add_m1 can fit almost anything).
#
# TIER0 — core ops: the most frequent and least ambiguous.
#          Ops from solver's TIER0 that appear in our 22.
# TIER1 — adds division/mod ops (more selective, rarer in practice).
#          Ops from solver's TIER1 that appear in our 22.
# TIER2 — adds offset variants (*_m1, *_p1, etc.).
#          Ops from solver's TIER2 that appear in our 22.
# TIER_FULL — all 22 ops, including the rarest ones not covered above
#             (absdiff_m2, add_p2, a2_plus_b).
# ---------------------------------------------------------------------------

# Core ops: high frequency, unambiguous semantics.
TIER0: set[str] = {
    "mul",
    "add",
    "absdiff",
    "sub_signed",
    "rsub_signed",
    "neg_absdiff",
    "gcd",
    "lcm",
    "concat_fwd",
    "concat_rev",
}

# TIER0 + division / mod ops.
TIER1: set[str] = TIER0 | {
    "fdiv",
    "mod",
    "rmod",
}

# TIER1 + offset variants (±1 / ±2 on common ops).
TIER2: set[str] = TIER1 | {
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "absdiff_m1",
    "absdiff_p1",
}

# Full set: everything in this module, including the rarest ops.
TIER_FULL: set[str] = set(OPERATIONS.keys())

# ---------------------------------------------------------------------------
# Op priority hints — filtered to the 22 ops in this module.
#
# Mirrors OP_PRIORITY / DEFAULT_PRIORITY in solver_eq_symbolic.py, with ops
# not present in OPERATIONS removed. Used to order candidates when the solver
# (or a data-gen pipeline) consults this module directly.
# ---------------------------------------------------------------------------

OP_PRIORITY: dict[str, list[str]] = {
    "*": [
        "mul", "mul_m1", "mul_p1",
        "absdiff", "add",
        "gcd", "lcm",
        "concat_fwd", "concat_rev",
    ],
    "+": [
        "add", "add_m1", "add_p1", "add_p2",
        "mul", "absdiff",
        "gcd", "lcm",
        "concat_fwd", "concat_rev",
    ],
    "-": [
        "absdiff", "sub_signed", "rsub_signed",
        "absdiff_m1", "absdiff_p1", "absdiff_m2",
        "mul", "add", "neg_absdiff",
        "gcd", "lcm",
        "concat_fwd", "concat_rev",
    ],
    "/": [
        "fdiv", "mul", "add", "absdiff",
        "concat_fwd", "concat_rev",
    ],
}

DEFAULT_PRIORITY: list[str] = [
    "add", "absdiff", "mul",
    "sub_signed", "rsub_signed",
    "add_m1", "add_p1",
    "mul_m1", "mul_p1",
    "absdiff_m1", "absdiff_p1",
    "gcd", "lcm",
    "concat_fwd", "concat_rev",
    "add_p2", "absdiff_m2",
    "a2_plus_b", "neg_absdiff",
    "fdiv", "mod", "rmod", "a2_plus_b"
]
