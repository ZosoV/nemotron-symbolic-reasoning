"""Operation *families* for the 1-5-example CoT generator.

The solver's op registry (src/operations.py) treats each offset variant as a
separate op (`mul`, `mul_p1`, `mul_m1`). The CoT template instead reasons about
a small set of op *families*, each carrying an offset *parameter*:

    mul(a, b) = a x b + offset, offset in {0, -1, +1}

This module is the single source of truth for that family view: the family
table (names, allowed offsets, natural-language labels/formulas), a value
function `apply_family`, magnitude-range bounds used by the structural
feasibility checks, base-digit helpers, and a reproducible seed recipe.

Self-contained: it does not import the project's op registry. Semantics mirror
src/operations.py exactly (guards included) so the rendered arithmetic matches
the solver.

Run (self-test):
    uv run python src/cot_generator/families.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import gcd as _gcd


# ---------------------------------------------------------------------------
# Family table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    """One operation family.

    name     — family key (also the offset-0 registry name)
    offsets  — allowed offsets, offset 0 first (the order candidates are tried)
    label    — short natural-language name for prose
    formula  — symbolic formula string, with `+ offset` where applicable
    kind     — 'plain' | 'signed' | 'neg' | 'concat'
               signed  → result may be negative (sub_signed/rsub_signed)
               neg     → result always <= 0 (neg_absdiff)
               concat  → structural digit-concatenation op
    """
    name: str
    offsets: tuple[int, ...]
    label: str
    formula: str
    kind: str

    @property
    def has_offset(self) -> bool:
        return len(self.offsets) > 1


# Ordered exactly as the template's <available_operations> block.
FAMILIES: tuple[Family, ...] = (
    Family("add", (0, 1, -1, 2), "add", "a + b + offset", "plain"),
    Family("sub_signed", (0,), "sub_signed", "a - b", "signed"),
    Family("rsub_signed", (0,), "rsub_signed", "b - a", "signed"),
    Family("mul", (0, -1, 1), "mul", "a x b + offset", "plain"),
    Family("absdiff", (0, 1, -1, -2), "absdiff", "|a - b| + offset", "plain"),
    Family("neg_absdiff", (0,), "neg_absdiff", "-|a - b|", "neg"),
    Family("gcd", (0,), "gcd", "gcd(a, b)", "plain"),
    Family("lcm", (0,), "lcm", "lcm(a, b)", "plain"),
    Family("fdiv", (0,), "fdiv", "a // b", "plain"),
    Family("mod", (0,), "mod", "a % b", "plain"),
    Family("rmod", (0,), "rmod", "b % a", "plain"),
    Family("concat_fwd", (0,), "concat_fwd", "a x 10^(digits of b) + b", "concat"),
    Family("concat_rev", (0,), "concat_rev", "b x 10^(digits of a) + a", "concat"),
)

FAMILY_BY_NAME: dict[str, Family] = {f.name: f for f in FAMILIES}
FAMILY_NAMES: tuple[str, ...] = tuple(f.name for f in FAMILIES)

SIGNED_FAMILIES: frozenset[str] = frozenset({"sub_signed", "rsub_signed"})
NEGATIVE_FAMILIES: frozenset[str] = frozenset({"sub_signed", "rsub_signed", "neg_absdiff"})
CONCAT_FAMILIES: frozenset[str] = frozenset({"concat_fwd", "concat_rev"})


# ---------------------------------------------------------------------------
# Registry-name <-> family resolution
#
# Maps a stored solver op name (e.g. "mul_p1") to (family, offset). Used to
# read the known feasible solution out of a parquet row's ops_mapping.
# ---------------------------------------------------------------------------

_REGISTRY_TO_FAMILY: dict[str, tuple[str, int]] = {
    "mul": ("mul", 0), "mul_m1": ("mul", -1), "mul_p1": ("mul", 1),
    "add": ("add", 0), "add_m1": ("add", -1), "add_p1": ("add", 1), "add_p2": ("add", 2),
    "absdiff": ("absdiff", 0), "absdiff_p1": ("absdiff", 1),
    "absdiff_m1": ("absdiff", -1), "absdiff_m2": ("absdiff", -2),
    "sub_signed": ("sub_signed", 0), "rsub_signed": ("rsub_signed", 0),
    "neg_absdiff": ("neg_absdiff", 0), "gcd": ("gcd", 0), "lcm": ("lcm", 0),
    "fdiv": ("fdiv", 0), "mod": ("mod", 0), "rmod": ("rmod", 0),
    "concat_fwd": ("concat_fwd", 0), "concat_rev": ("concat_rev", 0),
}


def resolve_registry_op(name: str) -> tuple[str, int]:
    """Map a registry op name like 'mul_p1' to (family_name, offset).

    Raises KeyError for names with no family (e.g. 'a2_plus_b').
    """
    return _REGISTRY_TO_FAMILY[name]


# ---------------------------------------------------------------------------
# Base-digit helpers
# ---------------------------------------------------------------------------

def int_to_base_digits(v: int, base: int) -> list[int]:
    """Decompose a non-negative int into base-`base` digits, MSB first.

    Mirrors src/encoding.py:_int_to_base_digits — returns [0] for v == 0.
    """
    if v < 0:
        raise ValueError(f"int_to_base_digits expects v >= 0, got {v}")
    if v == 0:
        return [0]
    out: list[int] = []
    while v:
        out.append(v % base)
        v //= base
    out.reverse()
    return out


def digit_count(v: int, base: int) -> int:
    """Number of base-`base` digits in |v| (0 has 1 digit)."""
    return len(int_to_base_digits(abs(v), base))


# ---------------------------------------------------------------------------
# Family value function
# ---------------------------------------------------------------------------

def apply_family(name: str, offset: int, a: int, b: int, base: int = 10) -> int | None:
    """Compute a family's value for operands (a, b) at a given offset.

    Returns None when the op's guard fails (division/mod by zero, or an offset
    that would drive an absolute-value op negative). Semantics mirror
    src/operations.py exactly. For signed/neg families the SIGNED value is
    returned (may be negative); callers compare |value| against the RHS
    magnitude and check the sign separately.
    """
    if name == "add":
        return a + b + offset
    if name == "mul":
        return a * b + offset
    if name == "absdiff":
        v = abs(a - b) + offset
        return v if v >= 0 else None
    if name == "sub_signed":
        return a - b
    if name == "rsub_signed":
        return b - a
    if name == "neg_absdiff":
        return -abs(a - b)
    if name == "gcd":
        return _gcd(a, b)
    if name == "lcm":
        return 0 if (a == 0 or b == 0) else a * b // _gcd(a, b)
    if name == "fdiv":
        return None if b == 0 else a // b
    if name == "mod":
        return None if b == 0 else a % b
    if name == "rmod":
        return None if a == 0 else b % a
    if name == "concat_fwd":
        return _concat(a, b, base)
    if name == "concat_rev":
        return _concat(b, a, base)
    raise KeyError(f"unknown family {name!r}")


def _concat(first: int, second: int, base: int) -> int:
    """Concatenate the base-`base` digit strings of two operands into one int."""
    fd = int_to_base_digits(first, base)
    sd = int_to_base_digits(second, base)
    digits = fd + sd
    val = 0
    for d in digits:
        val = val * base + d
    return val


# ---------------------------------------------------------------------------
# Magnitude ranges (for structural feasibility checks)
#
# Conservative [lo, hi] over operands in [0, base^2 - 1] and over the family's
# allowed offsets. Used to decide whether a family can possibly reach a result
# field of a given digit width:
#   - lower bound: lo > max representable  -> infeasible (too big a minimum)
#   - upper bound: hi < min for that width -> infeasible (can't reach enough
#                  digits), e.g. add/absdiff cannot make a 4-digit result.
# ---------------------------------------------------------------------------

def family_magnitude_range(name: str, offsets: tuple[int, ...], base: int) -> tuple[int, int]:
    """Achievable (min, max) result *magnitude* for the family over `offsets`.

    Magnitude = |result|. Operands range over [0, base^2 - 1] (leading zeros
    allowed). The result spans the union over the allowed offsets.
    """
    m = base * base - 1  # max 2-digit operand value
    off_lo = min(offsets)
    off_hi = max(offsets)
    if name == "mul":
        return (max(0, 0 + off_lo), m * m + off_hi)
    if name == "add":
        return (max(0, 0 + off_lo), 2 * m + off_hi)
    if name == "absdiff":
        # |a-b| in [0, m]; + offset, clamped at 0 by the guard.
        return (max(0, 0 + off_lo), m + off_hi)
    if name in ("sub_signed", "rsub_signed", "neg_absdiff"):
        return (0, m)
    if name == "gcd":
        return (0, m)
    if name == "lcm":
        return (0, m * m)
    if name == "fdiv":
        return (0, m)
    if name in ("mod", "rmod"):
        return (0, max(0, m - 1))
    if name in CONCAT_FAMILIES:
        return (0, base ** 4 - 1)
    return (0, m * m)


def max_result_digits(name: str, offsets: tuple[int, ...], base: int) -> int:
    """Max number of base-`base` digits the family can produce."""
    _lo, hi = family_magnitude_range(name, offsets, base)
    return digit_count(hi, base)


def min_result_magnitude(name: str, offsets: tuple[int, ...], base: int) -> int:
    """Min achievable result magnitude for the family."""
    lo, _hi = family_magnitude_range(name, offsets, base)
    return lo


# ---------------------------------------------------------------------------
# Reproducible seeding (PYTHONHASHSEED-independent)
#
# Mirrors the hash-mix recipe in src/solver_eq_symbolic.py:_stage_seed so the
# generator's random choices (digit-priority mode, perturbations) are stable.
# ---------------------------------------------------------------------------

def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a 64-bit seed from a base seed and arbitrary string parts."""
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(f"{base_seed}|{key}".encode()).digest()
    val = int.from_bytes(h[:8], "big")
    # xorshift mix with the golden-ratio multiplier (matches the solver).
    val ^= (val >> 30)
    val = (val * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    val ^= (val >> 27)
    return val & 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> None:
    assert apply_family("mul", 1, 56, 49) == 2745
    assert apply_family("add", 0, 32, 56) == 88
    assert apply_family("absdiff", 0, 32, 65) == 33
    assert apply_family("rsub_signed", 0, 32, 65) == 33
    assert apply_family("fdiv", 0, 32, 0) is None
    assert apply_family("concat_fwd", 0, 56, 18) == 5618
    assert apply_family("concat_rev", 0, 56, 18) == 1856
    assert int_to_base_digits(2745, 10) == [2, 7, 4, 5]
    assert digit_count(0, 10) == 1
    # add/absdiff cannot reach 4 digits at base 10; mul can.
    assert max_result_digits("add", (0, 1, -1, 2), 10) == 3
    assert max_result_digits("absdiff", (0, 1, -1, -2), 10) == 3  # |a-b|<=99, +1 -> 100
    assert max_result_digits("mul", (0, -1, 1), 10) == 4
    assert resolve_registry_op("mul_p1") == ("mul", 1)
    assert resolve_registry_op("absdiff") == ("absdiff", 0)
    assert derive_seed(0, "a", 1) == derive_seed(0, "a", 1)
    print("families self-test PASS")


if __name__ == "__main__":
    _selftest()
