"""Answer encoding helpers (spec §2.4 Step F) and program verifier.

Converts a (mapping, ops, mode, radix) program + query LHS into a
symbol-string answer, applying the digit-swap for little_endian mode,
magnitude rejection, zero-padding, and the inverse symbol lookup.

Also exposes:
  verify_equation  — check a single structured equation dict under a program.
  verify_program   — check every equation in a puzzle (examples + to_predict).
"""

from __future__ import annotations

from src.operations import OPERATIONS
from src.op_index import SIGNED_OPS, CONCAT_OPS


def _int_to_base_digits(v: int, base: int) -> list[int]:
    """Decompose v into base-N digits, most-significant first.

    Returns [0] for v == 0.
    """
    if v == 0:
        return [0]
    digits: list[int] = []
    while v > 0:
        digits.append(v % base)
        v //= base
    return digits[::-1]


def _is_reversed_digit_mode(mode: str) -> bool:
    """Return True for modes where the first symbol is the units digit."""
    return mode in ("little_endian", "alice")


def encode_answer(
    mapping: dict[str, int],
    ops: dict[str, str],
    op_char: str,
    query_lhs: dict,
    mode: str,
    radix: int,
    answer_hint: str | None = None,
) -> tuple[str, int] | None:
    """Encode the query answer given a solved program (spec §2.4 Step F).

    Args:
        mapping:      symbol -> digit (content symbols only).
        ops:          op_char -> op_name (winning candidate per operator).
        op_char:      operator character from the query LHS.
        query_lhs:    {'l0', 'l1', 'op', 'r0', 'r1'} from the query equation.
        mode:         'standard' | 'little_endian'.
        radix:        positional base for arithmetic.
        answer_hint:  gold answer string; used to determine zero-pad length.

    Returns:
        (answer_string, magnitude_int) on success, None on failure.
    """
    op_name = ops.get(op_char)
    if op_name is None:
        return None

    ql0 = query_lhs["l0"]
    ql1 = query_lhs["l1"]
    qr0 = query_lhs["r0"]
    qr1 = query_lhs["r1"]

    # Step 2: Concat ops — emit operand symbols directly, no arithmetic.
    if op_name == "concat_fwd":
        return ql0 + ql1 + qr0 + qr1, 0
    if op_name == "concat_rev":
        return qr0 + qr1 + ql0 + ql1, 0

    # Step 1: Decode operands.
    for ch in (ql0, ql1, qr0, qr1):
        if ch not in mapping:
            return None

    L = mapping[ql0] * radix + mapping[ql1]
    R = mapping[qr0] * radix + mapping[qr1]

    if _is_reversed_digit_mode(mode):
        # Swap digits: units becomes tens, tens becomes units.
        L = (L % radix) * radix + L // radix
        R = (R % radix) * radix + R // radix

    # Step 3: Apply operation.
    fn = OPERATIONS.get(op_name)
    if fn is None:
        return None

    result = fn(L, R)
    if result is None:
        return None

    if op_name in SIGNED_OPS:
        prefix = op_char if result < 0 else ""
        mag = abs(result)
    elif op_name == "neg_absdiff":
        # neg_absdiff(a, b) = -abs(a-b); always emits the sign prefix.
        prefix = op_char
        mag = abs(result)
    else:
        if result < 0:
            return None
        prefix = ""
        mag = result

    # Step 4: Convert magnitude to base-radix digits.
    # NOTE: When forcing to base 10, the base digits are the same as the magnitude digits
    digits = _int_to_base_digits(mag, radix)

    # Determine sign position from the gold answer_hint.
    # Sign position (leading vs trailing) is independent of mode — both can
    # appear in little_endian puzzles.  We read it directly from the hint so
    # we never have to guess.
    trailing_sign = False
    if answer_hint is not None:
        # Detect trailing sign: hint ends with op_char but does NOT start with it
        # (to avoid misidentifying a leading-sign hint of length 1).
        if (
            len(answer_hint) > 1
            and answer_hint[-1] == op_char
            and answer_hint[0] != op_char
        ):
            trailing_sign = True
            hint_mag = answer_hint[:-1]   # strip trailing sign
        elif len(answer_hint) > 1 and answer_hint[0] == op_char:
            hint_mag = answer_hint[1:]    # strip leading sign
        else:
            hint_mag = answer_hint
        target_len = len(hint_mag)
        if len(digits) > target_len:
            return None  # magnitude overflows the expected width
        digits = [0] * (target_len - len(digits)) + digits

    if _is_reversed_digit_mode(mode):
        digits = digits[::-1]

    # Step 5: Map digits back to symbols via the inverse mapping.
    inv: dict[int, str] = {v: k for k, v in mapping.items()}
    parts = [inv.get(d) for d in digits]
    if any(p is None for p in parts):
        return None

    digit_str = "".join(parts)  # type: ignore[arg-type]
    if trailing_sign:
        answer_str = digit_str + prefix   # sign at end (trailing-sign gold format)
    else:
        answer_str = prefix + digit_str   # sign at front (default / leading format)
    return answer_str, mag


# ---------------------------------------------------------------------------
# Program verifier
# ---------------------------------------------------------------------------

def verify_equation(
    eq: dict,
    mapping: dict[str, int],
    ops_mapping: dict[str, str],
    mode: str,
    radix: int,
) -> bool:
    """Return True iff a single structured equation holds under the program.

    Mirrors the Rust ``candidate_matches_eq`` semantics exactly:
      - Decodes L and R from the LHS symbols.
      - Applies the operand-digit swap for little_endian mode.
      - Applies the chosen operation.
      - Checks (result < 0) == has_sign for signed ops.
      - Reverses the magnitude digits for little_endian, then compares them
        symbol-by-symbol against actual_rhs.

    Concat ops are verified by structural symbol comparison (no arithmetic).

    Returns False for any program inconsistency (missing mapping, operation
    returning None, magnitude overflow, etc.).
    """
    from src.op_index import CONCAT_OPS  # local to avoid circular at module level

    lhs = eq["lhs"]
    op_char = lhs["op"]
    op_name = ops_mapping.get(op_char)
    if op_name is None:
        return False

    has_sign: bool = eq["rhs"]["op"] is not None
    actual_rhs: list[str] = eq["rhs"]["actual_rhs"]

    # --- Concat ops: purely structural ---
    if op_name in CONCAT_OPS:
        if has_sign or len(actual_rhs) != 4:
            return False
        if op_name == "concat_fwd":
            return actual_rhs == [lhs["l0"], lhs["l1"], lhs["r0"], lhs["r1"]]
        if op_name == "concat_rev":
            return actual_rhs == [lhs["r0"], lhs["r1"], lhs["l0"], lhs["l1"]]
        return False

    # --- Arithmetic ops ---
    try:
        L = mapping[lhs["l0"]] * radix + mapping[lhs["l1"]]
        R = mapping[lhs["r0"]] * radix + mapping[lhs["r1"]]
        expected_digits = [mapping[c] for c in actual_rhs]
    except KeyError:
        return False

    if _is_reversed_digit_mode(mode):
        L = (L % radix) * radix + L // radix
        R = (R % radix) * radix + R // radix

    fn = OPERATIONS.get(op_name)
    if fn is None:
        return False

    raw = fn(L, R)
    if raw is None:
        return False

    if op_name in SIGNED_OPS:
        # sub_signed / rsub_signed: result sign must match the RHS sign prefix.
        if (raw < 0) != has_sign:
            return False
        mag = abs(raw)
    elif op_name == "neg_absdiff":
        # neg_absdiff always emits the sign prefix (result is always <= 0).
        # Python's OPERATIONS["neg_absdiff"] returns -abs(a-b); Rust's apply(4)
        # returns abs(a-b) (positive) and skips the signed-op check.  Mirror
        # encode_answer's handling: require has_sign=True, take abs(raw).
        if not has_sign:
            return False
        mag = abs(raw)
    else:
        if raw < 0:
            return False
        mag = raw

    # Convert magnitude to base-radix digits (zero-padded to expected length).
    res_len = len(actual_rhs)
    digits = _int_to_base_digits(mag, radix)
    if len(digits) > res_len:
        return False  # magnitude overflows the expected width
    digits = [0] * (res_len - len(digits)) + digits

    if _is_reversed_digit_mode(mode):
        digits = digits[::-1]

    return digits == expected_digits


def verify_program(
    equations: list[dict],
    mapping: dict[str, int],
    ops_mapping: dict[str, str],
    mode: str,
    radix: int,
) -> bool:
    """Return True iff every equation (examples + to_predict) holds under the program.

    Iterates over all structured equation dicts — the same list produced by
    ``extract_all_equations`` and stored in the ``equations`` JSON column.
    Returns False on the first unsatisfied equation.
    """
    return all(
        verify_equation(eq, mapping, ops_mapping, mode, radix)
        for eq in equations
    )
