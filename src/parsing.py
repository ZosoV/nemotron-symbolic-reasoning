"""Equation extraction and parsing.

An equation string has the form:
    l0 l1 op r0 r1 = rhs
where l0, l1, r0, r1 are content symbols, op is an operator symbol,
and rhs may optionally begin with the op character (sign prefix).

Structured equation dict:
    {
        'type': 'example' | 'to_predict',
        'lhs': {'l0': str, 'l1': str, 'op': str, 'r0': str, 'r1': str},
        'rhs': {'op': str | None, 'actual_rhs': list[str]},
    }
"""

from __future__ import annotations


def extract_all_equations(row: dict) -> list[dict]:
    """Extract and parse all equations from a prompt row.

    Appends the gold answer to the query line to form the to_predict equation.
    Returns a list of structured equation dicts; the last entry is always
    the to_predict equation.
    """
    raw: list[tuple[str, str]] = []
    for line in row["prompt"].split("\n"):
        if line.startswith("Now"):
            question_eq = line.split(": ", 1)[1].strip()
            raw.append((f"{question_eq} = {row['answer']}", "to_predict"))
        elif " = " in line:
            raw.append((line.strip(), "example"))
    return [parse_equation(eq_str, kind) for eq_str, kind in raw]


def parse_equation(eq_str: str, kind: str) -> dict:
    """Parse a single equation string into a structured dict.

    Args:
        eq_str: e.g. "AB*CD = EF" or "AB*CD = *EF"
        kind:   'example' or 'to_predict'
    """
    lhs_str, rhs_str = eq_str.split(" = ", 1)
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()

    if len(lhs_str) != 5:
        raise ValueError(f"LHS must be exactly 5 characters, got {lhs_str!r}")

    op_char = lhs_str[2]
    lhs = {
        "l0": lhs_str[0],
        "l1": lhs_str[1],
        "op": op_char,
        "r0": lhs_str[3],
        "r1": lhs_str[4],
    }

    rhs_chars = list(rhs_str)
    if rhs_chars and rhs_chars[0] == op_char:
        # Leading sign: standard mode (op glyph at front of RHS).
        rhs = {"op": op_char, "sign_pos": "prefix", "actual_rhs": rhs_chars[1:]}
    elif len(rhs_chars) > 1 and rhs_chars[-1] == op_char:
        # Trailing sign: little_endian mode reverses the answer so the sign
        # glyph lands at the back.  Strip it and keep the magnitude in written
        # order; the Rust core (candidate_matches_eq) handles the positional
        # reversal independently of sign placement.
        rhs = {"op": op_char, "sign_pos": "postfix", "actual_rhs": rhs_chars[:-1]}
    else:
        rhs = {"op": None, "sign_pos": None, "actual_rhs": rhs_chars}

    return {"type": kind, "lhs": lhs, "rhs": rhs}


def content_symbols(equations: list[dict]) -> list[str]:
    """Return unique non-operator, non-'=' symbols in first-appearance order.

    Operator symbols are the characters found in the op position of any LHS.
    These are excluded even if they also appear in rhs or operand positions.
    """
    op_chars: set[str] = {eq["lhs"]["op"] for eq in equations}
    seen: dict[str, None] = {}
    for eq in equations:
        lhs = eq["lhs"]
        for ch in (lhs["l0"], lhs["l1"], lhs["r0"], lhs["r1"]):
            if ch not in op_chars and ch not in seen:
                seen[ch] = None
        for ch in eq["rhs"]["actual_rhs"]:
            if ch not in op_chars and ch not in seen:
                seen[ch] = None
    return list(seen.keys())


def build_op_groups(
    equations: list[dict],
    sym2i: dict[str, int],
) -> dict[str, list[tuple]]:
    """Group equations by operator and encode symbols as indices.

    Each entry in the group list is:
        (l0, l1, r0, r1, res_syms: tuple[int, ...], has_sign: bool, res_len: int)

    Equations whose operands or rhs contain characters absent from sym2i
    (i.e., operator characters that bleed into content positions) are silently
    skipped. Groups where all equations are skipped are omitted from the result.

    Returns dict mapping op_char -> list of encoded equation tuples.
    """
    op_eqs: dict[str, list[tuple]] = {}
    for eq in equations:
        lhs = eq["lhs"]
        op = lhs["op"]
        has_sign = eq["rhs"]["op"] is not None
        actual_rhs = eq["rhs"]["actual_rhs"]

        # Skip equations that reference characters outside the digit alphabet.
        try:
            entry = (
                sym2i[lhs["l0"]],
                sym2i[lhs["l1"]],
                sym2i[lhs["r0"]],
                sym2i[lhs["r1"]],
                tuple(sym2i[c] for c in actual_rhs),
                has_sign,
                len(actual_rhs),
            )
        except KeyError:
            continue

        op_eqs.setdefault(op, []).append(entry)

    return op_eqs
