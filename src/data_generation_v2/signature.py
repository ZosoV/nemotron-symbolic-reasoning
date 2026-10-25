"""Structural (symbol-invariant) signature for cryptarithm problems.

Shared by stats.py (building source_signatures) and synth.py (dedup check)
so canonicalization is guaranteed identical on both sides.
"""
from __future__ import annotations

from src.encoding import _is_reversed_digit_mode
from src.op_index import CONCAT_OPS


def problem_signature(
    equations: list[dict],
    sym2digit: dict[str, int],
    op_sym2name: dict[str, str],
    mode: str,
    strategy: str,
) -> tuple:
    """Return a structural, symbol-invariant signature for one problem.

    Arithmetic problems: each equation → (op_name, L_val, R_val) where
    L_val/R_val are recovered from digit symbols via sym2digit and mode.

    Pure-concat problems: symbols are replaced by ordinal labels (order of
    first appearance across the whole problem), yielding a pattern-based
    signature that avoids collapsing the concat space.
    """
    is_concat = strategy == "pure_concat"
    sig_elements: list = []

    if is_concat:
        sym_label: dict[str, int] = {}
        counter = 0

        def get_label(s: str) -> int:
            nonlocal counter
            if s not in sym_label:
                sym_label[s] = counter
                counter += 1
            return sym_label[s]

        for eq in equations:
            lhs = eq.get("lhs", {})
            op_name = op_sym2name.get(lhs.get("op", ""), "unknown")
            l0 = get_label(lhs.get("l0", ""))
            l1 = get_label(lhs.get("l1", ""))
            r0 = get_label(lhs.get("r0", ""))
            r1 = get_label(lhs.get("r1", ""))
            sig_elements.append((op_name, l0, l1, r0, r1))
    else:
        is_le = _is_reversed_digit_mode(mode)
        for eq in equations:
            lhs = eq.get("lhs", {})
            op_name = op_sym2name.get(lhs.get("op", ""), "unknown")
            l0_d = sym2digit.get(lhs.get("l0", ""), 0)
            l1_d = sym2digit.get(lhs.get("l1", ""), 0)
            r0_d = sym2digit.get(lhs.get("r0", ""), 0)
            r1_d = sym2digit.get(lhs.get("r1", ""), 0)
            # Inverse of synth.py decomposition
            if is_le:
                L_val = l1_d * 10 + l0_d  # l0=units, l1=tens in LE
                R_val = r1_d * 10 + r0_d
            else:
                L_val = l0_d * 10 + l1_d  # l0=tens, l1=units in standard
                R_val = r0_d * 10 + r1_d
            sig_elements.append((op_name, L_val, R_val))

    return (strategy, tuple(sig_elements))
