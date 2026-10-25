"""Python-side heuristics for the arithmetic search.

Each function implements one numbered heuristic from the spec. The caller
(solver_eq_symbolic._try_arithmetic) decides whether to apply or skip each
heuristic based on SolverConfig.disabled_heuristics.

Heuristics #1, #6, #7 live entirely inside the Rust core and cannot be
bypassed from Python.

Heuristic naming convention (for disabled_heuristics config key):
  sign_pattern_filter  → heuristic #2
  op_glyph_priority    → heuristic #3
  concat_optin         → heuristic #4
  order_ops            → heuristic #5
  tier_skip            → heuristic #9
"""

from __future__ import annotations

from src.operations import OPERATIONS, OP_PRIORITY, DEFAULT_PRIORITY


def op_glyph_priority(op_char: str) -> list[str]:
    """Return ordered candidate op-name list for this operator glyph (Heuristic #3).

    Uses OP_PRIORITY for the four standard arithmetic glyphs {*, +, -, /},
    DEFAULT_PRIORITY otherwise. Appends any OPERATIONS names not already
    present so the list always covers all available ops.
    """
    priority_base = OP_PRIORITY.get(op_char, DEFAULT_PRIORITY)
    seen: dict[str, None] = {}
    for name in priority_base:
        if name in OPERATIONS and name not in seen:
            seen[name] = None
    for name in OPERATIONS:
        if name not in seen:
            seen[name] = None
    return list(seen.keys())


def sign_pattern_filter(candidates: list[str], eqs: list[tuple]) -> list[str]:
    """Narrow candidate list based on the sign pattern of this op's equations (Heuristic #2).

    Each tuple in eqs: (l0, l1, r0, r1, res_syms, has_sign, res_len).
    has_sign is at index 5.

    Pattern → candidates kept:
      all signed   → {neg_absdiff, sub_signed, rsub_signed}
      all unsigned → everything except {neg_absdiff, sub_signed, rsub_signed}
                     (absdiff subsumes sub_signed/rsub_signed when all results
                     are non-negative, so signed-difference ops add no new
                     feasible programs)
      mixed        → {sub_signed, rsub_signed}
    """
    if not eqs:
        return candidates

    signs = [bool(e[5]) for e in eqs]
    all_signed = all(signs)
    all_unsigned = not any(signs)

    if all_signed:
        keep = {"neg_absdiff", "sub_signed", "rsub_signed"}
        return [c for c in candidates if c in keep]
    elif all_unsigned:
        exclude = {"neg_absdiff", "sub_signed", "rsub_signed"}
        return [c for c in candidates if c not in exclude]
    else:
        # Mixed: some signed, some not
        keep = {"sub_signed", "rsub_signed"}
        return [c for c in candidates if c in keep]


def concat_optin(candidates: list[str], eqs: list[tuple]) -> list[str]:
    """Append concat_fwd/concat_rev when structural conditions are met (Heuristic #4).

    Conditions: every equation for this op is unsigned AND has res_len == 4.
    Only appends ops that exist in OPERATIONS.
    """
    if not eqs:
        return candidates

    all_unsigned = all(not e[5] for e in eqs)   # has_sign at index 5
    all_res4 = all(e[6] == 4 for e in eqs)       # res_len at index 6

    if not (all_unsigned and all_res4):
        return candidates

    result = list(candidates)
    for op in ("concat_fwd", "concat_rev"):
        if op in OPERATIONS and op not in result:
            result.append(op)
    return result


def order_ops(
    op_eqs: dict[str, list],
    op_candidates: dict[str, list],
) -> list[str]:
    """Sort operator characters: fewest candidates first, ties by most equations (Heuristic #5).

    This is MRV at the operator level — the most-constrained group
    rejects bad digit assignments earliest.
    """
    return sorted(
        op_eqs.keys(),
        key=lambda op: (len(op_candidates[op]), -len(op_eqs[op])),
    )


def tier_skip(tier_cands: dict[str, list]) -> bool:
    """Return True if any operator group has zero candidates in this tier (Heuristic #9).

    Calling Rust with an empty group is guaranteed to fail; skipping saves
    the overhead and advances to the next (wider) tier immediately.
    """
    return any(len(cands) == 0 for cands in tier_cands.values())
