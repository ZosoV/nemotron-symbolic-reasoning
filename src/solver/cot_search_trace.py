"""Renderer for the solver's search-trace chain-of-thought.

Takes the Python-layer narration events (from _try_arithmetic) and the compact DFS spine
produced by arithmetic_search_traced, and renders them as a single explanatory prose string
saved in the `search_cot` column of the output parquet.

Spine format (from Rust search_traced):
    Each entry is (sym_idx, accepted_digit, pruned, backtracked) for one symbol on the
    accepted assignment chain (first solution only):
        pruned:      list of (digit, group_idx) — forward-check rejections at this depth
        backtracked: list of digits — passed forward-check but subtree failed

Python-layer event dicts (collected in _try_arithmetic):
    {"type": "candidates",  "op_char": str, "initial": list, "final": list, "changes": list[str]}
    {"type": "order_ops",   "order": list[str], "stats": list[tuple[str,int,int]]}
    {"type": "tier_skip",   "tier": str, "op_char": str}
    {"type": "stage_enter", "tier": str, "mode": str, "radix": int}
    {"type": "stage_result","tier": str, "mode": str, "radix": int, "n_solutions": int}

Section layout of the rendered CoT:
    [Candidate setup and search strategy]
        — operator candidate narrowing (heuristics #2/#3/#4)
        — operator ordering (heuristic #5)
        — tier/mode/radix escalation events
    [Digit-assignment DFS]
        — symbol assignment order (variable-ordering heuristic)
        — per-symbol digit trial log (accepted-path only)
    [Accepted candidates per operator]
        — per-operator accepted operation list with decoded arithmetic verification
    [Result]
        — final digit and operator mapping summary
"""

from __future__ import annotations

# Human-readable short labels for arithmetic operations.
_OP_LABEL: dict[str, str] = {
    "add":          "addition (a + b)",
    "mul":          "multiplication (a × b)",
    "absdiff":      "absolute difference |a − b|",
    "sub_signed":   "signed subtraction (a − b, may be negative)",
    "rsub_signed":  "signed reverse subtraction (b − a, may be negative)",
    "neg_absdiff":  "negated absolute difference (−|a − b|)",
    "gcd":          "greatest common divisor gcd(a, b)",
    "lcm":          "least common multiple lcm(a, b)",
    "fdiv":         "floor division ⌊a ÷ b⌋",
    "mod":          "modulo (a mod b)",
    "rmod":         "reverse modulo (b mod a)",
    "concat_fwd":   "forward concatenation (left‖right)",
    "concat_rev":   "reverse concatenation (right‖left)",
    "add_m1":       "addition minus 1 (a + b − 1)",
    "add_p1":       "addition plus 1 (a + b + 1)",
    "mul_m1":       "multiplication minus 1 (a × b − 1)",
    "mul_p1":       "multiplication plus 1 (a × b + 1)",
    "absdiff_m1":   "|a − b| minus 1",
    "absdiff_p1":   "|a − b| plus 1",
    "add_p2":       "addition plus 2 (a + b + 2)",
    "absdiff_m2":   "|a − b| minus 2",
    "a2_plus_b":    "a² + b",
}

_INDENT = "  "


# ---------------------------------------------------------------------------
# Python-layer narration
# ---------------------------------------------------------------------------

def _narrate_py_events(py_events: list[dict]) -> list[str]:
    """Convert Python-layer events into a list of prose sentences."""
    lines: list[str] = []
    for ev in py_events:
        t = ev["type"]

        if t == "candidates":
            op = ev["op_char"]
            initial = ev["initial"]
            final = ev["final"]
            changes = ev["changes"]
            # Only list up to 5 candidate names to keep prose brief.
            cand_str = ", ".join(final[:5]) + ("…" if len(final) > 5 else "")
            if changes:
                lines.append(
                    f"Operator '{op}': candidates prioritised by glyph → "
                    f"{', '.join(changes)} → {len(final)} candidate(s) remaining: [{cand_str}]."
                )
            else:
                lines.append(
                    f"Operator '{op}': {len(final)} candidate operation(s): [{cand_str}] "
                    f"(glyph priority, no further narrowing)."
                )

        elif t == "order_ops":
            stats = ev["stats"]  # list of (op_char, n_cands, n_eqs)
            parts = "; ".join(
                f"'{op}' ({n_c} cand{'s' if n_c != 1 else ''} / {n_e} eq{'s' if n_e != 1 else ''})"
                for op, n_c, n_e in stats
            )
            lines.append(
                f"Operator ordering (MRV heuristic #5, fewest candidates first): {parts}."
            )

        elif t == "tier_skip":
            lines.append(
                f"Tier '{ev['tier']}': operator '{ev['op_char']}' has no candidates in this "
                f"tier — skipped (tier_skip heuristic)."
            )

        elif t == "stage_enter":
            mode_str = "standard (big-endian)" if ev["mode"] == "standard" else "little-endian"
            radix = ev["radix"]
            tier = ev["tier"]
            lines.append(
                f"Searching tier '{tier}', {mode_str}, base {radix}:"
            )

        elif t == "stage_result":
            n = ev["n_solutions"]
            if n == 0:
                lines.append("  → No solution found in this stage; escalating.")
            else:
                lines.append(f"  → {n} solution(s) found.")

    return lines


# ---------------------------------------------------------------------------
# Rust DFS-layer narration (spine-based, per-digit trial log)
# ---------------------------------------------------------------------------

def _narrate_spine(
    spine: list[tuple[int, int, list, list]],
    content_syms: list[str],
    sorted_ops: list[str],
) -> list[str]:
    """Convert the DFS spine into a per-digit trial log — one sub-line per digit tried."""
    if not spine:
        return ["  (No DFS trace available — no solution was found in this stage.)"]

    lines: list[str] = []

    order_str = " → ".join(_sym(sym_idx, content_syms) for sym_idx, _, _, _ in spine)
    lines.append(
        f"  Symbol assignment order (most-constrained first, "
        f"variable-ordering heuristic): {order_str}."
    )
    lines.append("")

    for depth, (sym_idx, accepted_digit, pruned, backtracked) in enumerate(spine):
        sym = _sym(sym_idx, content_syms)
        indent = _INDENT * (depth + 1)

        # Build lookup structures for this symbol's trial record.
        pruned_map: dict[int, int] = {int(d): int(g) for d, g in pruned}
        backtracked_set: set[int] = {int(d) for d in backtracked}
        acc = int(accepted_digit)

        # All digits actually tried at this depth (sorted = trial order).
        all_tried = sorted(pruned_map.keys() | backtracked_set | {acc})

        lines.append(f"{indent}Symbol '{sym}' — digit trials:")
        for d in all_tried:
            trial_indent = indent + _INDENT
            if d in pruned_map:
                op = _group(pruned_map[d], sorted_ops)
                lines.append(
                    f"{trial_indent}try '{sym}'={d} → pruned: operator '{op}' "
                    f"has no surviving candidates (forward-checking)."
                )
            elif d in backtracked_set:
                lines.append(
                    f"{trial_indent}try '{sym}'={d} → explored; subtree yielded no solution (backtrack)."
                )
            else:
                lines.append(f"{trial_indent}try '{sym}'={d} → accepted ✓")
        lines.append("")

    lines.append(f"  [All {len(spine)} symbols assigned — solution accepted.]")
    return lines


def _sym(sym_idx: int, content_syms: list[str]) -> str:
    if 0 <= sym_idx < len(content_syms):
        return content_syms[sym_idx]
    return f"sym[{sym_idx}]"


def _group(group_idx: int, sorted_ops: list[str]) -> str:
    if 0 <= group_idx < len(sorted_ops):
        return sorted_ops[group_idx]
    return f"group[{group_idx}]"


def _digits_str(digits: list[int]) -> str:
    """Format a list of digits as a compact string, collapsing runs: [0,1,2,5] → '0–2, 5'."""
    if not digits:
        return ""
    digits = sorted(set(digits))
    runs: list[str] = []
    start = digits[0]
    end = digits[0]
    for d in digits[1:]:
        if d == end + 1:
            end = d
        else:
            runs.append(f"{start}–{end}" if end > start else str(start))
            start = end = d
    runs.append(f"{start}–{end}" if end > start else str(start))
    return ", ".join(runs)


# ---------------------------------------------------------------------------
# Accepted candidates per operator (with decoded arithmetic)
# ---------------------------------------------------------------------------

def _narrate_accepted_candidates(
    op_eqs: dict,
    ops_candidates: dict[str, list[str]],
    content_syms: list[str],
    digits_mapping: dict[str, int],
    mode: str,
    radix: int,
) -> list[str]:
    """Render every accepted candidate operation per operator with decoded arithmetic.

    For each operator group lists all surviving op names/labels and then, for each equation
    in the group, shows the decoded operand values and each candidate's computed result vs.
    the expected RHS — marking ✓ (matches) or the actual vs. expected values on mismatch.
    """
    from src.operations import OPERATIONS  # late import — avoids any potential circularity
    from src.op_index import CONCAT_OPS

    reversed_mode = mode in ("little_endian", "alice")
    lines: list[str] = []

    for op_char, accepted_ops in ops_candidates.items():
        eqs = op_eqs.get(op_char, [])

        # Header: list all accepted candidates with human-readable labels.
        labels = ", ".join(
            f"{name} ({_OP_LABEL.get(name, name)})" for name in accepted_ops
        )
        lines.append(
            f"  Operator '{op_char}' — {len(accepted_ops)} accepted "
            f"candidate(s): {labels}."
        )

        # Per-equation arithmetic verification.
        for eq_idx, eq in enumerate(eqs, 1):
            l0i, l1i, r0i, r1i, res_syms, has_sign, res_len = eq
            l0s = content_syms[l0i]
            l1s = content_syms[l1i]
            r0s = content_syms[r0i]
            r1s = content_syms[r1i]
            res_strs = [content_syms[si] for si in res_syms]
            sign_prefix = "−" if has_sign else ""
            eq_str = f"{l0s}{l1s} {op_char} {r0s}{r1s} = {sign_prefix}{''.join(res_strs)}"

            # Guard: skip equations where a symbol has no digit assignment (shouldn't occur).
            try:
                d_l0 = digits_mapping[l0s]
                d_l1 = digits_mapping[l1s]
                d_r0 = digits_mapping[r0s]
                d_r1 = digits_mapping[r1s]
                res_digits = [digits_mapping[content_syms[si]] for si in res_syms]
            except KeyError:
                lines.append(f"    eq[{eq_idx}]: '{eq_str}' — symbol not in mapping, skipped.")
                continue

            # Decode operand numeric values.
            if reversed_mode:
                L = d_l0 + d_l1 * radix
                R = d_r0 + d_r1 * radix
            else:
                L = d_l0 * radix + d_l1
                R = d_r0 * radix + d_r1

            # Decode expected RHS magnitude, then apply sign.
            rhs_mag = 0
            if reversed_mode:
                for j, dv in enumerate(res_digits):
                    rhs_mag += dv * (radix ** j)
            else:
                for dv in res_digits:
                    rhs_mag = rhs_mag * radix + dv
            expected = -rhs_mag if has_sign else rhs_mag

            sign_note = f" [signed, magnitude {rhs_mag}]" if has_sign else ""
            lines.append(
                f"    eq[{eq_idx}]: '{eq_str}'  L={L}, R={R}, expected={expected}{sign_note}"
            )

            # Per-candidate result.
            cand_parts: list[str] = []
            for op_name in accepted_ops:
                if op_name in CONCAT_OPS:
                    cand_parts.append(f"{op_name}=concat(structural)✓")
                else:
                    fn = OPERATIONS.get(op_name)
                    if fn is None:
                        cand_parts.append(f"{op_name}=??(unknown)")
                    else:
                        result = fn(L, R)
                        if result is None:
                            cand_parts.append(f"{op_name}=None(guard)✗")
                        elif result == expected:
                            cand_parts.append(f"{op_name}={result}✓")
                        else:
                            cand_parts.append(f"{op_name}={result}✗(exp {expected})")
            lines.append("           " + " ; ".join(cand_parts))

        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Solution summary
# ---------------------------------------------------------------------------

def _solution_summary(
    digits_mapping: dict[str, int] | None,
    ops_mapping: dict[str, str] | None,
    mode: str,
    radix: int,
    tier: str | None,
) -> list[str]:
    """Return 1–2 prose lines summarising the found solution."""
    if digits_mapping is None or ops_mapping is None:
        return ["No solution was found."]

    digit_str = ", ".join(
        f"'{s}' = {d}" for s, d in sorted(digits_mapping.items(), key=lambda x: x[1])
    )
    op_str = "; ".join(
        f"'{g}' → {_OP_LABEL.get(op, op)}"
        for g, op in ops_mapping.items()
    )
    mode_str = "standard (big-endian)" if mode == "standard" else "little-endian"
    tier_note = f", tier '{tier}'" if tier else ""
    return [
        f"Solution: digits [{digit_str}]; operators: [{op_str}].",
        f"(Strategy: {mode_str}, base {radix}{tier_note}.)",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_search_cot(
    py_events: list[dict],
    spine: list[tuple[int, int, list, list]],
    content_syms: list[str],
    sorted_ops: list[str],
    digits_mapping: dict[str, int] | None,
    ops_mapping: dict[str, str] | None,
    mode: str = "standard",
    radix: int = 10,
    tier: str | None = None,
    ops_candidates: dict[str, list[str]] | None = None,
    op_eqs: dict | None = None,
) -> str:
    """Assemble the full search-trace CoT string.

    Args:
        py_events:      Python-layer narration events (from _try_arithmetic).
        spine:          DFS spine from arithmetic_search_traced — one entry per symbol
                        in the accepted chain: (sym_idx, digit, pruned, backtracked).
        content_syms:   List of symbol strings in sym_idx order (sym2i inverse).
        sorted_ops:     Operator glyphs in group index order (same order as Rust groups).
        digits_mapping: Decoded symbol → digit assignment from the found solution.
        ops_mapping:    Decoded operator → operation name from the found solution.
        mode:           Strategy of the traced DFS stage ('standard' | 'little_endian').
        radix:          Base used in the traced DFS stage.
        tier:           Tier name that produced the solution.
        ops_candidates: All accepted candidate op names per operator (op_char → [op_name]).
                        When provided (with op_eqs), adds the [Accepted candidates per operator]
                        section with decoded arithmetic verification.
        op_eqs:         Encoded equation groups from build_op_groups — needed to decode
                        operand arithmetic in the accepted-candidates section.
    """
    sections: list[str] = []

    # --- Section 1: candidate setup and Python-level strategy ---
    py_lines = _narrate_py_events(py_events)
    if py_lines:
        sections.append("[Candidate setup and search strategy]")
        sections.extend(py_lines)

    # --- Section 2: DFS spine (per-digit trial log) ---
    sections.append("")
    sections.append("[Digit-assignment DFS]")
    sections.extend(_narrate_spine(spine, content_syms, sorted_ops))

    # --- Section 3: accepted candidates with arithmetic verification ---
    if ops_candidates and op_eqs and digits_mapping is not None:
        sections.append("")
        sections.append("[Accepted candidates per operator]")
        sections.extend(
            _narrate_accepted_candidates(
                op_eqs=op_eqs,
                ops_candidates=ops_candidates,
                content_syms=content_syms,
                digits_mapping=digits_mapping,
                mode=mode,
                radix=radix,
            )
        )

    # --- Section 4: solution summary ---
    sections.append("")
    sections.append("[Result]")
    sections.extend(_solution_summary(digits_mapping, ops_mapping, mode, radix, tier))

    return "\n".join(sections)


def render_concat_cot(op_glyph: str, op_name: str, answer: str) -> str:
    """Short CoT for puzzles solved by the concat fast-path."""
    direction = "forward (left‖right)" if op_name == "concat_fwd" else "reverse (right‖left)"
    return (
        f"[Concat fast-path]\n"
        f"Operator '{op_glyph}': every example equation is satisfied by {direction} "
        f"concatenation of its operand symbols — no digit search required.\n\n"
        f"[Result]\n"
        f"Answer: {answer} (pure concatenation)."
    )
