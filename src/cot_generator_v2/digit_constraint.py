"""Constraint-propagation digit picking for Fix 3 (revised).

Derives and narrates the digit-mapping entirely from per-equation allowed sets,
forced-value propagation, and progressive picking.  The oracle (first-attempt
mapping or solution) is used ONLY as a private digit selector — it never leaks
into the generated text.

Public API:
  build_allowed_table(examples, analyses, base, mode)
      → (allowed: sym->set[int], eq_allowed: (sym, eq_idx)->set[int])
  propagate(allowed)
      → dict[str, set[int]]
  order_by_allowed(allowed, all_syms, base)
      → list[str]   (fewest-options first, stable for ties)
  picking_sequence(allowed, order, oracle, base, sym_to_eqs)
      → list of pick dicts
  render_first_mapping_block(examples, allowed_pre, allowed, eq_allowed, order, picks, base,
                              analyses, mode)
      → str  (full content for <digit_mapping_1>)
  reduce_allowed_from_followup(running_allowed, step)
      → (new_allowed, removed: list[(symbol, digit)])
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    from src.cot_generator_v2.equations import Equation
    from src.cot_generator_v2.factual_observations import OperatorAnalysis
    from src.cot_generator_v2.perturbation import ChainStep


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eq_syms_unique(eq: "Equation") -> list[str]:
    """Unique symbols in an equation (l0,l1,r0,r1,rhs...) in first-seen order."""
    seen: dict[str, None] = {}
    for s in [eq.l0, eq.l1, eq.r0, eq.r1] + list(eq.rhs):
        seen.setdefault(s, None)
    return list(seen)


def _fmt_set(digits: set[int], base: int) -> str:
    """Format a digit set compactly (range form for large sets, hides holes)."""
    if not digits:
        return "∅"
    if len(digits) == base:
        return "*"
    s = sorted(digits)
    if len(s) <= 5:
        return "{" + ",".join(str(d) for d in s) + "}"
    return f"{{{s[0]}..{s[-1]}}}"


def _fmt_explicit(digits: set[int]) -> str:
    """Always list elements explicitly (no range form — shows holes after pool reduction)."""
    if not digits:
        return "∅"
    return "{" + ",".join(str(d) for d in sorted(digits)) + "}"


def _fmt_allowed_column(digits: set[int], base: int) -> str:
    """Format the 'allowed' column — full set shows as {0..N-1} instead of *."""
    if not digits:
        return "∅"
    if len(digits) == base:
        return "{0.." + str(base - 1) + "}"
    s = sorted(digits)
    if len(s) <= 5:
        return "{" + ",".join(str(d) for d in s) + "}"
    return f"{{{s[0]}..{s[-1]}}}"


# ---------------------------------------------------------------------------
# Step 1 — Build allowed-range table
# ---------------------------------------------------------------------------

def build_allowed_table(
    examples: list["Equation"],
    analyses: list["OperatorAnalysis"],
    base: int,
    mode: str,
) -> tuple[dict[str, set[int]], dict[tuple[str, int], set[int]]]:
    """Compute per-symbol allowed digit ranges from example equations.

    For each example equation, enumerate all bijection-consistent digit assignments
    to its symbols and record which (family, offset) candidates succeed. The union
    of digits appearing in succeeding assignments is the per-equation allowed range.
    Then intersect across equations to get the global allowed range per symbol.

    Returns:
      allowed:    symbol -> set of allowed digits (intersection across equations)
      eq_allowed: (symbol, eq_index) -> set of allowed digits for that equation
    """
    from src.cot_generator_v2.decode import family_matches

    # Map equation index → OperatorAnalysis
    eq_to_analysis: dict[int, "OperatorAnalysis"] = {}
    for a in analyses:
        for eq in a.eqs:
            eq_to_analysis[eq.index] = a

    eq_allowed: dict[tuple[str, int], set[int]] = {}
    sym_eq_sets: dict[str, list[set[int]]] = {}  # symbol -> [allowed set per equation]

    for eq in examples:
        syms = _eq_syms_unique(eq)
        a = eq_to_analysis.get(eq.index)
        if a is None or not a.candidate_families:
            # Unconstrained — all digits allowed
            for s in syms:
                eq_allowed[(s, eq.index)] = set(range(base))
                sym_eq_sets.setdefault(s, []).append(set(range(base)))
            continue

        sym_allowed_eq: dict[str, set[int]] = {s: set() for s in syms}
        full_count = {s: base for s in syms}  # can't have more than base values

        for perm in permutations(range(base), len(syms)):
            assignment = dict(zip(syms, perm))
            # Check if any surviving (family, offset) validates this assignment
            ok = False
            for fam in a.candidate_families:
                if ok:
                    break
                for off in a.survivors[fam]:
                    mr = family_matches(fam, off, eq, assignment, base, mode)
                    if mr.ok:
                        ok = True
                        break
            if ok:
                for s in syms:
                    sym_allowed_eq[s].add(assignment[s])
            # Early exit: if all symbols already have all possible digits, stop
            if all(len(sym_allowed_eq[s]) == base for s in syms):
                break

        for s in syms:
            eq_allowed[(s, eq.index)] = sym_allowed_eq[s]
            sym_eq_sets.setdefault(s, []).append(sym_allowed_eq[s])

    # Intersect across equations per symbol
    allowed: dict[str, set[int]] = {}
    for s, sets in sym_eq_sets.items():
        result = sets[0].copy()
        for st in sets[1:]:
            result &= st
        # Fallback: empty intersection (shouldn't happen for valid puzzles)
        allowed[s] = result if result else set(range(base))

    return allowed, eq_allowed


# ---------------------------------------------------------------------------
# Step 2 — Pre-picking propagation (forced-value propagation)
# ---------------------------------------------------------------------------

def propagate(
    allowed: dict[str, set[int]],
) -> dict[str, set[int]]:
    """Propagate forced values: if |allowed[s]| == 1, remove that digit from others.

    Simple bijection propagation — does not do full equation-level re-narrowing
    (that would require re-running build_allowed_table, which is expensive).
    Returns a NEW dict; does not mutate `allowed`.
    """
    current = {s: set(a) for s, a in allowed.items()}
    changed = True
    while changed:
        changed = False
        for s, digits in list(current.items()):
            if len(digits) == 1:
                d = next(iter(digits))
                for t, tdigits in current.items():
                    if t != s and d in tdigits:
                        tdigits.discard(d)
                        changed = True
    return current


# ---------------------------------------------------------------------------
# Step 3 — Symbol ordering and picking
# ---------------------------------------------------------------------------

def order_by_allowed(
    allowed: dict[str, set[int]],
    all_syms: list[str],
    base: int,
) -> list[str]:
    """Sort symbols fewest-options-first (stable: preserves appearance order for ties)."""
    return sorted(all_syms, key=lambda s: len(allowed.get(s, set(range(base)))))


def picking_sequence(
    allowed: dict[str, set[int]],
    order: list[str],
    oracle: dict[str, int],
    base: int,
    sym_to_eqs: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Pick digits in `order` using `oracle` to select the digit; track pool reduction.

    `oracle` is a private selector (first-attempt mapping or solution) — its
    values are never rendered directly in the CoT text.

    Returns a list of pick dicts per symbol:
      symbol          — the symbol being assigned
      digit           — the chosen digit (from oracle, within candidates)
      allowed_orig    — the full allowed set for this symbol (before pool reduction)
      candidates      — allowed_orig ∩ remaining pool at pick time
      consumed        — digits from allowed_orig already taken by prior picks
      case            — 'forced' | 'pool_forced' | 'free' | 'unconstrained'
      motivating_eq_index — first equation index that constrains this symbol (or None)

    case semantics:
      forced      — |allowed[s]| == 1: constraint analysis leaves only one option
      pool_forced — |allowed[s]| > 1 but prior picks left only 1 candidate in pool
      free        — multiple candidates remain in pool
      unconstrained — symbol not in any example equation (question-only)
    """
    if sym_to_eqs is None:
        sym_to_eqs = {}

    pool = set(range(base))
    picks = []

    for s in order:
        allowed_s = allowed.get(s, set(range(base)))
        candidates = sorted(allowed_s & pool)
        consumed = sorted(allowed_s - pool)  # taken by prior picks

        if not candidates:
            # Contradiction fallback — pick oracle value (or first in full set)
            d = oracle.get(s, sorted(allowed_s)[0] if allowed_s else 0)
            candidates = [d]

        # Oracle: select the target digit; fall back to first candidate if not available
        d = oracle.get(s, candidates[0])
        if d not in candidates:
            d = candidates[0]

        # Case classification
        in_examples = bool(sym_to_eqs.get(s))
        orig_size = len(allowed_s)

        if not in_examples:
            case = "unconstrained"
            mot_eq = None
        elif orig_size == 1:
            case = "forced"
            mot_eq = sym_to_eqs[s][0]
        elif len(candidates) == 1:
            case = "pool_forced"
            mot_eq = sym_to_eqs[s][0]
        else:
            case = "free"
            mot_eq = sym_to_eqs[s][0]

        picks.append({
            "symbol": s,
            "digit": d,
            "allowed_orig": sorted(allowed_s),
            "candidates": candidates,
            "consumed": consumed,
            "case": case,
            "motivating_eq_index": mot_eq,
        })

        pool.discard(d)

    return picks


# ---------------------------------------------------------------------------
# Analysis helpers — reason→conclusion bullets for digit_mapping_1
# ---------------------------------------------------------------------------

def _msd_of_operand(s: str, eq: "Equation", mode: str) -> str | None:
    """If `s` is the MSD of a multi-distinct-symbol operand in `eq`, return the operand
    string (e.g. 'AB'); otherwise return None.

    Standard:      MSD of left=(l0,l1) is l0; MSD of right=(r0,r1) is r0.
    Little-endian: MSD of left is l1; MSD of right is r1.
    """
    left_tok = (eq.l0, eq.l1)
    right_tok = (eq.r0, eq.r1)
    if mode == "standard":
        if s == eq.l0 and len(set(left_tok)) > 1:
            return "".join(left_tok)
        if s == eq.r0 and len(set(right_tok)) > 1:
            return "".join(right_tok)
    else:  # little_endian
        if s == eq.l1 and len(set(left_tok)) > 1:
            return "".join(left_tok)
        if s == eq.r1 and len(set(right_tok)) > 1:
            return "".join(right_tok)
    return None


def _derive_constraint_observations(
    examples: list["Equation"],
    eq_allowed: dict[tuple[str, int], set[int]],
    allowed_pre: dict[str, set[int]],
    order: list[str],
    analyses: list["OperatorAnalysis"],
    base: int,
    mode: str,
) -> list[str]:
    """Generate ~3–6 reason→conclusion bullets for the digit_mapping_1 analysis section.

    Three rule types:
      A — No-leading-zero: symbol is MSD of a multi-distinct-symbol operand and
          0 is excluded (but no operator upper bound tightens the cap further).
      B — Operator upper-bound: max(eq_allowed) < base-1 (or 0 excluded but not
          by a simple operand-position rule); attribute to surviving candidate families.
      C — Unconstrained: symbol's final allowed set is the full digit range.
    """
    # Build eq.index → OperatorAnalysis map
    eq_to_analysis: dict[int, "OperatorAnalysis"] = {}
    for a in analyses:
        for eq in a.eqs:
            eq_to_analysis[eq.index] = a

    # Classify each constrained (sym, eq) pair
    no_lz_by_sym: dict[str, list[tuple[int, str]]] = {}   # sym → [(eq_idx, operand_str)]
    op_bound_by_eq: dict[int, list[tuple[str, set[int]]]] = {}  # eq_idx → [(sym, aset)]

    for s in order:
        for eq in examples:
            key = (s, eq.index)
            if key not in eq_allowed:
                continue
            aset = eq_allowed[key]
            if len(aset) == base:
                continue  # fully unconstrained cell — skip

            zero_excluded = 0 not in aset
            upper_bounded = bool(aset) and max(aset) < base - 1
            token = _msd_of_operand(s, eq, mode)

            if zero_excluded and token is not None and not upper_bounded:
                # Clean no-leading-zero case: operand MSD, only 0 removed
                no_lz_by_sym.setdefault(s, []).append((eq.index, token))
            elif len(aset) <= base - 2:
                # Significant operator constraint: at least 2 values excluded from full set
                op_bound_by_eq.setdefault(eq.index, []).append((s, set(aset)))
            # else: ≤1 value excluded — too weak to warrant a bullet; appears correctly in table

    bullets: list[str] = []

    # --- Bullet A: no-leading-zero operands (grouped by symbol, dedup same token) ---
    if no_lz_by_sym:
        parts: list[str] = []
        for s in order:
            if s not in no_lz_by_sym:
                continue
            tok_eqs: dict[str, list[int]] = {}
            for eq_idx, tok in no_lz_by_sym[s]:
                tok_eqs.setdefault(tok, []).append(eq_idx)
            for tok, eq_idxs in tok_eqs.items():
                eq_list = " and ".join(f"eq[{i}]" for i in sorted(eq_idxs))
                parts.append(f"`{s}` leads `{tok}` in {eq_list}")
        joined = "; ".join(parts)
        bullets.append(
            f"{joined} → each leads a multi-symbol operand with distinct characters; "
            f"no operand carries a leading zero → 0 is excluded "
            f"(per-equation constraint: {{1..{base - 1}}})."
        )

    # --- Bullet B: operator-based constraints (one bullet per equation) ---
    for eq_idx in sorted(op_bound_by_eq):
        items = op_bound_by_eq[eq_idx]
        eq_obj = next((e for e in examples if e.index == eq_idx), None)
        if eq_obj is None:
            continue
        a = eq_to_analysis.get(eq_idx)
        fams = a.candidate_families if a else []
        fam_str = (
            "{" + ", ".join(fams[:4]) + ("…" if len(fams) > 4 else "") + "}"
            if fams else "all candidates"
        )
        sym_parts = [f"`{s}` ∈ {_fmt_explicit(aset)}" for s, aset in items]
        sym_joined = ", ".join(sym_parts)
        bullets.append(
            f"In eq[{eq_idx}] (operator `{eq_obj.op}`, surviving candidates: {fam_str}), "
            f"the set of assignments consistent with any candidate constrains: {sym_joined}."
        )

    # --- Closing bullet: unconstrained symbols ---
    unconstrained = [s for s in order if len(allowed_pre.get(s, set(range(base)))) == base]
    if unconstrained:
        sym_list = ", ".join(f"`{s}`" for s in unconstrained)
        bullets.append(
            f"Symbols {sym_list} appear in positions where no candidate operator "
            f"restricts their value → unconstrained ({{0..{base - 1}}})."
        )

    return bullets


# ---------------------------------------------------------------------------
# Step 4 — Render the full first-mapping block
# ---------------------------------------------------------------------------

def _fmt_eq_set(digits: set[int], base: int, in_table: bool = True) -> str:
    """Format a per-equation allowed set (table cell form)."""
    if not digits:
        return "∅"
    if len(digits) == base:
        return "*"
    s = sorted(digits)
    if len(s) <= 5:
        cell = "{" + ",".join(str(d) for d in s) + "}"
    else:
        cell = f"{{{s[0]}..{s[-1]}}}"
    return cell


def _render_range_table_internal(
    allowed_pre: dict[str, set[int]],
    eq_allowed: dict[tuple[str, int], set[int]],
    order: list[str],
    examples: list["Equation"],
    base: int,
) -> str:
    """Render the symbol × equation allowed-set table (pre-propagation allowed)."""
    eq_indices = [e.index for e in examples]
    rows = []
    for s in order:
        parts = []
        for eq in examples:
            key = (s, eq.index)
            if key in eq_allowed:
                parts.append(_fmt_eq_set(eq_allowed[key], base))
            else:
                parts.append("—")
        g = allowed_pre.get(s, set(range(base)))
        g_str = _fmt_allowed_column(g, base)
        eq_col = "  ".join(f"{p:>12}" for p in parts)
        rows.append(f"  {s:4s}  {eq_col}  →  {g_str}")

    header_cols = "  ".join(f"{'eq['+str(i)+']':>12}" for i in eq_indices)
    header_line = f"  {'sym':4s}  {header_cols}  →  allowed"
    return header_line + "\n" + "\n".join(rows)


def render_first_mapping_block(
    examples: list["Equation"],
    allowed_pre: dict[str, set[int]],
    allowed: dict[str, set[int]],
    eq_allowed: dict[tuple[str, int], set[int]],
    order: list[str],
    picks: list[dict],
    base: int,
    analyses: "list[OperatorAnalysis] | None" = None,
    mode: str = "standard",
) -> str:
    """Render the full reasoning for <digit_mapping_1>.

    Layout:
      1. Reason→conclusion analysis bullets (structural + operator constraints)
      2. Per-symbol × per-equation allowed-set table
      3. Forced-value propagation (emitted only when something propagates)
      4. Progressive picking from the pool
      5. Resulting first mapping
    """
    lines: list[str] = []

    # --- 1. Analysis bullets ----------------------------------------------------
    if analyses:
        obs = _derive_constraint_observations(
            examples, eq_allowed, allowed_pre, order, analyses, base, mode
        )
        if obs:
            lines.append(
                "Let's analyze the possible constraints for the digits given the operator "
                "candidates, example equations, and structural position of the operand "
                "symbols across the system of equations:\n"
            )
            for b in obs:
                lines.append(f"- {b}")
            lines.append("")

    # --- 2. Range table ---------------------------------------------------------
    lines.append(
        "Each cell shows which digits are consistent with that equation; "
        "'*' means unconstrained, '—' means the symbol does not appear.\n"
    )
    lines.append(_render_range_table_internal(allowed_pre, eq_allowed, order, examples, base))

    # --- 3. Propagation (only when something actually changed) ------------------
    forced_pre = {s: next(iter(allowed_pre[s])) for s in order
                  if s in allowed_pre and len(allowed_pre[s]) == 1}
    propagation_effects: list[str] = []
    for s in forced_pre:
        d = forced_pre[s]
        victims = [t for t in order if t != s
                   and d in allowed_pre.get(t, set())
                   and d not in allowed.get(t, {d})]
        if victims:
            propagation_effects.append(
                f"  {s} is forced to {d} — removed {d} from: {', '.join(victims)}."
            )

    if propagation_effects:
        lines.append("\nForced-value propagation:")
        lines.extend(propagation_effects)
        changed = [(s, allowed_pre[s], allowed[s])
                   for s in order
                   if allowed_pre.get(s) != allowed.get(s, allowed_pre.get(s))]
        if changed:
            updates = ", ".join(
                f"{s}: {_fmt_set(old, base)} → {_fmt_set(new, base)}"
                for s, old, new in changed
            )
            lines.append(f"  Updated: {updates}.")

    # --- 4. Progressive picking -------------------------------------------------
    full_pool_str = "{" + ",".join(str(d) for d in range(base)) + "}"
    lines.append(
        f"\nPicking from the initial pool {full_pool_str}, "
        f"starting with the symbol that has the fewest allowed options:"
    )
    pool_display = set(range(base))

    for i, p in enumerate(picks, 1):
        s = p["symbol"]
        d = p["digit"]
        orig = p["allowed_orig"]
        consumed = p["consumed"]

        orig_str = _fmt_allowed_column(set(orig), base)
        pool_display.discard(d)
        pool_after = _fmt_explicit(pool_display)

        if consumed:
            consumed_str = _fmt_explicit(set(consumed))
            line = (f"  {i}. {s} → reduce {consumed_str} from allowed {orig_str} "
                    f"→ pick randomly {d} → pool → {pool_after}")
        else:
            line = (f"  {i}. {s} → allowed: {orig_str} "
                    f"→ pick randomly {d} → pool → {pool_after}")
        lines.append(line)

    # --- 5. Resulting assignment -------------------------------------------------
    lines.append("\nResulting first mapping attempt:")
    for p in picks:
        lines.append(f"  {p['symbol']} = {p['digit']}")

    return "\n".join(lines)


def render_first_mapping_block_occurrences(occ_text: str, picks: list[dict], base: int) -> str:
    """Render <digit_mapping_1> from occurrence ordering + shared-pool picking (no allowed).

    `occ_text` is the pre-rendered occurrence table from compute_digit_priority.
    `picks` comes from picking_sequence called with a full-pool allowed map.
    """
    lines = [occ_text, ""]
    full_pool_str = "{" + ",".join(str(d) for d in range(base)) + "}"
    lines.append(
        f"Picking from the initial pool {full_pool_str} in occurrence order "
        "(each pick is removed from the pool):"
    )
    pool_display = set(range(base))
    for i, p in enumerate(picks, 1):
        s, d = p["symbol"], p["digit"]
        pool_display.discard(d)
        lines.append(
            f"  {i}. {s} → pick randomly {d} from pool → pool → {_fmt_explicit(pool_display)}"
        )
    lines.append("\nResulting first mapping attempt:")
    for p in picks:
        lines.append(f"  {p['symbol']} = {p['digit']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 5 — Progressive allowed-set reduction for subsequent mapping blocks
# ---------------------------------------------------------------------------

def reduce_allowed_from_followup(
    running_allowed: dict[str, set[int]],
    step: "ChainStep",
) -> tuple[dict[str, set[int]], list[tuple[str, int]]]:
    """Remove digits shown infeasible by step's fix(es) from running_allowed.

    For each fix symbol, the digit tried in step.mapping[s] is now known bad.
    Returns a new (copied) reduced dict and the list of (symbol, digit) removed.
    """
    new_allowed = {s: set(ds) for s, ds in running_allowed.items()}
    removed: list[tuple[str, int]] = []
    for fix in (step.fixes or []):
        for s in fix.symbols:
            d = step.mapping.get(s)
            if d is not None and s in new_allowed and d in new_allowed[s]:
                new_allowed[s].discard(d)
                removed.append((s, d))
    return new_allowed, removed
