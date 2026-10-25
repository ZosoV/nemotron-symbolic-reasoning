"""Equation-Symbolic Solver.

Entry point:
    uv run python src/solver/solver_eq_symbolic.py [config.yaml]

Reads a .parquet with columns (id, prompt, answer), infers a latent program
for each puzzle, and writes solver_results.parquet.

Public re-exports (for eval_from_parquet.py and other consumers):
    OPERATIONS, SIGNED_OPS, OP_NAME_TO_IDX,
    _int_to_base_digits, _is_reversed_digit_mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.solver.config import SolverConfig, load_config
from src.encoding import (
    _int_to_base_digits,
    _is_reversed_digit_mode,
    encode_answer,
    verify_program,
)
from src.solver.heuristics import (
    concat_optin,
    op_glyph_priority,
    order_ops,
    sign_pattern_filter,
    tier_skip,
)
from src.op_index import OP_IDX_TO_NAME, OP_NAME_TO_IDX, SIGNED_OPS
from src.operations import (
    OPERATIONS,
    TIER0,
    TIER1,
    TIER2,
    TIER_FULL,
)
from src.parsing import (
    build_op_groups,
    content_symbols,
    extract_all_equations,
)

# ---------------------------------------------------------------------------
# Tier name → set mapping
# ---------------------------------------------------------------------------

_TIER_SETS: dict[str, set[str]] = {
    "tier0": TIER0,
    "tier1": TIER1,
    "tier2": TIER2,
    "full": TIER_FULL,
}

_TIER_LABEL: dict[str, str] = {
    "tier0": "tier0",
    "tier1": "tier1",
    "tier2": "tier2",
    "full": "full",
}


# ---------------------------------------------------------------------------
# Rust bridge
# ---------------------------------------------------------------------------

def _stage_seed(base_seed: int, content_syms: list[str], mode: str, radix: int) -> int:
    """Derive a per-puzzle/per-stage seed from the base seed.

    Mixes the base_seed with a stable (deterministic across Python processes) hash of the
    symbol set, mode, and radix so that every (puzzle, mode, radix) triple gets a distinct
    shuffle while remaining fully reproducible given the same base_seed.

    Uses SHA-256 truncated to 64 bits for a stable, PYTHONHASHSEED-independent hash.
    """
    import hashlib
    key = f"{','.join(content_syms)}|{mode}|{radix}".encode()
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    # xorshift mix to combine base_seed and h
    s = (base_seed ^ (h * 0x9E3779B97F4A7C15)) & 0xFFFF_FFFF_FFFF_FFFF
    return s


def _candidate_seed(base_seed: int, content_syms: list[str], op_char: str) -> int:
    """Derive a per-puzzle/per-operator seed for candidate-list shuffling.

    Combines base_seed with a stable hash of the symbol set and operator character so
    each (puzzle, operator) pair gets a distinct, reproducible shuffle regardless of
    PYTHONHASHSEED.  Modeled on _stage_seed.
    """
    import hashlib
    key = f"{','.join(content_syms)}|{op_char}".encode()
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    s = (base_seed ^ (h * 0x9E3779B97F4A7C15)) & 0xFFFF_FFFF_FFFF_FFFF
    return s


def _search_traced(
    op_eqs: dict[str, list[tuple]],
    tier_cands: dict[str, list[str]],
    sorted_ops: list[str],
    n: int,
    mode: str,
    radix: int,
    digit_count: int,
    max_solutions: int = 1,
) -> tuple[list[tuple[list, list]], list[tuple[int, int, list, list]]]:
    """Like _search but uses arithmetic_search_traced and returns (results, spine).

    Always uses canonical weight-priority variable ordering (no randomize).
    spine is a list of (sym_idx, accepted_digit, pruned, backtracked) — one entry
    per symbol in the accepted assignment chain (first solution). Empty when no
    solution is found. See cot_search_trace.py for the per-step rendering.
    """
    from alice_sovler_helper import arithmetic_search_traced  # type: ignore[import]

    mode_int = 0 if mode == "standard" else 2

    groups = []
    for op_char in sorted_ops:
        cand_indices = [OP_NAME_TO_IDX[name] for name in tier_cands[op_char]]
        eqs_packed = []
        for l0, l1, r0, r1, res_syms, has_sign, res_len in op_eqs[op_char]:
            packed = [l0, l1, r0, r1, res_len, int(has_sign)] + list(res_syms)
            eqs_packed.append(packed)
        groups.append((cand_indices, eqs_packed))

    return arithmetic_search_traced(
        n, False, groups, radix, digit_count, mode_int, max_solutions
    )


def _search_full_trace(
    op_eqs: dict[str, list[tuple]],
    tier_cands: dict[str, list[str]],
    sorted_ops: list[str],
    n: int,
    mode: str,
    radix: int,
    digit_count: int,
    max_events: int = 0,
) -> tuple[
    tuple[list, list] | None,        # (perm, ops_valid) or None
    list[tuple[int, int]],           # var_order_scores: [(sym_idx, score), ...]
    list[tuple],                     # events raw from Rust
    int,                             # total_trials (digit_mappings_analyzed)
    bool,                            # truncated
]:
    """Wrap arithmetic_search_full_trace and return the raw Rust output.

    Returns a 5-tuple: (solution_opt, var_order_scores, events, total_trials, truncated).
    solution_opt is (perm, ops_valid) for the first solution, or None.
    events is a list of (depth, sym_idx, digit, passed, failed_group,
    surviving_per_group, on_solution_path) tuples.
    """
    from alice_sovler_helper import arithmetic_search_full_trace  # type: ignore[import]

    mode_int = 0 if mode == "standard" else 2

    groups = []
    for op_char in sorted_ops:
        cand_indices = [OP_NAME_TO_IDX[name] for name in tier_cands[op_char]]
        eqs_packed = []
        for l0, l1, r0, r1, res_syms, has_sign, res_len in op_eqs[op_char]:
            packed = [l0, l1, r0, r1, res_len, int(has_sign)] + list(res_syms)
            eqs_packed.append(packed)
        groups.append((cand_indices, eqs_packed))

    return arithmetic_search_full_trace(
        n, False, groups, radix, digit_count, mode_int, max_events
    )


def _search(
    op_eqs: dict[str, list[tuple]],
    tier_cands: dict[str, list[str]],
    sorted_ops: list[str],
    n: int,
    mode: str,
    radix: int,
    digit_count: int,
    max_solutions: int = 1,
    randomize: bool = False,
    seed: int = 0,
) -> list[tuple[list, list]]:
    """Call the Rust arithmetic_search core and return a list of (perm, ops_valid) tuples.

    Returns up to *max_solutions* distinct digit assignments. Returns an empty list when
    no assignment satisfies the constraints.

    Args:
        op_eqs:         {op_char: [encoded equation tuples]}
        tier_cands:     {op_char: [op names filtered to current tier]}
        sorted_ops:     op chars in MRV-first order (Heuristic #5)
        n:              number of content symbols
        mode:           'standard' | 'little_endian'
        radix:          positional base for arithmetic
        digit_count:    digit pool size (== radix for normal case)
        max_solutions:  stop after collecting this many digit assignments
        randomize:      when True, shuffle symbol-assignment and digit-iteration orders
                        inside the Rust DFS using the seeded PRNG (see seed)
        seed:           seed for the xorshift64 PRNG used when randomize=True
    """
    from alice_sovler_helper import arithmetic_search  # type: ignore[import]

    mode_int = 0 if mode == "standard" else 2  # standard=0, little_endian=2

    groups = []
    for op_char in sorted_ops:
        cand_indices = [OP_NAME_TO_IDX[name] for name in tier_cands[op_char]]
        eqs_packed = []
        for l0, l1, r0, r1, res_syms, has_sign, res_len in op_eqs[op_char]:
            packed = [l0, l1, r0, r1, res_len, int(has_sign)] + list(res_syms)
            eqs_packed.append(packed)
        groups.append((cand_indices, eqs_packed))

    return arithmetic_search(
        n, False, groups, radix, digit_count, mode_int, max_solutions, randomize, seed
    )


# ---------------------------------------------------------------------------
# Concat fast-path (spec §2.2)
# ---------------------------------------------------------------------------

def _try_concat(equations: list[dict]) -> tuple[str, str] | None:
    """Check whether every equation with the query's operator is pure concat.

    Returns ``(answer, op_name)`` if the puzzle is solved by concatenation,
    where ``op_name`` is ``"concat_fwd"`` or ``"concat_rev"``, or ``None`` to
    fall through to arithmetic search.
    """
    to_predict = equations[-1]
    op_of_interest = to_predict["lhs"]["op"]

    # Condition: no sign in query rhs AND exactly 4 rhs symbols.
    if to_predict["rhs"]["op"] is not None:
        return None
    if len(to_predict["rhs"]["actual_rhs"]) != 4:
        return None

    # Only use example equations (not to_predict itself) as evidence.
    relevant = [eq for eq in equations[:-1] if eq["lhs"]["op"] == op_of_interest]
    if not relevant:
        return None

    # Try forward concat: rhs == [l0, l1, r0, r1]
    fwd_ok = True
    for eq in relevant:
        lhs = eq["lhs"]
        expected = [lhs["l0"], lhs["l1"], lhs["r0"], lhs["r1"]]
        if eq["rhs"]["actual_rhs"] != expected:
            fwd_ok = False
            break
    if fwd_ok:
        q = to_predict["lhs"]
        return q["l0"] + q["l1"] + q["r0"] + q["r1"], "concat_fwd"

    # Try reverse concat: rhs == [r0, r1, l0, l1]
    rev_ok = True
    for eq in relevant:
        lhs = eq["lhs"]
        expected = [lhs["r0"], lhs["r1"], lhs["l0"], lhs["l1"]]
        if eq["rhs"]["actual_rhs"] != expected:
            rev_ok = False
            break
    if rev_ok:
        q = to_predict["lhs"]
        return q["r0"] + q["r1"] + q["l0"] + q["l1"], "concat_rev"

    return None


# ---------------------------------------------------------------------------
# Full-trace helpers
# ---------------------------------------------------------------------------

def _spine_from_full_trace(
    events: list[tuple],
    solution: tuple | None,
) -> list[tuple]:
    """Convert full-trace events to the compact spine format expected by render_search_cot.

    The spine format is [(sym_idx, accepted_digit, pruned, backtracked), ...] — one entry
    per symbol in assignment order along the solution path.

    pruned      = [(digit, group_idx), ...]  — forward-check rejections before accepted
    backtracked = [digit, ...]               — digits that passed forward-check but subtree failed

    When solution is None (no solution found), returns [].
    """
    if solution is None or not events:
        return []

    # Collect solution-path events in depth order.
    path_events = [ev for ev in events if ev[6]]  # on_solution_path == True
    # Sort by depth (should already be in order, but be safe).
    path_events.sort(key=lambda ev: ev[0])

    if not path_events:
        return []

    # For each depth, collect: the accepted digit (on_solution_path=True) and all
    # events at that depth that precede it (pruned or backtracked).
    from collections import defaultdict
    by_depth: dict[int, list[tuple]] = defaultdict(list)
    for ev in events:
        by_depth[ev[0]].append(ev)

    spine = []
    for path_ev in path_events:
        depth, sym_idx, accepted_digit = path_ev[0], path_ev[1], path_ev[2]
        pruned: list[tuple[int, int]] = []
        backtracked: list[int] = []
        for ev in by_depth[depth]:
            d = ev[2]
            if d == accepted_digit:
                continue
            passed = ev[3]
            failed_group = ev[4]
            if not passed:
                pruned.append((d, failed_group))
            else:
                backtracked.append(d)
        spine.append((sym_idx, accepted_digit, pruned, backtracked))

    return spine


def _write_search_trace_json(
    ft_raw: dict,
    content_syms: list[str],
    sorted_ops: list[str],
    problem_id: str | None,
    out_path: "Path",
) -> None:
    """Stream the search-trace JSON directly to disk one node at a time.

    Avoids holding the full nodes list or the serialised string in memory —
    peak extra allocation is O(DFS depth) rather than O(total nodes).

    Schema is identical to what _build_search_trace_json used to return.
    """
    var_order_scores: list[tuple[int, int]] = ft_raw["var_order_scores"]
    events: list[tuple] = ft_raw["events"]
    total_trials: int = ft_raw["total_trials"]
    truncated: bool = ft_raw["truncated"]
    tier_cands: dict[str, list[str]] = ft_raw["tier_cands"]

    var_order = [
        {"glyph": content_syms[sym_idx], "score": int(score)}
        for sym_idx, score in var_order_scores
    ]
    initial_candidates = {op: list(cands) for op, cands in tier_cands.items()}
    init_surviving_per_glyph = {op: list(cands) for op, cands in tier_cands.items()}

    def idx_list_to_names(idxs: list[int]) -> list[str]:
        return [OP_IDX_TO_NAME[int(i)] for i in idxs]

    header = {
        "problem_id": problem_id,
        "var_order": var_order,
        "initial_candidates": initial_candidates,
        "count": total_trials,
        "truncated": truncated,
    }

    mapping_stack: list[dict[str, int]] = [{}]

    with out_path.open("w", encoding="utf-8") as f:
        # Write all header fields then open the nodes array inline.
        header_str = json.dumps(header, ensure_ascii=False)
        f.write(header_str[:-1])  # strip trailing '}'
        f.write(', "nodes": [')

        first = True
        for ev in events:
            depth, sym_idx, digit, passed, failed_group, surviving_per_group, on_sol = ev
            depth = int(depth)
            sym_idx = int(sym_idx)
            digit = int(digit)
            failed_group = int(failed_group)
            on_sol = bool(on_sol)

            glyph = content_syms[sym_idx]

            while len(mapping_stack) <= depth:
                mapping_stack.append(dict(mapping_stack[-1]))

            parent = mapping_stack[depth] if depth < len(mapping_stack) else {}
            partial = {**parent, glyph: digit}

            surviving: dict[str, list[str]] = {}
            for i, op_char in enumerate(sorted_ops):
                surviving[op_char] = idx_list_to_names(surviving_per_group[i]) if i < len(surviving_per_group) else []

            eliminated: dict[str, list[str]] = {
                op: [c for c in init_surviving_per_glyph.get(op, []) if c not in set(surviving.get(op, []))]
                for op in sorted_ops
            }

            pruned_glyph = sorted_ops[failed_group] if 0 <= failed_group < len(sorted_ops) else None

            node = {
                "depth": depth,
                "glyph": glyph,
                "digit": digit,
                "partial_mapping": dict(partial),
                "passed": passed,
                "pruned_glyph": pruned_glyph,
                "surviving": surviving,
                "eliminated": eliminated,
                "on_solution_path": on_sol,
            }

            if not first:
                f.write(",")
            f.write(json.dumps(node, ensure_ascii=False))
            first = False

            if passed:
                if len(mapping_stack) <= depth + 1:
                    mapping_stack.append(partial)
                else:
                    mapping_stack[depth + 1] = partial

        f.write("]}")


# ---------------------------------------------------------------------------
# Arithmetic search (spec §2.4)
# ---------------------------------------------------------------------------

def _try_arithmetic(
    equations: list[dict],
    content_syms: list[str],
    sym2i: dict[str, int],
    gold_answer: str,
    config: SolverConfig,
) -> tuple[list[tuple[str, dict]], str | None, str | None, dict | None]:
    """Run the tiered arithmetic search and return (solutions, search_reached, search_cot, ft_raw).

    The to_predict equation (last in equations) is included as a constraint
    so the search is gold-conditioned.

    Collects up to config.max_solutions fully-verified programs. Each program is a
    distinct (digit mapping, mode, radix) combination. The search escalates in order:
      1. distinct digit mappings under 'standard' mode and the primary base
      2. 'little_endian' mode (same base)
      3. alternative base (when not force_base_10)

    search_reached reflects the furthest stage the loop advanced into:
      'digits'       — stayed within standard + primary base
      'little_endian'— advanced into little_endian (same base)
      'alt_base'     — advanced into a non-primary base

    search_cot is a prose chain-of-thought narrating the search path to the first
    solution (both Python-layer decisions and the Rust DFS trace). None when
    config.generate_cot is False or no solution was found.

    ft_raw is a dict of raw full-trace data from the first solution-producing stage,
    populated when config.generate_search_trace is True and a solution was found.
    None otherwise. The caller (solve_sample) decodes it and writes the JSON file.

    Returns ([], None, None, None) when no solution is found, falling back to
    ([first_partial], None, None, None) when a partial candidate exists (encode_answer
    succeeded but verify failed), so the caller can surface it as wrong_answer.
    """
    base = len(content_syms)

    # Accumulate Python-layer narration events when CoT is enabled.
    py_events: list[dict] = []

    # Step A: Build operator groups (includes to_predict as a constraint).
    op_eqs = build_op_groups(equations, sym2i)

    # Step B: Per-op candidate lists.
    op_candidates: dict[str, list[str]] = {}
    for op_char, eqs in op_eqs.items():
        if "op_glyph_priority" not in config.disabled_heuristics:
            cands = op_glyph_priority(op_char)        # #3
            initial_cands = list(cands)
        else:
            cands = list(OPERATIONS.keys())
            initial_cands = list(cands)

        changes: list[str] = []
        if "sign_pattern_filter" not in config.disabled_heuristics:
            before = len(cands)
            cands = sign_pattern_filter(cands, eqs)   # #2
            if len(cands) != before:
                changes.append(f"sign_pattern_filter ({before}→{len(cands)})")
        if "concat_optin" not in config.disabled_heuristics:
            before = len(cands)
            cands = concat_optin(cands, eqs)           # #4
            if len(cands) != before:
                changes.append(f"concat_optin ({before}→{len(cands)})")

        op_candidates[op_char] = cands
        if config.generate_cot:
            py_events.append({
                "type": "candidates",
                "op_char": op_char,
                "initial": initial_cands,
                "final": list(cands),
                "changes": changes,
            })

    # Step B': drop operations excluded via config.
    if config.filter_out_operations:
        blocked = set(config.filter_out_operations)
        op_candidates = {
            op: [c for c in cands if c not in blocked]
            for op, cands in op_candidates.items()
        }

    # Step B'': randomize candidate priority order (optional).
    if config.randomize_candidates:
        import random
        op_candidates = {
            op: random.Random(
                _candidate_seed(config.random_seed, content_syms, op)
            ).sample(cands, len(cands))
            for op, cands in op_candidates.items()
        }

    # Step C: Op ordering.
    if "order_ops" not in config.disabled_heuristics:
        sorted_ops = order_ops(op_eqs, op_candidates)  # #5
    else:
        sorted_ops = list(op_eqs.keys())

    if config.generate_cot:
        py_events.append({
            "type": "order_ops",
            "order": list(sorted_ops),
            "stats": [
                (op, len(op_candidates.get(op, [])), len(op_eqs.get(op, [])))
                for op in sorted_ops
            ],
        })

    # Steps D/E: Tiered search loop.
    tiers_to_try = [(name, _TIER_SETS[name]) for name in config.tiers]

    # Collect verified solutions; dedup key = (frozen digits, mode, radix).
    solutions: list[tuple[str, dict]] = []
    seen: set[tuple] = set()

    # First partial candidate found (encode_answer succeeded but verify failed).
    # Returned as wrong_answer only after all options are exhausted.
    first_partial: tuple[str, dict] | None = None

    # Track the furthest escalation stage the loop advances into.
    search_reached: str | None = None

    # For CoT: capture the Rust spine from the first stage that yields a solution.
    cot_spine: list[tuple[int, int, list, list]] = []
    cot_stage_mode: str = "standard"
    cot_stage_radix: int = 10
    cot_stage_tier: str | None = None
    cot_first_solution: dict | None = None  # details dict of the traced solution

    # For full-trace: raw Rust output from the first solution-producing stage.
    ft_raw: dict | None = None

    for tier_name, tier_set in tiers_to_try:
        if len(solutions) >= config.max_solutions:
            break

        tier_cands = {
            op: [c for c in cands if c in tier_set]
            for op, cands in op_candidates.items()
        }

        if "tier_skip" not in config.disabled_heuristics and tier_skip(tier_cands):  # #9
            if config.generate_cot:
                # Find which op caused the skip (first empty group).
                empty_op = next(
                    (op for op, cs in tier_cands.items() if not cs), None
                )
                if empty_op:
                    py_events.append({
                        "type": "tier_skip",
                        "tier": tier_name,
                        "op_char": empty_op,
                    })
            continue

        if config.force_base_10:
            radix_candidates = [(10, 10)]
        else:
            radix_candidates = [(base, base)]
            if base != 10:
                radix_candidates.append((10, 10))

        primary_radix = radix_candidates[0][0]

        for radix, digit_count in radix_candidates:
            if len(solutions) >= config.max_solutions:
                break

            for mode in ("standard", "little_endian"):
                if len(solutions) >= config.max_solutions:
                    break

                # Update search_reached to the furthest stage entered.
                if radix != primary_radix:
                    stage = "alt_base"
                elif mode == "little_endian":
                    stage = "little_endian"
                else:
                    stage = "digits"
                if search_reached != "alt_base":
                    if stage == "alt_base" or (
                        stage == "little_endian" and search_reached == "digits"
                    ) or search_reached is None:
                        search_reached = stage

                if config.generate_cot:
                    py_events.append({
                        "type": "stage_enter",
                        "tier": tier_name,
                        "mode": mode,
                        "radix": radix,
                    })

                remaining = config.max_solutions - len(solutions)
                stage_quota = remaining
                mdt = (
                    config.max_digit_trials_little_endian
                    if mode == "little_endian"
                    else config.max_digit_trials_standard
                )
                if mdt is not None:
                    stage_quota = min(remaining, mdt)

                # Choose the right Rust back-end for this stage.
                # Priority: full-trace (generate_search_trace) > CoT-traced > plain.
                # Both traced variants only run for the *first* solution-producing stage
                # (ft_raw / cot_spine are falsy until then).
                stage_spine: list = []
                if config.generate_search_trace and ft_raw is None:
                    # Full branch-by-branch trace — also feeds the CoT spine when enabled.
                    max_ev = config.search_trace_max_nodes or 0
                    ft_sol, ft_scores, ft_events, ft_count, ft_trunc = _search_full_trace(
                        op_eqs, tier_cands, sorted_ops, base, mode, radix, digit_count,
                        max_events=max_ev,
                    )
                    if ft_sol is not None:
                        results = [ft_sol]
                        # Stash raw output for decode + JSON write in solve_sample.
                        ft_raw = {
                            "var_order_scores": ft_scores,
                            "events": ft_events,
                            "total_trials": ft_count,
                            "truncated": ft_trunc,
                            "tier": tier_name,
                            "mode": mode,
                            "radix": radix,
                            # tier_cands keyed in sorted_ops order for deterministic output.
                            "tier_cands": {op: list(tier_cands[op]) for op in sorted_ops},
                            "sorted_ops": list(sorted_ops),
                        }
                    else:
                        results = []
                    # When CoT is also requested, build spine from the full-trace events.
                    if config.generate_cot and not cot_spine and ft_events:
                        stage_spine = _spine_from_full_trace(ft_events, ft_sol)
                elif config.generate_cot and not cot_spine:
                    results, stage_spine = _search_traced(
                        op_eqs, tier_cands, sorted_ops, base, mode, radix, digit_count,
                        max_solutions=stage_quota,
                    )
                else:
                    results = _search(
                        op_eqs, tier_cands, sorted_ops, base, mode, radix, digit_count,
                        max_solutions=stage_quota,
                        randomize=config.randomize_search,
                        seed=_stage_seed(config.random_seed, content_syms, mode, radix),
                    )

                for perm, ops_valid_indices in results:
                    # Decode mapping: symbol -> digit.
                    mapping: dict[str, int] = {
                        sym: int(perm[sym2i[sym]]) for sym in content_syms
                    }

                    # Dedup: skip if we've already recorded this (mapping, mode, radix).
                    dedup_key = (frozenset(mapping.items()), mode, radix)
                    if dedup_key in seen:
                        continue

                    # Decode surviving op candidates per group.
                    ops_candidates_found: dict[str, list[str]] = {
                        op_char: [OP_IDX_TO_NAME[int(idx)] for idx in ops_valid_indices[i]]
                        for i, op_char in enumerate(sorted_ops)
                    }
                    ops_mapping: dict[str, str] = {
                        op_char: ops_candidates_found[op_char][0]
                        for op_char in sorted_ops
                    }

                    # Step F: Encode the query answer.
                    to_predict = equations[-1]
                    encoded = encode_answer(
                        mapping=mapping,
                        ops=ops_mapping,
                        op_char=to_predict["lhs"]["op"],
                        query_lhs=to_predict["lhs"],
                        mode=mode,
                        radix=radix,
                        answer_hint=gold_answer,
                    )
                    if encoded is None:
                        continue

                    answer_str, _ = encoded
                    strategy = mode  # "standard" | "little_endian"

                    details = {
                        "solver_strategy": strategy,
                        "ops_mapping": ops_mapping,
                        "digits_mapping": mapping,
                        "solver_tier": tier_name,
                        "solver_radix": radix,
                        "ops_candidates": ops_candidates_found,
                    }

                    # Full-program verification: every equation (examples + to_predict)
                    # must hold under the found program.  When verification fails, keep
                    # searching in case a later combination gives a fully consistent
                    # solution.  Only fall back to first partial when all options are
                    # exhausted.
                    if not verify_program(equations, mapping, ops_mapping, mode, radix):
                        if first_partial is None:
                            first_partial = (answer_str, {**details, "_verify_failed": True})
                        continue

                    seen.add(dedup_key)
                    solutions.append((answer_str, details))
                    # Capture CoT spine on the first verified solution from the traced stage.
                    if (
                        config.generate_cot
                        and stage_spine
                        and not cot_spine
                        and len(solutions) == 1
                    ):
                        cot_spine = list(stage_spine)
                        cot_stage_mode = mode
                        cot_stage_radix = radix
                        cot_stage_tier = tier_name
                        cot_first_solution = details
                    if len(solutions) >= config.max_solutions:
                        break

                if config.generate_cot:
                    py_events.append({
                        "type": "stage_result",
                        "tier": tier_name,
                        "mode": mode,
                        "radix": radix,
                        "n_solutions": len(results),
                    })

    # Render the search CoT if enabled.
    search_cot: str | None = None
    if config.generate_cot and solutions:
        from src.solver.cot_search_trace import render_search_cot
        first_details = cot_first_solution or solutions[0][1]
        search_cot = render_search_cot(
            py_events=py_events,
            spine=cot_spine,
            content_syms=content_syms,
            sorted_ops=sorted_ops,
            digits_mapping=first_details.get("digits_mapping"),
            ops_mapping=first_details.get("ops_mapping"),
            mode=cot_stage_mode,
            radix=cot_stage_radix,
            tier=cot_stage_tier,
            ops_candidates=first_details.get("ops_candidates"),
            op_eqs=op_eqs,
        )

    if solutions:
        return solutions, search_reached, search_cot, ft_raw

    # No fully-verified solution found; return the first partial as wrong_answer.
    if first_partial is not None:
        return [first_partial], None, None, None
    return [], None, None, None


# ---------------------------------------------------------------------------
# Per-sample solver
# ---------------------------------------------------------------------------

def solve_sample(row: dict, config: SolverConfig) -> list[dict]:
    """Solve one puzzle row and return a list of result dicts (one per solution found).

    Each dict has a 'solution_count' key (same value across all dicts for this puzzle)
    and a 'search_reached' key indicating the furthest escalation stage entered.

    Concat puzzles → 1 dict (solution_count=1, search_reached=None).
    Arithmetic with N verified solutions → N dicts (solution_count=N).
    Wrong-answer / no-solution → 1 dict (solution_count=0, search_reached=None).
    """
    gold_answer = str(row["answer"]).strip()

    try:
        equations = extract_all_equations(row)
    except Exception as exc:
        return [_no_solution(equations=[], error=str(exc))]

    syms = content_symbols(equations)
    sym2i = {s: i for i, s in enumerate(syms)}

    # Concat fast-path.
    concat_result = None if config.disable_concat_fastpath else _try_concat(equations)
    if concat_result is not None:
        concat_ans, concat_op = concat_result
        op_glyph = equations[-1]["lhs"]["op"]
        concat_cot: str | None = None
        if config.generate_cot:
            from src.solver.cot_search_trace import render_concat_cot
            concat_cot = render_concat_cot(op_glyph, concat_op, concat_ans)
        return [{
            "solver_answer": concat_ans,
            "solver_correct": concat_ans == gold_answer,
            "solver_strategy": "pure_concat",
            "category": "concatenation",
            "ops_mapping": {op_glyph: concat_op},
            "digits_mapping": None,
            "solver_tier": None,
            "solver_radix": None,
            "ops_candidates": None,
            "equations": equations,
            "solution_count": 1,
            "standard_solution_count": 0,
            "little_endian_solution_count": 0,
            "search_reached": None,
            "search_cot": concat_cot,
            "digit_mappings_analyzed": None,
            "search_trace_path": None,
        }]

    # Arithmetic search.
    solutions, search_reached, search_cot, ft_raw = _try_arithmetic(
        equations, syms, sym2i, gold_answer, config
    )

    if not solutions:
        return [_no_solution(equations=equations)]

    # Build and write the full-trace JSON file when enabled.
    search_trace_path: str | None = None
    digit_mappings_analyzed: int | None = None
    if config.generate_search_trace and ft_raw is not None:
        problem_id = row.get("problem_id")
        digit_mappings_analyzed = ft_raw["total_trials"]

        # Sanitise problem_id for use as a filename.
        import re
        safe_id = re.sub(r"[^\w\-]", "_", str(problem_id)) if problem_id is not None else "unknown"
        trace_dir = _ROOT / config.search_trace_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{safe_id}.json"
        _write_search_trace_json(
            ft_raw=ft_raw,
            content_syms=syms,
            sorted_ops=ft_raw["sorted_ops"],
            problem_id=str(problem_id) if problem_id is not None else None,
            out_path=trace_file,
        )
        # Store path relative to inputs/ so the Vite /files/ handler can serve it.
        rel_to_inputs = trace_file.relative_to(_ROOT / "inputs")
        search_trace_path = str(rel_to_inputs)

    solution_count = len(solutions)
    standard_solution_count = sum(1 for _, d in solutions if d.get("solver_strategy") == "standard")
    little_endian_solution_count = sum(1 for _, d in solutions if d.get("solver_strategy") == "little_endian")
    result_dicts: list[dict] = []
    for answer_str, details in solutions:
        verify_failed = details.pop("_verify_failed", False)
        result_dicts.append({
            "solver_answer": answer_str,
            # solver_correct=False for verify failures (wrong_answer labeling).
            "solver_correct": (answer_str == gold_answer) and not verify_failed,
            **details,
            "category": "arithmetic",
            "equations": equations,
            "solution_count": solution_count if not verify_failed else 0,
            "standard_solution_count": standard_solution_count if not verify_failed else 0,
            "little_endian_solution_count": little_endian_solution_count if not verify_failed else 0,
            "search_reached": search_reached if not verify_failed else None,
            "search_cot": search_cot if not verify_failed else None,
            "digit_mappings_analyzed": digit_mappings_analyzed if not verify_failed else None,
            "search_trace_path": search_trace_path if not verify_failed else None,
        })
    return result_dicts


def _no_solution(equations: list, error: str | None = None) -> dict:
    return {
        "solver_answer": None,
        "solver_correct": False,
        "solver_strategy": None,
        "category": None,
        "ops_mapping": None,
        "digits_mapping": None,
        "solver_tier": None,
        "solver_radix": None,
        "ops_candidates": None,
        "equations": equations,
        "solution_count": 0,
        "standard_solution_count": 0,
        "little_endian_solution_count": 0,
        "search_reached": None,
        "search_cot": None,
        "digit_mappings_analyzed": None,
        "search_trace_path": None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _to_json(obj) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Equation-Symbolic Solver")
    parser.add_argument("config", nargs="?", default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    input_path = _ROOT / config.input_parquet
    output_path = _ROOT / config.output_parquet

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    _COLUMNS = [
        "problem_id", "solution_id", "prompt", "answer",
        "solver_answer", "solver_correct", "solver_strategy", "solver_category", "problem_category",
        "ops_mapping", "digits_mapping", "solver_tier", "solver_radix", "ops_candidates",
        "equations",
        "solution_count", "standard_solution_count", "little_endian_solution_count",
        "search_cot",
        "digit_mappings_analyzed", "search_trace_path",
    ]

    records = []
    solved = wrong = no_sol = 0
    pure_concat_count = standard_sols = little_endian_sols = 0
    solution_count_dist: dict[int, int] = {}
    _CHECKPOINT_EVERY = 50  # write parquet every N puzzles so a kill preserves partial results

    def _flush_parquet(recs: list[dict], path: Path) -> None:
        if not recs:
            return
        tmp = pd.DataFrame(recs)
        tmp["solution_id"] = tmp.groupby("problem_id").cumcount()
        cols = [c for c in _COLUMNS if c in tmp.columns]
        tmp = tmp[cols]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.to_parquet(path, index=False)

    for puzzle_idx, (_, row) in enumerate(df.iterrows()):
        result_list = solve_sample(row.to_dict(), config)
        gold = str(row["answer"]).strip()

        # Count per-puzzle outcome using the first result dict.
        first = result_list[0]
        solver_ans = first["solver_answer"]
        if solver_ans is None:
            no_sol += 1
        elif first["solver_correct"]:
            solved += 1
        else:
            wrong += 1

        # Strategy and solution-count stats.
        sc = first["solution_count"]
        solution_count_dist[sc] = solution_count_dist.get(sc, 0) + 1
        if first.get("solver_strategy") == "pure_concat":
            pure_concat_count += 1
        else:
            standard_sols += first.get("standard_solution_count", 0) or 0
            little_endian_sols += first.get("little_endian_solution_count", 0) or 0

        # Emit one record per solution (multiple rows per puzzle when solutions > 1).
        for res in result_list:
            records.append({
                "problem_id": row.get("problem_id"),
                "prompt": row.get("prompt"),
                "answer": gold,
                "solver_answer": res["solver_answer"],
                "solver_correct": res["solver_correct"],
                "solver_strategy": res["solver_strategy"],
                "solver_category": res["category"],
                "problem_category": row.get("problem_category"),
                "ops_mapping": _to_json(res["ops_mapping"]),
                "digits_mapping": _to_json(res["digits_mapping"]),
                "solver_tier": res["solver_tier"],
                "solver_radix": res["solver_radix"],
                "ops_candidates": _to_json(res["ops_candidates"]),
                "equations": _to_json(res["equations"]),
                "solution_count": res["solution_count"],
                "standard_solution_count": res["standard_solution_count"],
                "little_endian_solution_count": res["little_endian_solution_count"],
                "search_cot": res.get("search_cot"),
                "digit_mappings_analyzed": res.get("digit_mappings_analyzed"),
                "search_trace_path": res.get("search_trace_path"),
            })

        # Checkpoint every N puzzles so a kill preserves partial results.
        if (puzzle_idx + 1) % _CHECKPOINT_EVERY == 0:
            _flush_parquet(records, output_path)
            print(f"  checkpoint: {puzzle_idx + 1}/{len(df)} puzzles written")

    _flush_parquet(records, output_path)

    total = solved + wrong + no_sol
    print(f"\n=== solver results ===")
    print(f"solved correctly : {solved}")
    print(f"wrong answer     : {wrong}")
    print(f"no solution      : {no_sol}")
    print(f"total            : {total}")
    print(f"\n--- strategy breakdown ---")
    print(f"pure_concat      : {pure_concat_count}")
    print(f"standard         : {standard_sols}")
    print(f"little_endian    : {little_endian_sols}")
    print(f"\n--- solution count distribution ---")
    for sc in sorted(solution_count_dist):
        print(f"  {sc} solutions : {solution_count_dist[sc]} puzzles")
    print(f"\nwritten to {output_path}")


if __name__ == "__main__":
    main()
