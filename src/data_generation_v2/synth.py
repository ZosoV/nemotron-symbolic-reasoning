"""Strategy B: backward-encoded synthetic problems for n=3, 4, 5.

Generates brand-new synthetic cryptarithm problems using the statistics of the
803 solver-correct training rows. The backward encoding process:

  Arithmetic (standard / little_endian):
    1. Sample strategy.
    2. Pre-select n_ops distinct op_names ~ P(n_ops) × P(op_name); assign glyphs.
    3. For each equation: sample op from pre-selected set, operands ~ biased [10,99],
       compute result, assign symbols bijectively, apply zero-padding / sign-pos sampling.
    4. Reject if operand-symbol count ≠ sampled target.
    5. Reject if sign/pad demo requirement (Option 3) is not met.
    6. Validate with verify_program.

  Pure-concat:
    1. Pick concat op + glyph from P(op_char | op_name).
    2. Sample n_concat (# concat example equations) from empirical distribution.
    3. Build n_concat concat example equations + arithmetic distractor equations.
    4. Build 1 to_predict concat equation; shuffle examples.
    5. ops_mapping = {concat_glyph: concat_op} only (matching real data).
    6. digits_mapping = None (no global digit bijection).

Each synthetic problem produces N = n+1 equations, the last being to_predict.
The digit↔symbol bijection is shared across all equations in an arithmetic problem.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_generation.build_example_datasets import lhs_str, make_prompt, rhs_str
from src.data_generation_v2.schema import make_solver_row
from src.data_generation_v2.signature import problem_signature
from src.data_generation_v2.stats import DataStats
from src.encoding import _int_to_base_digits, _is_reversed_digit_mode, verify_program
from src.op_index import CONCAT_OPS, SIGNED_OPS
from src.operations import OPERATIONS

_MAX_RETRIES = 200        # per equation; reject whole problem and restart if exceeded
_MAX_BUCKET_ATTEMPTS_MULTIPLIER = 50  # per-bucket attempt cap = target * this


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_synthetic(
    stats: DataStats,
    target_counts: dict[int, int],   # {n: how_many_to_generate}
    seed: int = 42,
    include_pure_concat: bool = True,
    dedup_mode: str = "canonical",   # "canonical" (structural) or "prompt" (exact)
) -> list[dict]:
    """Return synthetic samples for the given bucket→count mapping.

    target_counts example: {3: 764, 4: 699, 5: 714}
    dedup_mode:
      "canonical" — structural (symbol-invariant) dedup against source + prior output
      "prompt"    — exact prompt-string match dedup
    """
    rng = random.Random(seed)
    results: list[dict] = []
    total_idx = 0

    if dedup_mode == "canonical":
        seen: set = set(stats.source_signatures)
    else:
        seen = set(stats.source_prompts)

    for n, target in sorted(target_counts.items()):
        if target <= 0:
            continue
        generated = 0
        rejected = 0
        n_equations = n + 1  # n examples + 1 to_predict
        max_attempts = target * _MAX_BUCKET_ATTEMPTS_MULTIPLIER

        for _ in range(max_attempts):
            if generated >= target:
                break

            row = _try_generate_problem(n_equations, stats, rng, include_pure_concat)
            if row is None:
                continue

            # Dedup check
            if dedup_mode == "canonical":
                mode = "little_endian" if row["strategy"] == "little_endian" else "standard"
                key = problem_signature(
                    row["equations"],
                    row.get("digits_mapping") or {},
                    row["op_sym2name"],
                    mode,
                    row["strategy"],
                )
            else:
                key = row["prompt"]

            if key in seen:
                rejected += 1
                continue
            seen.add(key)

            problem_id = f"synth_n{n}_{total_idx:06d}"
            total_idx += 1
            results.append(
                make_solver_row(
                    problem_id=problem_id,
                    prompt=row["prompt"],
                    answer=row["answer"],
                    equations=row["equations"],
                    digits_mapping=row.get("digits_mapping"),
                    ops_mapping=row.get("ops_mapping"),
                    radix=row["radix"],
                    strategy=row["strategy"],
                    dataset_category=f"examples_{n}",
                )
            )
            generated += 1

        if generated < target:
            print(f"  [synth] n={n}: WARNING only {generated}/{target} generated "
                  f"(rejected={rejected}, attempts exhausted)")
        else:
            print(f"  [synth] n={n}: generated {generated}  (rejected={rejected})")

    return results


# ---------------------------------------------------------------------------
# Problem generator — top level
# ---------------------------------------------------------------------------

def _try_generate_problem(
    n_equations: int,
    stats: DataStats,
    rng: random.Random,
    include_pure_concat: bool = True,
) -> dict | None:
    """Attempt to generate one valid synthetic problem. Returns None on failure."""

    strategy = _sample_strategy(stats, rng, include_pure_concat=include_pure_concat)

    # Pure-concat: delegate to dedicated builder
    if strategy == "pure_concat":
        return _build_concat_problem(n_equations, stats, rng)

    # Arithmetic (standard / little_endian)
    radix = 10
    mode = "little_endian" if strategy == "little_endian" else "standard"

    # (A) Pre-select operator set
    n_ops = _sample_from_int_probs(stats.n_operators_probs, rng)
    n_ops = max(1, min(n_ops, min(n_equations, len(stats.arith_ops))))
    pre_selected_ops = _select_ops(stats, rng, n_ops)
    if not pre_selected_ops:
        return None

    # Pre-assign op slots: each pre-selected op gets exactly one mandatory slot,
    # remaining slots are sampled weighted by op_probs (ensures all ops appear ≥1).
    op_weights = [stats.op_probs.get(op, 1e-6) for op in pre_selected_ops]
    n_extra = n_equations - len(pre_selected_ops)
    eq_op_assignments: list[str] = list(pre_selected_ops)
    if n_extra > 0:
        eq_op_assignments += rng.choices(pre_selected_ops, weights=op_weights, k=n_extra)
    rng.shuffle(eq_op_assignments)

    # (C) Operand-symbol target — sample once per problem for the post-build check
    _target_n_operand_syms = _sample_from_int_probs(stats.n_operand_syms_probs, rng)

    # Problem-level bijections
    digit2sym: dict[int, str] = {}
    sym2digit: dict[str, int] = {}
    op_name2sym: dict[str, str] = {}
    op_sym2name: dict[str, str] = {}
    used_syms: set[str] = set()

    # Build equations — each slot uses its pre-assigned op (guarantees n_ops coverage)
    equations: list[dict] = []
    padded_flags: list[bool] = []
    signed_flags: list[bool] = []
    eq_op_names: list[str] = []

    for eq_idx in range(n_equations):
        eq_type = "to_predict" if eq_idx == n_equations - 1 else "example"
        forced_op = eq_op_assignments[eq_idx]
        result = None
        for _ in range(_MAX_RETRIES):
            result = _generate_arithmetic_equation(
                mode=mode, radix=radix, stats=stats, rng=rng,
                digit2sym=digit2sym, sym2digit=sym2digit,
                op_name2sym=op_name2sym, op_sym2name=op_sym2name,
                used_syms=used_syms,
                forced_op=forced_op,
            )
            if result is not None:
                break
        if result is None:
            return None

        eq, is_padded = result
        eq["type"] = eq_type
        equations.append(eq)
        padded_flags.append(is_padded)
        signed_flags.append(eq["rhs"]["op"] is not None)
        eq_op_names.append(op_sym2name.get(eq["lhs"]["op"], ""))

    # Verify all equations
    if not verify_program(equations, sym2digit, op_sym2name, mode, radix):
        return None

    # (C) Operand-symbol count: enforce hard cap ≤ 10 (bijection allows max 10 digits);
    # exact match to sampled target omitted — distribution is already emergent.
    if _count_operand_syms(equations) > 10:
        return None

    # (G) Demonstrating-example check (Option 3: match real rate via Bernoulli)
    tp_signed = signed_flags[-1]
    tp_padded = padded_flags[-1]
    tp_op = eq_op_names[-1]

    if tp_signed:
        want_demo = rng.random() < stats.sign_demo_prob
        has_demo = any(
            s and op == tp_op
            for s, op in zip(signed_flags[:-1], eq_op_names[:-1])
        )
        if has_demo != want_demo:
            return None

    if tp_padded:
        want_demo = rng.random() < stats.pad_demo_prob
        has_demo = any(
            p and op == tp_op
            for p, op in zip(padded_flags[:-1], eq_op_names[:-1])
        )
        if has_demo != want_demo:
            return None

    # Build prompt + answer
    ex_eqs = equations[:-1]
    tp_eq = equations[-1]
    example_lines = [f"{lhs_str(e['lhs'])} = {rhs_str(e['rhs'])}" for e in ex_eqs]
    prompt = make_prompt(example_lines, lhs_str(tp_eq["lhs"]))
    answer = rhs_str(tp_eq["rhs"])

    return {
        "prompt": prompt,
        "answer": answer,
        "equations": equations,
        "digits_mapping": sym2digit,
        "ops_mapping": op_sym2name,
        "op_sym2name": op_sym2name,
        "radix": radix,
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Arithmetic equation builder
# ---------------------------------------------------------------------------

def _generate_arithmetic_equation(
    mode: str,
    radix: int,
    stats: DataStats,
    rng: random.Random,
    digit2sym: dict[int, str],
    sym2digit: dict[str, int],
    op_name2sym: dict[str, str],
    op_sym2name: dict[str, str],
    used_syms: set[str],
    pre_selected_ops: list[str] | None = None,
    forced_op: str | None = None,
) -> tuple[dict, bool] | None:
    """Generate one arithmetic equation. Returns (eq_dict, is_padded) or None."""

    # (A) Use forced op (pre-assigned slot) > pre-selected pool > global arith pool
    if forced_op is not None:
        op_name = forced_op
    elif pre_selected_ops:
        weights = [stats.op_probs.get(op, 1e-6) for op in pre_selected_ops]
        op_name = rng.choices(pre_selected_ops, weights=weights, k=1)[0]
    else:
        op_name = _sample_op(stats, rng, concat=False)

    # (B) Sample operands biased toward [10, 99]
    L_val = _sample_operand(stats, rng)
    R_val = _sample_operand(stats, rng)

    # Decompose operand values based on mode
    # Standard:      l0 = tens (MSD), l1 = units (LSD)
    # Little_endian: l0 = units (LSD), l1 = tens (MSD)
    if _is_reversed_digit_mode(mode):
        l0_d, l1_d = L_val % radix, L_val // radix
        r0_d, r1_d = R_val % radix, R_val // radix
    else:
        l0_d, l1_d = L_val // radix, L_val % radix
        r0_d, r1_d = R_val // radix, R_val % radix

    fn = OPERATIONS.get(op_name)
    if fn is None:
        return None
    result = fn(L_val, R_val)
    if result is None:
        return None

    # Sign handling
    has_sign = False
    sign_pos = None
    sign_sym: str | None = None

    if op_name in SIGNED_OPS:
        if result < 0:
            has_sign = True
        mag = abs(result)
    elif op_name == "neg_absdiff":
        has_sign = True
        mag = abs(result)
    else:
        if result < 0:
            return None
        mag = result

    # Magnitude → digit list (MSB first)
    mag_digits = _int_to_base_digits(mag, radix)

    # (F) Zero-padding: prepend a leading-zero digit with empirical probability
    is_padded = False
    if rng.random() < stats.zero_pad_prob:
        mag_digits = [0] + mag_digits
        is_padded = True

    # Convert to symbol-order for the encoding mode
    if _is_reversed_digit_mode(mode):
        res_digits = list(reversed(mag_digits))  # LSB first
    else:
        res_digits = mag_digits                   # MSB first

    # Assign symbols to operand positions
    positions = {"l0": l0_d, "l1": l1_d, "r0": r0_d, "r1": r1_d}
    for pos, d in positions.items():
        if not _assign_digit_symbol(d, pos, stats, rng, digit2sym, sym2digit, used_syms):
            return None

    # Assign symbols to result digits
    for d in res_digits:
        if not _assign_digit_symbol(d, None, stats, rng, digit2sym, sym2digit, used_syms):
            return None

    # Assign operator symbol
    op_sym = _assign_op_symbol(op_name, stats, rng, op_name2sym, op_sym2name, used_syms)
    if op_sym is None:
        return None

    # (E) Sign position: sample prefix/postfix from empirical distribution
    if has_sign:
        sign_pos_keys = list(stats.sign_pos_probs.keys())
        sign_pos_weights = [stats.sign_pos_probs[k] for k in sign_pos_keys]
        sign_pos = rng.choices(sign_pos_keys, weights=sign_pos_weights, k=1)[0]
        sign_sym = op_sym

    # Build symbol lists
    l0_sym = digit2sym[l0_d]
    l1_sym = digit2sym[l1_d]
    r0_sym = digit2sym[r0_d]
    r1_sym = digit2sym[r1_d]
    res_syms = [digit2sym[d] for d in res_digits]

    lhs = {"l0": l0_sym, "l1": l1_sym, "op": op_sym, "r0": r0_sym, "r1": r1_sym}
    rhs = {"op": sign_sym, "sign_pos": sign_pos, "actual_rhs": res_syms}

    return {"lhs": lhs, "rhs": rhs}, is_padded


# ---------------------------------------------------------------------------
# Pure-concat problem builder
# ---------------------------------------------------------------------------

def _build_concat_problem(
    n_equations: int,
    stats: DataStats,
    rng: random.Random,
) -> dict | None:
    """Build a realistic pure_concat problem (mixed: concat examples + fake distractors).

    Structure matches real data:
      - to_predict uses the concat op
      - n_concat example equations use the same concat op
      - (n_examples - n_concat) are fake distractor equations (look arithmetic but
        RHS is arbitrary — solver ignores them since digits_mapping=None)
      - All equations share ONE symbol pool (~8-10 symbols, matching real n_osym)
      - ops_mapping contains only the concat op; digits_mapping is None
    """
    n_examples = n_equations - 1

    # 1. Pick concat op and assign glyph from P(op_char | concat_op_name)
    if not stats.concat_ops:
        return None
    concat_op_name = rng.choices(
        stats.concat_ops,
        weights=[stats.op_probs.get(op, 1e-6) for op in stats.concat_ops],
        k=1,
    )[0]
    op_used: set[str] = set()
    dist = stats.op_char_probs.get(concat_op_name, {})
    concat_glyph = _weighted_sample_unused(dist, op_used, rng)
    if concat_glyph is None:
        avail = [s for s in stats.operator_syms if s not in op_used]
        if not avail:
            return None
        concat_glyph = rng.choice(avail)
    op_used.add(concat_glyph)

    # 2. Sample n_concat
    n_concat = _sample_from_int_probs(stats.concat_n_concat_probs, rng)
    n_concat = max(1, min(n_concat, n_examples))
    n_distractor = n_examples - n_concat

    # 3. Pre-select distractor op glyphs (distinct from concat glyph)
    distractor_op_glyphs: list[str] = []  # one glyph per distractor eq
    if n_distractor > 0:
        n_ops_total = _sample_from_int_probs(stats.n_operators_probs, rng)
        n_distractor_ops = max(1, min(n_ops_total - 1, n_distractor, len(stats.arith_ops)))
        d_op_names = _select_ops(stats, rng, n_distractor_ops)
        d_op_name2glyph: dict[str, str] = {}
        for op_name in d_op_names:
            g_dist = stats.op_char_probs.get(op_name, {})
            glyph = _weighted_sample_unused(g_dist, op_used, rng)
            if glyph is None:
                avail = [s for s in stats.operator_syms if s not in op_used]
                if not avail:
                    return None
                glyph = rng.choice(avail)
            op_used.add(glyph)
            d_op_name2glyph[op_name] = glyph
        # Assign one glyph per distractor slot (round-robin if fewer ops than slots)
        for i in range(n_distractor):
            op = d_op_names[i % len(d_op_names)]
            distractor_op_glyphs.append(d_op_name2glyph[op])

    # 4. Build shared symbol pool (all operand symbols across the problem)
    n_osym = _sample_from_int_probs(stats.n_operand_syms_probs, rng)
    n_osym = max(4, min(n_osym, 10))
    # Exclude op glyphs from operand pool
    avail_operand = [s for s in stats.operand_syms if s not in op_used]
    if len(avail_operand) < n_osym:
        avail_operand = list(stats.operand_syms)  # relax if needed
    # Sample n_osym symbols weighted by global_sym_probs
    weights = [stats.global_sym_probs.get(s, 1e-6) for s in avail_operand]
    sym_pool: list[str] = []
    used_pool: set[str] = set()
    for _ in range(n_osym):
        remaining = [(s, w) for s, w in zip(avail_operand, weights) if s not in used_pool]
        if not remaining:
            break
        syms, ws = zip(*remaining)
        chosen = rng.choices(list(syms), weights=list(ws), k=1)[0]
        sym_pool.append(chosen)
        used_pool.add(chosen)

    if len(sym_pool) < 4:
        return None

    # 5. Build concat example equations using sym_pool
    concat_example_eqs: list[dict] = []
    for _ in range(n_concat):
        eq = _generate_concat_eq_from_pool(rng, concat_glyph, concat_op_name, sym_pool)
        if eq is None:
            return None
        eq["type"] = "example"
        concat_example_eqs.append(eq)

    # 6. Build fake distractor equations using sym_pool (no arithmetic constraint;
    #    solver ignores them — only concat equations matter for pure_concat)
    distractor_eqs: list[dict] = []
    for d_glyph in distractor_op_glyphs:
        eq = _generate_fake_distractor(rng, d_glyph, sym_pool, stats)
        if eq is None:
            return None
        eq["type"] = "example"
        distractor_eqs.append(eq)

    # 7. Build to_predict concat equation
    tp_eq = _generate_concat_eq_from_pool(rng, concat_glyph, concat_op_name, sym_pool)
    if tp_eq is None:
        return None
    tp_eq["type"] = "to_predict"

    # 8. Shuffle examples; append to_predict
    example_eqs = concat_example_eqs + distractor_eqs
    rng.shuffle(example_eqs)
    equations = example_eqs + [tp_eq]

    # 9. Build prompt + answer
    ex_lines = [f"{lhs_str(e['lhs'])} = {rhs_str(e['rhs'])}" for e in example_eqs]
    prompt = make_prompt(ex_lines, lhs_str(tp_eq["lhs"]))
    answer = rhs_str(tp_eq["rhs"])

    op_sym2name: dict[str, str] = {concat_glyph: concat_op_name}
    # Include distractor glyphs in op_sym2name for signature computation,
    # but NOT in ops_mapping (matching real data)
    for d_g in set(distractor_op_glyphs):
        op_sym2name[d_g] = "unknown"

    return {
        "prompt": prompt,
        "answer": answer,
        "equations": equations,
        "digits_mapping": None,
        "ops_mapping": {concat_glyph: concat_op_name},  # concat only, matching real
        "op_sym2name": op_sym2name,
        "radix": 10,
        "strategy": "pure_concat",
    }


def _generate_concat_eq_from_pool(
    rng: random.Random,
    op_glyph: str,
    op_name: str,
    sym_pool: list[str],
) -> dict | None:
    """Generate one concat equation drawing 4 distinct symbols from sym_pool."""
    if len(sym_pool) < 4:
        return None
    idxs = list(range(len(sym_pool)))
    rng.shuffle(idxs)
    l0, l1, r0, r1 = (sym_pool[i] for i in idxs[:4])
    lhs = {"l0": l0, "l1": l1, "op": op_glyph, "r0": r0, "r1": r1}
    actual_rhs = [l0, l1, r0, r1] if op_name == "concat_fwd" else [r0, r1, l0, l1]
    return {"lhs": lhs, "rhs": {"op": None, "sign_pos": None, "actual_rhs": actual_rhs}}


def _generate_fake_distractor(
    rng: random.Random,
    op_glyph: str,
    sym_pool: list[str],
    stats: DataStats,
) -> dict | None:
    """Generate a fake distractor equation using only symbols from sym_pool.

    The RHS is arbitrary pool symbols (not arithmetically derived) — the solver
    ignores distractor equations in pure_concat problems (digits_mapping=None).
    """
    if len(sym_pool) < 4:
        return None

    # Pick 4 distinct operand symbols
    idxs = list(range(len(sym_pool)))
    rng.shuffle(idxs)
    l0, l1, r0, r1 = (sym_pool[i] for i in idxs[:4])
    lhs = {"l0": l0, "l1": l1, "op": op_glyph, "r0": r0, "r1": r1}

    # RHS: 1-4 symbols from pool, length sampled from real mag_len distribution
    mag_weights = [7, 40, 25, 28]  # approx real 1:7%,2:40%,3:25%,4:28%
    n_rhs = rng.choices([1, 2, 3, 4], weights=mag_weights, k=1)[0]
    actual_rhs = rng.choices(sym_pool, k=n_rhs)

    # Optionally add a sign (matching sign_demo_prob roughly)
    sign_sym: str | None = None
    sign_pos: str | None = None
    if rng.random() < 0.15:  # approx real sign rate
        sign_candidates = [s for s in stats.operator_syms if s != op_glyph]
        if sign_candidates:
            sign_sym = rng.choice(sign_candidates)
            sign_pos = rng.choices(
                ["prefix", "postfix"],
                weights=[stats.sign_pos_probs.get("prefix", 0.947),
                         stats.sign_pos_probs.get("postfix", 0.053)],
                k=1,
            )[0]

    return {"lhs": lhs, "rhs": {"op": sign_sym, "sign_pos": sign_pos, "actual_rhs": actual_rhs}}


# ---------------------------------------------------------------------------
# Symbol assignment helpers
# ---------------------------------------------------------------------------

def _assign_digit_symbol(
    digit: int,
    position: str | None,
    stats: DataStats,
    rng: random.Random,
    digit2sym: dict[int, str],
    sym2digit: dict[str, int],
    used_syms: set[str],
) -> bool:
    """Assign a symbol to `digit` if not already mapped. Returns True on success."""
    if digit in digit2sym:
        return True

    if position and position in stats.pos_probs:
        dist = stats.pos_probs[position]
        sym = _weighted_sample_unused(dist, used_syms, rng)
    elif stats.global_sym_probs:
        sym = _weighted_sample_unused(stats.global_sym_probs, used_syms, rng)
    else:
        sym = None

    if sym is None:
        avail = [s for s in stats.operand_syms if s not in used_syms]
        if not avail:
            return False
        sym = rng.choice(avail)

    digit2sym[digit] = sym
    sym2digit[sym] = digit
    used_syms.add(sym)
    return True


def _assign_op_symbol(
    op_name: str,
    stats: DataStats,
    rng: random.Random,
    op_name2sym: dict[str, str],
    op_sym2name: dict[str, str],
    used_syms: set[str],
) -> str | None:
    """Assign an operator symbol to `op_name` if not already mapped."""
    if op_name in op_name2sym:
        return op_name2sym[op_name]

    dist = stats.op_char_probs.get(op_name, {})
    sym = _weighted_sample_unused(dist, used_syms, rng)

    if sym is None:
        avail = [s for s in stats.operator_syms if s not in used_syms]
        if not avail:
            return None
        sym = rng.choice(avail)

    op_name2sym[op_name] = sym
    op_sym2name[sym] = op_name
    used_syms.add(sym)
    return sym


def _weighted_sample_unused(
    dist: dict[str, float],
    used: set[str],
    rng: random.Random,
) -> str | None:
    """Sample from `dist` skipping already-used symbols. Returns None if all used."""
    candidates = [s for s in dist if s not in used]
    if not candidates:
        return None
    weights = [dist[s] for s in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_strategy(
    stats: DataStats,
    rng: random.Random,
    include_pure_concat: bool = True,
) -> str:
    strats = [s for s in stats.strategy_probs if include_pure_concat or s != "pure_concat"]
    weights = [stats.strategy_probs[s] for s in strats]
    return rng.choices(strats, weights=weights, k=1)[0]


def _sample_op(stats: DataStats, rng: random.Random, *, concat: bool) -> str:
    """Sample an op_name from P(op_name), restricted to concat or arithmetic ops."""
    ops = stats.concat_ops if concat else stats.arith_ops
    if not ops:
        return "mul"
    weights = [stats.op_probs.get(op, 1e-6) for op in ops]
    return rng.choices(ops, weights=weights, k=1)[0]


def _sample_operand(stats: DataStats, rng: random.Random) -> int:
    """(B) Sample an operand value biased to match real data (rarely single-digit)."""
    if rng.random() < stats.single_digit_operand_prob:
        return rng.randint(0, 9)
    return rng.randint(10, 99)


def _sample_from_int_probs(probs: dict[int, float], rng: random.Random) -> int:
    """Sample an integer key from a probability dict."""
    keys = list(probs.keys())
    weights = [probs[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _select_ops(stats: DataStats, rng: random.Random, n_ops: int) -> list[str]:
    """Sample n_ops unique op_names weighted by op_probs (without replacement)."""
    remaining = list(stats.arith_ops)
    weights = [stats.op_probs.get(op, 1e-6) for op in remaining]
    selected: list[str] = []
    for _ in range(min(n_ops, len(remaining))):
        if not remaining:
            break
        idx = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        selected.append(remaining.pop(idx))
        weights.pop(idx)
    return selected


def _count_operand_syms(equations: list[dict]) -> int:
    """Count unique symbols appearing in operand/magnitude positions across all equations."""
    syms: set[str] = set()
    for eq in equations:
        lhs = eq.get("lhs", {})
        rhs = eq.get("rhs", {})
        for pos in ("l0", "l1", "r0", "r1"):
            syms.add(lhs[pos])
        sign_sym = rhs.get("op")
        for s in rhs.get("actual_rhs", []):
            if s != sign_sym:
                syms.add(s)
    return len(syms)
