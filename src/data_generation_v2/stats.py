"""Compute empirical distributions from the 823 solved dataset.

Distributions extracted and used by synth.py:
  op_probs            P(op_name)
  op_char_probs       P(op_char | op_name)
  pos_probs           P(symbol | position)
  strategy_probs      P(strategy)
  global_sym_probs    P(symbol) global fallback
  n_operators_probs   P(# distinct operators per problem)
  n_operand_syms_probs P(# unique operand symbols per problem)
  concat_n_concat_probs P(# concat example-eqs | pure_concat problem)
  single_digit_operand_prob  P(operand value < 10)
  sign_pos_probs      P(prefix | postfix) among signed equations
  zero_pad_prob       P(result is zero-padded) among arithmetic equations
  sign_demo_prob      P(same-op signed example | to_predict is signed)
  pad_demo_prob       P(same-op padded example | to_predict is padded)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataStats:
    op_probs: dict[str, float]               # op_name -> probability
    op_char_probs: dict[str, dict[str, float]]  # op_name -> {op_char -> prob}
    pos_probs: dict[str, dict[str, float]]   # position -> {symbol -> prob}
    global_sym_probs: dict[str, float]       # symbol -> prob (union of positions)
    strategy_probs: dict[str, float]         # strategy -> prob
    arith_ops: list[str]                     # non-concat ops sorted by P(op) desc
    concat_ops: list[str]                    # concat op names present
    operand_syms: list[str]                  # symbols in l0/l1/r0/r1 + magnitude
    operator_syms: list[str]                 # symbols in lhs.op position
    source_signatures: set[tuple]            # canonical structural sigs of source rows
    source_prompts: set[str]                 # raw prompt strings of source rows
    # --- fidelity distributions ---
    n_operators_probs: dict[int, float]      # P(# distinct op glyphs per problem)
    n_operand_syms_probs: dict[int, float]   # P(# unique operand/magnitude syms per problem)
    concat_n_concat_probs: dict[int, float]  # P(# concat example-eqs | pure_concat)
    single_digit_operand_prob: float         # P(operand value < 10) ≈ 0.012
    sign_pos_probs: dict[str, float]         # P(prefix|postfix) among signed eqs
    zero_pad_prob: float                     # P(zero-padded result) per arithmetic eq
    sign_demo_prob: float                    # P(same-op signed example | tp signed)
    pad_demo_prob: float                     # P(same-op padded example | tp padded)


def compute_stats(parquet_path: str | Path) -> DataStats:
    """Load solver_results.parquet and compute the empirical distributions."""
    import pandas as pd
    from src.data_generation_v2.signature import problem_signature
    from src.encoding import _int_to_base_digits, _is_reversed_digit_mode
    from src.op_index import CONCAT_OPS
    from src.operations import OPERATIONS

    df = pd.read_parquet(parquet_path)
    df = df[df["solver_correct"] == True].copy()

    op_counter: Counter[str] = Counter()
    op_char_counter: dict[str, Counter[str]] = defaultdict(Counter)
    pos_counter: dict[str, Counter[str]] = {
        "l0": Counter(), "l1": Counter(), "r0": Counter(), "r1": Counter()
    }
    global_sym_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    operand_sym_set: set[str] = set()
    operator_sym_set: set[str] = set()
    source_signatures: set[tuple] = set()
    source_prompts: set[str] = set()

    n_ops_counter: Counter[int] = Counter()
    n_operand_syms_counter: Counter[int] = Counter()
    concat_n_concat_counter: Counter[int] = Counter()
    sign_pos_counter: Counter[str] = Counter()
    single_digit_total = 0; single_digit_count = 0
    zero_pad_total = 0; zero_pad_count = 0
    sign_demo_total = 0; sign_demo_count = 0
    pad_demo_total = 0; pad_demo_count = 0

    for _, row in df.iterrows():
        strategy = str(row.get("solver_strategy") or "")
        strategy_counter[strategy] += 1
        is_arith = strategy in ("standard", "little_endian")
        is_concat_prob = strategy == "pure_concat"
        le = _is_reversed_digit_mode("little_endian" if strategy == "little_endian" else "standard")

        # Parse ops_mapping (op_char -> op_name)
        om_raw = row.get("ops_mapping")
        om: dict[str, str] = {}
        if isinstance(om_raw, str):
            try:
                om = json.loads(om_raw)
            except (json.JSONDecodeError, TypeError):
                om = {}
        elif isinstance(om_raw, dict):
            om = om_raw
        for op_char, op_name in om.items():
            op_char_counter[op_name][op_char] += 1

        # Parse digits_mapping (sym -> digit)
        dm_raw = row.get("digits_mapping")
        dm: dict[str, int] = {}
        if isinstance(dm_raw, str):
            try:
                dm = json.loads(dm_raw)
            except (json.JSONDecodeError, TypeError):
                dm = {}
        elif isinstance(dm_raw, dict):
            dm = dm_raw

        # Parse equations
        eq_raw = row.get("equations")
        equations: list[dict] = []
        if isinstance(eq_raw, str):
            try:
                equations = json.loads(eq_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(eq_raw, list):
            equations = eq_raw

        if not equations:
            continue

        # Per-problem accumulators
        prob_op_glyphs: set[str] = set()
        prob_operand_syms: set[str] = set()
        prob_concat_eq_count = 0  # for pure_concat rows

        # For demo tracking: list of (eq_type, op_name, signed, padded)
        eq_infos: list[tuple] = []

        for eq in equations:
            lhs = eq.get("lhs", {})
            rhs = eq.get("rhs", {})
            op_char = lhs.get("op", "")
            op_name = om.get(op_char, "")

            if op_char and op_name:
                op_counter[op_name] += 1
            if op_char:
                operator_sym_set.add(op_char)
                prob_op_glyphs.add(op_char)

            for pos in ("l0", "l1", "r0", "r1"):
                sym = lhs.get(pos, "")
                if sym:
                    pos_counter[pos][sym] += 1
                    global_sym_counter[sym] += 1
                    operand_sym_set.add(sym)
                    prob_operand_syms.add(sym)

            sign_sym = rhs.get("op")
            for sym in rhs.get("actual_rhs", []):
                if sym:
                    global_sym_counter[sym] += 1
                    if sym != sign_sym:
                        operand_sym_set.add(sym)
                        prob_operand_syms.add(sym)

            # Sign position
            signed = sign_sym is not None
            if signed:
                sp = rhs.get("sign_pos") or "prefix"
                sign_pos_counter[str(sp)] += 1

            # Pure-concat: count example equations that use the concat op
            if is_concat_prob and eq.get("type") == "example":
                if op_name in CONCAT_OPS:
                    prob_concat_eq_count += 1

            # Zero-padding and operand magnitudes (arithmetic only, has dm)
            padded = False
            if is_arith and dm and op_name and op_name not in CONCAT_OPS:
                try:
                    l0d, l1d = dm[lhs["l0"]], dm[lhs["l1"]]
                    r0d, r1d = dm[lhs["r0"]], dm[lhs["r1"]]
                except KeyError:
                    pass
                else:
                    L = l1d * 10 + l0d if le else l0d * 10 + l1d
                    R = r1d * 10 + r0d if le else r0d * 10 + r1d

                    # Single-digit operand tracking
                    single_digit_total += 2
                    if L < 10: single_digit_count += 1
                    if R < 10: single_digit_count += 1

                    # Zero-padding detection
                    fn = OPERATIONS.get(op_name)
                    if fn is not None:
                        res = fn(L, R)
                        if isinstance(res, (int, float)):
                            nat_len = len(_int_to_base_digits(abs(int(res)), 10))
                            ar_len = len(rhs.get("actual_rhs", []))
                            zero_pad_total += 1
                            if ar_len > nat_len:
                                padded = True
                                zero_pad_count += 1

            eq_infos.append((eq.get("type"), op_name, signed, padded))

        # Per-problem stats
        n_ops_counter[len(prob_op_glyphs)] += 1
        n_operand_syms_counter[len(prob_operand_syms)] += 1

        if is_concat_prob and prob_concat_eq_count > 0:
            concat_n_concat_counter[prob_concat_eq_count] += 1

        # Demo tracking (arithmetic only)
        if is_arith and dm:
            tp_entries = [(s, p, op) for (t, op, s, p) in eq_infos if t == "to_predict"]
            ex_entries = [(s, p, op) for (t, op, s, p) in eq_infos if t == "example"]
            if tp_entries:
                tp_signed, tp_padded, tp_op = tp_entries[0]
                if tp_signed:
                    sign_demo_total += 1
                    if any(s and op == tp_op for s, p, op in ex_entries):
                        sign_demo_count += 1
                if tp_padded:
                    pad_demo_total += 1
                    if any(p and op == tp_op for s, p, op in ex_entries):
                        pad_demo_count += 1

        # Source signature and prompt for dedup
        mode = "little_endian" if strategy == "little_endian" else "standard"
        try:
            sig = problem_signature(equations, dm, om, mode, strategy)
            source_signatures.add(sig)
        except Exception:
            pass
        prompt = row.get("prompt")
        if prompt:
            source_prompts.add(str(prompt))

    def _normalise_str(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.most_common()}

    def _normalise_int(counter: Counter[int]) -> dict[int, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(counter.items())}

    op_probs = _normalise_str(op_counter)
    op_char_probs = {op: _normalise_str(ch) for op, ch in op_char_counter.items()}
    pos_probs = {pos: _normalise_str(cnt) for pos, cnt in pos_counter.items()}
    global_sym_probs = _normalise_str(global_sym_counter)
    strategy_probs = _normalise_str(strategy_counter)
    arith_ops = [op for op in op_probs if op not in CONCAT_OPS]
    concat_ops = [op for op in op_probs if op in CONCAT_OPS]

    return DataStats(
        op_probs=op_probs,
        op_char_probs=op_char_probs,
        pos_probs=pos_probs,
        global_sym_probs=global_sym_probs,
        strategy_probs=strategy_probs,
        arith_ops=arith_ops,
        concat_ops=concat_ops,
        operand_syms=sorted(operand_sym_set),
        operator_syms=sorted(operator_sym_set),
        source_signatures=source_signatures,
        source_prompts=source_prompts,
        n_operators_probs=_normalise_int(n_ops_counter),
        n_operand_syms_probs=_normalise_int(n_operand_syms_counter),
        concat_n_concat_probs=_normalise_int(concat_n_concat_counter),
        single_digit_operand_prob=single_digit_count / max(single_digit_total, 1),
        sign_pos_probs=_normalise_str(sign_pos_counter),
        zero_pad_prob=zero_pad_count / max(zero_pad_total, 1),
        sign_demo_prob=sign_demo_count / max(sign_demo_total, 1),
        pad_demo_prob=pad_demo_count / max(pad_demo_total, 1),
    )
