"""
rebalance_stage0.py
-------------------
Transforms data/solver_examples_0_k_20_extended.parquet into a balanced
Stage-0 dataset at data/solver_examples_0_k_20_balanced.parquet.

Two operations:
  1. Cap neg_absdiff, sub_signed, rsub_signed to 4,500 rows each via
     per-equation water-filling: trim the equations that contribute the
     most rows of the target op first (variant rows dropped before originals).
  2. Synthesize 4,500 concat_fwd + 4,500 concat_rev rows on top of the
     existing 419 real pure_concat rows.
     - Operator glyph: frequency-weighted draw from the base dataset
       (solver_results_forcing_base_10_k_1.parquet).
     - Operand glyphs: drawn uniformly from the pool of all glyphs that
       appear as l0/l1/r0/r1 in the real data.
     - synthetic rows carry ops_mapping = {"<op>": "concat_fwd/rev"}
       so they register in op-frequency counts. Real pure_concat rows
       carry ops_mapping = None (unchanged).

Run:  uv run python rebalance_stage0.py
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TARGET = 4_500
CAP_OPS = {"neg_absdiff", "sub_signed", "rsub_signed"}
SOURCE = Path("data/solver_examples_0_k_20_extended.parquet")
BASE   = Path("data/solver_results_forcing_base_10_k_1.parquet")
OUT    = Path("data/solver_examples_0_k_20_balanced.parquet")

PROMPT_PREFIX = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:\n\nNow, determine the result for: "
)

random.seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def op_of(ops_mapping_val) -> str | None:
    """Return the single op name from an ops_mapping JSON string, or None."""
    if ops_mapping_val is None:
        return None
    try:
        if pd.isna(ops_mapping_val):
            return None
    except TypeError:
        pass
    d = json.loads(ops_mapping_val)
    if not d:
        return None
    return next(iter(d.values()))


def glyph_of(ops_mapping_val) -> str | None:
    """Return the single operator glyph from an ops_mapping JSON string, or None."""
    if ops_mapping_val is None:
        return None
    try:
        if pd.isna(ops_mapping_val):
            return None
    except TypeError:
        pass
    d = json.loads(ops_mapping_val)
    if not d:
        return None
    return next(iter(d.keys()))


def make_equations_json(l0, l1, op, r0, r1, answer_syms: list[str]) -> str:
    """Build the equations JSON string for a synthetic concat row."""
    eq = {
        "type": "to_predict",
        "lhs": {"l0": l0, "l1": l1, "op": op, "r0": r0, "r1": r1},
        "rhs": {"op": None, "sign_pos": None, "actual_rhs": answer_syms},
    }
    return json.dumps([eq], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 1 — load + annotate
# ---------------------------------------------------------------------------

print(f"Loading {SOURCE} …")
df = pd.read_parquet(SOURCE)
print(f"  {len(df):,} rows, {len(df.columns)} columns")

df["_op"]   = df["ops_mapping"].apply(op_of)
df["_mult"] = df.groupby("problem_id")["problem_id"].transform("size")

before_counts: Counter = Counter(df["_op"].dropna())
print("\nBefore — op counts for capped ops and concat:")
for op in sorted(CAP_OPS | {"concat_fwd", "concat_rev"}):
    print(f"  {op:18}: {before_counts.get(op, 0):6,}")


# ---------------------------------------------------------------------------
# Step 2 — water-filling cap for the three over-represented ops
# ---------------------------------------------------------------------------

keep_mask = pd.Series(True, index=df.index)

for target_op in sorted(CAP_OPS):
    sub = df[df["_op"] == target_op].copy()
    n_current = len(sub)
    n_drop = n_current - TARGET
    assert n_drop > 0, f"{target_op} already at/below {TARGET}"
    print(f"\nCapping {target_op}: {n_current:,} → {TARGET:,} (drop {n_drop:,})")

    # Per-equation count of this op's rows
    per_eq = sub.groupby("problem_id").size().rename("x_e")

    # Find the water-fill threshold t such that sum min(x_e, t) == TARGET
    # Binary search on t in [0, max(x_e)]
    x_vals = per_eq.values
    lo, hi = 0, int(x_vals.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(x_vals, mid).sum() <= TARGET:
            lo = mid
        else:
            hi = mid - 1
    t = lo
    # sum_keep at threshold t
    sum_at_t = int(np.minimum(x_vals, t).sum())
    # sum_at_t may be < TARGET because raising t by 1 would overshoot.
    # We need (TARGET - sum_at_t) additional rows: pick that many equations
    # with x_e > t and allow them to keep t+1 instead of t.
    extra_keep = TARGET - sum_at_t
    print(f"  water-fill t={t}, sum@t={sum_at_t}, extra={extra_keep}")

    # Among all equations with x_e > t, give the extra row to equations
    # with the LOWEST total multiplicity first (least prolific get the bonus;
    # most prolific equations are trimmed hardest).
    above_t_ids = per_eq[per_eq > t].index
    above_t_mult = df.loc[df["problem_id"].isin(above_t_ids), ["problem_id", "_mult"]].drop_duplicates("problem_id")
    above_t_mult = above_t_mult.sort_values(["_mult", "problem_id"], ascending=[True, True])
    extra_keep_ids = set(above_t_mult["problem_id"].iloc[:extra_keep])

    def rows_to_keep_for_eq(eq_id, eq_rows):
        x_e = len(eq_rows)
        if x_e <= t:
            return eq_rows.index.tolist()       # keep all
        elif eq_id in extra_keep_ids:
            k = t + 1                            # gets one extra row
        else:
            k = t                                # trimmed to threshold

        # Within the equation, drop variants before originals;
        # sort: is_original DESC (True=1 first), then variant_index ASC
        ordered = eq_rows.sort_values(
            ["is_original", "variant_index"],
            ascending=[False, True],
        )
        return ordered.index[:k].tolist()

    keep_indices = []
    for eq_id, eq_rows in sub.groupby("problem_id"):
        keep_indices.extend(rows_to_keep_for_eq(eq_id, eq_rows))

    # Guard: the dropped rows must not eliminate any equation entirely.
    # Because these ops co-occur with other ops, a row dropped here only
    # removes one op-labeling; the equation still has other rows in keep_mask.
    drop_indices = set(sub.index) - set(keep_indices)
    keep_mask.loc[list(drop_indices)] = False

    actual_remaining = int((df.loc[keep_mask & (df["_op"] == target_op)]).shape[0])
    assert actual_remaining == TARGET, (
        f"After cap {target_op}: expected {TARGET}, got {actual_remaining}"
    )
    print(f"  ✓ {target_op} now at {actual_remaining:,}")

df_capped = df[keep_mask].copy()
# Guard: no equation has been fully deleted
assert df_capped["problem_id"].nunique() == df["problem_id"].nunique(), "An equation was fully deleted!"
print(f"\nAfter capping: {len(df_capped):,} rows ({df_capped['problem_id'].nunique():,} equations)")


# ---------------------------------------------------------------------------
# Step 3 — build operator-glyph weights from base dataset
# ---------------------------------------------------------------------------

print(f"\nLoading base dataset {BASE} …")
base = pd.read_parquet(BASE)

fwd_glyphs: Counter = Counter()
rev_glyphs: Counter = Counter()
for v in base["ops_mapping"]:
    try:
        if pd.isna(v):
            continue
    except TypeError:
        pass
    if v is None:
        continue
    for glyph, op in json.loads(v).items():
        if op == "concat_fwd":
            fwd_glyphs[glyph] += 1
        elif op == "concat_rev":
            rev_glyphs[glyph] += 1

print("  concat_fwd operator-glyph weights:", dict(fwd_glyphs.most_common()))
print("  concat_rev operator-glyph weights:", dict(rev_glyphs.most_common()))

fwd_op_symbols, fwd_op_weights = zip(*fwd_glyphs.items())
rev_op_symbols, rev_op_weights = zip(*rev_glyphs.items())
fwd_op_weights = np.array(fwd_op_weights, dtype=float)
rev_op_weights = np.array(rev_op_weights, dtype=float)
fwd_op_weights /= fwd_op_weights.sum()
rev_op_weights /= rev_op_weights.sum()


# ---------------------------------------------------------------------------
# Step 4 — operand glyph pool from the real data
# ---------------------------------------------------------------------------

operand_pool: Counter = Counter()
for v in df["equations"]:
    try:
        for eq in json.loads(v):
            for k in ("l0", "l1", "r0", "r1"):
                operand_pool[eq["lhs"][k]] += 1
    except Exception:
        pass

operand_glyphs = list(operand_pool.keys())
operand_weights = np.array([operand_pool[g] for g in operand_glyphs], dtype=float)
operand_weights /= operand_weights.sum()
print(f"\nOperand glyph pool: {len(operand_glyphs)} distinct glyphs")


# ---------------------------------------------------------------------------
# Step 5 — existing real concat LHS set (to avoid collisions)
# ---------------------------------------------------------------------------

real_lhs_set: set[tuple] = set()
for v in df_capped[df_capped["solver_strategy"] == "pure_concat"]["equations"]:
    try:
        eq = json.loads(v)[0]["lhs"]
        real_lhs_set.add((eq["l0"], eq["l1"], eq["op"], eq["r0"], eq["r1"]))
    except Exception:
        pass
print(f"Real concat LHSes to avoid: {len(real_lhs_set)}")


# ---------------------------------------------------------------------------
# Step 6 — synthesize concat rows
# ---------------------------------------------------------------------------

def synthesize_concat(direction: str, n: int) -> pd.DataFrame:
    assert direction in ("fwd", "rev")
    if direction == "fwd":
        op_symbols, op_weights_arr = fwd_op_symbols, fwd_op_weights
    else:
        op_symbols, op_weights_arr = rev_op_symbols, rev_op_weights

    rows = []
    seen: set[tuple] = set()
    attempts = 0
    max_attempts = n * 20

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        # Draw operator glyph
        op = str(rng.choice(list(op_symbols), p=op_weights_arr))
        # Draw 4 operand glyphs
        l0, l1, r0, r1 = [
            str(g) for g in rng.choice(operand_glyphs, size=4, p=operand_weights)
        ]
        # Require fwd != rev answer (i.e. (l0,l1) != (r0,r1))
        if (l0, l1) == (r0, r1):
            continue
        key = (l0, l1, op, r0, r1)
        if key in seen or key in real_lhs_set:
            continue
        seen.add(key)

        lhs_str = l0 + l1 + op + r0 + r1
        if direction == "fwd":
            answer_syms = [l0, l1, r0, r1]
            answer = l0 + l1 + r0 + r1
            op_name = "concat_fwd"
        else:
            answer_syms = [r0, r1, l0, l1]
            answer = r0 + r1 + l0 + l1
            op_name = "concat_rev"

        rows.append({
            "problem_id":       f"synth_concat_{direction}_{len(rows)}",
            "prompt":           PROMPT_PREFIX + lhs_str,
            "answer":           answer,
            "solver_strategy":  "pure_concat",
            "solver_category":  "concatenation",
            "problem_category": None,
            "digits_mapping":   None,
            "solver_tier":      None,
            "solver_radix":     None,
            "ops_candidates":   None,
            "equations":        make_equations_json(l0, l1, op, r0, r1, answer_syms),
            "ops_mapping":      json.dumps({op: op_name}, ensure_ascii=False),
            "solver_answer":    answer,
            "solver_correct":   True,
            "variant_index":    0,
            "is_original":      False,
            "num_variants":     1,
            "fully_validated":  None,
        })

    if len(rows) < n:
        raise RuntimeError(
            f"Could only synthesize {len(rows)}/{n} {direction} rows "
            f"after {max_attempts} attempts. Increase max_attempts or relax constraints."
        )
    return pd.DataFrame(rows)


print(f"\nSynthesizing {TARGET:,} concat_fwd rows …")
df_fwd = synthesize_concat("fwd", TARGET)
print(f"  ✓ generated {len(df_fwd):,}")

print(f"Synthesizing {TARGET:,} concat_rev rows …")
df_rev = synthesize_concat("rev", TARGET)
print(f"  ✓ generated {len(df_rev):,}")


# ---------------------------------------------------------------------------
# Step 7 — assemble and write
# ---------------------------------------------------------------------------

df_balanced = pd.concat([df_capped, df_fwd, df_rev], ignore_index=True)

# Drop the helper columns before writing
df_balanced = df_balanced.drop(columns=["_op", "_mult"], errors="ignore")

# Re-assign solution_id after rows added/removed (concat synthesis changes counts).
df_balanced["solution_id"] = df_balanced.groupby("problem_id").cumcount()

# Ensure column order matches source schema
col_order = list(df.drop(columns=["_op", "_mult"], errors="ignore").columns)
df_balanced = df_balanced[col_order]

print(f"\nWriting {OUT} …")
df_balanced.to_parquet(OUT, index=False)
print(f"  ✓ {len(df_balanced):,} rows written")


# ---------------------------------------------------------------------------
# Step 8 — verification
# ---------------------------------------------------------------------------

print("\n=== Verification ===")

after_counts: Counter = Counter(df_balanced["ops_mapping"].apply(op_of).dropna())

print(f"\n{'op':20} {'before':>8} {'after':>8}")
print("-" * 40)
all_ops = sorted(set(before_counts) | set(after_counts))
for op in all_ops:
    b = before_counts.get(op, 0)
    a = after_counts.get(op, 0)
    flag = " ← changed" if b != a else ""
    print(f"{op:20} {b:8,} {a:8,}{flag}")

# Hard assertions
for op in CAP_OPS:
    assert after_counts[op] == TARGET, f"{op} = {after_counts[op]}, expected {TARGET}"

assert after_counts["concat_fwd"] == TARGET, f"concat_fwd = {after_counts['concat_fwd']}"
assert after_counts["concat_rev"] == TARGET, f"concat_rev = {after_counts['concat_rev']}"

# Unchanged ops (any op not in CAP_OPS and not concat_fwd/rev)
for op, cnt in before_counts.items():
    if op not in CAP_OPS:
        assert after_counts[op] == cnt, (
            f"{op}: before {cnt}, after {after_counts[op]} — unexpectedly changed"
        )

# No source equation fully deleted
df_real = df_balanced[~df_balanced["problem_id"].astype(str).str.startswith("synth_")]
assert df_real["problem_id"].nunique() == df["problem_id"].nunique(), "A source equation was fully deleted!"

# Synthetic rows: answer length 4, answer == solver_answer, fwd != rev verified above
synth = df_balanced[df_balanced["problem_id"].astype(str).str.startswith("synth_")]
assert (synth["answer"].str.len() == 4).all(), "A synthetic answer is not length 4"
assert (synth["answer"] == synth["solver_answer"]).all(), "answer != solver_answer in synthetic"

# Schema / column dtypes match source
assert list(df_balanced.columns) == col_order, "Column order mismatch"

print(f"\n✓ All assertions passed")
print(f"  Total rows: {len(df_balanced):,} (real {len(df_real):,} + synthetic {len(synth):,})")
print(f"  Source equations preserved: {df_real['problem_id'].nunique():,}")
print(f"  Output: {OUT}")
