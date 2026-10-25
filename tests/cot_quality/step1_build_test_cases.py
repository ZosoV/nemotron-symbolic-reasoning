"""Step 1 — Build a labeled test-case parquet for CoT quality evaluation.

Reads the combined canonical parquet and selects one representative row per
case of interest, covering every structural variant the CoT generator must
handle.  Writes a small parquet with an added ``test_case`` column.

Usage:
    uv run python tests/cot_quality/step1_build_test_cases.py [SOURCE] [--out OUT]

Defaults:
    SOURCE  inputs/data/v3/combined_1k_canonical.parquet
    --out   inputs/data/cot_quality/test_cases.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SOURCE_DEFAULT = _ROOT / "inputs" / "data" / "v3" / "combined_1k_canonical.parquet"
_OUT_DEFAULT    = _ROOT / "inputs" / "data" / "cot_quality" / "test_cases.parquet"

_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_eq(eq_str) -> list:
    if isinstance(eq_str, str):
        return json.loads(eq_str)
    return []

def _parse_dm(dm_str) -> dict:
    if isinstance(dm_str, str):
        return json.loads(dm_str)
    return {}


# ── Predicates ────────────────────────────────────────────────────────────────

def _zero_pad_in_answer(row) -> bool:
    dm = _parse_dm(row["digits_mapping"])
    ans = str(row["answer"]) if pd.notna(row["answer"]) else ""
    return bool(ans) and ans[0] in dm and dm[ans[0]] == 0

def _zero_pad_in_example(row) -> bool:
    dm = _parse_dm(row["digits_mapping"])
    if not dm:
        return False
    for eq in _parse_eq(row["equations"]):
        if eq["type"] == "example":
            rhs = eq["rhs"].get("actual_rhs", [])
            if rhs and isinstance(rhs[0], str) and rhs[0] in dm and dm[rhs[0]] == 0:
                return True
    return False

def _answer_sign_pos(row):
    for eq in _parse_eq(row["equations"]):
        if eq["type"] == "to_predict":
            return eq["rhs"].get("sign_pos")
    return None

def _sign_in_example(row, sign_type: str) -> bool:
    for eq in _parse_eq(row["equations"]):
        if eq["type"] == "example" and eq["rhs"].get("sign_pos") == sign_type:
            return True
    return False

def _no_demo_for_answer_op(row) -> bool:
    eqs = _parse_eq(row["equations"])
    tp_op = next(
        (eq["lhs"].get("op") for eq in eqs if eq["type"] == "to_predict"), None
    )
    if tp_op is None:
        return False
    return not any(
        eq["type"] == "example" and eq["lhs"].get("op") == tp_op for eq in eqs
    )

def _unseen_in_answer(row) -> bool:
    """True when the answer contains a symbol never seen in any example equation
    or the to_predict LHS (the CoT must assign a digit for it with no grounding)."""
    dm = _parse_dm(row["digits_mapping"])
    if not dm:
        return False
    example_syms: set = set()
    tp_lhs_syms: set = set()
    tp_rhs_syms: set = set()
    for eq in _parse_eq(row["equations"]):
        lhs = eq.get("lhs", {})
        lhs_set = {v for v in [lhs.get("l0"), lhs.get("l1"), lhs.get("r0"), lhs.get("r1")] if v}
        rhs_set = {s for s in eq["rhs"].get("actual_rhs", []) if s}
        if eq["type"] == "example":
            example_syms |= lhs_set | rhs_set
        elif eq["type"] == "to_predict":
            tp_lhs_syms |= lhs_set
            tp_rhs_syms |= rhs_set
    return bool(tp_rhs_syms - (example_syms | tp_lhs_syms))


# ── Cases ─────────────────────────────────────────────────────────────────────

# Each entry: (case_name, description, predicate)
_CASES: list[tuple[str, str, object]] = [
    (
        "case_1_little_endian",
        "Little-endian solution (digits read right-to-left)",
        lambda r: r["solver_strategy"] == "little_endian",
    ),
    (
        "case_2_standard",
        "Standard solution (digits read left-to-right)",
        lambda r: r["solver_strategy"] == "standard",
    ),
    (
        "case_3_pure_concat",
        "Pure-concat puzzle (no digit mapping needed)",
        lambda r: r["solver_strategy"] == "pure_concat",
    ),
    (
        "case_4_zero_pad_demo",
        "Answer requires zero-padding AND an example demonstrates zero-padded width",
        lambda r: _zero_pad_in_answer(r) and _zero_pad_in_example(r),
    ),
    (
        "case_5_sign_prefix_demo",
        "Answer has sign prefix AND an example shows prefix sign",
        lambda r: _answer_sign_pos(r) == "prefix" and _sign_in_example(r, "prefix"),
    ),
    (
        "case_6_sign_postfix_demo",
        "Answer has sign postfix AND an example shows postfix sign",
        lambda r: _answer_sign_pos(r) == "postfix" and _sign_in_example(r, "postfix"),
    ),
    (
        "case_7a_zero_pad_no_demo",
        "Answer requires zero-padding but NO example demonstrates the width",
        lambda r: _zero_pad_in_answer(r) and not _zero_pad_in_example(r),
    ),
    (
        "case_7b_sign_prefix_no_demo",
        "Answer has sign prefix but NO example shows a prefix sign",
        lambda r: _answer_sign_pos(r) == "prefix" and not _sign_in_example(r, "prefix"),
    ),
    (
        "case_7c_sign_postfix_no_demo",
        "Answer has sign postfix but NO example shows a postfix sign",
        lambda r: _answer_sign_pos(r) == "postfix" and not _sign_in_example(r, "postfix"),
    ),
    (
        "case_8_examples_3",
        "3-example puzzle (dataset_category = examples_3)",
        lambda r: r["dataset_category"] == "examples_3",
    ),
    (
        "case_8_examples_4",
        "4-example puzzle (dataset_category = examples_4)",
        lambda r: r["dataset_category"] == "examples_4",
    ),
    (
        "case_8_examples_5",
        "5-example puzzle (dataset_category = examples_5)",
        lambda r: r["dataset_category"] == "examples_5",
    ),
    (
        "case_9_no_op_demo",
        "No example equation shares the question's operator (CoT gets no grounding for it)",
        _no_demo_for_answer_op,
    ),
    (
        "case_10_unseen_digit_sym",
        "Answer symbol never appears in example equations or to_predict LHS",
        _unseen_in_answer,
    ),
]


# ── Selection ─────────────────────────────────────────────────────────────────

def _pick(df: pd.DataFrame, pred, exclude_ids: set) -> int | None:
    candidates = [i for i, row in df.iterrows() if pred(row) and row["problem_id"] not in exclude_ids]
    if not candidates:
        candidates = [i for i, row in df.iterrows() if pred(row)]
    if not candidates:
        return None
    return random.choice(candidates)


def build_test_cases(df: pd.DataFrame) -> pd.DataFrame:
    random.seed(_SEED)
    used_ids: set = set()
    rows = []

    for case_name, description, pred in _CASES:
        idx = _pick(df, pred, used_ids)
        if idx is None:
            print(f"  WARNING: no row found for {case_name}")
            continue
        row = df.loc[idx].copy()
        row["test_case"] = case_name
        row["test_case_description"] = description
        rows.append(row)
        used_ids.add(row["problem_id"])

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Counts summary ────────────────────────────────────────────────────────────

def print_counts(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"\n{'Case':<35} {'Count':>7}  {'%':>7}")
    print("-" * 54)
    for case_name, description, pred in _CASES:
        count = sum(1 for _, row in df.iterrows() if pred(row))
        print(f"{case_name:<35} {count:>7}  {count / total * 100:>6.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=str(_SOURCE_DEFAULT))
    parser.add_argument("--out", default=str(_OUT_DEFAULT))
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {src} ...")
    df = pd.read_parquet(src)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("\nCounts per case in source:")
    print_counts(df)

    print("\nSelecting test cases ...")
    out_df = build_test_cases(df)
    print(f"  Selected {len(out_df)} rows")
    print(out_df[["test_case", "problem_id", "solver_strategy", "dataset_category"]].to_string(index=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
