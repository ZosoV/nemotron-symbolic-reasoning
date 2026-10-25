"""Re-evaluate solver answers by re-applying stored mappings from solver_results.parquet.

For every row that has a non-null ops_mapping and digits_mapping, parses the query
from the equations field (or falls back to prompt parsing), re-applies the stored
symbol->digit mapping and op->function rules, and checks whether the computed
answer matches the gold answer.

Rows without stored mappings (pure_concat or no-solution) are skipped.

Usage:
    uv run python src/eval_from_parquet.py
    uv run python src/eval_from_parquet.py data/solver_results.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-exports from solver_eq_symbolic for backward compatibility.
from src.solver.solver_eq_symbolic import (  # noqa: F401
    OPERATIONS,
    SIGNED_OPS,
    _int_to_base_digits,
    _is_reversed_digit_mode,
)
from src.encoding import encode_answer, verify_program

DEFAULT_PARQUET = _ROOT / "data/solver_results_new.parquet"


def _get_query_lhs(row: pd.Series) -> dict | None:
    """Extract query LHS from the stored equations JSON or fall back to prompt."""
    equations_json = row.get("equations")
    if equations_json and not pd.isna(equations_json):
        try:
            equations = json.loads(equations_json)
            to_predict = next(
                (eq for eq in equations if eq.get("type") == "to_predict"), None
            )
            if to_predict:
                return to_predict["lhs"]
        except Exception:
            pass

    # Fallback: parse query from prompt.
    prompt = row.get("prompt", "")
    for line in prompt.strip().split("\n"):
        low = line.lower()
        if "determine the result for:" in low:
            idx = low.index("determine the result for:")
            query = line[idx + len("determine the result for:"):].strip()
            if len(query) >= 5:
                return {
                    "l0": query[0],
                    "l1": query[1],
                    "op": query[2],
                    "r0": query[3],
                    "r1": query[4],
                }
    return None


def _eval_row(row: pd.Series) -> dict:
    out: dict = {
        "problem_id": row.get("problem_id"),
        "gold": str(row["answer"]).strip(),
        "solver_answer": row.get("solver_answer"),
        "solver_correct": row.get("solver_correct"),
        "re_eval_answer": None,
        "re_eval_correct": False,
        "fully_validated": False,
        "skipped": False,
        "skip_reason": None,
        "solver_strategy": row.get("solver_strategy"),
    }

    # Skip rows without digit/op mappings (pure_concat or no_solution).
    digits_json = row.get("digits_mapping")
    ops_json = row.get("ops_mapping")
    strategy = row.get("solver_strategy")

    if pd.isna(digits_json) if isinstance(digits_json, float) else digits_json is None:
        out["skipped"] = True
        out["skip_reason"] = str(strategy) if strategy else "no_solution"
        return out
    if pd.isna(ops_json) if isinstance(ops_json, float) else ops_json is None:
        out["skipped"] = True
        out["skip_reason"] = str(strategy) if strategy else "no_solution"
        return out

    try:
        mapping: dict[str, int] = json.loads(digits_json)
        ops: dict[str, str] = json.loads(ops_json)
    except Exception as exc:
        out["skipped"] = True
        out["skip_reason"] = f"json_error: {exc}"
        return out

    # Keep "litte_endian" for backward compat with parquet files written before the typo was fixed.
    mode = "little_endian" if strategy in ("litte_endian", "little_endian") else "standard"
    radix_raw = row.get("solver_radix")
    radix = int(radix_raw) if radix_raw is not None and not pd.isna(radix_raw) else len(mapping)

    query_lhs = _get_query_lhs(row)
    if query_lhs is None:
        out["skipped"] = True
        out["skip_reason"] = "no_query"
        return out

    op_char = query_lhs["op"]
    gold = out["gold"]

    # Derive target_len from the gold answer so zero-padding is reproduced.
    gold_mag = gold[1:] if (len(gold) > 1 and gold[0] == op_char) else gold
    target_len = len(gold_mag)

    # Build an answer_hint string of the correct magnitude length for zero-padding.
    # Reuse the actual gold answer as the hint.
    re_ans_result = encode_answer(
        mapping=mapping,
        ops=ops,
        op_char=op_char,
        query_lhs=query_lhs,
        mode=mode,
        radix=radix,
        answer_hint=gold,
    )

    if re_ans_result is not None:
        re_ans, _ = re_ans_result
        out["re_eval_answer"] = re_ans
        out["re_eval_correct"] = re_ans == gold

    # Full-program verification: all equations (examples + to_predict) must hold.
    # Requires the stored equations JSON (same dicts produced by extract_all_equations).
    equations_json = row.get("equations")
    if equations_json and not (isinstance(equations_json, float) and pd.isna(equations_json)):
        try:
            equations = json.loads(equations_json)
            out["fully_validated"] = verify_program(
                equations=equations,
                mapping=mapping,
                ops_mapping=ops,
                mode=mode,
                radix=radix,
            )
        except Exception:
            pass  # leave fully_validated=False if parsing fails

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate solver answers from parquet")
    parser.add_argument(
        "parquet",
        nargs="?",
        default=str(DEFAULT_PARQUET),
        help="path to solver_results.parquet",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")

    results = [_eval_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(results)

    total = len(out)
    skipped = int(out["skipped"].sum())
    evaluated = total - skipped
    correct = int(out["re_eval_correct"].sum())
    wrong = evaluated - correct

    print("\n=== re-evaluation results ===")
    print(f"total rows : {total}")
    print(f"skipped    : {skipped}  (pure_concat or no stored mapping)")

    skipped_df = out[out["skipped"]]
    for reason, grp in skipped_df.groupby("skip_reason", dropna=False):
        print(f"  {str(reason):30s}  {len(grp)}")

    print(f"evaluated  : {evaluated}")
    if evaluated:
        fully_val = int(out["fully_validated"].sum())
        print(f"correct    : {correct}  ({correct / evaluated * 100:.1f}%)")
        print(f"wrong      : {wrong}  ({wrong / evaluated * 100:.1f}%)")
        print(f"fully validated (all eqs): {fully_val}  ({fully_val / evaluated * 100:.1f}%)")

        # Sanity: rows where query answer matches gold but program fails some equation.
        evaled = out[~out["skipped"]]
        partial = evaled[evaled["re_eval_correct"] & ~evaled["fully_validated"]]
        if len(partial):
            print(f"\n  ⚠  query-correct but not fully validated: {len(partial)}")
            print(partial[["problem_id", "gold", "re_eval_answer", "solver_strategy"]].to_string(index=False))

    if evaluated:
        evaled = out[~out["skipped"]]
        print("\n--- by strategy ---")
        for strat, grp in evaled.groupby("solver_strategy", dropna=False):
            n = len(grp)
            c = int(grp["re_eval_correct"].sum())
            fv = int(grp["fully_validated"].sum())
            print(f"  {str(strat or 'unknown'):40s}  correct {c:4d}/{n:<4d}  fully_val {fv:4d}/{n:<4d}")

    # Compare with original solver_correct to surface any discrepancies.
    if "solver_correct" in df.columns:
        orig_total = int(df["solver_correct"].fillna(False).sum())
        print(f"\noriginal solver_correct : {orig_total}/{total}  ({orig_total / total * 100:.1f}%)")

        evaled = out[~out["skipped"]]
        if len(evaled):
            disagree = evaled[
                evaled["re_eval_correct"] != evaled["solver_correct"].astype(bool)
            ]
            if len(disagree):
                print(f"discrepancies (re_eval ≠ solver_correct): {len(disagree)}")
                print(
                    disagree[
                        ["problem_id", "gold", "solver_answer", "re_eval_answer", "solver_strategy"]
                    ].to_string(index=False)
                )
            else:
                print("no discrepancies between re_eval and original solver_correct")


if __name__ == "__main__":
    main()
