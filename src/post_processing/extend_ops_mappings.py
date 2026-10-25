"""Expand solver-results parquet: enumerate all op-mapping combinations from ops_candidates.

For each puzzle that has a non-null ops_candidates, computes the Cartesian product of the
candidate-op lists per operator glyph and emits one row per combination.  Each variant is
re-verified with encode_answer + verify_program.

Rows without ops_candidates (pure_concat, no_solution) pass through as a single variant.

Because the original solver search is gold-conditioned (the to_predict equation is included
as a constraint), every op in ops_candidates already satisfies all equations including the
query.  The Cartesian product therefore yields only valid alternative solutions, all of which
should reproduce the gold answer.

Usage:
    uv run python src/post_processing/extend_ops_mappings.py
    uv run python src/post_processing/extend_ops_mappings.py data/solver_results.parquet
    uv run python src/post_processing/extend_ops_mappings.py data/solver_results.parquet -o data/out.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.encoding import encode_answer, verify_program  # noqa: E402

DEFAULT_PARQUET = _ROOT / "data/solver_results_forcing_base_10.parquet"


def _get_query_lhs(row: pd.Series) -> dict | None:
    """Extract query LHS from the stored equations JSON or fall back to prompt parsing.

    Mirrors eval_from_parquet._get_query_lhs.
    """
    equations_json = row.get("equations")
    if equations_json and not (isinstance(equations_json, float) and pd.isna(equations_json)):
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


def _passthrough(row: pd.Series, base: dict) -> list[dict]:
    """Return a single pass-through variant for rows without arithmetic mappings."""
    return [{
        **base,
        "ops_mapping":     row.get("ops_mapping"),
        "solver_answer":   row.get("solver_answer"),
        "solver_correct":  row.get("solver_correct"),
        "variant_index":   0,
        "is_original":     True,
        "num_variants":    1,
        "fully_validated": None,
    }]


def _expand_row(row: pd.Series) -> list[dict]:
    """Expand one input row into one or more variant rows.

    Returns one pass-through dict for pure_concat / no_solution rows, or one dict per
    Cartesian-product op-mapping combination for arithmetic rows.
    """
    gold = str(row.get("answer", "")).strip()

    # Columns carried unchanged to every variant.
    base: dict = {
        "problem_id":      row.get("problem_id"),
        "prompt":          row.get("prompt"),
        "answer":          row.get("answer"),
        "solver_strategy": row.get("solver_strategy"),
        "solver_category": row.get("solver_category"),
        "problem_category": row.get("problem_category"),
        "digits_mapping":  row.get("digits_mapping"),
        "solver_tier":     row.get("solver_tier"),
        "solver_radix":    row.get("solver_radix"),
        "ops_candidates":  row.get("ops_candidates"),
        "equations":       row.get("equations"),
    }

    # --- Pass-through: no ops_candidates (pure_concat / no_solution) ---
    cands_json = row.get("ops_candidates")
    if cands_json is None or (isinstance(cands_json, float) and pd.isna(cands_json)):
        return _passthrough(row, base)

    # --- Parse required fields ---
    try:
        ops_candidates: dict[str, list[str]] = json.loads(cands_json)
    except Exception:
        return _passthrough(row, base)

    digits_json = row.get("digits_mapping")
    ops_json    = row.get("ops_mapping")
    strategy    = row.get("solver_strategy")

    if (
        digits_json is None or (isinstance(digits_json, float) and pd.isna(digits_json))
        or ops_json  is None or (isinstance(ops_json,    float) and pd.isna(ops_json))
    ):
        return _passthrough(row, base)

    try:
        mapping: dict[str, int]   = json.loads(digits_json)
        original_ops: dict[str, str] = json.loads(ops_json)
    except Exception:
        return _passthrough(row, base)

    # Derive mode and radix — mirrors eval_from_parquet.py logic including legacy typo.
    mode  = "little_endian" if strategy in ("litte_endian", "little_endian") else "standard"
    radix_raw = row.get("solver_radix")
    radix = int(radix_raw) if radix_raw is not None and not pd.isna(radix_raw) else len(mapping)

    # Parse equations for verify_program.
    equations: list[dict] = []
    equations_json = row.get("equations")
    if equations_json and not (isinstance(equations_json, float) and pd.isna(equations_json)):
        try:
            equations = json.loads(equations_json)
        except Exception:
            pass

    query_lhs = _get_query_lhs(row)

    # --- Build Cartesian product ---
    keys   = list(ops_candidates.keys())
    combos = [dict(zip(keys, choice)) for choice in itertools.product(*ops_candidates.values())]
    num_variants = len(combos)

    results: list[dict] = []
    for idx, combo in enumerate(combos):
        is_original = (combo == original_ops)

        # Recompute solver_answer for this combo.
        solver_answer: str | None = None
        solver_correct            = False
        if query_lhs is not None:
            encoded = encode_answer(
                mapping=mapping,
                ops=combo,
                op_char=query_lhs["op"],
                query_lhs=query_lhs,
                mode=mode,
                radix=radix,
                answer_hint=gold,
            )
            if encoded is not None:
                solver_answer, _ = encoded
                solver_correct   = solver_answer == gold

        # Full-program verification.
        fully_validated = False
        if equations:
            try:
                fully_validated = verify_program(
                    equations=equations,
                    mapping=mapping,
                    ops_mapping=combo,
                    mode=mode,
                    radix=radix,
                )
            except Exception:
                pass

        results.append({
            **base,
            "ops_mapping":     json.dumps(combo),
            "solver_answer":   solver_answer,
            "solver_correct":  solver_correct,
            "variant_index":   idx,
            "is_original":     is_original,
            "num_variants":    num_variants,
            "fully_validated": fully_validated,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand solver-results parquet with all ops_candidates combinations."
    )
    parser.add_argument(
        "parquet",
        nargs="?",
        default=str(DEFAULT_PARQUET),
        help=(
            "Path to input solver_results parquet "
            "(default: data/solver_results_forcing_base_10.parquet)"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output parquet path "
            "(default: <input_dir>/<input_stem>_extended.parquet)"
        ),
    )
    args = parser.parse_args()

    input_path  = Path(args.parquet)
    output_path = Path(args.output) if args.output else (
        input_path.parent / f"{input_path.stem}_extended.parquet"
    )

    print(f"Loading {input_path} …")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} puzzles")

    # Expand every row.
    all_rows: list[dict] = []
    for _, row in df.iterrows():
        all_rows.extend(_expand_row(row))

    out_df = pd.DataFrame(all_rows)
    # Re-assign solution_id after variant expansion (row count per puzzle changes).
    out_df["solution_id"] = out_df.groupby("problem_id").cumcount()

    # --- Summary ---
    total_in  = len(df)
    total_out = len(out_df)
    per_puzzle = out_df.groupby("problem_id")["num_variants"].first()
    multi      = int((per_puzzle > 1).sum())
    mean_var   = per_puzzle.mean()

    print(f"\n=== expansion summary ===")
    print(f"input puzzles       : {total_in}")
    print(f"output rows (total) : {total_out}")
    print(f"mean variants/puzzle: {mean_var:.2f}")
    print(f"puzzles with >1 var (ambiguous samples): {multi}")

    # Flag anomalies (expected: 0 for a gold-conditioned parquet).
    # Only check arithmetic variants (fully_validated is not None); pass-through rows
    # (pure_concat / no_solution) have fully_validated=None and are excluded.
    evaled = out_df[out_df["fully_validated"].notna()]
    failed_val  = int((evaled["fully_validated"] == False).sum())  # noqa: E712
    failed_corr = int((evaled["solver_correct"] == False).sum())   # noqa: E712

    if failed_val:
        print(f"\n  ⚠  variants failing full verification : {failed_val}")
        print(evaled[evaled["fully_validated"] == False][  # noqa: E712
            ["problem_id", "variant_index", "ops_mapping", "solver_answer", "answer"]
        ].to_string(index=False))
    else:
        print("fully_validated failures    : 0  ✓")

    if failed_corr:
        print(f"\n  ⚠  arithmetic variants with wrong solver_answer : {failed_corr}")
        print(evaled[evaled["solver_correct"] == False][  # noqa: E712
            ["problem_id", "variant_index", "ops_mapping", "solver_answer", "answer"]
        ].to_string(index=False))
    else:
        print("solver_correct failures     : 0  ✓")

    # Report pass-through rows (no ops_candidates) separately.
    passthrough = out_df[out_df["fully_validated"].isna()]
    if len(passthrough):
        print(f"pass-through rows (pure_concat/no_solution): {len(passthrough)}")

    out_df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(out_df)} rows → {output_path}")


if __name__ == "__main__":
    main()
