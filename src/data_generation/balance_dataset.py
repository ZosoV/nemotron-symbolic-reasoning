"""Balance standard / little_endian solutions in a solver-results parquet dataset.

For each puzzle (problem_id) the script ensures exactly one solution row is kept and that
the overall standard vs. little_endian row counts are as equal as possible.

Strategy
--------
- ``solution_count == 1`` rows (single-solution puzzles): kept as-is — their strategy is
  fixed and cannot be changed.
- ``solution_count == 2`` rows (one standard + one little_endian row per puzzle): the
  flexible pool — for each puzzle we choose which strategy row to keep, giving us a lever
  to push the global standard / little_endian ratio toward 50-50.
- ``pure_concat`` rows: conserved unchanged.
- All other rows (no-solution, solution_count >= 3): **dropped**.

If the fixed single-solution rows are already too skewed for the 2-solution pool to fully
compensate (|S - L| > T), a warning is printed and the best achievable balance is returned
without dropping any single-solution puzzles.

Run:
    uv run python src/data_generation/balance_dataset.py inputs/data/dataset_examples_3.parquet
    uv run python src/data_generation/balance_dataset.py inputs/data/dataset_examples_3.parquet --output-dir outputs/ --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Canonical strategy names used throughout this script.
_STANDARD = "standard"
_LITTLE_ENDIAN = "little_endian"
_PURE_CONCAT = "pure_concat"

# Legacy typo emitted by older solver versions — normalised on load.
_LEGACY_LITTLE_ENDIAN = "litte_endian"


def _normalize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy 'litte_endian' typo with 'little_endian' in-place (copy)."""
    df = df.copy()
    df["solver_strategy"] = df["solver_strategy"].replace(
        _LEGACY_LITTLE_ENDIAN, _LITTLE_ENDIAN
    )
    return df


def balance(input_path: Path, output_dir: Path, seed: int) -> dict[str, int]:
    """Balance the dataset and write output. Returns a stats dict for the summary."""
    print(f"Loading {input_path} …")
    df = pd.read_parquet(input_path)
    n_input = len(df)
    print(f"  {n_input} rows")

    # ------------------------------------------------------------------ #
    # 1. Normalise problem_category and legacy strategy typo.
    # ------------------------------------------------------------------ #
    df["problem_category"] = "Cryptarithm (Symbolic Equations)"
    df = _normalize_strategy(df)

    # ------------------------------------------------------------------ #
    # 2. Partition rows.
    # ------------------------------------------------------------------ #
    mask_pure_concat = df["solver_strategy"] == _PURE_CONCAT
    mask_one = (df["solution_count"] == 1) & df["solver_strategy"].isin(
        {_STANDARD, _LITTLE_ENDIAN}
    )
    mask_two = df["solution_count"] == 2

    pure_concat = df[mask_pure_concat].copy()
    one_solution = df[mask_one].copy()
    two_solution = df[mask_two].copy()

    n_dropped = n_input - len(pure_concat) - len(one_solution) - len(two_solution)

    print(f"  pure_concat rows   : {len(pure_concat)}")
    print(f"  one-solution rows  : {len(one_solution)}")
    print(f"  two-solution rows  : {len(two_solution)}")
    if n_dropped:
        print(f"  dropped rows       : {n_dropped}  (no-solution / solution_count>=3)")

    # ------------------------------------------------------------------ #
    # 3. Validate the two-solution partition.
    # ------------------------------------------------------------------ #
    if not two_solution.empty:
        grp = two_solution.groupby("problem_id")
        row_counts = grp.size()
        assert (row_counts == 2).all(), (
            f"Expected exactly 2 rows per problem_id in two-solution set; "
            f"found problems with other counts: {row_counts[row_counts != 2].to_dict()}"
        )
        strategy_counts = grp["solver_strategy"].nunique()
        assert (strategy_counts == 2).all(), (
            "Expected each two-solution problem_id to have one 'standard' and one "
            "'little_endian' row; found duplicates or unexpected strategies."
        )

    # ------------------------------------------------------------------ #
    # 4. Counts.
    # ------------------------------------------------------------------ #
    S = int((one_solution["solver_strategy"] == _STANDARD).sum())
    L = int((one_solution["solver_strategy"] == _LITTLE_ENDIAN).sum())
    T = int((two_solution["solver_strategy"] == _STANDARD).sum())  # == T le as well

    print(f"\n  Fixed single-solution — standard    : {S}")
    print(f"  Fixed single-solution — little_endian: {L}")
    print(f"  Flexible two-solution puzzles         : {T}")

    # ------------------------------------------------------------------ #
    # 5. Balance math.
    # ------------------------------------------------------------------ #
    total_length = S + L + T
    balanced_half = total_length // 2
    desired = balanced_half - S  # how many 2-sol puzzles to keep as standard
    n_sample_std = max(0, min(desired, T))  # best-effort clamp

    if n_sample_std != desired:
        residual = abs(S - L) - T
        print(
            f"\n  WARNING: cannot fully balance — fixed single-solution rows too skewed "
            f"(|S-L|={abs(S - L)} > T={T}); residual imbalance = {residual} rows. "
            f"Returning best-achievable balance (no rows dropped)."
        )

    print(f"\n  balanced_half target  : {balanced_half}")
    print(f"  2-sol puzzles → std   : {n_sample_std}")
    print(f"  2-sol puzzles → le    : {T - n_sample_std}")

    # ------------------------------------------------------------------ #
    # 6. Sample which two-solution puzzle ids keep the standard row.
    # ------------------------------------------------------------------ #
    two_problem_ids = list(two_solution["problem_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(two_problem_ids)
    std_ids = set(two_problem_ids[:n_sample_std])
    le_ids = set(two_problem_ids[n_sample_std:])

    # ------------------------------------------------------------------ #
    # 7. Collapse two-solution puzzles to one row each.
    # ------------------------------------------------------------------ #
    kept_std = two_solution[
        (two_solution["solver_strategy"] == _STANDARD)
        & two_solution["problem_id"].isin(std_ids)
    ].copy()
    kept_std["solution_count"] = 1
    kept_std["standard_solution_count"] = 1
    kept_std["little_endian_solution_count"] = 0

    kept_le = two_solution[
        (two_solution["solver_strategy"] == _LITTLE_ENDIAN)
        & two_solution["problem_id"].isin(le_ids)
    ].copy()
    kept_le["solution_count"] = 1
    kept_le["standard_solution_count"] = 0
    kept_le["little_endian_solution_count"] = 1

    # ------------------------------------------------------------------ #
    # 8. Concatenate and reassign solution_id.
    # ------------------------------------------------------------------ #
    out = pd.concat(
        [one_solution, kept_std, kept_le, pure_concat], ignore_index=True
    )
    out["solution_id"] = out.groupby("problem_id").cumcount()
    out = out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 9. Write output.
    # ------------------------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    out_path = output_dir / f"{stem}_balanced.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  → {out_path}  ({len(out)} rows)")

    final_std = int((out["solver_strategy"] == _STANDARD).sum())
    final_le = int((out["solver_strategy"] == _LITTLE_ENDIAN).sum())
    final_pc = int((out["solver_strategy"] == _PURE_CONCAT).sum())

    return {
        "n_input": n_input,
        "n_dropped": n_dropped,
        "final_standard": final_std,
        "final_little_endian": final_le,
        "final_pure_concat": final_pc,
        "final_total": len(out),
        "balanced_half": balanced_half,
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Balance standard / little_endian solution counts in a solver-results "
            "parquet dataset. Produces exactly one solution row per problem_id."
        )
    )
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same directory as input)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    stats = balance(input_path, output_dir, args.seed)

    print(f"\n=== summary ===")
    print(f"  input rows     {stats['n_input']:>6}")
    print(f"  dropped rows   {stats['n_dropped']:>6}  (no-solution / solution_count>=3)")
    print(f"  standard       {stats['final_standard']:>6}")
    print(f"  little_endian  {stats['final_little_endian']:>6}")
    print(f"  pure_concat    {stats['final_pure_concat']:>6}")
    print(f"  total output   {stats['final_total']:>6}")
    print(f"  balanced_half  {stats['balanced_half']:>6}")
    print(f"  output → {stats['out_path']}")


if __name__ == "__main__":
    main()
