"""Merge per-n-shot balanced example datasets into one parquet.

Reads ``dataset_examples_{n}{suffix}.parquet`` for each n in ``--ns`` from
``--input-dir``, tags each row with a ``dataset_category`` string column
(``"examples_{n}"``), rewrites ``problem_id`` to ``"{problem_id}_examples_{n}"``
so it is globally unique in the merged file, and writes the result to ``--output``.

Run:
    uv run python src/data_generation/merge_example_datasets.py
    uv run python src/data_generation/merge_example_datasets.py --ns 1 2 3 --output inputs/data/my_merged.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def merge(
    input_dir: Path,
    ns: list[int],
    suffix: str,
    output: Path,
) -> None:
    frames: list[pd.DataFrame] = []
    ref_cols: list[str] | None = None

    for n in ns:
        path = input_dir / f"dataset_examples_{n}{suffix}.parquet"
        print(f"Loading {path} …")
        df = pd.read_parquet(path)
        print(f"  {len(df)} rows")

        # Schema guard — all files must share identical columns.
        cols = list(df.columns)
        if ref_cols is None:
            ref_cols = cols
        elif cols != ref_cols:
            missing = set(ref_cols) - set(cols)
            extra = set(cols) - set(ref_cols)
            raise ValueError(
                f"Column mismatch for n={n}.\n"
                f"  Missing vs n={ns[0]}: {sorted(missing)}\n"
                f"  Extra   vs n={ns[0]}: {sorted(extra)}"
            )

        category = f"examples_{n}"
        df = df.copy()
        df["dataset_category"] = category
        df["problem_id"] = df["problem_id"] + f"_{category}"
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Uniqueness assertion — fail fast if assumptions break.
    if not merged["problem_id"].is_unique:
        dupes = merged["problem_id"].value_counts()
        dupes = dupes[dupes > 1]
        raise AssertionError(
            f"problem_id not unique after suffixing — {len(dupes)} duplicate(s):\n"
            f"{dupes.head(10).to_dict()}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    print(f"\n→ {output}  ({len(merged)} rows)")

    # Summary
    print("\n=== summary ===")
    print(f"  {'category':<20}  rows")
    for cat, cnt in merged["dataset_category"].value_counts().sort_index().items():
        print(f"  {cat:<20}  {cnt:>6}")
    print(f"  {'TOTAL':<20}  {len(merged):>6}")
    print()
    print("  solver_strategy distribution:")
    for strat, cnt in merged["solver_strategy"].value_counts(dropna=False).items():
        print(f"    {str(strat):<20}  {cnt:>6}")
    no_sol = merged["solver_answer"].isna().sum()
    print(f"\n  no-solution rows (solver_answer null): {no_sol}")
    if no_sol:
        print("  WARNING: no-solution rows present — check source files.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-n-shot balanced example datasets into one parquet. "
            "Adds dataset_category column and suffixes problem_id for global uniqueness."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(_ROOT / "inputs/data"),
        help="Directory containing the source parquet files (default: inputs/data)",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        metavar="N",
        help="Which n-shot variants to merge (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--suffix",
        default="_k_2_balanced",
        help="Filename suffix appended to 'dataset_examples_{n}' (default: _k_2_balanced)",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "inputs/data/dataset_examples_merged.parquet"),
        help="Output parquet path (default: inputs/data/dataset_examples_merged.parquet)",
    )
    args = parser.parse_args(argv)

    merge(
        input_dir=Path(args.input_dir),
        ns=args.ns,
        suffix=args.suffix,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
