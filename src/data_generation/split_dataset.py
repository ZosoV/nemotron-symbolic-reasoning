"""Randomly shuffle a parquet dataset and split it in half.

Writes two output files named {stem}_split_0.parquet and {stem}_split_1.parquet
in the same directory as the input (or --output-dir if specified).

Run:
    uv run python src/data_generation/split_dataset.py inputs/data/dataset_examples_3.parquet
    uv run python src/data_generation/split_dataset.py inputs/data/dataset_examples_3.parquet --output-dir outputs/ --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def split(input_path: Path, output_dir: Path, seed: int) -> tuple[int, int]:
    """Shuffle and split input parquet 50/50. Returns (len_split_0, len_split_1)."""
    print(f"Loading {input_path} …")
    df = pd.read_parquet(input_path)
    print(f"  {len(df)} rows")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    mid = len(df) // 2
    split_0 = df.iloc[:mid]
    split_1 = df.iloc[mid:]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    out_0 = output_dir / f"{stem}_split_0.parquet"
    out_1 = output_dir / f"{stem}_split_1.parquet"

    split_0.to_parquet(out_0, index=False)
    split_1.to_parquet(out_1, index=False)
    print(f"  split_0 → {out_0}  ({len(split_0)} rows)")
    print(f"  split_1 → {out_1}  ({len(split_1)} rows)")
    return len(split_0), len(split_1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Randomly shuffle a parquet dataset and split it in half."
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

    n0, n1 = split(input_path, output_dir, args.seed)
    print(f"\n=== summary ===")
    print(f"  split_0  {n0:>6} rows  →  {output_dir}/{input_path.stem}_split_0.parquet")
    print(f"  split_1  {n1:>6} rows  →  {output_dir}/{input_path.stem}_split_1.parquet")


if __name__ == "__main__":
    main()
