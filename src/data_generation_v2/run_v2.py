"""CLI orchestrator for the v2 data generation pipeline.

Stages:
  stats   — compute distributions from solver results parquet (cached to JSON)
  base    — build random-drop base pools (~804/804/804/570/278)
  augment — gap-fill augmentation (Strategy A for n=1,2; Strategy B for n=3,4,5)
  emit    — write parquets and self-validate
  coverage— simulate the curriculum schedule and print coverage table
  all     — run all stages in sequence

Usage:
    uv run python src/data_generation_v2/run_v2.py --stage all
    uv run python src/data_generation_v2/run_v2.py --stage coverage \\
        inputs/data/v2/dataset_v2_base.parquet \\
        inputs/data/v2/dataset_v2_augmentation.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Default gap-fill targets (derived from 250-step coverage analysis).
# Adjust these if verify_coverage shows coverage ≠ ~1× after generation.
DEFAULT_GAP_COUNTS: dict[int, int] = {
    1: 1669,  # n=1 augmentation target (fixed-point, ~1x coverage at 250 steps)
    2: 234,   # n=2 augmentation target
    3: 293,   # n=3 augmentation target
    4: 221,   # n=4 augmentation target
    5: 323,   # n=5 augmentation target
    # Derived by iterating: pools = base + gap; coverage = draws(pools)/pools ≈ 1×.
    # n=1 is large because the curriculum fallback gives bucket 1 extra draws until
    # bucket 2's CDF threshold is reached (steps 0-~90 with these pool sizes).
    # Re-run --stage coverage after any pool change to verify.
}

DEFAULT_SOLVER_RESULTS = _ROOT / "inputs" / "data" / "eval_data_k_1.parquet"
DEFAULT_OUTPUT_DIR = _ROOT / "inputs" / "data"


def main(argv=None):
    parser = argparse.ArgumentParser(description="v2 data generation pipeline")
    parser.add_argument(
        "--stage",
        choices=["stats", "base", "augment", "emit", "coverage", "all"],
        default="all",
    )
    parser.add_argument("--solver-results", default=str(DEFAULT_SOLVER_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gaps",
        nargs="*",
        default=None,
        help="Override gap counts as 'n:count' pairs, e.g. --gaps 1:742 2:478",
    )
    parser.add_argument(
        "--num-steps", type=int, default=250,
        help="Curriculum steps for coverage simulation",
    )
    parser.add_argument(
        "--no-pure-concat", action="store_true", default=False,
        help="Exclude pure_concat from synthetic augmentation (Strategy B)",
    )
    # For coverage-only stage: accept extra positional parquet paths
    parser.add_argument("parquets", nargs="*", help="Parquet files for coverage stage")
    args = parser.parse_args(argv)

    parquet_path = Path(args.solver_results)
    output_dir = Path(args.output_dir)
    seed = args.seed

    if args.stage in ("base", "augment", "emit", "all"):
        if not parquet_path.exists():
            print(f"ERROR: {parquet_path} not found.")
            sys.exit(1)

    # Parse custom gap counts
    gap_counts = dict(DEFAULT_GAP_COUNTS)
    if args.gaps:
        for item in args.gaps:
            n_str, count_str = item.split(":")
            gap_counts[int(n_str)] = int(count_str)

    # ------------------------------------------------------------------
    # Stage: stats
    # ------------------------------------------------------------------
    if args.stage in ("stats", "all"):
        print("=== Stage: stats ===")
        from src.data_generation_v2.stats import compute_stats
        stats = compute_stats(parquet_path)
        print(f"  Top 5 ops by P(op): "
              f"{list(stats.op_probs.items())[:5]}")
        print(f"  Strategies: {stats.strategy_probs}")
        print(f"  Symbol pool size (global): {len(stats.global_sym_probs)}")

    # ------------------------------------------------------------------
    # Stage: base
    # ------------------------------------------------------------------
    base_samples = None
    usage_map = None
    if args.stage in ("base", "emit", "all"):
        print("\n=== Stage: base ===")
        from src.data_generation_v2.base import build_base_pools
        base_samples, usage_map = build_base_pools(parquet_path, seed=seed)
        from collections import Counter
        cats = Counter(r["dataset_category"] for r in base_samples)
        for n in sorted(cats):
            print(f"  {n}: {cats[n]}")
        print(f"  Total: {len(base_samples)}")

    # ------------------------------------------------------------------
    # Stage: augment
    # ------------------------------------------------------------------
    aug_samples = None
    if args.stage in ("augment", "all"):
        if usage_map is None:
            print("\n=== Rebuilding base (needed for usage_map) ===")
            from src.data_generation_v2.base import build_base_pools
            base_samples, usage_map = build_base_pools(parquet_path, seed=seed)

        print("\n=== Stage: augment ===")
        from src.data_generation_v2.stats import compute_stats
        stats = compute_stats(parquet_path)

        # Strategy A: n=1, n=2
        print("--- Strategy A (recombine n=1,2) ---")
        from src.data_generation_v2.recombine import build_recombination
        recomb = build_recombination(
            parquet_path,
            usage_map,
            target_counts={n: gap_counts[n] for n in (1, 2)},
            seed=seed,
        )

        # Strategy B: n=3, n=4, n=5
        print("--- Strategy B (synthetic n=3,4,5) ---")
        from src.data_generation_v2.synth import build_synthetic
        synth = build_synthetic(
            stats,
            target_counts={n: gap_counts[n] for n in (3, 4, 5)},
            seed=seed,
            include_pure_concat=not args.no_pure_concat,
        )

        aug_samples = recomb + synth
        print(f"\n  Total augmentation: {len(aug_samples)}")

    # ------------------------------------------------------------------
    # Stage: emit
    # ------------------------------------------------------------------
    if args.stage in ("emit", "all"):
        if base_samples is None or aug_samples is None:
            print("ERROR: run with --stage all to generate base + augment before emit")
            sys.exit(1)
        print("\n=== Stage: emit ===")
        from src.data_generation_v2.emit import emit_parquets
        base_path, aug_path = emit_parquets(base_samples, aug_samples, output_dir)

    # ------------------------------------------------------------------
    # Stage: coverage
    # ------------------------------------------------------------------
    if args.stage in ("coverage", "all"):
        print("\n=== Stage: coverage ===")
        from src.data_generation_v2.verify_coverage import verify_coverage

        if args.stage == "coverage" and args.parquets:
            parquet_files = args.parquets
        else:
            v2_dir = output_dir / "v2"
            parquet_files = [
                str(v2_dir / "dataset_v2_base.parquet"),
                str(v2_dir / "dataset_v2_augmentation.parquet"),
            ]

        verify_coverage(
            parquet_files,
            num_steps=args.num_steps,
        )


if __name__ == "__main__":
    main()
