"""Augmentation 3 — Synthetic v3 problem generation.

Generates brand-new cryptarithm problems by sampling from the empirical
distributions of `only_cryptharithm_solved_k_1.parquet` (823 rows).
Reuses `src/data_generation_v2/` libraries as-is — no edits to v2.

This is a standalone script; the output parquet is then fed through
`cot_generator_v2/generate.py` to produce training CoTs. Output rows are
tagged with `cot_variant_source="synthetic_v3"` and `dataset_category`
encoding the number of examples (examples_3 / examples_4 / examples_5).

Usage:
    uv run python src/data_generation_v3/run_v3.py [--output PATH] [--seed N] \\
        [--n3 N] [--n4 N] [--n5 N] [--no-concat]

Defaults:
    --output inputs/data/v3/synthetic_v3.parquet
    --seed   42
    --n3/4/5 use natural ratio derived from k=1 source (~equal thirds)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_generation_v2.stats import compute_stats
from src.data_generation_v2.synth import build_synthetic

_SOURCE_PARQUET = _ROOT / "inputs" / "data" / "v2" / "only_cryptharithm_solved_k_1.parquet"

# Default target counts per bucket — roughly equal thirds, tunable per coverage needs.
_DEFAULT_TARGETS = {3: 300, 4: 300, 5: 300}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic v3 cryptarithm problems.")
    parser.add_argument("--source", default=str(_SOURCE_PARQUET),
                        help="Source parquet for computing empirical stats")
    parser.add_argument("--output", default="inputs/data/v3/synthetic_v3.parquet",
                        help="Output parquet path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n3", type=int, default=_DEFAULT_TARGETS[3],
                        help="Number of 3-example problems to generate")
    parser.add_argument("--n4", type=int, default=_DEFAULT_TARGETS[4],
                        help="Number of 4-example problems to generate")
    parser.add_argument("--n5", type=int, default=_DEFAULT_TARGETS[5],
                        help="Number of 5-example problems to generate")
    parser.add_argument("--no-concat", action="store_true",
                        help="Exclude pure-concat synthetic problems")
    parser.add_argument("--dedup-mode", default="canonical", choices=["canonical", "prompt"],
                        help="Dedup strategy: 'canonical' (structural, symbol-invariant) "
                             "or 'prompt' (exact prompt-string match). Default: canonical")
    args = parser.parse_args(argv)

    import pandas as pd

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Source parquet not found: {source}", file=sys.stderr)
        sys.exit(1)

    target_counts = {3: args.n3, 4: args.n4, 5: args.n5}
    total = sum(target_counts.values())

    print(f"Computing stats from {source} ...")
    stats = compute_stats(source)
    print(f"  strategies:     {stats.strategy_probs}")
    print(f"  arith ops:      {stats.arith_ops}")
    print(f"  concat ops:     {stats.concat_ops}")
    print(f"  operand_syms:   ({len(stats.operand_syms)}) {sorted(stats.operand_syms)}")
    print(f"  operator_syms:  ({len(stats.operator_syms)}) {sorted(stats.operator_syms)}")
    print(f"  source sigs:    {len(stats.source_signatures)}")
    print(f"  source prompts: {len(stats.source_prompts)}")
    print(f"  dedup mode:     {args.dedup_mode}")

    print(f"Generating {total} synthetic problems ({target_counts}) ...")
    rows = build_synthetic(
        stats=stats,
        target_counts=target_counts,
        seed=args.seed,
        include_pure_concat=not args.no_concat,
        dedup_mode=args.dedup_mode,
    )
    print(f"  Generated {len(rows)} rows")

    df = pd.DataFrame(rows)
    # Tag all rows as synthetic_v3
    df["cot_variant_source"] = "synthetic_v3"

    # dataset_category follows examples_{n} convention
    if "n_examples" in df.columns:
        df["dataset_category"] = df["n_examples"].apply(lambda n: f"examples_{n}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}  ({len(df)} rows, columns: {list(df.columns)[:8]}...)")

    # Post-generation distribution summary
    print("\n── Synthetic output distributions ──")
    if "dataset_category" in df.columns:
        print("dataset_category:")
        for cat, cnt in df["dataset_category"].value_counts().sort_index().items():
            print(f"  {cat}: {cnt}")
    if "solver_strategy" in df.columns:
        print("solver_strategy:")
        for strat, cnt in df["solver_strategy"].value_counts().sort_index().items():
            print(f"  {strat}: {cnt}")


if __name__ == "__main__":
    main()
