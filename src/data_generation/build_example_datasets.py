"""Build n-shot Alice's Wonderland example datasets (n = 1..5).

Ports the leave-one-out rotation logic from
notebooks/data_generation/generation_of_data_examples.ipynb to a
runnable script. For each source row, every equation takes a turn as the
query; n examples are drawn from the remaining equations with coverage
balanced to a target fraction.

Source:  inputs/data/eval_data_k_1.parquet  (filtered to Cryptarithm rows)
Outputs: inputs/data/dataset_examples_{n}.parquet  for each n in --ns

Output schema per file:
  original_problem_id  — problem_id from the source row
  problem_id           — new per-sample id: "{original_problem_id}_q{qi}"
  prompt               — Alice prompt (intro + n example lines + query)
  answer               — expected result for the query equation

Run:
    uv run python src/data_generation/build_example_datasets.py
    uv run python src/data_generation/build_example_datasets.py --ns 1 3 5
    uv run python src/data_generation/build_example_datasets.py --input inputs/data/eval_data_k_1.parquet --output-dir inputs/data/
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

# ---------------------------------------------------------------------------
# Prompt helpers (ported from notebook cell 4ddbf09f)
# ---------------------------------------------------------------------------

INTRO = (
    "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
    "Below are a few examples:"
)


def lhs_str(l: dict) -> str:
    return l["l0"] + l["l1"] + l["op"] + l["r0"] + l["r1"]


def rhs_str(rh: dict) -> str:
    body = "".join(rh["actual_rhs"])
    if rh["op"] is None:
        return body
    return rh["op"] + body if rh["sign_pos"] == "prefix" else body + rh["op"]


def make_prompt(example_lines: list[str], target_lhs: str) -> str:
    """Build the Alice prompt. example_lines is a list of 'lhs = rhs' strings."""
    body = "\n".join(example_lines)
    sep = "\n" if example_lines else ""
    return f"{INTRO}\n{body}{sep}\nNow, determine the result for: {target_lhs}"


# ---------------------------------------------------------------------------
# Core builder (ported from notebook cell add54ba4)
# ---------------------------------------------------------------------------

def build_nshot(
    src_all: pd.DataFrame,
    n: int,
    cov_frac: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-one-out rotation dataset with n shown examples.

    Unit classification by pool (all other equations in the row):
      forced_cov  — ≥1 matching op, <n non-matching  → must be covered
      forced_unc  — 0 matching ops                    → must be uncovered
      flexible    — ≥1 matching AND ≥n non-matching   → assigned to hit target

    Target coverage: cov_frac.
    All eligible units are kept (no down-selection).
    If forced units already exceed the target for either class, flexible units
    all go to the other class and coverage drifts naturally (n=5 case).
    """
    rng = random.Random(seed)

    # Phase 1: enumerate and classify all eligible units ─────────────────────
    forced_cov: list = []
    forced_unc: list = []
    flexible: list = []

    for _, row in src_all.iterrows():
        eqs = json.loads(row["equations"])
        for qi, q in enumerate(eqs):
            pool_idx = [j for j in range(len(eqs)) if j != qi]
            if len(pool_idx) < n:
                continue
            qop = q["lhs"]["op"]
            match_idx = [j for j in pool_idx if eqs[j]["lhs"]["op"] == qop]
            nomatch_idx = [j for j in pool_idx if eqs[j]["lhs"]["op"] != qop]
            unit = (row["problem_id"], qi, eqs, q, pool_idx, match_idx, nomatch_idx)
            if match_idx and len(nomatch_idx) < n:
                forced_cov.append(unit)
            elif not match_idx:
                forced_unc.append(unit)
            else:
                flexible.append(unit)

    # Phase 2: assign flexible units to hit cov_frac (keep all units) ────────
    all_units = len(forced_cov) + len(forced_unc) + len(flexible)
    tC = round(all_units * cov_frac)
    tU = all_units - tC

    need_cov_flex = tC - len(forced_cov)
    need_unc_flex = tU - len(forced_unc)

    if need_cov_flex < 0:
        cov_flex, unc_flex = 0, len(flexible)
    elif need_unc_flex < 0:
        cov_flex, unc_flex = len(flexible), 0
    else:
        cov_flex, unc_flex = need_cov_flex, need_unc_flex

    rng.shuffle(flexible)
    chosen_cov = forced_cov + flexible[:cov_flex]
    chosen_unc = forced_unc + flexible[cov_flex: cov_flex + unc_flex]

    C, U = len(chosen_cov), len(chosen_unc)

    # Phase 3: build samples ──────────────────────────────────────────────────
    rows_out = []
    for label, units in [("cov", chosen_cov), ("unc", chosen_unc)]:
        for orig_id, qi, eqs, q, pool_idx, match_idx, nomatch_idx in units:
            if label == "cov":
                anchor = rng.sample(match_idx, 1)
                rest = rng.sample([j for j in pool_idx if j not in anchor], n - 1)
                ex_idx = anchor + rest
            else:
                ex_idx = rng.sample(nomatch_idx, n)
            rng.shuffle(ex_idx)
            ex_lines = [
                f"{lhs_str(eqs[j]['lhs'])} = {rhs_str(eqs[j]['rhs'])}"
                for j in ex_idx
            ]
            rows_out.append({
                "original_problem_id": orig_id,
                "problem_id":          f"{orig_id}_q{qi}",
                "prompt":              make_prompt(ex_lines, lhs_str(q["lhs"])),
                "answer":              rhs_str(q["rhs"]),
            })

    df = pd.DataFrame(rows_out)
    print(f"  n={n}: {len(df)} rows  C={C} ({C / len(df) * 100:.2f}%)  U={U} ({U / len(df) * 100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, n: int) -> None:
    """Structural checks — raise AssertionError on failure."""
    assert df.isnull().sum().sum() == 0, f"n={n}: nulls found"
    assert df["problem_id"].nunique() == len(df), f"n={n}: duplicate problem_id"

    bad_ex = df["prompt"].apply(
        lambda p: sum(1 for ln in p.split("\n") if " = " in ln) != n
    ).sum()
    assert bad_ex == 0, f"n={n}: {bad_ex} prompts have wrong number of example lines"

    bad_tail = df["prompt"].apply(lambda p: " = " in p.split("\n")[-1]).sum()
    assert bad_tail == 0, f"n={n}: {bad_tail} prompts have '=' in the query line"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build n-shot Alice example datasets (n=1..5) via leave-one-out rotation."
    )
    parser.add_argument(
        "--input",
        default=str(_ROOT / "inputs/data/eval_data_k_1.parquet"),
        help="Source parquet with an 'equations' column (default: inputs/data/eval_data_k_1.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_ROOT / "inputs/data"),
        help="Directory for output parquet files (default: inputs/data/)",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        metavar="N",
        help="Which n-shot variants to build (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--category",
        default="Cryptarithm (Symbolic Equations)",
        help="Filter rows by problem_category; pass '' to disable (default: Cryptarithm (Symbolic Equations))",
    )
    parser.add_argument(
        "--cov-frac",
        type=float,
        default=659 / 823,
        metavar="FRAC",
        help="Target covered fraction (default: 659/823 ≈ 0.8007)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Loading {input_path} …")
    src = pd.read_parquet(input_path)
    print(f"  Loaded {len(src)} rows")

    if args.category:
        before = len(src)
        src = src[src["problem_category"] == args.category].reset_index(drop=True)
        print(f"  Category filter '{args.category}': kept {len(src)} / {before} rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, int]] = []
    for n in args.ns:
        df = build_nshot(src, n=n, cov_frac=args.cov_frac, seed=args.seed)
        validate(df, n)
        out_path = output_dir / f"dataset_examples_{n}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}  ✓")
        results.append((n, len(df)))

    print("\n=== summary ===")
    for n, rows in results:
        print(f"  n={n}  →  {rows:>5} rows  →  {output_dir}/dataset_examples_{n}.parquet")


if __name__ == "__main__":
    main()
