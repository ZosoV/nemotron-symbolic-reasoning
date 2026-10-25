"""Generate 1-5-example chain-of-thought text for feasible solver-results rows.

Reads a solver-results .parquet, renders a CoT for every row that carries a
feasible arithmetic solution (both `digits_mapping` and `ops_mapping` present),
self-checks that the rendered conclusion reproduces the row's `solver_answer`,
and writes the result to a new column (default `cot_1_5`) in the output parquet.

Rows are skipped (CoT = None) when:
  * the solution uses an op with no family (e.g. a2_plus_b);
  * a question operand symbol is unconstrained (absent from digits_mapping);
  * rendering or the self-check fails (counted and reported).

Rows without a digit solution (pure-concat fast-path) are handled by a separate
concise renderer (`render_concat_cot`) that re-derives the concat rule from the
example equations directly — never trusting the `solver_strategy` label.

Aug 1 (order variants): for each base CoT, generate up to cfg.max_order_variants
extra rows with alternative break orderings (different reasoning chains, same answer).
Aug 4 (guess samples): gated on cfg.generate_guess_samples; for k≥2 rows emit a
truncated companion that ends with a "budget exhausted" conclusion.

Run:
    uv run python src/cot_generator_v2/generate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.config import CotGeneratorConfig, load_cot_config, perturbation_rounds_for_row
from src.cot_generator_v2.decode import detect_pure_concat
from src.cot_generator_v2.equations import parse_prompt
from src.cot_generator_v2.families import resolve_registry_op
from src.cot_generator_v2.guess import try_build_guess_sample
from src.cot_generator_v2.render import render_concat_cot, render_cot
from src.cot_generator_v2.variants import generate_order_variants

# Rough char-to-token ratio for the rendered CoT (empirical, for sizing only).
CHARS_PER_TOKEN = 2.81


def _is_feasible(row: dict) -> bool:
    import pandas as pd

    dm, om = row.get("digits_mapping"), row.get("ops_mapping")
    return isinstance(dm, str) and isinstance(om, str) and pd.notna(dm) and pd.notna(om)


def _try_concat_variant(row: dict, cfg: CotGeneratorConfig):
    """Attempt a concise pure-concat CoT for a row that has no digit solution.

    Returns a CotResult when the corrected evidence rule is satisfied, else None.
    Never reads solver_strategy — the label may be stale.
    """
    try:
        examples, q = parse_prompt(row["prompt"], row["answer"])
    except Exception:  # noqa: BLE001
        return None
    fam = detect_pure_concat(examples, q)
    if fam is None:
        return None
    try:
        return render_concat_cot(examples, q, fam)
    except Exception:  # noqa: BLE001
        return None


def _render_capped(row: dict, cfg: CotGeneratorConfig, rounds: int | None = None,
                   break_order: list[int] | None = None):
    """Render the arithmetic CoT, shrinking it under the token cap if set.

    Fix 9d: uses per-row perturbation count from perturbation_rounds_for_row().
    Then — while the approximate token size exceeds cfg.max_cot_tokens and at
    least one perturbation remains — re-renders requesting one fewer perturbation
    than was actually realised, re-checking each time.
    Returns (CotResult, approx_tokens).
    """
    row_rounds = rounds if rounds is not None else perturbation_rounds_for_row(row, cfg)
    res = render_cot(row, cfg, rounds=row_rounds, break_order=break_order)
    tokens = round(len(res.text) / CHARS_PER_TOKEN)
    cap = cfg.max_cot_tokens
    while cap is not None and tokens > cap and res.num_perturbations > 0:
        res = render_cot(row, cfg, rounds=res.num_perturbations - 1, break_order=break_order)
        tokens = round(len(res.text) / CHARS_PER_TOKEN)
    return res, tokens


def _renderable(row: dict) -> tuple[bool, str]:
    """Cheap pre-checks before attempting a full render."""
    try:
        om = json.loads(row["ops_mapping"])
        solution = json.loads(row["digits_mapping"])
    except (TypeError, json.JSONDecodeError):
        return False, "unparseable mapping"
    for name in om.values():
        try:
            resolve_registry_op(name)
        except KeyError:
            return False, f"no family for op {name!r}"
    examples, q = parse_prompt(row["prompt"], row["answer"])
    if any(s not in solution for s in (q.l0, q.l1, q.r0, q.r1)):
        return False, "unconstrained question symbol"
    return True, ""


def generate(cfg: CotGeneratorConfig) -> dict:
    import pandas as pd

    df = pd.read_parquet(cfg.input_parquet)
    col = cfg.cot_column

    # Base output list (one entry per source row, matching df index).
    out_rows: list[dict] = []

    total_pure_concat = int((df.get("solver_strategy", "") == "pure_concat").sum())
    stats = {"total": len(df), "feasible": 0, "generated": 0,
             "skipped": 0, "selfcheck_fail": 0, "render_error": 0,
             "concat_generated": 0, "total_pure_concat": total_pure_concat,
             "order_variants_generated": 0, "guess_samples_generated": 0}
    done = 0
    total_rows = len(df)
    _LOG_EVERY = 100

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        rowd = row.to_dict()
        rowd.setdefault("cot_variant_index", 0)
        rowd.setdefault("cot_variant_source", "base")

        if not _is_feasible(rowd):
            # Try concise pure-concat variant (no digit solution required).
            if cfg.limit is None or done < cfg.limit:
                res = _try_concat_variant(rowd, cfg)
                if res is not None:
                    if res.answer != rowd.get("solver_answer"):
                        stats["selfcheck_fail"] += 1
                        print(f"  [concat self-check fail] id={rowd.get('id')}: "
                              f"got {res.answer!r} expected {rowd.get('solver_answer')!r}")
                    else:
                        entry = dict(rowd)
                        entry[col] = res.text
                        entry["cot_approx_tokens"] = round(len(res.text) / CHARS_PER_TOKEN)
                        entry["cot_num_perturbations"] = res.num_perturbations
                        entry["cot_variant_index"] = 0
                        entry["cot_variant_source"] = "base"
                        out_rows.append(entry)
                        stats["generated"] += 1
                        stats["concat_generated"] += 1
                        done += 1
                    continue
            stats["skipped"] += 1
            continue

        stats["feasible"] += 1
        if cfg.limit is not None and done >= cfg.limit:
            stats["skipped"] += 1
            continue

        ok, _why = _renderable(rowd)
        if not ok:
            stats["skipped"] += 1
            continue

        # --- Base CoT (variant 0, default coupling-sorted break order) ---
        try:
            res, tokens = _render_capped(rowd, cfg, break_order=None)
        except Exception as exc:  # noqa: BLE001
            stats["render_error"] += 1
            print(f"  [render error] id={rowd.get('id')}: {type(exc).__name__}: {exc}")
            continue

        if res.answer != rowd.get("solver_answer"):
            stats["selfcheck_fail"] += 1
            print(f"  [self-check fail] id={rowd.get('id')}: "
                  f"got {res.answer!r} expected {rowd.get('solver_answer')!r}")
            continue

        base_entry = dict(rowd)
        base_entry[col] = res.text
        base_entry["cot_approx_tokens"] = tokens
        base_entry["cot_num_perturbations"] = res.num_perturbations
        base_entry["cot_variant_index"] = 0
        base_entry["cot_variant_source"] = "base"
        out_rows.append(base_entry)
        stats["generated"] += 1
        done += 1

        # --- Aug 4: guess sample from the base CoT ---
        if cfg.generate_guess_samples:
            guess_row = try_build_guess_sample(res, base_entry, col)
            if guess_row is not None:
                out_rows.append(guess_row)
                stats["guess_samples_generated"] += 1

        # --- Aug 1: order-variant CoTs ---
        if cfg.max_order_variants > 1 and res.num_perturbations >= 2:
            # num_candidates ≈ res.num_perturbations (build_chain caps at this)
            row_rounds = perturbation_rounds_for_row(rowd, cfg)
            seed = cfg.random_seed ^ hash(rowd.get("id", "")) & 0xFFFFFF
            orders = generate_order_variants(
                num_candidates=res.num_perturbations,
                num_rounds=row_rounds,
                seed=seed,
                max_variants=cfg.max_order_variants,
            )
            for v_idx, break_order in enumerate(orders[1:], start=1):
                try:
                    v_res, v_tokens = _render_capped(rowd, cfg, rounds=row_rounds,
                                                     break_order=break_order)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [variant render error] id={rowd.get('id')} v={v_idx}: {exc}")
                    continue
                if v_res.answer != rowd.get("solver_answer"):
                    continue  # silently drop — different ordering produced wrong answer
                v_entry = dict(rowd)
                v_entry[col] = v_res.text
                v_entry["cot_approx_tokens"] = v_tokens
                v_entry["cot_num_perturbations"] = v_res.num_perturbations
                v_entry["cot_variant_index"] = v_idx
                v_entry["cot_variant_source"] = "order_variant"
                out_rows.append(v_entry)
                stats["order_variants_generated"] += 1

        if row_idx % _LOG_EVERY == 0:
            print(f"  [progress] {row_idx}/{total_rows} rows processed, {done} CoTs generated")

    # Build output DataFrame.
    out_df = pd.DataFrame(out_rows)
    if col not in out_df.columns:
        out_df[col] = None
    Path(cfg.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(cfg.output_parquet, index=False)

    # Token stats from base rows only.
    base_tokens = [r.get("cot_approx_tokens") for r in out_rows
                   if r.get("cot_variant_source") == "base"
                   and r.get("cot_approx_tokens") is not None]
    if base_tokens:
        stats["cot_tokens_mean"] = round(sum(base_tokens) / len(base_tokens))
        stats["cot_tokens_max"] = max(base_tokens)
    return stats, out_df


def _print_stats(stats: dict, out_df, cfg: CotGeneratorConfig) -> None:
    """Print all end-of-run stats."""
    col = cfg.cot_column
    arithmetic_generated = stats["generated"] - stats["concat_generated"]
    print(f"Wrote {cfg.output_parquet} (column {col!r})")
    print("  " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    print(
        f"\n  concat_generated / generated / total_pure_concat  =  "
        f"{stats['concat_generated']} / {stats['generated']} / {stats['total_pure_concat']}"
        f"  ({arithmetic_generated} arithmetic)"
    )

    # --- Stat 1: CoTs by strategy ---
    print("\n=== CoTs by strategy (base rows only) ===")
    base_df = out_df[out_df.get("cot_variant_source", None) == "base"] if "cot_variant_source" in out_df.columns else out_df
    has_cot = base_df[base_df[col].notna()] if col in base_df.columns else base_df
    if "solver_strategy" in has_cot.columns:
        strat_counts = has_cot["solver_strategy"].value_counts()
        pure_concat_n = strat_counts.get("pure_concat", 0)
        le_n = strat_counts.get("little_endian", 0)
        std_n = strat_counts.get("standard", strat_counts.get("std", 0))
        # Also bucket by mode for arithmetic rows
        print(f"  pure_concat:   {pure_concat_n}")
        print(f"  little_endian: {le_n}")
        print(f"  standard:      {std_n}")
        other = len(has_cot) - pure_concat_n - le_n - std_n
        if other:
            print(f"  other:         {other}")
    print(f"  total base CoTs with content: {len(has_cot)}")
    print(f"  order_variants: {stats['order_variants_generated']}")
    print(f"  guess_samples:  {stats['guess_samples_generated']}")

    # --- Stat 2: CoTs per source sample (base rows, grouping by problem_id) ---
    print("\n=== Distribution of CoTs generated per source sample ===")
    all_has_cot = out_df[out_df[col].notna()] if col in out_df.columns else out_df
    pid_col = "problem_id" if "problem_id" in all_has_cot.columns else ("id" if "id" in all_has_cot.columns else None)
    if pid_col is not None:
        cots_per_problem = all_has_cot[pid_col].value_counts()
        hist = cots_per_problem.value_counts().sort_index(ascending=False)
        print(f"  {'cots_per_sample':>15}  {'#problems':>10}")
        for count, n_problems in hist.items():
            print(f"  {count:>15}  {n_problems:>10}")
    else:
        print("  (no problem_id/id column — cannot compute per-sample histogram)")

    # --- Stat 3: Perturbation count × dataset_category cross-tab ---
    print("\n=== Perturbation count × dataset_category (all rows) ===")
    cat_col = next(
        (c for c in ("dataset_category", "solver_category", "problem_category")
         if c in out_df.columns),
        None,
    )
    if "cot_num_perturbations" in out_df.columns and cat_col is not None:
        pert_df = out_df[out_df[col].notna()].copy()
        if "cot_variant_source" in pert_df.columns:
            pert_df = pert_df[pert_df["cot_variant_source"].isin(["base", "order_variant"])]
        cross = pert_df.groupby(
            ["cot_num_perturbations", cat_col], dropna=False
        ).size().unstack(fill_value=0)
        print(f"  (using column: {cat_col!r})")
        print(cross.to_string())
    else:
        print("  (missing cot_num_perturbations or category column)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate 1-5-example CoT for feasible rows.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (cot_generator block)")
    parser.add_argument("--input", default=None, help="Override input_parquet")
    parser.add_argument("--output", default=None, help="Override output_parquet")
    parser.add_argument("--limit", type=int, default=None, help="Override row limit")
    args = parser.parse_args(argv)

    cfg = load_cot_config(args.config)
    if args.input:
        cfg.input_parquet = args.input
    if args.output:
        cfg.output_parquet = args.output
    if args.limit is not None:
        cfg.limit = args.limit

    print(f"Reading {cfg.input_parquet}")
    stats, out_df = generate(cfg)
    _print_stats(stats, out_df, cfg)


if __name__ == "__main__":
    main()
