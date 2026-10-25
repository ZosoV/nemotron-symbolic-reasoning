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

Run:
    uv run python src/cot_generator/generate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator.config import CotGeneratorConfig, load_cot_config
from src.cot_generator.decode import detect_pure_concat
from src.cot_generator.equations import parse_prompt
from src.cot_generator.families import resolve_registry_op
from src.cot_generator.render import render_concat_cot, render_cot

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


def _render_capped(row: dict, cfg: CotGeneratorConfig):
    """Render the arithmetic CoT, shrinking it under the token cap if set.

    Renders with the configured perturbation count, then — while the approximate
    token size exceeds cfg.max_cot_tokens and at least one perturbation remains —
    re-renders requesting one fewer perturbation than was actually realised,
    re-checking each time. Returns (CotResult, approx_tokens).
    """
    res = render_cot(row, cfg)
    tokens = round(len(res.text) / CHARS_PER_TOKEN)
    cap = cfg.max_cot_tokens
    while cap is not None and tokens > cap and res.num_perturbations > 0:
        res = render_cot(row, cfg, rounds=res.num_perturbations - 1)
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
    col = [None] * len(df)
    tok_col: list[int | None] = [None] * len(df)
    pert_col: list[int | None] = [None] * len(df)

    total_pure_concat = int((df.get("solver_strategy", "") == "pure_concat").sum())
    stats = {"total": len(df), "feasible": 0, "generated": 0,
             "skipped": 0, "selfcheck_fail": 0, "render_error": 0,
             "concat_generated": 0, "total_pure_concat": total_pure_concat}
    done = 0
    for i, (_, row) in enumerate(df.iterrows()):
        rowd = row.to_dict()
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
                        col[i] = res.text
                        tok_col[i] = round(len(res.text) / CHARS_PER_TOKEN)
                        pert_col[i] = res.num_perturbations
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
        try:
            res, tokens = _render_capped(rowd, cfg)
        except Exception as exc:  # noqa: BLE001 - report and continue
            stats["render_error"] += 1
            print(f"  [render error] id={rowd.get('id')}: {type(exc).__name__}: {exc}")
            continue
        if res.answer != rowd.get("solver_answer"):
            stats["selfcheck_fail"] += 1
            print(f"  [self-check fail] id={rowd.get('id')}: "
                  f"got {res.answer!r} expected {rowd.get('solver_answer')!r}")
            continue
        col[i] = res.text
        tok_col[i] = tokens
        pert_col[i] = res.num_perturbations
        stats["generated"] += 1
        done += 1

    df[cfg.cot_column] = col
    df["cot_approx_tokens"] = tok_col
    df["cot_num_perturbations"] = pert_col
    Path(cfg.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.output_parquet, index=False)
    sized = [t for t in tok_col if t is not None]
    if sized:
        stats["cot_tokens_mean"] = round(sum(sized) / len(sized))
        stats["cot_tokens_max"] = max(sized)
    return stats


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
    stats = generate(cfg)
    print(f"Wrote {cfg.output_parquet} (column {cfg.cot_column!r})")
    print("  " + "  ".join(f"{k}={v}" for k, v in stats.items()))
    arithmetic_generated = stats["generated"] - stats["concat_generated"]
    total_pure_concat = stats["total_pure_concat"]
    print(
        f"\n  concat_generated / generated / total_pure_concat  =  "
        f"{stats['concat_generated']} / {stats['generated']} / {total_pure_concat}"
        f"  ({arithmetic_generated} arithmetic)"
    )


if __name__ == "__main__":
    main()
