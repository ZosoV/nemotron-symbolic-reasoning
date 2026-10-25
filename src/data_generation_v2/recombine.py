"""Strategy A augmentation for n=1 and n=2 buckets.

For each original problem (k examples + 1 to_predict), generate additional
samples by picking novel (query, example_subset) combinations that were NOT
used by base.py.  Any equation can serve as the new to_predict; the remaining
k equations form the pool from which n examples are drawn.

Novelty rule: the pair (to_predict_index, frozenset(example_indices)) must not
appear in the base usage_map for this problem.

Algorithm:
  1. Enumerate all C(k+1, n+1) * (n+1) candidates per problem per n.
  2. Skip already-used combos.
  3. Pick candidates in round-robin across problems until target count is reached.

All equations already carry the correct answer in their rhs (verified by the
original solver), so no re-encoding is needed.  A verify_program call confirms
consistency before emission.
"""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_generation.build_example_datasets import lhs_str, make_prompt, rhs_str
from src.data_generation_v2.base import _parse_json
from src.data_generation_v2.schema import make_solver_row
from src.encoding import verify_program


def build_recombination(
    parquet_path: str | Path,
    usage_map: dict[str, dict[int, frozenset]],
    target_counts: dict[int, int],   # {n: how_many_to_generate}
    seed: int = 42,
) -> list[dict]:
    """Return augmentation samples for n=1 and n=2.

    target_counts example: {1: 742, 2: 478}
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["solver_correct"] == True].copy()

    id_col = "problem_id" if "problem_id" in df.columns else "id"

    # Build per-problem candidate lists per n
    per_problem: list[dict] = []

    for _, row in df.iterrows():
        original_id = str(row[id_col])
        try:
            equations = json.loads(row["equations"])
        except (json.JSONDecodeError, TypeError):
            continue

        examples = [e for e in equations if e["type"] == "example"]
        all_eqs = examples + [e for e in equations if e["type"] == "to_predict"]
        k_all = len(all_eqs)
        if k_all < 2:
            continue

        dm = _parse_json(row.get("digits_mapping"))
        om = _parse_json(row.get("ops_mapping"))
        radix = int(row["solver_radix"]) if pd.notna(row.get("solver_radix")) else 10
        strategy = str(row.get("solver_strategy") or "")
        mode = "little_endian" if strategy == "little_endian" else "standard"

        used_by_base = usage_map.get(original_id, {})

        per_problem.append({
            "id": original_id,
            "all_eqs": all_eqs,
            "dm": dm, "om": om, "radix": radix, "strategy": strategy, "mode": mode,
            "used": used_by_base,
        })

    rng = random.Random(seed)
    results: list[dict] = []

    for n, target in target_counts.items():
        if target <= 0:
            continue
        needed = target
        # Shuffle problems for fair distribution
        prob_order = list(range(len(per_problem)))
        rng.shuffle(prob_order)

        generated = 0
        # Multiple passes until we reach target (some problems have many candidates)
        while generated < needed:
            progress_this_pass = 0
            for pi in prob_order:
                if generated >= needed:
                    break
                p = per_problem[pi]
                all_eqs = p["all_eqs"]
                k_all = len(all_eqs)
                if k_all < n + 1:
                    continue

                used_combos: set[tuple] = set()
                # Reconstruct base-used combo as (to_predict_idx_in_all_eqs, frozenset)
                # Base always uses the last equation (original to_predict = index k_all-1)
                # and example indices are indices in the examples list (0..k_all-2)
                # In all_eqs: indices 0..k_all-2 are examples, k_all-1 is original to_predict
                base_n = p["used"].get(n)
                if base_n is not None:
                    # base_n contains frozenset of example-list indices
                    # in all_eqs: example i → all_eqs index i
                    used_combos.add((k_all - 1, frozenset(base_n)))

                # Enumerate all (to_predict_idx, frozenset(example_idxs)) combinations
                # that are NOT in used_combos
                candidates = []
                for tp_idx in range(k_all):
                    remaining = [i for i in range(k_all) if i != tp_idx]
                    for ex_combo in combinations(remaining, n):
                        combo = (tp_idx, frozenset(ex_combo))
                        if combo not in used_combos:
                            candidates.append(combo)

                if not candidates:
                    continue

                # Pick one random novel candidate
                chosen = rng.choice(candidates)
                tp_idx, ex_idxs = chosen
                # Mark it used so the same combo isn't re-picked in future passes
                used_combos.add(chosen)

                tp_eq = {**all_eqs[tp_idx], "type": "to_predict"}
                ex_eqs = [{**all_eqs[i], "type": "example"} for i in sorted(ex_idxs)]
                new_eqs = ex_eqs + [tp_eq]

                example_lines = [f"{lhs_str(e['lhs'])} = {rhs_str(e['rhs'])}" for e in ex_eqs]
                prompt = make_prompt(example_lines, lhs_str(tp_eq["lhs"]))
                answer = rhs_str(tp_eq["rhs"])

                # Self-check (arithmetic rows only; pure_concat passes mapping=None)
                if p["dm"] is not None and p["om"] is not None:
                    ok = verify_program(new_eqs, p["dm"], p["om"], p["mode"], p["radix"])
                    if not ok:
                        continue

                aug_idx = generated
                results.append(
                    make_solver_row(
                        problem_id=f"{p['id']}_aug_n{n}_{aug_idx}",
                        prompt=prompt,
                        answer=answer,
                        equations=new_eqs,
                        digits_mapping=p["dm"],
                        ops_mapping=p["om"],
                        radix=p["radix"],
                        strategy=p["strategy"],
                        dataset_category=f"examples_{n}",
                        original_problem_id=p["id"],
                    )
                )
                generated += 1
                progress_this_pass += 1

            if progress_this_pass == 0:
                # No more novel candidates exist
                print(f"  [recombine] n={n}: exhausted candidates after {generated}/{needed}")
                break

        print(f"  [recombine] n={n}: generated {generated}/{needed}")

    return results
