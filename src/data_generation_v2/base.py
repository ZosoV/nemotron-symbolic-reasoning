"""Build base random-drop pools: one sample per (problem, n) for n ≤ k.

For each solved problem in solver_results.parquet, for each target shot-count n,
randomly drop k-n examples (keeping the original to_predict) and emit one
training sample with dataset_category="examples_{n}".

Approximate pool sizes (solver_correct=True, k distribution 234/292/278):
  n=1: 804   n=2: 804   n=3: 804   n=4: 570   n=5: 278

Returns:
  A list of row dicts (schema.make_solver_row format) and a usage_map:
    usage_map[original_id][n] = frozenset of chosen example INDICES
  The usage_map is consumed by recombine.py to avoid re-using the same subset.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_generation.build_example_datasets import lhs_str, make_prompt, rhs_str
from src.data_generation_v2.schema import make_solver_row


def build_base_pools(
    parquet_path: str | Path,
    seed: int = 42,
) -> tuple[list[dict], dict[str, dict[int, frozenset]]]:
    """Return (samples, usage_map).

    usage_map[original_id][n] = frozenset of example indices chosen for bucket n.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["solver_correct"] == True].copy()

    # Support both "problem_id" (eval_data_k_1.parquet) and "id" (legacy)
    id_col = "problem_id" if "problem_id" in df.columns else "id"

    samples: list[dict] = []
    # usage_map[original_id][n] = frozenset of example indices used
    usage_map: dict[str, dict[int, frozenset]] = {}

    for _, row in df.iterrows():
        original_id = str(row[id_col])
        try:
            equations = json.loads(row["equations"])
        except (json.JSONDecodeError, TypeError):
            continue

        examples = [e for e in equations if e["type"] == "example"]
        to_predict_eqs = [e for e in equations if e["type"] == "to_predict"]
        if not to_predict_eqs:
            continue
        to_predict = to_predict_eqs[0]
        k = len(examples)
        if k == 0:
            continue

        # Deterministic per-problem seed derived from the problem id hash
        prob_rng = random.Random(seed + abs(hash(original_id)) % (2**32))

        dm = _parse_json(row.get("digits_mapping"))
        om = _parse_json(row.get("ops_mapping"))
        radix = int(row["solver_radix"]) if pd.notna(row.get("solver_radix")) else 10
        strategy = str(row.get("solver_strategy") or "")

        usage_map.setdefault(original_id, {})

        for n in range(1, k + 1):
            # Randomly pick n examples out of k
            chosen_indices = sorted(prob_rng.sample(range(k), n))
            usage_map[original_id][n] = frozenset(chosen_indices)

            chosen_examples = [examples[i] for i in chosen_indices]
            new_eqs = [
                {**e, "type": "example"} for e in chosen_examples
            ] + [{**to_predict, "type": "to_predict"}]

            example_lines = [
                f"{lhs_str(e['lhs'])} = {rhs_str(e['rhs'])}" for e in chosen_examples
            ]
            prompt = make_prompt(example_lines, lhs_str(to_predict["lhs"]))
            answer = rhs_str(to_predict["rhs"])

            samples.append(
                make_solver_row(
                    problem_id=f"{original_id}_base_n{n}",
                    prompt=prompt,
                    answer=answer,
                    equations=new_eqs,
                    digits_mapping=dm,
                    ops_mapping=om,
                    radix=radix,
                    strategy=strategy,
                    dataset_category=f"examples_{n}",
                    original_problem_id=original_id,
                )
            )

    return samples, usage_map


def _parse_json(v) -> dict | None:
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(v, dict):
        return v
    return None
