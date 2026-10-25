"""Canonical row schema for v2-generated solver-results parquets.

Every emitted row (base and augmentation) must pass through make_solver_row()
so the column contract is enforced in one place.  Downstream tools
(cot_generator/generate.py, add_mask_clues.py) expect these columns.
"""

from __future__ import annotations

import json


def make_solver_row(
    problem_id: str,
    prompt: str,
    answer: str,
    equations: list[dict],
    digits_mapping: dict[str, int] | None,   # symbol -> digit; None for pure_concat
    ops_mapping: dict[str, str] | None,       # op_char -> op_name; None for pure_concat
    radix: int,
    strategy: str,                            # "arithmetic" | "little_endian" | "pure_concat"
    dataset_category: str,                    # "examples_n"
    original_problem_id: str | None = None,
) -> dict:
    """Return a solver-results-compatible row dict.

    Sets derived fields (solver_answer, solver_correct, solver_category, etc.)
    and JSON-stringifies mappings/equations so the row is parquet-ready.
    """
    # "standard" and "arithmetic" both map to arithmetic solver_category
    solver_category = "concatenation" if strategy == "pure_concat" else "arithmetic"

    # JSON-stringify mappings (None stays None → cot_generator uses _is_feasible check)
    dm_json = json.dumps(digits_mapping) if digits_mapping is not None else None
    om_json = json.dumps(ops_mapping) if ops_mapping is not None else None
    eq_json = json.dumps(equations)

    return {
        "id": problem_id,
        "problem_id": problem_id,
        "original_problem_id": original_problem_id,
        "prompt": prompt,
        "answer": answer,
        "solver_answer": answer,          # our known answer IS the solver answer
        "solver_correct": True,
        "solver_strategy": strategy,
        "solver_category": solver_category,
        "problem_category": "Cryptarithm (Symbolic Equations)",
        "ops_mapping": om_json,
        "digits_mapping": dm_json,
        "solver_tier": "full",
        "solver_radix": float(radix),
        "ops_candidates": None,
        "equations": eq_json,
        "solution_count": 1,
        "standard_solution_count": 1 if strategy in ("arithmetic", "standard") else 0,
        "little_endian_solution_count": 1 if strategy == "little_endian" else 0,
        "search_cot": None,
        "dataset_category": dataset_category,
    }
