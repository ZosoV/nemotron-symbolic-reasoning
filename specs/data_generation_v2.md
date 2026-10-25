### Data Generation v2

Implementation: `src/data_generation_v2/`
Entry point: `uv run python src/data_generation_v2/run_v2.py --stage all`
Default source: `inputs/data/eval_data_k_1.parquet` (829 rows, `problem_id` column, strategy=`"standard"`/`"little_endian"`/`"pure_concat"`).

---

# "Other variant" base pools

One random-drop sample per (problem, n) where n ≤ k.  The original to_predict
is kept; k−n examples are dropped at random (seed=42).

Source: `inputs/data/eval_data_k_1.parquet` (solver_correct=True, 805 rows).
k distribution: k=3→235, k=4→292, k=5→278.

**Actual base pool sizes (solver_correct=True):**

- n=5 : 278
- n=4 : 292 + 278 = 570
- n=3 : 234 + 292 + 278 = 804
- n=2 : 804
- n=1 : 804

---

# Options per bucket (original spec, approximate)

- n=5 : 278 (5 examples)
- n=4 : 299 + 278 (drop 1 randomly)
- n=3 : 243 + 299 (drop 1 randomly) + 278 (drop 2 randomly)
- n=2 : 243 (drop 1 randomly) + 299 (drop 2 randomly) + 278 (drop 3 randomly)
- n=1 : 243 (drop 2 randomly) + 299 (drop 3 randomly) + 278 (drop 4 randomly)

---

# Generated variations (combinatorial variant)

  ┌────────┬──────────────────────────────────────┬─────────┐
  │ Bucket │               Formula                │ Samples │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ n=5    │ 278                                  │     278 │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ n=4    │ 299 + C(5,4)×278                     │   1,689 │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ n=3    │ 243 + C(4,3)×299 + C(5,3)×278        │   4,219 │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ n=2    │ C(3,2)×243 + C(4,2)×299 + C(5,2)×278 │   5,303 │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ n=1    │ C(3,1)×243 + C(4,1)×299 + C(5,1)×278 │   3,315 │
  ├────────┼──────────────────────────────────────┼─────────┤
  │ Total  │                                      │  14,804 │
  └────────┴──────────────────────────────────────┴─────────┘

---

# Other variation (single random drop — "other variant")

- n=5 : 278
- n=4 : 299 + 278 (drop 1 randomly)
- n=3 : 243 + 299 (drop 1 randomly) + 278 (drop 2 randomly)
- n=2 : 243 (drop 1 randomly) + 299 (drop 2 randomly) + 278 (drop 3 randomly)
- n=1 : 243 (drop 2 randomly) + 299 (drop 3 randomly) + 278 (drop 4 randomly)

---

# Gap-fill augmentation for ~1× coverage at 250 curriculum steps

**Key insight:** the curriculum fallback mechanism gives bucket n=1 disproportionately
many draws (it is the fallback bucket until bucket n=2's CDF threshold is reached,
~step 90). The fixed-point iteration accounts for this.

**Target pool sizes for ~1× coverage (250 steps, T=175, n_crypt=24):**

| Bucket | Base | Aug gap | Target pool | Coverage |
|--------|------|---------|-------------|---------|
| n=1 | 804 | +1,669 | 2,473 | ~1.00× |
| n=2 | 804 | +234 | 1,038 | ~0.99× |
| n=3 | 804 | +293 | 1,097 | ~1.01× |
| n=4 | 570 | +221 | 791 | ~0.98× |
| n=5 | 278 | +323 | 601 | ~1.02× |

**Augmentation strategies:**

- **n=1,2 (Strategy A — recombine):** novel (query_eq, example_subset) combinations
  from the original problem's k+1 equations.  No re-encoding; uses existing digit/op
  mappings.  Validated with `verify_program` before emission.

- **n=3,4,5 (Strategy B — backward-encoded synthetic):** new problems generated from
  distributions P(op_name), P(op_char|op_name), P(sym|position) computed from the 823
  training set.  Backward encoding via arithmetic + `encode_answer` path; no solver run.
  Strategy mix includes ~9% pure_concat (matching empirical distribution).
  Limitation: puzzles may be non-unique (`solution_count=1` = a known solution, not unique).

**Outputs:** `inputs/data/v2/dataset_v2_base.parquet` and `inputs/data/v2/dataset_v2_augmentation.parquet`.

To verify after any pool change:
```
uv run python src/data_generation_v2/run_v2.py --stage coverage \
    inputs/data/v2/dataset_v2_base.parquet \
    inputs/data/v2/dataset_v2_augmentation.parquet
```
