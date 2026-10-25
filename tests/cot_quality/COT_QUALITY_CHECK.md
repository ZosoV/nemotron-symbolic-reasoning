# CoT Quality Check

End-to-end test suite for verifying the output of `src/cot_generator_v2/`.
Steps are run in order; each step reads the output of the previous one.

---

## Quick Start

```bash
# Step 1 — build labeled test-case parquet
uv run python tests/cot_quality/step1_build_test_cases.py

# (future steps go here)
```

All steps read/write under `inputs/data/cot_quality/`.

---

## Step 1 — Build Test Cases

**Script:** `tests/cot_quality/step1_build_test_cases.py`  
**Input:** `inputs/data/v3/combined_1k_canonical.parquet`  
**Output:** `inputs/data/cot_quality/test_cases.parquet`

Selects one representative row per structural case that the CoT generator
must handle correctly.  Adds two columns to the output parquet:

| Column | Content |
|---|---|
| `test_case` | Short slug identifying the case (e.g. `case_4_zero_pad_demo`) |
| `test_case_description` | Human-readable description of what makes the row interesting |

### Cases

| # | Slug | Description | Count in 3k dataset |
|---|---|---|---|
| 1 | `case_1_little_endian` | Little-endian solution (digits read right-to-left) | 1105 (36.8%) |
| 2 | `case_2_standard` | Standard solution (digits read left-to-right) | 1637 (54.6%) |
| 3 | `case_3_pure_concat` | Pure-concat puzzle — no digit mapping needed | 238 (7.9%) |
| 4 | `case_4_zero_pad_demo` | Answer requires zero-padding AND an example demonstrates the padded width | 75 (2.5%) |
| 5 | `case_5_sign_prefix_demo` | Answer has a sign prefix AND an example shows prefix sign | 125 (4.2%) |
| 6 | `case_6_sign_postfix_demo` | Answer has a sign postfix AND an example shows postfix sign | **1 (0.03%)** |
| 7a | `case_7a_zero_pad_no_demo` | Answer requires zero-padding but NO example demonstrates the width | 108 (3.6%) |
| 7b | `case_7b_sign_prefix_no_demo` | Answer has sign prefix but NO example shows a prefix sign | 99 (3.3%) |
| 7c | `case_7c_sign_postfix_no_demo` | Answer has sign postfix but NO example shows a postfix sign | 9 (0.3%) |
| 8a | `case_8_examples_3` | 3-example puzzle | 1000 (33.3%) |
| 8b | `case_8_examples_4` | 4-example puzzle | 1000 (33.3%) |
| 8c | `case_8_examples_5` | 5-example puzzle | 1000 (33.3%) |
| 9 | `case_9_no_op_demo` | No example equation shares the question's operator | 637 (21.2%) |
| 10 | `case_10_unseen_digit_sym` | Answer contains a symbol never seen in examples or to_predict LHS | 319 (10.6%) |

**Selection policy:** one row per case, preferring distinct `problem_id`s
across cases (soft constraint — relaxed when the candidate pool is small,
as with case 6 which has only one row in the dataset).  Randomised with
`seed=42` for reproducibility.

### What each case exercises

- **Cases 1–2:** encoding direction (`solver_strategy`). The CoT must
  correctly identify standard vs little-endian and render the conclusion's
  digit decoding accordingly.

- **Case 3:** pure-concat fast-path. The CoT must detect forward/reverse
  concatenation from examples and conclude without entering the arithmetic
  path.

- **Cases 4–6 / 7a–7c:** answer-format edge cases. Cases with a demo test
  whether the CoT correctly observes and applies the format from examples.
  Cases without a demo test the fallback behaviour (best-guess from the
  question field only).  Sign postfix (cases 6 / 7c) is extremely rare in
  the dataset.

- **Case 8a–8c:** n-shot count. Controls the number of example equations
  available for grounding (and therefore the `perturbation_rounds` ceiling
  via Fix 9d).

- **Case 9:** operator with no demonstration. The CoT must apply Fix 8's
  "Case C" fallback for zero-padding width and sign reasoning when no
  example uses the question's operator.

- **Case 10:** unseen answer symbol. The CoT must produce a digit assignment
  for a symbol that appears nowhere in the example equations or the question
  LHS — it is only constrained by the bijection pool.

---

## Planned Steps

| Step | What it checks |
|---|---|
| Step 1 | Build labeled test-case parquet *(this step)* |
| Step 2 | Run `cot_generator_v2` on test cases; attach `cot` column |
| Step 3 | Structural checks — required XML tags present, `\boxed{}` present |
| Step 4 | Semantic checks — conclusion answer matches `solver_answer` |
| Step 5 | Case-specific checks — zero-pad width matches observed examples, sign position correct, etc. |
