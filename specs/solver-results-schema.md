# Solver Results Schema

Reference for every column in `inputs/data/*.parquet`, produced by `src/solver/solver_eq_symbolic.py`.

> **Multi-row layout:** A puzzle may span **multiple rows** (one per solution found). All rows
> for the same puzzle share the same `problem_id`, `problem_category`, and `solution_count`.
> `solution_id` is a sequential integer (0, 1, 2 …) restarting for each `problem_id`.
> `eval_from_parquet.py` evaluates each row independently and handles multiple rows per
> `problem_id` correctly.

---

## Identity

| Column | Type | Meaning |
|--------|------|---------|
| `problem_id` | string | Puzzle ID from the original input dataset |
| `solution_id` | int | Per-puzzle sequential index: 0 for the first solution row, 1 for the second, … |
| `prompt` | string | Full puzzle text as given to the model |
| `answer` | string | Gold answer from the training data |
| `problem_category` | string \| null | High-level domain of the puzzle (`"Cryptarithm (Symbolic Equations)"`, `"Numeral"`, `"Unit conversion"`, …). Carried through from the input parquet. |

---

## Solver output

| Column | Type | Meaning |
|--------|------|---------|
| `solver_answer` | string \| null | Answer returned by the solver (`null` = no solution found) |
| `solver_correct` | bool | Whether `solver_answer == answer` |

---

## Solver classification

| Column | Type | Meaning |
|--------|------|---------|
| `solver_category` | string \| null | High-level puzzle family assigned by the solver: `arithmetic` or `concatenation` (null = no solution) |
| `solver_strategy` | string \| null | Sub-strategy within the category (see table below) |

### `solver_category` values

| Value | Meaning |
|-------|---------|
| `arithmetic` | Puzzle solved by assigning digit values to symbols and evaluating arithmetic operators |
| `concatenation` | Operator literally concatenates the left and right operand strings |

### `solver_strategy` values

| Value | Category | Meaning |
|-------|----------|---------|
| `standard` | arithmetic | Digits read in normal (big-endian) order |
| `little_endian` | arithmetic | Digit order is reversed before arithmetic is applied |
| `pure_concat` | concatenation | All operators are simple string concatenation |

---

## Multi-solution metadata

| Column | Type | Meaning |
|--------|------|---------|
| `solution_count` | int | Number of distinct programs found for this puzzle (`0` = no solution, `1–k` = programs found; same value across all rows for a puzzle) |
| `standard_solution_count` | int | Count of solutions found under `standard` mode (same across all rows for a puzzle) |
| `little_endian_solution_count` | int | Count of solutions found under `little_endian` mode (same across all rows for a puzzle) |

---

## Arithmetic search details

These columns are `null` for concatenation puzzles and no-solution rows.

| Column | Type | Meaning |
|--------|------|---------|
| `ops_mapping` | JSON string | Map of operator symbol → arithmetic function found, e.g. `{"*": "mul", "-": "absdiff"}` |
| `digits_mapping` | JSON string | Map of symbol → assigned digit (the bijection found), e.g. `{"@": 5, "&": 8}` |
| `solver_radix` | float | Number base used (almost always `10`) |
| `solver_tier` | string | Tier set that produced the solution: `tier0`, `tier1`, `tier2`, or `full` |
| `ops_candidates` | JSON string | Map of operator symbol → list of candidate operations that were tried |
| `equations` | JSON string | Parsed equation list (examples + query), including the gold `actual_rhs`. **Retained for post-processing** (`extend_ops_mappings`, `eval_from_parquet`); not included in the final CoT/training dataset. |

---

## CoT dataset addition

The `cot` column is added by `notebooks/cot_generation/cot_dataset_0.ipynb` and does not
appear in the raw solver output.

| Column | Type | Meaning |
|--------|------|---------|
| `cot` | string | Generated chain-of-thought reasoning for model training (natural language only — no code identifiers) |

---

## Search-trace CoT (solver-generated)

The `search_cot` column is written directly by the solver when `generate_cot: true` is set
in `config.yaml`. It is distinct from the notebook-added `cot` column above.

| Column | Type | Meaning |
|--------|------|---------|
| `search_cot` | string \| null | Prose narration of the solver's search process: Python-layer candidate narrowing and operator ordering, tier/mode/radix escalation, the Rust DFS per-digit trial log, accepted-candidate arithmetic verification, and the final solution summary. `null` when `generate_cot` is disabled or no solution was found. |

---

## Full search trace (solver-generated)

Written when `generate_search_trace: true` in `config.yaml`. Produces a JSON file per puzzle
and two new parquet columns.

| Column | Type | Meaning |
|--------|------|---------|
| `digit_mappings_analyzed` | int \| null | Total number of `(symbol, digit)` trials (partial and complete assignments) explored by the DFS before the first solution was found. Counts all trials regardless of whether they are recorded in the JSON. `null` for concat puzzles and no-solution cases. |
| `search_trace_path` | string \| null | Path to the per-puzzle JSON trace file, relative to `inputs/`. Fetchable by the web UI via `/files/<path>`. `null` when the trace was not generated. |

### JSON trace schema (`inputs/<search_trace_dir>/<problem_id>.json`)

```json
{
  "problem_id": "...",
  "var_order": [{"glyph": ")", "score": 189}, ...],
  "initial_candidates": {"!": ["add", "absdiff", ...], ...},
  "count": 83897,
  "truncated": false,
  "nodes": [
    {
      "depth": 0,
      "glyph": ")",
      "digit": 0,
      "partial_mapping": {")": 0},
      "passed": true,
      "pruned_glyph": null,
      "surviving": {"!": ["add", ...], ...},
      "eliminated": {"!": [], ...},
      "on_solution_path": false
    }, ...
  ]
}
```

**Top-level fields:**
- `var_order` — DFS symbol assignment order (most-constrained first), with the weight/score computed by `variable_order_with_scores`.
- `initial_candidates` — per-operator-glyph candidate operation names before any digit is assigned.
- `count` — total `(symbol, digit)` trials (equals `digit_mappings_analyzed` in parquet).
- `truncated` — true when `search_trace_max_nodes` was reached; DFS still ran to the first solution, but recording stopped. `count` still reflects all trials.

**Per-node fields:**
- `depth` — DFS depth (0 = first symbol assigned).
- `glyph` / `digit` — the symbol being assigned and the digit value tried.
- `partial_mapping` — mapping of all glyphs assigned so far (up to and including this trial).
- `passed` — forward-check result: did every operator group still have at least one surviving candidate?
- `pruned_glyph` — operator glyph of the first group that lost all candidates when `passed=false`; `null` when `passed=true`.
- `surviving` — per-glyph surviving candidate names after this assignment (empty for groups ≥ `pruned_glyph` when `passed=false`).
- `eliminated` — per-glyph candidate names removed by this trial (diff vs. parent node's surviving list; approximate — uses initial candidates for depth-0 nodes).
- `on_solution_path` — true for the unique chain of nodes from root to the first solution leaf (these form the accepted digit mapping).

The `search_cot` string is divided into four labeled sections:

**`[Candidate setup and search strategy]`** — Python layer:
- Operator candidate lists after heuristics #3 (`op_glyph_priority`), #2 (`sign_pattern_filter`), #4 (`concat_optin`), showing candidate counts and narrowing steps.
- Operator ordering by heuristic #5 (`order_ops` / MRV): each operator listed with its candidate count and equation count, most-constrained first.
- Tier-skip events (heuristic #9) and tier/mode/radix stage entries and results.

**`[Digit-assignment DFS]`** — Rust `search_traced` layer (accepted path only):
- Symbol assignment order (variable-ordering heuristic #6 / most-constrained first).
- Per-symbol digit trial log showing every digit actually tried at that depth (in ascending order), classified as:
  - **pruned** — operator `X` lost all surviving candidates (forward-checking).
  - **explored / backtracked** — passed forward-check but subtree yielded no solution.
  - **accepted ✓** — the digit that leads to the first solution.
- Closing "[All N symbols assigned — solution accepted.]"

**`[Accepted candidates per operator]`** — post-search verification:
- For each operator: all surviving candidate operation names with human-readable labels.
- For each equation in the group: decoded operand values (L, R), expected RHS, and each candidate's computed result marked ✓ or ✗ (with actual vs. expected on mismatch).
- Concat operations show "structural ✓" (no arithmetic).

**`[Result]`** — final summary: digit assignment, operator → operation mapping, strategy/base/tier.

Unlike `cot`, `search_cot` names operations (`add`, `rmod`, …) and heuristics by label for
readability. It is an explanatory debugging/analysis artifact, not a training CoT.

---

## Merged dataset (`dataset_examples_merged.parquet`)

Produced by `src/data_generation/merge_example_datasets.py`, which concatenates
`dataset_examples_{n}_k_2_balanced.parquet` for n = 1..5.

| Column | Type | Meaning |
|--------|------|---------|
| `dataset_category` | string | Source n-shot variant: `"examples_1"` … `"examples_5"` |
| `problem_id` | string | **Rewritten** to `"{original_problem_id}_examples_{n}"` for global uniqueness |

All other columns carry their original values unchanged. The effective primary key in the
merged file is the rewritten `problem_id` (unique). The original per-source `problem_id`
can be recovered by stripping the `_examples_{n}` suffix.

---

## Removed columns

| Column | Reason |
|--------|--------|
| `id` | Renamed to `problem_id` (was null in practice because the input column is `problem_id`) |
| `category` | Renamed to `solver_category` to distinguish from `problem_category` |
| `search_reached` | Removed — dead weight in the training corpus; internal solver detail |
