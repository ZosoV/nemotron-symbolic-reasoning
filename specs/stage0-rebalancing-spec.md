# Stage-0 Dataset Rebalancing Spec

## 1. Motivation

The raw Stage-0 dataset `data/solver_examples_0_k_20_extended.parquet` (96,794 rows,
4,165 distinct source equations) has a severely skewed operation-frequency distribution:

| Root cause | Effect |
|---|---|
| The solver's k=20 cap produces ~26 rows per arithmetic equation (alternative op/digit-mapping interpretations of the **same** prompt→answer) | Arithmetic ops are over-represented by ×20 relative to their true task frequency |
| `pure_concat` equations go through a deterministic fast-path with no alternatives | Each of the 419 concat source equations stays at exactly 1 row |
| `neg_absdiff` never appears in the original 823 equations — it only arises as an ambiguity candidate | 10,988 rows, all from variants (`is_original=False`) |

Concretely, the before-rebalancing op counts among the top ops:

```
neg_absdiff    10,988   (all ambiguity variants; 0 in original data)
sub_signed      7,995
rsub_signed     7,756
mul             5,332
lcm             5,094
…
concat_fwd          0   (filtered out — pure_concat rows have ops_mapping=None)
concat_rev          0
```

The goal is an approximately balanced dataset where no op swamps the others and concat
is correctly represented at roughly the same scale.

---

## 2. Inputs and Output

| File | Role |
|---|---|
| `data/solver_examples_0_k_20_extended.parquet` | Source (96,794 rows, 16-col schema below) |
| `data/solver_results_forcing_base_10_k_1.parquet` | Base dataset used only to derive concat operator-glyph weights |
| `data/solver_examples_0_k_20_balanced.parquet` | Output (92,555 rows) |

**Schema (19 columns):** `problem_id, solution_id, prompt, answer, solver_strategy,
solver_category, problem_category, digits_mapping, solver_tier, solver_radix, ops_candidates,
equations, ops_mapping, solver_answer, solver_correct, variant_index, is_original,
num_variants, fully_validated`.

---

## 3. Rebalancing Algorithm

### 3.1  Operation-count definition

Each arithmetic row carries a single-entry JSON `ops_mapping = {"<op_glyph>": "<op_name>"}`.
The op count for an operation is simply the row count where `ops_mapping` maps to that name.
`pure_concat` rows have `ops_mapping = null` and do not contribute to arithmetic op counts.

Synthetic concat rows are given `ops_mapping = {"<op_glyph>": "concat_fwd/rev"}` (unlike
real `pure_concat` rows) so that they register in the balanced count.

### 3.2  Water-filling cap (three over-represented ops)

Applied to: `neg_absdiff`, `sub_signed`, `rsub_signed`. Target: **4,500 rows each**.

For each target op X with current count N > 4500:

1. Let `x_e` = number of X-rows belonging to source equation `e`.
2. Binary-search for the largest integer threshold `t` such that
   `Σ_e min(x_e, t) ≤ 4500`.
3. `sum_at_t = Σ_e min(x_e, t)`.  `extra_keep = 4500 − sum_at_t`.
4. Among all equations with `x_e > t`, sort by **total equation multiplicity ascending**
   (least prolific equations get the bonus row; most prolific are trimmed hardest),
   then by `problem_id` for tie-break (deterministic, fixed seed=42).
   Grant the first `extra_keep` of these equations a cap of `t+1`; the rest are capped at `t`.
5. For each equation, **within the capped rows**, prefer dropping `is_original=False`
   (variant) rows before `is_original=True` originals. Sort by
   `is_original DESC, variant_index ASC` and keep the first `k` rows.
6. Guard: never drop an equation's last row. (Not needed in practice because the three
   target ops always co-occur with other ops per equation.)

Empirical thresholds from the source data:

| op | before | t | sum@t | extra_keep | after |
|---|---|---|---|---|---|
| `neg_absdiff` | 10,988 | 7 | 4,067 | 433 | 4,500 |
| `rsub_signed` |  7,756 | 4 | 4,207 | 293 | 4,500 |
| `sub_signed`  |  7,995 | 4 | 4,334 | 166 | 4,500 |

After capping: **83,555 rows** from 4,165 equations (all source tasks preserved).

### 3.3  Synthetic concat rows

Synthesize **4,500 `concat_fwd`** + **4,500 `concat_rev`** rows on top of the existing
419 real `pure_concat` rows.

**Operator glyph — frequency-weighted from the base dataset.**
Read `solver_results_forcing_base_10_k_1.parquet` and count how often each glyph maps
to `concat_fwd` / `concat_rev` in `ops_mapping`. Use those counts as sampling weights
(separate distribution per direction):

| glyph | fwd count | rev count |
|---|---|---|
| `+` | 44 | 12 |
| `*` | 40 | 16 |
| `@` |  3 | — |
| `\` |  2 | — |
| `^` |  2 | — |
| `'` |  2 | — |
| `!` |  2 | — |
| `[` |  2 | — |
| `{` |  2 | — |
| `"` |  2 | — |
| `(` |  2 | — |
| other singles | 1 each | — |
| **total** | **110** | **28** |

So for `concat_rev` only `*` and `+` are used (weighted 16:12).

**Operand glyphs** (`l0`, `l1`, `r0`, `r1`): drawn uniformly-weighted from the pool of
23 glyphs that appear as operands in the real data:
`! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ \` { | }`.

**Per-sample construction:**
- Draw 4 operand glyphs and 1 operator glyph (with the weights above).
- Require `(l0,l1) ≠ (r0,r1)` so the fwd and rev answers differ.
- Deduplicate on `(l0,l1,op,r0,r1)`; reject any tuple that collides with the 419 real
  concat LHSes.
- `lhs_str = l0+l1+op+r0+r1`;
  fwd: `answer = l0+l1+r0+r1`; rev: `answer = r0+r1+l0+l1`.

**Synthetic row schema:**

| column | value |
|---|---|
| `problem_id` | `synth_concat_fwd_{i}` / `synth_concat_rev_{i}` |
| `solution_id` | re-assigned via `groupby("problem_id").cumcount()` after assembly |
| `prompt` | standard Alice prefix + `lhs_str` |
| `answer` / `solver_answer` | concat string (length 4) |
| `solver_strategy` | `"pure_concat"` |
| `solver_category` | `"concatenation"` |
| `problem_category` | `None` |
| `ops_mapping` | `json.dumps({op: "concat_fwd/rev"})` |
| `equations` | one `to_predict` eq, `rhs.op=null, sign_pos=null, actual_rhs=[4 symbols]` |
| `digits_mapping / solver_tier / solver_radix / ops_candidates` | `None` |
| `solver_correct` | `True` |
| `variant_index` | `0` |
| `is_original` | `False` |
| `num_variants` | `1` |
| `fully_validated` | `None` |

---

## 4. Resulting Distribution

```
op                   before    after
-----------------------------------
neg_absdiff          10,988    4,500  ← capped
sub_signed            7,995    4,500  ← capped
rsub_signed           7,756    4,500  ← capped
mul                   5,332    5,332
lcm                   5,094    5,094
absdiff               4,609    4,609
absdiff_p1            4,570    4,570
mul_m1                4,528    4,528
mul_p1                4,483    4,483
add_p2                4,256    4,256
add_m1                3,888    3,888
add_p1                4,061    4,061
add                   3,958    3,958
a2_plus_b             3,858    3,858
gcd                   3,379    3,379
fdiv                  3,362    3,362
rmod                  2,753    2,753
mod                   2,660    2,660
absdiff_m2            4,451    4,451
absdiff_m1            4,394    4,394
concat_fwd                0    4,500  ← synthesized
concat_rev                0    4,500  ← synthesized
(pure_concat real)      419      419  (unchanged, ops_mapping=null)
-----------------------------------
Total rows           96,794   92,555
Real rows            96,794   83,555  (4,165 source equations preserved)
Synthetic rows            0    9,000
```

---

## 5. Reproduce

```bash
uv run python rebalance_stage0.py
```

The script reads the two source parquets, applies capping + synthesis (SEED=42), writes
the output, then self-verifies with hard assertions and prints the before/after table.

---

## 6. Known Caveats

- **Notebook frequency cell** (`result_analysis.ipynb` Section 4) filters
  `solver_strategy in {standard, little_endian}`. This hides all `pure_concat` rows
  (both real and synthetic). To see concat counts in the notebook, widen the filter or
  count `ops_mapping` directly.
- **`mul` and `lcm` still sit above 4,500** (5,332 and 5,094). These were left untouched
  by design — the imbalance is modest and deliberate.
- **`mod`, `rmod`, `fdiv`** remain below 3,000 — these ops are genuinely rare in the
  original task data and the low counts reflect that faithfully.
