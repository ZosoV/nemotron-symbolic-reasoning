---
name: cot-num-perturbations-distribution
description: What drives the cot_num_perturbations distribution — concat-variant 0s, and a monotonic example-count → fewer-perturbations trend
metadata:
  type: project
---

`cot_num_perturbations` (in `dataset_examples_merged_cot.parquet`) = realised failed-attempt count, `len(chain.steps)-1` in `render.py`. Config asks for up to `perturbation_rounds: 4`, but the real count is capped by puzzle structure. Two non-obvious facts:

**1. The `0` bucket is mostly a different CoT type, not empty chains.** Rows with no digit solution fall through `_try_concat_variant` → `render_concat_cot` (no perturbation chain → default `num_perturbations=0`). 1455/1604 zeros are concat-variant; only 149 are genuine arithmetic rows that realised 0 perturbations. **True marker = the `<concat_evidence>` tag in `cot_1_5`.** The word "concat" appears in *every* CoT's basic-knowledge vocab, so it is NOT a valid marker. The 0-share is ~flat (~9%) across all dataset_category values because concat is independent of example count.

**2. Among arithmetic rows, more example equations → fewer perturbations (monotonic).** Row-normalized % per `dataset_category`:

| category | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| examples_1 | 9.8 | 0.9 | 4.4 | 42.1 | 42.8 |
| examples_2 | 9.1 | 0.1 | 20.3 | 34.9 | 35.5 |
| examples_3 | 8.8 | 9.5 | 38.5 | 24.8 | 18.4 |
| examples_4 | 8.3 | 22.4 | 40.9 | 17.1 | 11.4 |
| examples_5 | 9.2 | 38.3 | 34.1 | 12.9 | 5.5 |

The **1-perturbation bucket (1942 rows)** is 77% examples_4+examples_5, ~21% examples_3, and almost no examples_1/2 — despite examples_1 and examples_2 each being ~23% of the dataset overall.

**Why:** more example equations constrain more symbols → fewer spare digits → reassigns become impossible (`perturbation.py:108`, one reassign per spare), leaving only swaps, so the realised chain bottoms out at 1. 71% of 1-perturbation rows are full permutations (0 spares).

**How to apply:** when using `cot_num_perturbations` as a difficulty/length signal (e.g. curriculum), filter concat rows via `<concat_evidence>` first, and remember the count is driven mainly by spare-digit count (example-equation count), not by intrinsic arithmetic hardness. See [[pure_concat_rule]].
