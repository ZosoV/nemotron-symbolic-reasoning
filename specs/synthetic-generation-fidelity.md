# Synthetic Generation Fidelity

Documents every statistic profiled from the real dataset and every rule applied by the v3 generator. Written after verification against 600-row test runs.

**Source dataset:** `inputs/data/v2/only_cryptharithm_solved_k_1.parquet` (803 rows, 744 arithmetic + 59 pure_concat)
**Generator:** `src/data_generation_v3/run_v3.py` → `src/data_generation_v2/{stats,synth}.py`
**Op registry:** `src/operations.py` (22 arithmetic ops + 2 concat; see [[project_solver_op_registry_source]])

---

## Strategy Distribution

| Strategy | Real | Synthetic |
|---|---|---|
| standard | 55.5% | ~55% |
| little_endian | 37.1% | ~37% |
| pure_concat | 7.3% | ~7% |

**Mechanism: Enforced** — sampled from `stats.strategy_probs`.

---

## Operator Count per Problem (n_ops)

Number of distinct operator glyphs per problem.

| n_ops | Real | Synthetic |
|---|---|---|
| 1 | 1% | 1% |
| 2 | 36% | 37% |
| 3 | 61% | 60% |

**Mechanism: Enforced** — pre-select n_ops from `stats.n_operators_probs`; pre-assign one equation slot per op (mandatory coverage) so each pre-selected op appears ≥ 1 time. The remaining slots are filled by weighted-random sampling from the pre-selected set. This guarantees exact n_ops in output.

*Key DataStats field:* `n_operators_probs`

---

## Unique Operand Symbol Count per Problem (n_osym)

| n_osym | Real | Synthetic |
|---|---|---|
| 8 | 6% | 4% |
| 9 | 29% | 34% |
| 10 | 63% | 61% |

**Mechanism: Enforced (cap only)** — arithmetic problems: hard cap at 10 (global digit↔symbol bijection allows max 10 digits); pure_concat problems: shared symbol pool of n_osym symbols sampled from `stats.n_operand_syms_probs`. No exact-match rejection; distribution is emergent from bijection constraints.

*Key DataStats field:* `n_operand_syms_probs`

---

## Operand Magnitude Bias

Single-digit operand (value < 10) rate.

| Metric | Real | Synthetic |
|---|---|---|
| P(operand < 10) | ~0.6% | ~0.6% |

**Mechanism: Enforced** — `_sample_operand` draws from [0,9] with probability `stats.single_digit_operand_prob` (~0.006), else from [10,99].

*Key DataStats field:* `single_digit_operand_prob`

---

## Result Magnitude Length

Number of symbols in `actual_rhs` (excluding sign symbol).

| Length | Real | Synthetic |
|---|---|---|
| 1 | 7% | 6% |
| 2 | 40% | 39% |
| 3 | 25% | 26% |
| 4 | 26% | 27% |

**Mechanism: Emergent** — naturally follows from op selection + operand magnitude bias toward [10,99]. No direct enforcement.

---

## Sign Appearance

Rate of signed results (negative values) among arithmetic equations.

| Metric | Real | Synthetic |
|---|---|---|
| Sign rate | 15.8% | ~10% |
| Prefix | 94.7% | ~94.5% |
| Postfix | 5.3% | ~5.5% |

**Sign rate mechanism: Partially emergent** — driven by op selection probabilities (`op_probs`). The real data sign rate (15.8%) is higher than the theoretical expectation from op_probs (~11%) because the original real-data generation was biased toward producing negative results for `sub_signed` (selecting L < R more often). This bias cannot be replicated from stored data alone. The sign direction (prefix/postfix) is **Enforced** by sampling from `stats.sign_pos_probs`.

*Key DataStats field:* `sign_pos_probs`

---

## Zero Padding

Rate of leading-zero-padded results among arithmetic equations.

| Metric | Real | Synthetic |
|---|---|---|
| Zero-pad rate | 1.65% | ~1.7% |

**Mechanism: Enforced** — with probability `stats.zero_pad_prob`, prepend a leading-zero digit to `mag_digits` before LE reversal. The zero digit is encoded via the global digit↔symbol bijection.

*Key DataStats field:* `zero_pad_prob`

---

## Demonstrating Examples (Sign/Pad)

Among problems where `to_predict` is signed or padded, the fraction that also have a same-op example exhibiting the same behavior (sign/pad) in the examples list.

| Metric | Real | Synthetic |
|---|---|---|
| Sign demo rate | 54% | ~40–65% (high variance) |
| Pad demo rate | 35% | ~0–35% (high variance) |

**Mechanism: Enforced (Bernoulli rejection, Option 3)** — after building a problem, if `to_predict` is signed with op O: sample `want_demo ~ Bernoulli(stats.sign_demo_prob)`; let `has_demo = ∃ example with op O that is signed`; reject if `want_demo ≠ has_demo`. Same for padded. High variance is expected at n=600 due to small number of signed/padded to_predict problems.

*Key DataStats fields:* `sign_demo_prob`, `pad_demo_prob`

---

## Operand-Symbol Coverage (Deducibility)

Among `to_predict` LHS operand symbols, how many are absent from ALL example LHS/RHS symbols.

| Missing syms | Real | Synthetic |
|---|---|---|
| 0 missing | 78% | ~75% |
| 1 missing | 20% | ~23% |
| 2 missing | 2% | ~3% |

**Mechanism: Emergent** — consequence of the global digit↔symbol bijection + random operand sampling. No enforcement needed; distribution matches within ~3%.

---

## Pure-Concat Problem Structure

| Property | Real | Synthetic |
|---|---|---|
| n_concat examples (1) | 47% | ~50% |
| n_concat examples (2) | 31% | ~29% |
| n_concat examples (3) | 19% | ~17% |
| n_concat examples (4) | 3% | ~4% |
| Total unique n_osym | 7–10 | 7–10 |
| ops_mapping | concat glyph only | concat glyph only |
| digits_mapping | null | null |

**Mechanism: Enforced** —

- **n_concat**: sampled from `stats.concat_n_concat_probs`; remaining example slots get fake distractor equations.
- **Fake distractors**: operand and RHS symbols drawn from the shared symbol pool (same pool as concat equations) → realistic n_osym. Distractors are structurally valid but not arithmetically verified (solver ignores them for pure_concat problems).
- **Shared symbol pool**: n_osym symbols sampled upfront from `operand_syms`, used by all equations. This matches real data where all equations share one pool.
- **ops_mapping**: only the concat glyph is stored (matching real data); distractor op glyphs appear in equations but not in ops_mapping.
- **digits_mapping**: always null (pure_concat solver path does not need it).

*Key DataStats fields:* `concat_n_concat_probs`, `n_operand_syms_probs`

---

## Symbol Pool Restrictions

| Pool | Symbols (n) | Never-used symbols excluded |
|---|---|---|
| operand_syms | 23 | `, . ; _ ~` |
| operator_syms | 26 | `, . ; _ ~` excluded; `* + -` included |

**Mechanism: Enforced** — `stats.operand_syms` and `stats.operator_syms` are computed directly from real data (l0/l1/r0/r1/magnitude and lhs.op positions respectively). The 5 symbols `, . ; _ ~` never appear in real data and are therefore absent from both pools.

---

## Deduplication

Two modes available via `--dedup-mode`:

| Mode | Mechanism |
|---|---|
| `canonical` (default) | Symbol-invariant structural signature: (strategy, ((op_name, L_val, R_val), ...)) for arithmetic; ordinal label sequence for concat. Seeded from 803 real-data signatures. |
| `prompt` | Exact prompt-string match. Seeded from 803 real-data prompts. |

Both modes prevent duplicates within the synthetic batch AND against the source 803 rows.

---

## Output Files

| File | n_rows | Dedup mode |
|---|---|---|
| `inputs/data/v3/synthetic_v3_1k_canonical.parquet` | 1000 | canonical |
| `inputs/data/v3/synthetic_v3_1k_prompt.parquet` | 1000 | prompt |
| `inputs/data/v3/synthetic_v3_1500_canonical.parquet` | 1500 | canonical |
| `inputs/data/v3/synthetic_v3_1500_prompt.parquet` | 1500 | prompt |

All files have `cot_variant_source="synthetic_v3"` and `dataset_category` ∈ {examples_3, examples_4, examples_5}.
