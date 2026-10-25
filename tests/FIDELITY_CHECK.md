# Fidelity Check — Synthetic vs Real Dataset

Validates that a synthetic augmented dataset shares the same structural and
statistical properties as the real source (`only_cryptharithm_solved_k_1.parquet`).
Every constraint enforced by the v3 generator has a corresponding automated check.

See also: `specs/synthetic-generation-fidelity.md` for full rationale and observed values.

---

## Quick Start

```bash
# Check default file (synthetic_v3_1k_canonical.parquet vs real source)
uv run python tests/check_fidelity.py

# Check a specific synthetic file
uv run python tests/check_fidelity.py inputs/data/v3/synthetic_v3_1500_prompt.parquet

# Check against a non-default source
uv run python tests/check_fidelity.py MY_SYNTH.parquet \
    --source inputs/data/v2/only_cryptharithm_solved_k_1.parquet
```

Exit code is `0` if all checks pass, `1` if any fail.

---

## What Is Checked

The script runs 28 checks grouped into 10 categories. Each compares the synthetic
dataset against the real source and prints PASS (✓) or FAIL (✗) with the real
value, synthetic value, and tolerance.

### 1 — Strategy Distribution

Checks that the fraction of `standard`, `little_endian`, and `pure_concat` problems
matches the real distribution within **±5 pp**.

| Strategy | Real | Target |
|---|---|---|
| standard | 55.5% | ±5 pp |
| little_endian | 37.1% | ±5 pp |
| pure_concat | 7.3% | ±5 pp |

**How it is enforced:** sampled from `stats.strategy_probs` in `synth.py`.

---

### 2 — Operator Count per Problem (n_ops)

Checks that the distribution of distinct operator glyphs per problem matches within
**±5 pp** per bucket.

| n_ops | Real | Target |
|---|---|---|
| 1 | 1.5% | ±5 pp |
| 2 | 37.0% | ±5 pp |
| 3 | 61.5% | ±5 pp |

**How it is enforced:** `n_ops` is sampled from `stats.n_operators_probs`; each
pre-selected op is guaranteed exactly one equation slot (mandatory coverage), with
remaining slots filled by weighted sampling.

---

### 3 — Unique Operand Symbol Count per Problem (n_osym)

Checks the distribution of unique symbols in operand positions (l0/l1/r0/r1 +
magnitude) within **±7 pp** per bucket.

| n_osym | Real | Target |
|---|---|---|
| 8 | 6.6% | ±7 pp |
| 9 | 29.9% | ±7 pp |
| 10 | 63.1% | ±7 pp |

Also includes a **hard-cap check**: no arithmetic problem may have more than 10 unique
operand symbols. Any violation is an immediate FAIL (the digit↔symbol bijection
allows at most 10 digits).

**How it is enforced:** bijection naturally caps at 10 digits; pure_concat uses a
shared symbol pool of `n_osym` symbols sampled from `stats.n_operand_syms_probs`.

---

### 4 — Result Magnitude Length (mag_len)

Checks the distribution of the number of symbols in `actual_rhs` (result magnitude,
excluding sign symbol) within **±6 pp** per bucket.

| Length | Real | Target |
|---|---|---|
| 1 | 7.8% | ±6 pp |
| 2 | 40.5% | ±6 pp |
| 3 | 25.2% | ±6 pp |
| 4 | 26.5% | ±6 pp |

**Status: Emergent.** Naturally follows from op selection + operand magnitude bias
toward [10, 99]. Not directly enforced.

---

### 5 — Sign Rate

Checks the overall fraction of arithmetic equations that produce a signed (negative)
result, within **±8 pp**.

| Metric | Real | Target |
|---|---|---|
| Sign rate | 15.8% | ±8 pp |

> **Note:** The tolerance is wider here by design. Synthetic sign rate is naturally
> ~10% (driven by op probability weights). The real data is 15.8% because the
> original generation biased `sub_signed` toward L < R — a bias not recoverable
> from stored data. The ±8 pp tolerance accepts this known gap.

---

### 6 — Sign Position (Prefix vs Postfix)

Checks that the split of sign glyphs appearing before (prefix) vs after (postfix)
the magnitude matches within **±3 pp**.

| Position | Real | Target |
|---|---|---|
| prefix | 94.7% | ±3 pp |
| postfix | 5.3% | ±3 pp |

**How it is enforced:** sampled from `stats.sign_pos_probs` in `_generate_arithmetic_equation`.

---

### 7 — Zero Padding Rate

Checks the fraction of arithmetic equations where the result has a leading-zero
prepended (result longer than the natural digit length), within **±1.5 pp**.

| Metric | Real | Target |
|---|---|---|
| Zero-pad rate | 1.65% | ±1.5 pp |

**How it is enforced:** `Bernoulli(stats.zero_pad_prob)` per equation in
`_generate_arithmetic_equation`; the zero digit is prepended to `mag_digits` before
little-endian reversal.

---

### 8 — Pure-Concat n_concat Distribution

For pure_concat problems, checks the distribution of how many example equations use
the concat operator (i.e. provide direct evidence for the fast path), within
**±15 pp** per bucket (wider tolerance due to small counts).

| n_concat | Real | Target |
|---|---|---|
| 1 | 47.5% | ±15 pp |
| 2 | 30.5% | ±15 pp |
| 3 | 18.6% | ±15 pp |
| 4 | 3.4% | ±15 pp |

**How it is enforced:** sampled from `stats.concat_n_concat_probs`; remaining
example slots become fake distractor equations.

---

### 9 — Symbol Pool Restrictions

**Hard checks — any violation is a FAIL.**

Verifies that no symbol from the set `, . ; _ ~` appears in any operand position
or operator position in the synthetic data. These 5 symbols are never used in the
real dataset.

| Check | Expected |
|---|---|
| Forbidden operator symbols | ∅ |
| Forbidden operand symbols | ∅ |

**How it is enforced:** `stats.operand_syms` (23 symbols) and `stats.operator_syms`
(26 symbols) are computed directly from real data; all sampling uses these pools.

---

### 10 — Pure-Concat Structural Integrity

**Hard checks — any violation is a FAIL.**

Verifies that every pure_concat row satisfies:
- `digits_mapping` is null / None (pure_concat solver does not need it)
- `ops_mapping` contains only the concat glyph — no arithmetic operators

| Check | Expected |
|---|---|
| `digits_mapping` null | 0 violations |
| `ops_mapping` concat-only | 0 violations |

---

## Informational Metrics (Not Checked)

Two metrics are printed at the end but not included in the PASS/FAIL count because
their sample size is too small for stable estimates at n=600–1500:

- **Sign demo rate**: among problems where `to_predict` is signed, fraction that also
  have a same-op signed example. Target ≈ 54% (real). High variance at n < 200.
- **Pad demo rate**: same concept for zero-padded results. Target ≈ 35%. Very small
  sample (< 20 cases even at n=1500).

Both are enforced via Bernoulli rejection in `synth.py` but cannot be reliably
verified at typical run sizes.

---

## Adding New Checks

To add a check for a new statistic:

1. Compute the statistic inside `_extract_stats()` (one pass over all rows).
2. Add a tolerance constant at the top of `check_fidelity.py` (`_TOL_*`).
3. Call `_check_scalar()` or `_check_dist()` in `main()`.
4. Document the real value and target range in this file.

---

## Expected Output (All Pass)

```
Source : only_cryptharithm_solved_k_1.parquet  (803 rows)
Synth  : synthetic_v3_1k_canonical.parquet  (1000 rows)

Check                                          Real       Synth  Note
──────────────────────────────────────────────────────────────────────
✓ strategy[standard]                          55.5%      56.1%  Δ=0.6pp (tol ±5pp)
✓ strategy[little_endian]                     37.1%      35.0%  Δ=2.1pp (tol ±5pp)
✓ strategy[pure_concat]                        7.3%       8.9%  Δ=1.6pp (tol ±5pp)
✓ n_ops[1]                                     1.5%       1.4%  Δ=0.1pp (tol ±5pp)
...
✓ forbidden_operand_syms                          ∅          ∅  must be empty
✓ pc_structural (...)                      0 issues   0 issue(s) OK
──────────────────────────────────────────────────────────────────────

PASSED  28/28 checks passed
```
