# Pure-Concat Evidence Rule

`_try_concat()` in `src/solver/solver_eq_symbolic.py` triggers the concat fast-path only when **all three conditions hold**:

1. `to_predict` RHS has no sign and exactly 4 symbols.
2. At least one **example** equation (not `to_predict` itself) shares the same operator symbol.
3. All those matching examples consistently follow `concat_fwd` (`l0l1r0r1`) or `concat_rev` (`r0r1l0l1`).

If condition 2 fails (no matching examples), return `None` → fall through to arithmetic search.

**Why:** `to_predict` alone trivially passes the pattern check (its gold RHS IS the answer), so it cannot serve as evidence in an inference context. Dataset analysis of 824 problems found 16 such cases — all routed correctly to arithmetic search after this fix.

**Stats (eval_data_k_1.parquet, 824 problems):**
- 60 problems: fast-path valid — consistent direction, 0 wrong inferences
- 16 problems: no matching examples → arithmetic search
- 748 problems: to_predict is not concat

**CoT generator usage:** `decode.detect_pure_concat(examples, q)` in `src/cot_generator/decode.py` re-derives this rule from the parsed equations — never trusting the `solver_strategy` label. Rows passing the corrected rule get `render_concat_cot` (3-tag concise CoT); those failing the rule remain skipped.
