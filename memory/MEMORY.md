# Memory Index

Small findings and rules discovered through analysis that should not be re-derived. Check this index first; open a file only when relevant.

- [pure_concat_rule.md](pure_concat_rule.md) — `_try_concat` requires at least one matching *example* equation as evidence; `to_predict` alone does not qualify. 16 no-match cases fall through to arithmetic search.
- [cot_num_perturbations_distribution.md](cot_num_perturbations_distribution.md) — `0`s are mostly concat-variant CoTs (marker: `<concat_evidence>` tag, not the word "concat"); among arithmetic rows more example equations → fewer perturbations (1-pert bucket is 77% examples_4/5, ~0% examples_1/2), driven by spare-digit count.
- [cot_token_budget.md](cot_token_budget.md) — Use `max_cot_tokens: 6000`; Nemotron scorer has max_model_len=8192/max_tokens=7680; natural CoT max is ~5750 approx tokens; 6000 leaves ~1500 token headroom for inference overrun before boxed answer is cut off.
- [cot_symbol_visibility.md](cot_symbol_visibility.md) — Visible symbols = example equations + to_predict lhs symbols constrained by examples; answer-only and unexplained lhs symbols are invisible to the LLM and must never appear in CoT reasoning steps.
