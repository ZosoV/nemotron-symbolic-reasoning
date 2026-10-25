---
name: cot-token-budget
description: Recommended max_cot_tokens cap for Nemotron competition scorer (max_model_len=8192, max_tokens=7680)
metadata:
  type: project
---

Set `max_cot_tokens: 6000` for the Nemotron scoring config.

**Why:** The scorer uses `max_model_len=8192` and `max_tokens=7680`. The model output must include the full CoT *plus* the `\boxed{answer}` line plus any extra thinking; if generation hits 7680 before the box is written, `extract_final_answer` gets no `\boxed{}` and the row scores zero. Measured `cot_approx_tokens` (2.81 ch/tok) tracks real tokens at ~1.02× (verified via cl100k_base on 8 samples). Natural training-target max is ~5750 approx tokens (4-perturbation CoTs). 6000 leaves ~1500 token headroom (~19%) for inference-time overrun without forcing perturbation shedding on any current puzzle.

**How to apply:** Keep `max_cot_tokens: 6000` in `config.yaml → cot_generator:`. Drop to ~5500 if truncated / boxless rows appear in a `debug=True` scorer run. The `cot_num_perturbations` column shows which rows shed perturbations.
