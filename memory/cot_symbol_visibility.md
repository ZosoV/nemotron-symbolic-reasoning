---
name: cot_symbol_visibility
description: CoT may only reference symbols visible to the LLM — those in the example equations or the to_predict lhs that are also explained by at least one example; never the answer
metadata:
  type: project
---

**Visible symbols** = those appearing in the example equations + the to_predict lhs
symbols that are also constrained by at least one example equation.

When generating a CoT, do NOT mention or reason about:
1. Symbols that appear only in the **answer** — not visible in the LLM prompt at all.
2. Symbols that appear only on the **LHS of the `to_predict` equation** and are not
   explained (i.e., do not appear in any example equation). These are unconstrained
   and must not appear in reasoning steps (only flagged in the header as "unconstrained").

The answer is known only to the internal rendering logic; it must **never** be written
into any CoT reasoning step — only in the final `\boxed{}` conclusion.

**Why:** These symbols are invisible to the LLM at inference time. Referencing them
leaks information the model cannot access, making the CoT invalid as a reasoning trace.

**How to apply:**
- Before writing any symbol into a CoT step, verify it appears in at least one example
  equation on the prompt side.
- If it only appears in the answer or solely on the to_predict LHS without sample
  support, omit it from the reasoning.
- In `_render_initial_reasoning` (`render.py`), unconstrained lhs symbols are flagged
  but not used in observations. Factual-observation generators operate solely on example
  equations, so they are inherently safe — but check any prose that references operand
  symbols to ensure none come from the answer field.
