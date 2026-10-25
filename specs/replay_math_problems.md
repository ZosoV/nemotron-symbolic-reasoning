# Ideas from the training of nemotron
For math, code, and science, we start from curated problem sets and use open source permissive models such as GPT-OSS-120B to produce step-by-step reasoning traces, candidate solutions, best-of-n selection traces, and verified CUDA kernels.

For all domains, we apply a unified data filtering pipeline to ensure that only high-quality, license-compliant, and verifiable samples are used for post-training. We first discard malformed examples using structural checks (e.g., missing tool definitions when tool calls are present). We then aggressively filter reasoning traces exhibiting pathological repetition, such as repeated n-grams within a sliding window or across the entire trajectory, which we found to be a strong indicator of malformed or low-quality reasoning.

# Replay Dataset Specification

**Purpose:** Define what a "good" replay dataset looks like for each core skill, so that a
LoRA adapter trained on the *Alice's Wonderland* latent-rule-induction puzzles does not
catastrophically forget the base capabilities of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

**Scope:** These specs describe *retention* data (keeping base skills alive), **not** task
data (the puzzle examples that teach the new task). The puzzle set is the supervised signal;
replay data is the regularizer.

---

## 0. Global criteria (apply to every skill)

A replay dataset, regardless of skill, MUST satisfy:

| Property | Requirement | Why |
| --- | --- | --- |
| Source provenance | Drawn from, or distributionally close to, the model's disclosed post-training corpus | In-distribution replay is what actually prevents drift; off-distribution data can *cause* forgetting |
| License | Permissively licensed / publicly downloadable | Several corpus datasets (TAUS, Scale HLE, HackerRank) are private and unusable |
| Format match | Chat-formatted with the model's template; reasoning traces wrapped in `<think>...</think>` where applicable | The adapter must keep parsing/emitting the model's native structure |
| Reasoning on/off balance | Mix of `enable_thinking=True` and `=False` samples | Your task is run in thinking mode, but you do not want to lose the no-think pathway |
| Answer verifiability | Final answer programmatically checkable (regex / numeric / exact-match) | Lets you filter bad traces and measure retention objectively |
| Decontamination | No overlap with your held-out puzzle eval set | Prevents inflated retention metrics |
| Volume | Small relative to task data (target ≈ 15–50% of the training mix) | Replay is a regularizer, not the lesson |
| Difficulty spread | Span easy→hard; do not replay only hard items | Preserves the full competence curve, not just the tail |

---

## 1. Math & Quantitative Reasoning

**Protects puzzle categories:** Numeral, Unit Conversion, Gravity, Equation Numeric, Cryptarithm.

These categories are all "infer a latent numeric/symbolic mapping from few examples, then
apply it." The shared base skill is multi-step arithmetic + symbolic manipulation + base/unit
conversion.

### Required characteristics
- **Digit-level arithmetic present.** Must include raw arithmetic (addition, multiplication,
  modular ops), not only word problems — your Equation Numeric and Numeral tasks are
  digit-/base-level, not narrative.
- **Step-by-step traces.** Solutions show intermediate work, not just final answers, so the
  model retains chain-of-thought arithmetic rather than memorized lookups.
- **Base / representation conversion coverage.** At least a subset touching non-decimal
  representations (binary, hex, Roman-numeral-like mappings) to protect the Numeral category.
- **Multi-operand symbolic equations.** Items where an operator's meaning must be tracked,
  protecting Cryptarithm and Equation Numeric.
- **Unit-bearing quantities.** Some fraction with physical units and scaling factors, to
  protect Unit Conversion and Gravity (which is `d = 0.5*g*t^2` with a hidden `g`).
- **Verifiable numeric answers.** Exact-match or tolerance-based checking on a final number.

### Disqualifiers
- Pure proof / olympiad prose with no numeric final answer (low retention value here).
- Answer-only datasets with no reasoning trace.

### Candidate sources (disclosed corpus)
Nemotron-PrismMath · OpenMathReasoning · Big-Math-RL-Verified · NuminaMath CoT ·
MetaMathQA · GSM8K · simple-arithmetic-problems · EleutherAI/arithmetic

> The last two are unusually well-matched: they are digit-level and map almost directly onto
> Equation Numeric and Numeral.

---

## 2. Bit / Logical Manipulation

**Protects puzzle categories:** Bit Manipulation.

Boolean ops, shifts, rotations, XOR/AND/OR/NOT, majority/choice functions. This is
code-flavored deterministic reasoning over fixed-width binary strings.

### Required characteristics
- **Explicit bitwise / boolean operations.** Datasets containing shift, mask, XOR, AND, OR,
  NOT, rotate operations — ideally on fixed-width words.
- **Deterministic, fully specified transforms.** Output must be uniquely determined by input
  (no ambiguity), matching the puzzle's exactness.
- **Trace shows the operation sequence.** Retains the *procedure*, not just input→output pairs.
- **Fixed-width / positional reasoning.** Bonus for tasks reasoning per-bit or per-position,
  since the puzzle is 8-bit string→8-bit string.
- **Code-execution-style verification.** Answers checkable by running the equivalent operation.

### Disqualifiers
- High-level application code (web, frameworks) with no low-level bit logic.
- Natural-language "logic puzzles" with no binary/boolean substrate.

### Candidate sources (disclosed corpus)
OpenCodeReasoning-2 · opc-sft-stage2 · synthetic OpenCodeReasoning traces ·
AGIEval-seeded logic (AQUA-RAT / LogiQA / AR-LSAT) for abstract rule-inference

---

## 3. Cipher / Substitution Decoding

**Protects puzzle categories:** Cipher.

Token-to-token / symbol-to-symbol mapping inferred from parallel examples. Structurally this
is sequence transduction (the same machinery translation uses). There is **no disclosed
cryptography dataset**, so translation pairs are the closest in-distribution proxy.

### Required characteristics
- **Parallel sequence pairs.** Aligned source→target where a consistent positional/token
  mapping must be applied — mirrors decrypt(`ciphertext`) → `plaintext`.
- **Consistent, learnable mapping.** The transform is stable across the example set (as in a
  substitution cipher), not free paraphrase.
- **Word- and character-level granularity.** Include both, since the cipher operates per-word
  but is recoverable per-character.
- **Order preservation.** Tasks where token order is preserved between source and target
  (substitution, not reordering) best match the puzzle.
- **Exact-match verifiable output.** Decoded string checkable verbatim.

### Disqualifiers
- Loose / creative translation where output is not a deterministic function of input.
- Summarization or open-ended generation (wrong structure entirely).

### Candidate sources (disclosed corpus)
News Commentary · synthetic translation pairs (En↔It/De/Es/Fr/Ja) · TAUS Translation Memory
*(private — likely unavailable; plan around the open pairs)*

> Treat translation as a *structural* proxy for substitution decoding. It keeps the
> sequence-transduction pathway warm; it will not by itself teach ciphering — that is the
> job of your task data.

---

## 4. Few-Shot Instruction Following & Output Format

**Protects:** the cross-cutting ability to parse the `"In Alice's Wonderland... examples...
Now determine X"` structure and emit a clean, correctly formatted answer.

### Required characteristics
- **In-context / few-shot examples in the prompt.** Samples that demonstrate "here are
  examples, now do the new one," matching your puzzle envelope.
- **Strict output-format adherence.** Includes structured-output / constrained-format
  samples so the adapter keeps emitting exactly the expected answer shape (no preamble).
- **Multi-turn + single-turn mix.** Preserves general conversational competence.
- **Instruction diversity.** Varied phrasings so the model does not overfit one template.
- **Both reasoning-on and reasoning-off examples.** As per the global criteria.

### Disqualifiers
- Heavily templated single-format data (causes format overfitting).
- Toxic / low-quality chat logs without filtering.

### Candidate sources (disclosed corpus)
HelpSteer3 · WildChat-1M · LMSYS-Chat-1M · synthetic structured-output data

---

## 5. Suggested replay mix

A starting point for the replay portion of the LoRA training mix (tune empirically):

| Skill bucket | Share of replay | Primary sources |
| --- | --- | --- |
| Math & quantitative | ~50% | PrismMath, OpenMathReasoning, simple-arithmetic-problems, arithmetic |
| Bit / logical | ~20% | OpenCodeReasoning-2, opc-sft-stage2 |
| Cipher / transduction | ~15% | translation pairs (open subset) |
| Instruction / format | ~15% | HelpSteer3, structured outputs |

**Replay-to-task ratio:** keep total replay at roughly 15–50% of the overall training tokens;
the puzzle task data should remain the majority signal.

---

## 6. Acceptance checklist (per dataset, before inclusion)

- [ ] Publicly downloadable / permissively licensed
- [ ] In or near the model's disclosed post-training distribution
- [ ] Chat-formatted to the model template; `<think>` blocks where relevant
- [ ] Contains reasoning traces, not answer-only (where the skill warrants it)
- [ ] Final answers programmatically verifiable
- [ ] Difficulty spread easy→hard
- [ ] Decontaminated against the puzzle eval set
- [ ] Mix of reasoning-on / reasoning-off samples