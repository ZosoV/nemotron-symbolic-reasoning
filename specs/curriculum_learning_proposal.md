# Curriculum SFT for nemotron

This document covers the design of the n-shot strategy for learning different symbolic equation puzzles using a curriculum learning strategy.

---

## 1. Context

This training is part of a Kaggle competition by Nvidia, using the Nemotron model and GPU that they provide. They have some restrictions regarding the usage of the GPU: weekly, we only have access to 30h of execution, and the notebooks are offline.

- The goal is to train a LoRA adapter on top of the Nemotron model to solve some puzzles that have the following format

In Alice's Wonderland, {a given task}. 

{some examples are provided}

Now, determine the output for: {additional_input}

- The examples have been divided into some categories:

**Numeral**:

In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
52 -> LII
14 -> XIV
97 -> XCVII
23 -> XXIII
Now, write the number 43 in the Wonderland numeral system.

**Unit conversion**:
In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
33.81 m becomes 18.87
32.74 m becomes 18.28
35.04 m becomes 19.56
25.86 m becomes 14.44
45.98 m becomes 25.67
Now, convert the following measurement: 27.74 m

**Gravity**
In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = 3.54s, distance = 82.75 m
For t = 1.28s, distance = 10.82 m
For t = 3.21s, distance = 68.04 m
Now, determine the falling distance for t = 3.31s given d = 0.5*g*t^2.


**Cipher**
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
kqlz twhjohm lhjw pjejth -> king creates near palace
iwjzfl ufeefsm oah zfeihl iffw -> dragon follows the golden door
pwqlthmm qcjzqlhm oah mowjlzh owhjmbwh -> princess imagines the strange treasure
Now, decrypt the following text: obwoeh mhhm jwfbli cfblojql


**Bit Manipulation**
The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01110000 -> 11100111
11011000 -> 11001101
00010001 -> 00111011
01100000 -> 11100111
01010101 -> 00101111
11101111 -> 10001110
00101000 -> 11110111
00101001 -> 00110111

Now, determine the output for: 11110000

**Equation Numeric**
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
64*49 = 4234
01{83 = 8
33@74 = 08
87{78 = 9
Now, determine the result for: 36*53

**Cryptarithm (Symbolic Equations)**
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
?|-<" = -?<
"<+`" = <!
`^+?^ = {{}
{|+{: = "<
^{+?} = {"}
Now, determine the result for: <!-`{


Currently, the community has managed to solve the contest partially; when it is evaluated on the real test data, the leaderboard gets around 87%. When I copied a notebook from the Kaggle competition and ran it, I got around 84% in the real test set (probably I can test others and get 86%). My team is composed of me and my colleague; my colleague already got 86%.

To achieve more accuracy, the problem resides in solving the "Cryptarithm (Symbolic Equations)" problems. Currently, there are around 823 problems out of 9500 in the train dataset that correspond to "Cryptarithm (Symbolic Equations)" problems. Of these 823 problems, only 65 have been solved.

My objective is to train the LoRA adapter to learn to reason and solve these problems. In that way,
- I want somehow to keep the previous learning, let's assume I can get a model with 86% of accuracy. I'm very sure what method I could use to keep the learning.
  - TODO: Write an approach to keep the learning.

But for now, I have focused on solving the Cryptarithm (Symbolic Equations) problems.

My intention is to make the LLM learn to solve these issues by following a curriculum learning strategy. A problem can have between 3 and 5 examples and one additional lhs to be solved. In theory, the examples are adding constraints to solve the question. If we keep this in mind, it is easier to solve a problem with 0 examples than to solve a problem with 5 examples because the latter is more constrained.

The assumption for getting the solution is that the equations are somehow encrypted, so that there exist a solving strategy, a mapping of operand symbols to digits, and a mapping from operator symbols to operations, that can be used to infer the solution.

- IMPORTANT: in most of the cases the solutions are undetermined with this approach. There are different possible combinations that can find a solution following this approach

The solution that I want to get is something like this:
- solving strategy
- operation mapping
- digit mapping (optional)
- heuristics to format the solution inferred from the examples:
  - zero-padding required (optional)
  - pre or post sign required (optional)

Coming back to my curriculum learning strategy, the idea is the following:

1. Use the dataset with 0 examples to learn some basics about the solver, some assumptions, factual observations, and possible heuristics based on the form of the equation. One equation by itself tells us a lot.
2. Use the dataset with 1 example to start learning about solving the question given the example as a constraint.
3. Progressively, add more examples to learn to fill more constraints to solve the question.


**Curriculum direction:** training starts at n=0 (no examples to reconcile) and
progresses toward n=5 (most examples). More examples = harder *generation task*,
because the CoT must explicitly derive and cross-check more constraints, manage
covered/uncovered reasoning, and produce a longer rationale — even though more
examples also pin down the mapping more tightly.

> **Adopted training variant (see `specs/curriculum_training_pipeline.md`).** The
> competence-based pipeline implements this direction but **diverges from vanilla
> Platanios steps 8–9** ("keep all unlocked buckets in the pool as built-in
> replay"). Because the real 823-puzzle eval set has only 3–5 examples (no 1- or
> 2-example puzzles), the adopted Dial-C schedule **tapers ex_1/2 to zero in the
> final phase** (`t ≥ T`), training on ex_3/4/5 only to match the eval
> distribution. Buckets still unlock progressively (1→5); only the late-phase
> *mixing weight* changes. The implemented buckets are n=1..5 (n=0 is not part of
> the merged curriculum parquet).

## 2. What to learn from Dataset with 0 examples?

- Starting with the dataset with zero examples. NOTICE: with this dataset, you can solve directly because you don't have examples; you only have 1 example that can admit different possible solutions. Instead, we can focus on learning the basics and provide this learning to the LoRA. I want to use this dataset to learn the basic ideas:
  - structure of the equation
  - possible operations
    - maybe it is not necessary to add all the operations. Only the most critical for these cases, for example the ones that are present in the different solutions of this sample
  - solver strategies
  - ops mapping
  - digit mappings
  - possible factual observations based on the structure of the equation that reduce the number of assumptions
    - ops sign on the rhs => ops should be something related with difference
    - concatenation => same symbols (possible reverted) on the rhs without the op symbol
    - ops sign used as post fix => little endian strategy (but the other direction is not true)
    - understand what digits are more probable:
      - if the eq involves a difference, there should be a rule to pick more or less probable symbols, so that the difference can still be satisfied
      - if the operation is sum or mult, there should also be a rule to pick more or less probable symbols, so that the operation doesn't overflow, considering that you know the rhs
    - understand that given the op symbols, it is more probable to have a specific operation over the others.
      - e.g. with symbols * the operation mult (or similar) are more probable
      - e.g. with symbols - the operations related with difference are more probable
      - e.g. with symbols + the operation add or concat are more probable
      - There is even a way to calculate this probability from (solver_results_forcing_base_10_k_1.parquet)
    - understand that sometimes the rhs can contain a symbol mapping to 0 in the left side zero-padding for standard learning strategies
    - 
  - make the LLM understand that there are different possible mappings
    - maybe instead of separating in different solutions from 1 samples
    - use only 1 time and show different possible solutions
  - heuristic strategies applied to the solution to format it correctly based on the examples:
    - zero padding
    - pre or post sign required
    - NOTE: this strategy only work with dataset with 1 or more examples

## 3. How to create the Datasets?

### Dataset with 0 examples

- Pick every equation on the 823 sample and get the solution using the `solver_eq_symbolic.py` → output parquet.
- Balance the data if necessary to have enough standard and little endian cases, and to cover the necessary operations.
- Create a chain of thought: using the conventions on "What to learn from Dataset with 0 examples?"

Every equation in every row becomes a separate zero-shot sample — both
`type == "example"` and `type == "to_predict"` equations. Total: **4,165
samples**.

- **id:** `{row_id}_{equation_index}`
- **prompt:** intro header + blank examples section + query line
- **answer:** `rhs_str(eq["rhs"])` (lossless reconstruction validated on 100%
  of rows)

### n = 1..5 datasets — leave-one-out query rotation

**Source:** all 823 rows — arithmetic (`solver_strategy` ∈ {standard,
little_endian}) and `pure_concat` treated **uniformly in one pipeline**.
No branching is done by row type. The downstream solver is responsible for
determining which exercises fall under `pure_concat`; a rotated concat
equation can produce a *covered* or *uncovered* sample that the solver handles
differently.

**Design (design ④):** for each row, *every* equation takes a turn as the
query; n examples are drawn from the remaining equations. One sample per
`(row, query)` unit.

- More distinct query→answer targets than alternatives (no repeated answers).
- Naturally balanced: rows with more equations contribute more samples, but
  without the combinatorial explosion of drawing all subsets.
- Consistent with n=0: every equation appears as a query at every n.

#### Unit classification

Each `(row, query)` unit is classified by examining its **pool** (all other
equations in the row):

| class | condition | coverage forced to |
|---|---|---|
| `forced_cov` | pool has ≥1 matching op; <n non-matching ops | covered |
| `forced_unc` | pool has 0 matching ops | uncovered |
| `flexible` | pool has ≥1 matching op AND ≥n non-matching ops | either |

**Matching op** = another equation whose `lhs['op']` glyph equals the query's
`lhs['op']` glyph.

#### Keep-all-units rule

All eligible units (those with `≥ n` pool equations) are kept — no
down-selection. The flexible units are assigned to covered/uncovered to hit the
target ratio; if forced units already exceed the target for a class, all
flexible units go to the other class and coverage drifts naturally (n=5 case).

#### Achieved sizes

**Target ratio:** 659/164 ≈ **80.07 % covered / 19.93 % uncovered** (coverage
of the full 823-row source on its original `to_predict` equations).

| n | total | covered | uncovered | C% | note |
|---|---|---|---|---|---|
| 1 | 4165 | 3335 | 830 | 80.07% | all eligible units |
| 2 | 4165 | 3335 | 830 | 80.07% | all eligible units |
| 3 | 4165 | 3335 | 830 | 80.07% | all eligible units |
| 4 | 3221 | 2579 | 642 | 80.07% | all eligible units |
| 5 | 1716 | 1489 | 227 | 86.77% | natural drift — no flexible units |

n=1..3 match the n=0 dataset size (4165): the whole n=0..5 family spans the
same 823-row source. At n=5 there are no flexible units, so coverage drifts to
the ratio determined entirely by forced_cov and forced_unc.

#### Example selection (seeded, deterministic)

- **Covered unit:** pick 1 matching example at random, then fill n−1 slots
  from the remaining pool.
- **Uncovered unit:** pick n non-matching examples at random.
- Shuffle the final example order. Seed = 42.

#### Output schema (minimal)

| column | description |
|---|---|
| `id` | `{row_id}_q{query_index}` — unique; puzzle id is the prefix before `_q` |
| `prompt` | Alice intro + n example lines + query line |
| `answer` | `rhs_str(query["rhs"])` |

Files: `data/dataset_examples_{n}.parquet`.

#### Prompt format

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
<lhs_1> = <rhs_1>
...
<lhs_n> = <rhs_n>
Now, determine the result for: <query_lhs>
```

Reconstruction helpers (validated 100% against source):
```python
def lhs_str(l):   return l["l0"] + l["l1"] + l["op"] + l["r0"] + l["r1"]
def rhs_str(rh):
    body = "".join(rh["actual_rhs"])
    if rh["op"] is None: return body
    return rh["op"] + body if rh["sign_pos"] == "prefix" else body + rh["op"]
```

---

## 4. Coverage and Difficulty Axes

**Two independent axes make the generation task harder or easier:**

| axis | easier end | harder end |
|---|---|---|
| n (examples shown) | n=0 — no examples to reconcile | n=5 — most constraints to reconcile |
| coverage | covered — query op is demonstrated | uncovered — must generalise |

**Why coverage is held constant at 80.07/19.93 across n=1..4 (drift at n=5):**
- If coverage drifts with n, difficulty moves on *two* axes at once, making
  the curriculum signal ambiguous.
- Holding coverage constant means *n* is the sole varying difficulty axis.

**Covered vs uncovered:**
- *Covered*: at least one shown example uses the same operator glyph as the
  query. The model can observe how that operator behaves before answering.
- *Uncovered*: no example demonstrates the query's operator. The model must
  generalise from digit-mapping constraints alone.

---

## 4. Ambiguous Candidate Expansion

Puzzles are **underdetermined**: fewer shown examples leave more valid mappings
consistent with the prompt. The k=10 solver run shows:

- Mean **4.58** distinct solutions per puzzle; many puzzles hit the cap of 10.
- At low n (especially n=0), a single prompt may admit dozens of consistent
  solutions.

**Implications for SFT:**

- Labelling an underdetermined sample with a single ground-truth answer trains
  the model to *recall* the secret rather than *deduce* it from the shown
  examples — the model cannot actually solve it from the prompt alone.
- Two principled options:
  1. **State the assumption in the CoT:** "The examples don't uniquely pin
     down `-`; I assume absdiff …" — then derive from there.
  2. **Accept any solver-consistent answer:** treat the label as a set. Any
     mapping that satisfies all shown examples produces a valid answer.

Option 1 is practical for SFT (single target); option 2 breaks
query→answer determinism and reduces memorisation pressure more aggressively.

**Underdetermination signal:** use `solution_count` from the solver output.
Samples where `solution_count > 1` are ambiguous; the CoT should reflect this.

---

## 5. Memorisation Analysis

### The irreducible risk

All samples spawned from one source row share **one** ground-truth mapping
(digits + operators). No matter how many samples you mint from a row, the
underlying information is bounded by the **748 original puzzles**. The model
can learn "this glyph set → this mapping" as a lookup table.

### Three risk types

| risk | what is memorised | does ④ reduce it? | does CoT help? |
|---|---|---|---|
| **Answer memorisation** | query lhs → answer string | partially (more distinct targets) | only if CoT derives, not asserts |
| **Mapping memorisation** | glyph cluster → digit/op assignment | no — same mapping repeats across all row's samples | no — mapping facts repeat in CoT too |
| **CoT-template memorisation** | the deduction procedure | ✅ yes — this is *productive* | ✅ yes — this is the goal |

### What design ④ improves

- **More distinct targets:** ~3,785 distinct query→answer facts per stage
  instead of 748. A simple lookup table is harder to memorise.
- **Dual-role equations:** the same equation appears as an *answer* in one
  sample and as a *shown example* in another. This weakens direct
  input→output shortcutting.

### What stays irreducible

- The per-row secret mapping still recurs across all samples from that row.
- Augmentation inflates sample counts but not *information*; real diversity
  is bounded by 748 puzzles.

### Four controls

1. **Puzzle-level train/test split.** Always split by the source row (`id`
   prefix before `_q`), never by sample. Otherwise test answers appear
   verbatim in training prompts as shown examples.

2. **Derive-not-assert CoT.** The chain of thought must *deduce* each symbol
   value from the shown examples, not state the mapping as a known fact. See
   section 6.

3. **Handle underdetermined samples.** At low n, state ambiguity explicitly
   in the CoT ("examples leave `-` unconstrained; assuming absdiff") rather
   than asserting a uniquely determined answer.

4. **Novel-mapping holdout.** Ideal test puzzles use glyph sets and
   digit/operator assignments not seen during training. Under this condition,
   good scores can only come from generalised reasoning, not recall.
