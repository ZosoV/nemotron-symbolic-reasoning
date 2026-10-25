# Curriculum Training Pipeline — Nemotron LoRA

How `notebooks/training_eval/end-to-end-curriculum_learning.ipynb` trains the
cryptarithm curriculum **directly from parquet**, with anti-forgetting replay, a
competence-based curriculum that targets the real eval distribution, and a fully
reproducible data order.

Related design docs (read these for the *why*):
- `specs/curriculum_learning_proposal.md` — n-shot dataset design (the examples_1..5
  buckets, leave-one-out rotation, memorisation controls).
- `specs/competence_based_curriculum_learning.md` — the Platanios competence algorithm
  this pipeline adapts (difficulty → CDF → competence → filtered sampling).
- `specs/nemotron-finetuning-pipeline.md` — the underlying LoRA finetuning mechanics
  (Unsloth + CCE + MoE tying) that this notebook reuses unchanged.

---

## 1. Goal

Lift the cryptarithm (Symbolic Equations) score **without forgetting** the ~86% the
base LoRA already earns on the other six problem categories. Training starts from
that 86% adapter (`RESET_WEIGHTS=False`, `ADAPTER_SRC` → the 86% submission).

---

## 2. Data sources

| dataset | rows | role | key columns |
|---|---|---|---|
| `dataset_examples_merged.parquet` | 17,265 | cryptarithm curriculum | `problem_id, prompt, answer, cot, mask_clues, dataset_category` |
| `train_corpus_no_cryptarithm.parquet` | 7,049 | anti-forgetting replay | `problem_id, prompt, answer, cot, mask_clues, problem_category` |

- `dataset_category ∈ {examples_1..5}` → curriculum difficulty `n = 1..5`.
  Bucket counts: 4165 / 4165 / 4121 / 3146 / 1668 (unbalanced).
- `problem_id` of a cryptarithm row is `{puzzle}_q{idx}_examples_{n}`; the source
  puzzle id is the prefix before `_q` (use for any puzzle-level holdout — see the
  memorisation controls in `curriculum_learning_proposal.md`).
- The cryptarithm parquet receives `cot` and `mask_clues` out-of-band (the CoT may
  arrive in a `cot_1_5` column — it is renamed to `cot` on load).
- Replay categories (6): Bit Manipulation 24.9% … Numeral 10.4%.

---

## 3. Tokenization + masking (from parquet)

`build_examples_from_parquet(tok, path, source, max_seq_len)` →
`{pid: {tokens, targets, weights, source, bucket}}`.

Per row:
1. Render the chat turn in **prediction mode** (`EMIT_BOXED`): user =
   `prompt + BOXED_INSTRUCTION`, assistant = `cot`; `_rearrange_think` places the
   CoT inside `<think>…</think>` and appends `\boxed{answer}` before `<|im_end|>`.
   (Same rearrange as `prepare_kaggle_corpus.ipynb`.)
2. **Mask from `mask_clues`** (not a single `COT_CONST_MARKER`). `mask_clues` is an
   array of `{mask_begin_clue, mask_end_clue}` string anchors over the rendered
   text. `render_and_mask` masks (loss=0) everything up to the end of the last
   `mask_end_clue` (which ends at the opening `<think>\n`), trains (loss=1) on the
   CoT + boxed answer. Implemented by a prefix-walk of `tok.encode` (handles the
   single start-anchored span the corpus uses). *Extension point:* a future row
   masking an extra mid-CoT span (e.g. an ops boilerplate block) needs a
   token-offset union over all clue spans.
3. Drop rows with no trainable tokens; truncate to `MAX_SEQ_LEN`.

`source='replay'` rows are keyed `replay::<problem_id>` so the cryptarithm and
replay id namespaces never collide; `bucket` is set only for cryptarithm rows.

---

## 4. Curriculum schedule

`build_schedule(...)` precomputes, **before training**, a list of `num_steps`
batches, each a list of `BATCH_SIZE` problem_ids. Three dials turn per step:

- **Dial A — difficulty.** `d = n`; normalized via the empirical CDF of bucket
  population (cumulative fraction = unlock threshold per bucket).
- **Dial B — competence `c(t)`.** Sqrt: `c(t)=min(1, √(t·(1−c₀²)/T + c₀²))`, rising
  from `C0` to 1.0 at `T = T_FRAC·NUM_STEPS`. A bucket is *unlocked* when its CDF
  threshold ≤ `c(t)` (bucket 1 always unlocked).
- **Dial C — mixing weight (non-stationary).** While `t < T`: **equal weight across
  unlocked buckets** (class-balanced — decouples difficulty from bucket size, so the
  small examples_5 isn't starved). At `t ≥ T`: restrict to `TARGET_BUCKETS = {3,4,5}`
  — the 823-puzzle eval distribution has **no 1/2-example puzzles**, so training
  there concentrates on the real target.

**Replay (separate, always on).** Each batch is `round(BATCH_SIZE·REPLAY_RATIO)`
non-cryptarithm rows (default 25% → 8 of 32) drawn **randomly/proportionally** from
the full replay corpus — never curriculum-gated. Random (not stratified) replay
reproduces the data mix the model already scores 86% on. The remaining slots are the
cryptarithm picks from Dials A/B/C. The batch is shuffled so crypt+replay interleave.

Per-bucket and replay draws use `_Cycler` (seeded shuffled cycling: even,
no-replacement coverage with reshuffle on exhaustion). The old
`max_steps = len//BATCH` clamp is removed — cycling lets `NUM_STEPS` run freely;
require `T < NUM_STEPS` so a target-fit phase exists.

---

## 5. Reproducible data order

Everything random is driven by one `random.Random(SEED)` inside `build_schedule`,
and the training loop only *consumes* the schedule (no in-loop shuffle).

- `SCHEDULE_PATH = None` → build from `SEED` + curriculum config, then dump to
  `SCHEDULE_DUMP` (`schedule.jsonl`, `{step, ids}` per line — mirrors the legacy
  `index.jsonl` replay).
- `SCHEDULE_PATH = <file>` → load that frozen schedule and ignore the knobs. This
  **pins the exact per-step batch order across code changes**, so a change to the
  learning logic (LR, loss, masking, model) can be A/B-tested against the 86%
  baseline with identical data order.

Caveat: identical data order ≠ bit-identical accuracy (CUDA kernels are not fully
deterministic). For the closest comparison also keep `MICRO_BATCH_SIZE` fixed — the
per-micro-batch loss normalization in the training loop makes the exact gradient
depend on how examples group into micro-batches. `LORA_DROPOUT=0.0` already.

---

## 6. Notebook layout

| cell | purpose |
|---|---|
| `curric_config` | `REPLAY_RATIO, C0, T_FRAC, COMPETENCE, TARGET_BUCKETS, SEED, SCHEDULE_PATH, SCHEDULE_DUMP, BOXED_INSTRUCTION` |
| `curric_helpers` | `_rearrange_think, render_and_mask, build_examples_from_parquet, competence_value, _Cycler, build_schedule, dump_schedule, load_schedule` |
| `run_training` (`ba69409a`) | env paths add `CRYPT_PARQUET`/`REPLAY_PARQUET`; corpus block tokenizes from parquet + builds/loads the schedule; loop iterates the schedule |

`ORIGINAL_PROBLEMS_ONLY` and `SHUFFLE_DATASET` are **not used** on this path (order
comes from the schedule).

---

## 7. Verification

1. **Schedule (offline, pure Python):** deterministic across two builds; 32/batch
   with 8 replay; only bucket 1 early, only {3,4,5} after `T`; dump/load round-trip.
2. **Masking (offline, mock or real tokenizer):** exactly one 0→1 mask transition;
   masked region ends at `<think>\n`; `\boxed{answer}` lands in the trainable region;
   the prompt does not. (Real-tokenizer check runs on Kaggle.)
3. **End-to-end (Kaggle):** start from the 86% adapter; compare LB. Re-run with a
   changed learning knob but the **same** `SCHEDULE_PATH` to confirm the order pins.
