# Publication evidence

This is an aggregate-only audit of the latest local `experience_extractor/oscar_analysis` snapshot. Raw puzzles, model outputs, answer strings, IDs, model weights and private credentials are excluded.

## Saved model results

The six result files are enumerated in `results.json`, including source SHA-256, filesystem modification time, symbolic numerator/denominator, boxed-answer count, and methodology. Every saved file has 829 rows: 823 symbolic equations and one example from each other category.

| Saved experiment label (filename shorthand) | Symbolic correct | Accuracy | Boxed answer present |
|---|---:|---:|---:|
| Zero-example / unmasked basic CoT | 3 / 823 | 0.36% | 691 / 823 |
| Zero-example / masked analysis CoT | 3 / 823 | 0.36% | 479 / 823 |
| Zero-example / masked basic CoT | 7 / 823 | 0.85% | 464 / 823 |
| Merged (unqualified filename) | 5 / 823 | 0.61% | 619 / 823 |
| Merged / mask-prefix / 250 steps | 1 / 823 | 0.12% | 179 / 823 |
| Merged / mask-prefix-initial / 400 steps | 29 / 823 | 3.52% | 81 / 823 |

All 823 symbolic IDs match the public `inputs/data/v2/train.csv` (9,500 rows). These are diagnostics on public training puzzles, not a held-out benchmark. The best saved diagnostic is 29/823. The 250-step file has a later filesystem modification date than the 400-step file, so do not call the best run the latest run. All results predate the June 14–15 CoT generator-v2 changes; no final model evaluation of those changes was found. The experiment labels are filenames, not immutable training manifests. No controlled curriculum-versus-flat comparison can be claimed from them.

`notebooks/training_eval/evaluate_lora_adapter.ipynb` describes greedy vLLM adapter evaluation (temperature 0, max model length 8,192, generation limit 7,680). It extracts boxed answers, then uses fallback patterns and its `verify` helper. Numeric answers have tolerance; symbolic strings are compared case-insensitively after stripping. Saved `correct` values are used rather than re-evaluating model text with today's possibly changed parser.

The best saved run emits a boxed answer in only 81/823 cases, also the count with a closing `</think>`. This supports discussing answer-completion/format problems. It does not prove token-cap truncation as the sole cause: these artifacts have no finish-reason or exact token-count column.

The ~0.86 competition baseline appears in `specs/nemotron-finetuning-pipeline.md` (intro) and `specs/curriculum_training_pipeline.md:20`. It is a documented starting adapter, not the result of the symbolic curriculum. No new leaderboard improvement is evidenced by these saved result files.

## Implemented curriculum

`specs/curriculum_training_pipeline.md:69` documents the schedule implemented in `notebooks/training_eval/end-to-end-curriculum_learning.ipynb`. Current notebook settings: 250 optimizer steps, batch size 32, microbatch size 4, replay ratio 0.25, c0=0.01, T=0.70*250=175, square-root competence. The adapter starts from existing pretrained weights (`RESET_WEIGHTS=False`).

Difficulty means **number of shown example equations**, n=1..5; the query is additional. This can reconcile the user's recollection of starting at two equations: one example + one query. More shown equations demand more explicit reconciliation and longer explanations, but also provide more constraints and can reduce ambiguity. Avoid asserting that fewer constraints always make mathematical inference easier.

Unlock buckets by empirical CDF threshold <= c(t), with bucket 1 always available. Before T, sample equally across unlocked buckets; after T, focus symbolic slots on buckets 3,4,5. Each batch retains 8/32 non-symbolic replay rows (24 symbolic slots) throughout. Earlier symbolic buckets remain mixed while unlocking, but 1/2 are deliberately removed in the final phase to match the evaluation distribution. Seeded shuffled cycling makes schedules reproducible.

The current curriculum and flat-shuffle notebooks have no execution outputs. Notebook implementations are evidence of code, not by themselves proof of a successful training run or final score.

## RL status

Generic GRPO is implemented in `notebooks/training_eval/end-to-end-grpo.ipynb` and `notebooks/training_eval/grpo_training_vllm.py`. The latter's reward registry at lines 590–615 contains only:

- `cosine_reward`, weight 1.0: correct boxed answers earn more if short; wrong/absent answers receive length-sensitive penalties.
- `think_format_reward`, weight 0.15: checks the `<think>...</think>` format.

Group-relative advantage normalization is at `grpo_training_vllm.py:870`. There is no implemented semantic reward parser for equation checks, candidate eliminations, mapping consistency, repair steps or the generator's structured tags. Publish that richer RL phase as **proposed future work**, while acknowledging existing generic RL scaffolding.

The earlier `/home/zosov/workspace/nemotron_rl_approach/logs/train.log` records a generation call taking 1,909.19968 seconds at global_step 0, then a Torch Dynamo tensor-shape error. `reported_issues/grpo_issue.md` describes generation/cache integration difficulties. This supports a factual statement that early RL was slow and technically difficult. Sparse semantic feedback is the user's research motivation, not a quantified experiment result in that log.

## Solver and synthetic trajectories

`src/solver/solver_eq_symbolic.py` + `rust/alice_sovler_helper/src/lib.rs` implement symbolic program search. `data/solver_results.parquet` contains 804/823 marked correct (97.69%); this is solver coverage, not LLM accuracy. Solver/oracle use of known answers and differing search conditions must be explained before presenting this number.

`src/cot_generator_v2/` is the active generator; `specs/cot-generator.md` explains choosing a feasible target program, creating backward perturbations, then narrating forward failed attempts and repairs. `specs/cot-generator-fixes.md` records constraint propagation, visibility/leakage and answer-format issues. Generated trajectories are constructed teaching explanations and should not be described as faithful transcripts of the solver search or evidence of an LLM's internal reasoning.

The active config `config_eval_cot.yaml` specifies three perturbation rounds, token cap 6,500, up to three order variants and guess-sample companions. The number of realized perturbations can be lower because of structural eligibility and token caps. Historic `memory/cot_token_budget.md` recommends 6,000; label a cap as configuration-dependent rather than claiming one universal number.
