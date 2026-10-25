# Specs Index

Authoritative implementation specs. Check this index first; open a file only when relevant.

## Solver

- [solver_eq_symbolic.md](solver_eq_symbolic.md) — Full solver spec: Python driver + Rust DFS search core, concat fast-path, arithmetic search tiers, heuristics, CoT rendering.
- [solver-results-schema.md](solver-results-schema.md) — Every column in output `*.parquet` files produced by `solver_eq_symbolic.py`.
- [operation-feasibility-constraints.md](operation-feasibility-constraints.md) — Structural pre-filter that discards impossible ops before any digit value is known.
- [operations_spec.md](operations_spec.md) — Raw list of all 22 arithmetic operations (no code coupling; reference only).

## Dataset & Post-processing

- [data_generation_v2.md](data_generation_v2.md) — v2 dataset: "other variant" random-drop base pools + gap-fill augmentation (Strategy A recombine n=1,2; Strategy B backward-encoded synthetic n=3,4,5).
- [synthetic-generation-fidelity.md](synthetic-generation-fidelity.md) — Every statistic profiled from the real 803-row set and every rule enforced or naturally emergent in the v3 synthetic generator (operator count, operand magnitude, sign, zero-pad, pure_concat structure, dedup).
- [stage0-rebalancing-spec.md](stage0-rebalancing-spec.md) — Stage-0 op-frequency rebalancing: caps over-represented ops, synthesizes concat rows.
- [digit_mapping.md](digit_mapping.md) — Example digit-mapping priority analysis (illustrative, not normative).

## Chain-of-Thought (CoT)

- [cot-generator.md](cot-generator.md) — Authoritative spec for `src/cot_generator/`: programmatic 1–5-example CoT generator.
- [cot-generator-fixes.md](cot-generator-fixes.md) — Fixes and augmentations implemented in `src/cot_generator_v2/` (v2 is the active generator; v1 kept as baseline).
- [cot-template_0_examples.md](cot-template_0_examples.md) — CoT column template, zero-demonstration (prediction) variant.
- [cot-template_0_examples_analysis.md](cot-template_0_examples_analysis.md) — CoT template, analysis/candidate-enumeration variant (no `solution:` line).
- [cot-template_1_5_examples.md](cot-template_1_5_examples.md) — CoT template with worked failed-attempt chain (1–5 examples, backward-generated).
- [cot_template_1_5_examples_placeholders.md](cot_template_1_5_examples_placeholders.md) — Placeholder variant of the 1–5-example template.
- [cot-basic_knowledge.md](cot-basic_knowledge.md) — Background knowledge reference used when generating CoT (operation definitions).
- [search_cot.md](search_cot.md) — Example search-trace CoT output (illustrative, not normative).
- [search-trace-schema.md](search-trace-schema.md) — Schema for the per-puzzle full-trace JSON files generated when `generate_search_trace: true`; documents the node event format and invariants.

## Training

- [curriculum_learning_proposal.md](curriculum_learning_proposal.md) — n-shot curriculum SFT design for Nemotron LoRA.
- [curriculum_training_pipeline.md](curriculum_training_pipeline.md) — End-to-end curriculum LoRA training pipeline from parquet.
- [nemotron-finetuning-pipeline.md](nemotron-finetuning-pipeline.md) — Full LoRA finetuning pipeline (LB-0.86 submission).
- [competence_based_curriculum_learning.md](competence_based_curriculum_learning.md) — Competence-based curriculum learning strategy adapted from Platanios et al.

## UI

- [web_ui_inspector.md](web_ui_inspector.md) — Vite + React inspector for browsing dataset and results parquet files.

## Scratchpad *(non-authoritative)*

- [ideas.md](ideas.md) — Exploratory ideas only. Do not execute anything from here.

---

## Spec ↔ Code Relationship Table

| Spec file | Related code |
|---|---|
| `solver_eq_symbolic.md` | `rust/alice_sovler_helper/src/lib.rs` · `src/solver/solver_eq_symbolic.py` · `src/operations.py` · `src/op_index.py` · `src/solver/heuristics.py` · `src/encoding.py` · `src/parsing.py` · `config.yaml` · `src/solver/config.py` |
| `solver-results-schema.md` | `src/solver/solver_eq_symbolic.py` (producer) · `src/eval_from_parquet.py` (consumer) · `src/post_processing/extend_ops_mappings.py` |
| `search-trace-schema.md` | `rust/alice_sovler_helper/src/lib.rs` (`search_full_trace`, `arithmetic_search_full_trace`) · `src/solver/solver_eq_symbolic.py` (`_search_full_trace`, `_build_search_trace_json`, `_spine_from_full_trace`) · `src/web_ui_inspector/src/components/SearchTraceView.tsx` |
| `data_generation_v2.md` | `src/data_generation_v2/` (schema, stats, base, recombine, synth, emit, run_v2, verify_coverage) |
| `synthetic-generation-fidelity.md` | `src/data_generation_v2/{stats,synth,signature}.py` · `src/data_generation_v3/run_v3.py` |
| `stage0-rebalancing-spec.md` | `src/post_processing/rebalance_stage0.py` · `src/post_processing/extend_ops_mappings.py` |
| `operation-feasibility-constraints.md` | `src/solver/heuristic_constraints/operation_feasibility.py` · `src/parsing.py` · `src/operations.py` · `src/op_index.py` |
| `cot-generator.md` | `src/cot_generator/` (v1 baseline, read-only) · `src/cot_generator_v2/` (active; fixes 1–9, Aug 1/3/4) · `src/data_generation_v3/run_v3.py` (synthetic Aug 3) |
| `cot-template_0_examples.md` | `notebooks/cot_generation/cot_dataset_0.ipynb` · `src/solver/heuristics.py` · `src/operations.py` · `src/encoding.py` |
| `cot-template_0_examples_analysis.md` | `notebooks/cot_generation/cot_dataset_0_analysis.ipynb` · `src/operations.py` · `src/encoding.py` |
| `cot-template_1_5_examples.md` | `notebooks/cot_generation/*` (1–5 example variant) |
| `curriculum_learning_proposal.md` | `notebooks/training_eval/prepare_kaggle_corpus.ipynb` · `notebooks/training_eval/end-to-end-finetuning-*.ipynb` |
| `curriculum_training_pipeline.md` | `notebooks/training_eval/end-to-end-curriculum_learning.ipynb` |
| `nemotron-finetuning-pipeline.md` | `notebooks/training_eval/end-to-end-finetuning-*.ipynb` · `notebooks/training_eval/evaluate_lora_adapter.ipynb` · `notebooks/training_eval/nvidia-nemotron-metric.ipynb` |
| `web_ui_inspector.md` | `src/web_ui_inspector/` |
