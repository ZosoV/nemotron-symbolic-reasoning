# Nemotron: teaching symbolic reasoning

**A research exploration of curriculum SFT, backward-generated reasoning trajectories, and a proposed phase of reinforcement learning with verifiable process rewards.**

[Interactive project page](https://zosov.github.io/portfolio/nemotron/) · [Competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) · [Saved experiment evidence](docs/EXPERIMENT_EVIDENCE.md)

## The idea

Symbolic equation puzzles were a weak spot in our NVIDIA Nemotron competition work. Direct reinforcement learning was slow, and final-answer feedback gave little guidance when most attempts failed. The intended approach was:

1. **Find feasible programs.** Search for symbol-to-digit assignments, operator functions and encoding conventions that explain the equations. Solutions need not be unique.
2. **Construct short teaching trajectories backward.** Select a feasible solution, perturb its mapping, then render the reverse sequence as attempts, verifiable failures and suggested repairs. This is constructed supervision, not a transcript of the brute-force search.
3. **Warm up with curriculum SFT.** Progress from one shown equation plus a query to three–five shown equations. Mix earlier examples and replay other puzzle categories to limit forgetting.
4. **Refine with RL (proposed).** Parse structured tags, independently verify arithmetic and mapping claims, and reward useful progress as well as the final answer.

The contribution is the experimental design and the problems it exposes. **This is not a successful reasoning model.** The best saved symbolic diagnostic was **29/823 (3.52%)**, on public training puzzles, not a held-out benchmark. Those evaluations predate the last generator changes. No improvement from the proposed trace-aware RL stage was demonstrated.

## What is here

| Directory | Contents |
|---|---|
| `src/solver/`, `rust/alice_sovler_helper/` | Python orchestration and Rust symbolic search |
| `src/cot_generator_v2/` | Latest synthetic trajectory generator |
| `src/cot_generator/` | Earlier generator baseline |
| `src/data_generation*`, `src/post_processing/` | Example buckets, synthetic puzzles and data preparation |
| `notebooks/training_eval/` | Curriculum SFT, flat-shuffle comparison, generic GRPO and evaluation notebooks |
| `src/web_ui_inspector/` | Local React/Vite inspection tool |
| `specs/`, `memory/` | Original design documents, iterations and known issues |
| `tests/` | Original research checks; several require local datasets |
| `docs/` | Publication provenance and aggregate experiment evidence |

Generic GRPO scaffolding exists. Its answer/length and format rewards are distinct from the **proposed semantic trace rewards** described on the project page. Tags alone do not validate reasoning: an independent checker must recompute each claim.

## Using the research snapshot

This is a curated copy of the latest local research tree, including previously uncommitted work. The algorithm implementation has not been repaired or retrained for publication. Some historical specifications and notebook paths reflect earlier iterations; consult the actual modules and configuration together.

The Python project targets Python 3.12+ and uses `uv`, with a Rust/PyO3 extension built by `maturin`. Dependencies are recorded in `pyproject.toml` and `uv.lock`; the inspector has its own package manifest. Training additionally requires the notebook-specific GPU stack, model access and input corpus. **This is a research archive, not a one-command reproducible training release.**

Competition data, generated corpora, adapters, credentials and local environments are excluded. Obtain competition inputs from Kaggle under its access conditions and adapt the input paths. Notebook source is retained; saved outputs and execution metadata were cleared for publication. Aggregate results and their source hashes are in [`docs/results.json`](docs/results.json).

## Provenance and acknowledgement

Oscar Guarnizo's symbolic-equation work, developed during March–June 2026 for the NVIDIA Nemotron Model Reasoning Challenge. Extracted from the `oscar_analysis` subdirectory of the local `experience_extractor` project on 25 October 2026. The separate collaborator analysis directory is not included.

The solver work builds on the public [equation-symbolic solver by lkevincc0](https://github.com/lkevincc0/kaggle-nemotron-equation-symbolic). Training notebooks build on competition/community pipelines; existing references are preserved. The curriculum schedule adapts [Platanios et al. (2019)](https://aclanthology.org/N19-1119/). See [provenance](docs/PROVENANCE.md). The existing license notice is retained; it does not replace the terms of external models, datasets or third-party source.
