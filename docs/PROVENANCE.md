# Snapshot provenance

- Published from Oscar Guarnizo's local `experience_extractor/oscar_analysis` working tree on 2026-10-25.
- Source repository HEAD: `747d9e1`; local edits, staged moves and untracked generator-v2 files are included. That commit alone does not reconstruct this snapshot.
- Research implementation copied without modification. Notebook cell source preserved, outputs/execution metadata removed. The existing source license is retained.
- Included: Python/Rust source, UI source/manifests, configurations, dependency locks, original specifications, research checks and notebook source.
- Excluded: raw competition data, generated parquet corpora, trained models, notebook output, caches, credentials, local agent settings, build artifacts and the collaborator's separate analysis directory.
- Documentation added for publication: this README, provenance, aggregate experiment evidence, results JSON and source checksums. These explain the archive and do not change the research methods.
- Upstream reference: https://github.com/lkevincc0/kaggle-nemotron-equation-symbolic. The original May solver checkout points to that repository. This archive credits that starting point and does not claim all solver or community training code originated here.
- Existing design specifications can be historical or inconsistent with later code. `src/cot_generator_v2/` is the latest trace implementation; richer semantic RL rewards are a proposal.

`source-manifest.json` records SHA-256 hashes of copied research files. Notebook hashes refer to the published output-cleared copies; their cell sources are preserved.
