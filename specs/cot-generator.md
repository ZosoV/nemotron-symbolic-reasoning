# CoT Generator (`src/cot_generator/`)

Authoritative spec for the programmatic 1–5-example chain-of-thought generator.
The golden design reference is [`specs/cot-template_1_5_examples.md`](cot-template_1_5_examples.md)
(a hand-authored example); this document describes the code that reproduces that
structure automatically from a solver-results `.parquet` row that already
contains a feasible solution.

## Goal

Given one feasible row (a known digit mapping + op mapping + encoding strategy),
emit a tagged CoT that *looks like* a solver gradually homing in on the answer:
discard infeasible operations per operator, score symbols by constraint, then
walk a chain of digit mappings where each attempt fixes one perturbation until
the known solution is reached — ending with the labelled answer.

The generator is **self-contained**: it re-implements the parsing, decoding,
feasibility and heuristic logic it needs rather than importing the solver's
modules.

## Pipeline & modules

| Module | Responsibility |
|---|---|
| `families.py` | Single source of truth for op **families** (offset *parameters*, not per-offset registry instances). Family table, `apply_family`, magnitude bounds, base-digit helpers, registry-name→(family,offset) resolution, reproducible seed recipe. |
| `equations.py` | Self-contained equation parsing (`Equation`, `parse_prompt`, `content_symbols`, `operator_groups`, `occurrences`). |
| `decode.py` | Operand decode + result checking for `standard`/`little_endian`; `family_matches` (concat handled structurally), `predict_answer`, `verify_program`, `detect_pure_concat` (re-derives the corrected evidence rule — see §Correctness anchors). |
| `feasibility.py` | Family-level structural filter (relocated + rewritten from `src/solver/heuristic_constraints/operation_feasibility.py`). Sign / concat / magnitude-lower-bound / **digit-length upper-bound** / self-application checks; folds in the solver's sign-pattern (#2) and concat opt-in (#4) heuristics. The digit-length check is the bijection-sound *upper* half of the solver's magnitude pre-prune (#6) — it drops gcd/fdiv/mod/rmod when a wide rhs would need leading zeros it cannot supply; self-application now also wires `fdiv→1` (`ONE_ON_EQUAL`) and `gcd,lcm→x`. For all-unsigned groups drops `neg_absdiff`, `sub_signed`, and `rsub_signed` together (unless in the `protect` frozenset). Emits `reason -> conclusion` one-liner observations. |
| `factual_observations.py` | **Script #1**: per-operator orchestration over `feasibility.py`; question operator first; keeps the question operator even when unconstrained. Accepts a `protect` dict `{op_glyph: family}` to shield gold-op families from the sign heuristic drop. |
| `redundancy_observations.py` | **Script #2**: identical-output collapse on all-equal-operand groups (`gcd≡lcm`, `mod≡rmod` — keep one), with each gold op's family protected. Note: `sub_signed`/`rsub_signed` elimination for unsigned rhs was moved to `feasibility.py`'s sign check (Script #1), not here. |
| `digit_priority.py` | *(unused in v2 arithmetic path)* Two scorers — weighted most-constrained and plain occurrences. No longer called from `render_cot`; symbol ordering now uses `order_by_allowed` from `digit_constraint.py`. |
| `digit_constraint.py` | Constraint-propagation digit picking. Public API: `build_allowed_table`, `propagate`, `order_by_allowed` (fewest-options-first sort), `picking_sequence` (fixed order, `pool_forced` case), `render_first_mapping_block` (full `<digit_mapping_1>` content), `reduce_allowed_from_followup` (progressive narrowing for chain blocks). |
| `perturbation.py` | Backward chain from the solution: swap / reassign perturbations (disjoint symbol sets), incremental (commutativity-safe) construction, records each fix and the unused-digit candidates per mapping. |
| `analysis.py` | Renders `<digit_mapping_analysis_N>`: per-equation operand/target decode then per-candidate computation with **progressive narrowing** across the operator's equations (candidates, offsets and encoding modes are eliminated equation-by-equation, the surviving offset carried forward — Option-A, template-faithful); keeps the "unchanged from previous attempt" shortcut; `solution_program` anchors to the solver's program. |
| `tells.py` | Renders `<digit_mapping_analysis_follow_up_N>`: algebraic operand-gap / result-position tells when an isolating equation exists, simple fallback otherwise (or when `tell_mode=simple`). When a fix lies wholly inside one operand, the operand-gap tell **back-solves** that operand (the other operand and result are pinned by the passing equations) and names the exact digit each symbol must take, instead of merely asserting the swap. |
| `render.py` | Assembles all tags into the full CoT; `render_cot(row, cfg) -> CotResult`. Also `render_concat_cot(examples, q, fam) -> CotResult` for the concise pure-concat variant (three tags: `<initial_reasoning>`, `<concat_evidence>`, `<conclusion>`). |
| `config.py` | `CotGeneratorConfig` + `load_cot_config` (reads the `cot_generator:` block of `config.yaml`). |
| `generate.py` | CLI: read parquet → render feasible rows → self-check answer == `solver_answer` → write the `cot_1_5` column plus a `cot_approx_tokens` size column (`len(cot)/2.81`, rounded; `None` on skipped rows) and a `cot_num_perturbations` column (realised failed-attempt count per row). When `max_cot_tokens` is set, `_render_capped` re-renders the arithmetic CoT with one fewer perturbation than realised until it fits the cap or none remain. Rows without a digit solution fall through to `_try_concat_variant` (which calls `detect_pure_concat` + `render_concat_cot`) before being skipped; `concat_generated` stat tracks how many were handled this way. |

## Op families

`add` (offsets 0,+1,−1,+2) · `mul` (0,−1,+1) · `absdiff` (0,+1,−1,−2) ·
`sub_signed` · `rsub_signed` · `neg_absdiff` · `gcd` · `lcm` · `fdiv` · `mod` ·
`rmod` · `concat_fwd` · `concat_rev`. The registry op `a2_plus_b` has no family;
rows whose solution uses it are skipped.

## Correctness anchors (important design decisions)

- **Anchor to the solver's program.** Puzzles are underdetermined — several
  feasible ops may satisfy the examples but disagree on the question answer. The
  conclusion therefore uses the solver's own `(family, offset)` per operator
  (its `ops_mapping`), and `generate.py` self-checks that the rendered answer
  equals the row's `solver_answer`.
- **Leading-zero results are allowed, but the bijection makes an upper bound
  sound.** A result may carry leading zeros (e.g. value `71` filling a 4-symbol
  rhs as `0071`), so a *naive* upper bound on result digit count is **not** a
  sound elimination on its own. However, digits are injective (distinct symbols →
  distinct digits, `base = #content_symbols`), so the digit `0` has exactly **one**
  symbol: any required leading-zero positions must all be that single repeated
  symbol. The `feasibility.py` **digit-length check** uses this — a family whose
  result has ≤ D significant digits is infeasible for an L-symbol rhs (L > D) when
  *neither* encoding can supply an equal-symbol leading run (standard `rhs[:k]`,
  little_endian `rhs[L-k:]`, `k = L-D`). This is the bijection-sound *upper* half
  of the solver's magnitude pre-prune (heuristic #6, `lib.rs`); the magnitude
  *lower* bound is its lower half. It drops gcd/fdiv/mod/rmod on wide rhs (lcm,
  `D=4`, is never pruned this way). The template's hand-authored "add can't make a
  4-digit rhs" remains unsound (add *can*, via leading zeros) and is **not** used.
- **Concatenation is structural.** Concat result symbols equal the operand
  symbols in written order (mode-independent); checked by symbol identity
  (`decode.concat_pattern`), not through the integer/mode machinery.
  `feasibility.py` applies this **per direction across the operator's whole
  equation group**: the direction whose pattern matches the rhs (fwd `l0 l1 r0 r1`
  or rev `r0 r1 l0 l1`) is kept and the other dropped; both survive only when the
  operands are identical, so the two patterns coincide and the orientation is
  ambiguous. (Earlier code kept both on a count-only "rhs has 4 symbols" check —
  the bug this fixes.)
- **Pure-concat rows get a concise CoT, not a skip.** Rows whose solver fast-path
  found a concat solution (`digits_mapping = None`) formerly produced no CoT. They
  are now handled by `render_concat_cot` (three tags only: `<initial_reasoning>`,
  `<concat_evidence>`, `<conclusion>`). The `solver_strategy == "pure_concat"` label
  is **not trusted** — `detect_pure_concat` (`decode.py`) re-derives the corrected
  evidence rule directly from the equations: (a) question rhs unsigned + 4 symbols,
  (b) ≥ 1 *example* equation shares the question operator, (c) all such examples
  unanimously follow one direction. Rows that pass the old (buggy) label but fail the
  corrected rule remain skipped. The `concat_generated` stat in `generate.py` tracks
  how many rows were handled this way.
- **Self-application (L≡R) is offset-aware and constant-aware.** `absdiff`
  collapses to `0` only at offset 0; `absdiff_p1` (→ 1) survives `L≡R`. The other
  forced constants are now wired too: `mod,rmod → 0` (zero-on-equal), `fdiv → 1`
  (`ONE_ON_EQUAL`), and `gcd,lcm → x` (the operand value). Each drops when the rhs
  cannot encode its forced value in *either* encoding under the bijection.
- **Unconstrained question operator.** When the question operator appears in no
  example equation it cannot be narrowed; the block says so and the conclusion
  adopts the solver's choice as one valid option.

## Config knobs (`cot_generator:` in `config.yaml`)

`input_parquet`, `output_parquet`, `cot_column` (default `cot_1_5`),
`digit_priority_mode` (kept in config schema; **ignored** by v2 arithmetic path —
symbol ordering uses `order_by_allowed` instead),
`perturbation_rounds` (max; capped by puzzle structure), `tell_mode`
(`algebraic|simple`), `random_seed`, `limit`, `max_cot_tokens` (cap on
`cot_approx_tokens`; re-renders the CoT one perturbation shorter until it fits
or none remain — `null` = no cap).

## Output tags

**Arithmetic CoT (1–5-example variant):**
`<initial_reasoning>` · `<available_operations>` ·
`<operator_N_candidates>` + `<factual_observations_operator_N>` ·
`<digit_mapping_N>` / `<digit_mapping_analysis_N>` /
`<digit_mapping_analysis_follow_up_N>` · `<conclusion>`.
(`<digit_priority>` removed — see Fix 3 in `cot-generator-fixes.md`)

**Pure-concat CoT (concise variant):**
`<initial_reasoning>` · `<concat_evidence>` · `<conclusion>`.

## Notation (token-saving)

`<initial_reasoning>` defines: `eq[N]` for the N-th example equation; `std` / `le`
for the `standard` / `little_endian` encoding strategies.


## Verification

- Per-module self-tests: `uv run python src/cot_generator/<module>.py`
  (and `feasibility.py --smoke`).
- End-to-end: `uv run python src/cot_generator/generate.py --config config.yaml`
  — every generated row self-checks against `solver_answer`; the run reports
  `generated / skipped / selfcheck_fail / render_error / concat_generated` counts.

## Relation to `operation_feasibility.py`

`feasibility.py` is a relocated, family-level rewrite of
`src/solver/heuristic_constraints/operation_feasibility.py` (which is not imported
anywhere else). The original module and its spec
(`specs/operation-feasibility-constraints.md`) remain in the tree; this spec
supersedes that description for the CoT generator's purposes.
