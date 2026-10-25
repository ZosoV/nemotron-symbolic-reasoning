# Equation-Symbolic Solver — Build Specification

This is the authoritative spec for the solver. It is written so you can reimplement it from
scratch (Python driver + Rust search core) without reading the existing code. File:line
citations point at the reference implementation:
`rust/alice_sovler_helper/src/lib.rs`.

---

## 1. The Problem

Each puzzle gives a few demonstration equations built from arbitrary Unicode symbols, then
asks for the result of a query equation under the same hidden rule. Example:

```
`!*[{ = '"[`
\'*'> = ![@
\'-!` = \\
`!*\& = '@'{
Determine the result for: [[-!'        →  answer: @&
```

There is no stated rule. The solver must infer a **latent program** —
`(symbol→digit mapping, operator→function mapping, interpretation mode)` — that explains every
demonstration simultaneously, then apply it to the query.

There are two puzzle families:

- **Type 1 — Concatenation:** the operator glues its two operands (forward or reversed).
  Detected structurally; no search.
- **Type 2 — Arithmetic:** each unique symbol maps bijectively to a digit; each operator maps
  to one of arithmetic functions defined on `operations.py`. Solved by brute-force CSP search.

--- 

## 2. Execution Flow

### 2.1 Construction & Analysis

On construction:

- Read a `.parquet` with columns `prompt` and `answer`
- Extract all the equations on the prompt, including the one needs to be predict appending the answer.
  ```
  def extract_all_equations(row):
    equations = []
    for line in row['prompt'].split('\n'):
        if line.startswith('Now'):
            question_eq = line.split(': ', 1)[1].strip()
            equations.append(f"{question_eq} = {row['answer']}")
        if ' = ' in line:
            equations.append(line.strip())
    return equations
  ```
- Given an equation l0 l1 op r0 r1 = rhs, parser each equation to the following format:
  ```
  {
    type: example | to_predict (the last equation that is built with the answer column is of type to_predict)
    lhs : {
      l0: symbol
      l1: symbol
      op: operator symbol
      r0: symbol
      r1: symbol
    }
    rhs : {
      op: optional operator symbol in the first position of rhs
      actual_rhs: the rhs without the operator symbol if any. It is a list that can contain several values.
    }
  }
  ```
  Each sample will contain a list of structured equations

 - available operators are choosen from OPERATIONS in `operations.py`

### 2.2 Concat Fast-Path — `_try_concat()`

For each sample, take the last equation corresponding to equation that is expected to be predicted.

1. Take the op symbol of the lhs on the last equation. This is our op_of_interest.
2. If there is no op in rhs and rhs contains exactly 4 symbols:
  - Collect all **example** equations (i.e. all equations except `to_predict`) whose op equals `op_of_interest`. If none exist, **abort and fall through to arithmetic search** — `to_predict` alone is not sufficient evidence.
  - For every collected example equation:
    - Check if the solution corresponds to a simple forward concatenation (`l0l1r0r1`) or reverse concatenation (`r0r1l0l1`) between the symbols. If any equation fails both checks, abort early.
  - If all examples pass the same check, the problem is solved with concatenation; record the solution.

**Evidence requirement:** at least one example equation must share `op_of_interest` for the fast-path to trigger. Dataset analysis (824 problems) found 16 cases where `to_predict`'s operator never appears in examples; those are now routed to arithmetic search. Of the remaining 60 concat problems, all show perfectly consistent direction (0 wrong inferences).

### 2.4 Arithmetic Search — `_try_arithmetic()`


**Step A — Build operator groups**

For every equation:
- Extract `op`.
- Detect `has_sign` if op in rhs.
- Map every character to an integer index via `sym2i = {s: i for i, s in enumerate(content_symbols)}`.
- Append to `op_eqs[op]` the tuple:
  ```
  (sym2i[l0], sym2i[l1], sym2i[r0], sym2i[r1],
   tuple(sym2i[c] for c in actual_rhs), has_sign, len(actual_rhs))
  ```

Result: `op_eqs` is `{op_char: [equation_tuples]}` — one group per distinct example operator.

**Step B — Per-op candidate lists**

For each `op` in `op_eqs`:

1. Start from `_priority(op)`: use the glyph list from `OP_PRIORITY` (Heuristic #3)
   if `op ∈ {* + - /}`, else `DEFAULT_PRIORITY`.

2. Apply the sign-pattern filter (Heuristic #2):
   | Sign pattern | Candidates kept |
   |---|---|
   | All signed (every `rhs` starts with `op`) | keep `neg_absdiff, sub_signed, rsub_signed` only |
   | All unsigned | every op except `neg_absdiff` |
   | Mixed | `sub_signed, rsub_signed` only |

3. Append `concat_fwd / concat_rev` if every example for that op is unsigned **and**
   `res_len == 4` and they are in `op_types` (Heuristic #4).

---

**Step C — Op ordering**

```python
sorted_ops = sorted(op_eqs.keys(), key=lambda op: (len(op_candidates[op]), -len(op_eqs[op])))
```

Fewest candidates first, ties broken by most equations. This is the order in which groups are
handed to Rust (Heuristic #5).

---

**Step D — Tiers**

Four op-sets, each a superset of the previous:

| Tier | Ops (new additions in **bold**) |
|------|--------------------------------|
| TIER0 | `mul, add, absdiff, sub_signed, rsub_signed, neg_absdiff, gcd, lcm, concat_fwd, concat_rev` |
| TIER1 | TIER0 + **`fdiv, mod, rmod`** |
| TIER2 | TIER1 + **`add_m1, add_p1, mul_m1, mul_p1, absdiff_m1, absdiff_p1`** |
| FULL | all 22 ops in `OPERATIONS` (adds `add_p2, absdiff_m2, a2_plus_b`) |

These tiers mirror `TIER0/TIER1/TIER2/TIER_FULL` defined in `src/operations.py`.
`operations.py` is the authoritative source; Rust indices supported beyond these 22 ops
(e.g. `sub, rsub, min, max, rdiv`) are unused by the Python driver.

Start with TIER0 and add progressively if the solution is not found with that TIER. It needs to backtrack if the solution is not found, and perform again the whole search in the extended TIER.

---

**Step E — The search loops**

Nesting, outermost → innermost:

```
for tier in tiers:
    tier_cands = {op: [c for c in cands if c in tier] for op, cands in op_candidates.items()}
    if any group is empty: continue              # Heuristic #9

    radix_candidates = [(base, base)]
    if base != 10:
        radix_candidates.append((10, 10))        # see Section 4

    for (radix, digit_count) in radix_candidates:
        for mode in ("standard", "little_endian"):
            sol = _search(op_eqs, tier_cands, sorted_ops, n=base, mode, radix, digit_count)
            if sol is not None:
                → Step F
```

Failure backtracks **mode → radix → tier** (not radix → mode → tier).

---

**Step F — Encode the query answer** (`_encode_answer`, `:1374-1435`)

Given `(mapping, ops, mode, radix)`:

1. Decode query operands `L = mapping[ql0]*radix + mapping[ql1]`,
   `R = mapping[qr0]*radix + mapping[qr1]`. If reversed-digit mode (`little_endian`), swap each operand's two digits: `X → (X % radix)*radix + X // radix`. In little_endian mode, the puzzle's encoding is reversed — the first symbol is actually the units digit and the second is the tens digit. So AB really means BA as a number. The swap corrects for that:

  L = (L % base) * base + L // base
  #    ^^^^^^^^          ^^^^^^^^^
  #    units digit  →    becomes tens    (i.e., the original "second" digit)
  #    tens digit   →    becomes units   (i.e., the original "first" digit)

  Concrete example with base=10, A=3, B=7 → L=37:
  - L % 10 = 7 (units) → becomes tens: 7 * 10 = 70
  - L // 10 = 3 (tens) → becomes units: 3
  - Result: 73
2. If query op is `concat_fwd/rev`, emit the operand symbols directly (no arithmetic).
3. Else apply `OPERATIONS[op](L, R)`. Handle sign for `sub_signed / rsub_signed /
   neg_absdiff`. Reject if result is `None` or negative for unsigned ops.
4. Convert magnitude to base-`radix` digits via `_int_to_base_digits`. 
  - If `answer_hint` is set, left-pad with zeros to match the gold answer's digit length. If reversed-digit mode,
   reverse the digit list (Heuristic #8)
5. Map each digit back to a symbol via the inverse mapping. Reject if any digit has no symbol.

Return `(answer_string, numeric_value)`.

The `details` dict in the successful return carries:
- solver_strategy: `pure_concat`, `arithmetic` or `little_endian`
- ops_mapping: mapping of the op symbol to the actual operation
- digits_mapping: mapping of the symbols to the digits
- solver_tier (str): tier0, tier1, tier2, full
- solver_radix
- ops_candidates: the candidates per operations after the search (it can potentially contain multiple candidates).
- equations: all the symbolic equation including the to_predict one.

---

## 3. Rust Search Core

Python's `_search` packages the call and invokes `arithmetic_search` in
`lib.rs:615`.

### Signature (`lib.rs:615-660`)

```rust
fn arithmetic_search(
    n: usize,                                   // number of content symbols (variables)
    alice: bool,                                // legacy; only used if `mode` is None
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,       // per-op (candidate op-idxs, equations)
    radix: Option<usize>,                       // positional base for arithmetic; default n
    digit_count: Option<usize>,                 // size of the digit pool; default n
    mode: Option<u8>,                           // 0=standard, 1=alice, 2=little_endian
    max_solutions: Option<usize>,
    randomize: Option<bool>,
    seed: Option<u64>,
) -> Vec<(Vec<u8>, Vec<Vec<u8>>)>
```

Each equation is encoded as `[l0, l1, r0, r1, res_len, has_sign, res_syms…]`.

### Algorithm (`search` / `dfs`, `lib.rs:319-452`)

1. Compute `var_order = variable_order(groups, n)` — symbol assignment order (Heuristic #1).
2. Initialise state: `perm[sym] = digit`, `used[0..digit_count]` (bijection enforcer),
   `assigned[sym]` (which symbols have been assigned so far).
3. DFS: at depth `d`, take `sym = var_order[d]`; try each digit `d in 0..digit_count` not
   yet in `used`; tentatively set `perm[sym] = d, used[d] = true, assigned[sym] = true`.
4. **Forward-check** (`group_has_candidate_for_assigned_eqs`, `lib.rs:301`): for every group,
   check equations whose symbols are *all* assigned (`eq_is_assigned`, `lib.rs:220`); a
   candidate survives if it matches every such equation; if a group has zero survivors →
   undo, try next digit (prune this branch).
5. If forward-check passes, recurse to depth `d+1`.
6. **Leaf** (`depth == n`, `lib.rs:369`): re-verify every candidate against every equation in
   its group; collect survivors; if every group has ≥1 survivor, push `(perm, ops_valid)` to solutions.

### Return value

`(perm, ops_valid)` — `perm[sym_idx] = digit`, `ops_valid[group_i] = [surviving_op_idx, …]`.
**Mode and radix are not returned**; the Python loop already holds them.

### Equation matching (`candidate_matches_eq`, `lib.rs:201-263`)

- Concat ops: matched structurally on symbol-index sequences, no arithmetic (Heuristic #7).
- Arithmetic ops:
  - Decode `L = perm[l0]*base + perm[l1]`, `R = perm[r0]*base + perm[r1]`.
  - If reversed-digit mode, swap each operand's two digits.
  - Apply `apply(cand, L, R)`.
  - Magnitude pre-prune: if `v >= base^res_len`, reject (Heuristic #6).
  - If reversed-digit mode, reverse the result digit list.
  - Compare to expected digits decoded from `res_syms`.

---

## 4. `digit_count` vs `radix`

**`digit_count` is the size of the digit *alphabet* the mapping draws from — not a way to
shrink the mapping below the number of symbols.**

In Rust, `used = vec![false; digit_count]` (`lib.rs:330`) and the assignment loop is
`for &d in digit_order` (`lib.rs:393`). So each symbol is assigned a **unique** value from the
pool `{0, 1, …, digit_count − 1}`. `used` enforces the injection (no digit reused).

`radix` is the **positional base** for arithmetic: `L = perm[l0]*base + perm[l1]`.
It governs the math, not which digits exist.

Both default to `n` (`lib.rs:654-655`).

| `digit_count` vs `n` | Meaning |
|---|---|
| `digit_count == n` | **Bijection** — every symbol gets a distinct digit and every digit is used. Normal case: `(base, base)`. |
| `digit_count > n` | **Injection** — symbols draw from a larger pool; some digits go unused. This is the `(10, 10)` fallback for puzzles with fewer than 10 symbols that are really base-10. |
| `digit_count < n` | **Impossible** — DFS runs out of unused digits, returns no solution. `digit_count ≥ n` must hold. |

### How `radix_candidates` is constructed

`base` = `len(content_symbols)` — the number of unique non-operator (neither =) characters across all
examples. Each symbol must map to a distinct digit, so `base` also equals `n` (the permutation width passed to search).

The candidate list is always:

```python
radix_candidates = [(base, base)]
if base != 10:
    radix_candidates.append((10, 10))
```

**`(base, base)` — the natural case.** `radix = digit_count = base`. The digit pool is
exactly `{0, 1, …, base−1}`, giving a bijection: every symbol gets a unique digit and every
digit is used. For a puzzle with 7 distinct symbols this is `(7, 7)` — base-7 arithmetic
over digits `{0,…,6}`.

**`(10, 10)` — the decimal fallback.** Added whenever `base ≠ 10`. The puzzle may have
fewer than 10 symbols but the hidden arithmetic is still decimal — e.g., 7 symbols whose
true digit values are `{2, 3, 5, 6, 7, 8, 9}`, skipping `0, 1, 4`. In base-7 that mapping
is unreachable (the pool only goes up to 6), but with `(10, 10)` the pool is `{0,…,9}` and
the 7 symbols are assigned 7-of-10 values (injection, not bijection). The search tries this
as a second attempt if `(base, base)` yields no solution.

---

## 5. Heuristics

Each heuristic reduces the search space or speeds up evaluation. The following is a list of heuristics applied to this search.

### 1. Symbol variable ordering (Rust)
- **Where:** `lib.rs:388-405`, function `variable_order`.
- **Rule:** `score[s] = Σ_groups Σ_eqs_in_group (1 + |candidates(group)|) × occurrences(s in eq)`.
  Sort symbols by descending score, ties by index. This order governs which symbol gets a
  digit assigned first in the DFS.
- **Example:** Group `+` has 4 candidates and 3 equations; group `*` has 18 candidates and 1
  equation. Symbol `A` appears 6× in `+` equations and 2× in the `*` equation:
  `score[A] = (1+4)·6 + (1+18)·2 = 30 + 38 = 68`. A symbol appearing only 6× in `+` scores
  30. `A` is assigned first.
- **Why:** Maximises expected pruning per committed digit. Assigning a symbol that appears in
  a group with 18 candidates can collapse those 18 to zero in one step (instant backtrack).
  This is the *inverse* of classic MRV — we want each assignment to have the largest
  downstream impact on the candidate count.

### 2. Sign-pattern op narrowing (Python)
- **Where:**
- **Rule:**
  | Sign pattern across all of an op's examples | Candidates kept |
  |---|---|
  | All signed (every `rhs` starts with the op char) | `neg_absdiff, sub_signed, rsub_signed` only |
  | All unsigned | everything except `neg_absdiff` |
  | Mixed (some signed, some not) | `sub_signed, rsub_signed` only |
- **Example:** Op `-` with examples `AB-CD = -EF` and `GH-IJ = -KL` (both signed). The
  all-signed branch fires and only three candidates survive — `add`, `mul`, `absdiff`, etc.
  are all dropped before the search begins.
- **Why:** A leading sign character is structural proof. Only signed-subtraction ops and
  `neg_absdiff` can produce a sign prefix. Mixed-sign examples require a *truly* signed op
  because `neg_absdiff` always signs.

### 3. Op-glyph priority ordering (Python)
- **Where:** `OP_PRIORITY`, `DEFAULT_PRIORITY`, applied in
  `_priority(op)`.
- **Rule:** Order (not filter) each op's candidate list by glyph: `*`→`mul` first,
  `+`→`add` first, `-`→`absdiff` first, `/`→`fdiv` first. Other glyphs use
  `DEFAULT_PRIORITY`. The list is appended with any `op_types` entries not already in it.
- **Example:** For `*` the order is `mul, mul_m1, mul_p1, mul_m2, mul_p2, absdiff, add, …`.
  At the leaf, `mul` is checked first and returned before offset variants that might also fit.
- **Why:** The leaf returns the *first* matching op, so ordering biases toward the canonical
  interpretation. Without it, generic ops like `mul_p1` (`a*b+1`) could win over `mul` when
  both fit a small example set.

### 4. Structural concat opt-in (Python)
- **Where:** 
- **Rule:** `concat_fwd` and `concat_rev` are added to an op's candidate list only if every
  example for that op is unsigned **and** `res_len == 4`.
- **Example:** A 3-symbol result rules concat out — two 2-symbol operands always concatenate
  to exactly 4 symbols. Three 4-symbol unsigned examples are a strong concat signal.
- **Why:** Cheap structural test that removes concat from consideration in virtually all
  arithmetic puzzles, avoiding the symbol-comparison work at the leaf.

### 5. Op-character ordering — `sorted_ops` (Python)
- **Where:**
- **Rule:** Sort op chars by `(len(candidates), -len(eqs))` — fewest candidates first, ties
  broken by most equations. This controls the order in which groups are handed to Rust.
- **Example:** Op `*` with 4 candidates and 1 equation comes before op `-` with 30 candidates
  and 3 equations. The Rust forward-check hits `*` first on every DFS node.
- **Why:** MRV at the operator level — the most-constrained group rejects bad digit
  assignments earliest.

### 6. Magnitude pre-prune (Rust)
- **Where:** `lib.rs:244`.
- **Rule:** After computing the raw arithmetic result `v`, check `if v >= base^res_len →
  false` before converting to digits.
- **Example:** Base 10, 2-digit expected result. `add(78, 89) = 167 ≥ 100` → rejected with
  one comparison. No digit loop, no symbol lookup.
- **Why:** Width-based rejection is O(1) and short-circuits the digit conversion + symbol
  comparison, which is significantly more expensive.

### 7. Concat structural fast-check (Rust)
- **Where:** `lib.rs:203-217` inside `candidate_matches_eq`.
- **Rule:** For concat candidates, compare symbol-index sequences directly:
  `concat_fwd` ⟺ `res_syms == [l0, l1, r0, r1]`; `concat_rev` ⟺ `[r0, r1, l0, l1]`.
  Skip `apply()` entirely.
- **Example:** `res_syms = [1, 2, 3, 4]` with operands `[1, 2] OP [3, 4]` → `concat_fwd`
  matches in one comparison. No digit conversion.
- **Why:** Concat is a structural relation on symbol indices, not a numeric operation.
  Routing it through the arithmetic `apply` path would be both slower and conceptually wrong.

### 8. Zero-Padding Strategy (Python + Rust)
- **Where:** Python `_encode_answer`; Rust `candidate_matches_eq` (`:247-255`).
- **Rule:** Arithmetic results often have fewer digits than the expected result width (e.g., `3 + 2 = 05`). Before comparing or emitting:
  1. **Left-pad with zeros** to match `res_len` (the number of result symbols from the equation).
  2. In **reversed-digit mode** (`little_endian`/`alice`), pad first then reverse the whole digit list — so leading zeros become trailing zeros after the flip.
  3. In **`_encode_answer`** (gold-conditioned path only): if `answer_hint` is set, derive `target_len` from the gold answer's character count (stripping a leading sign character when present), then left-pad the digit list to that length before reversing.
- **Rejection guard:** If `v >= base^res_len` the result cannot fit in `res_len` digits even with zero-padding → immediately reject (`false` in Rust, skip in Python) without doing any digit work. (See Heuristic #6.)
- **Example:** Base 10, `res_len = 2`, result value `5`. `_int_to_base_digits(5, 10)` returns `[5]` (one digit). Left-pad to length 2 → `[0, 5]`. In `little_endian` mode reverse → `[5, 0]`. Map back via inverse symbol table to emit the answer.
- **Why:** Symbol sequences always have a fixed length (one symbol per digit position). A raw arithmetic result may be smaller than that width. Zero-fill closes the gap, matching the positional encoding the puzzle uses. Reversing after padding (not before) ensures leading zeros land in the correct position for the reversed-digit encoding.

### 9. Empty-Group Tier Skip (Python)
- **Where:**
- **Rule:** After filtering each op's candidate list to the current tier (`tier_cands`), if **any** op ends up with zero surviving candidates, skip the entire tier immediately — do not call `_search` at all — and advance to the next (wider) tier.
- **Example:** TIER0 contains `add, sub, …` but not `fdiv`. If an op's sign pattern left only `fdiv` as a candidate (via Heuristic #2), that op's `tier_cands` list is empty for TIER0. The skip fires, and the solver jumps straight to TIER1 where `fdiv` is present.
- **Why:** Calling Rust with a group that has zero candidates is guaranteed to return no solution — the DFS has nothing to try for that group. Skipping saves the overhead of spawning the search and lets the tier loop advance to a wider candidate set where a solution is actually possible.

---

## 6. Module Layout (From-Scratch Build)

```
rust/alice_sovler_helper/src/lib.rs
  ├── EqData, OpGroup                   # data structures
  ├── arithmetic_search()               # PyO3 entry; forwards to search()
  ├── search() → dfs()                 # §3 algorithm
  ├── variable_order()                  # Heuristic #1
  ├── candidate_matches_eq()            # §3 equation matching
  ├── group_has_candidate_for_assigned_eqs()  # §3 forward-check
  ├── apply(op_idx, L, R)              # arithmetic dispatch by index
  ├── is_concat_op(), is_signed_op()   # structural helpers
  └── pow_i32(base, exp)               # fast integer power
```

**Critical lockstep requirement:** `OP_NAME_TO_IDX` in Python and `apply()` in Rust must
index the same ops at the same integer positions. Drift between the two causes silent wrong
answers.

Build commands:
```bash
uv sync
uv run maturin develop --release   # compile Rust extension
```

---

## 7. Configuration File (config.yaml)

The solver reads an optional `config.yaml` from the project root (or any path
passed as a CLI argument). Available keys and their defaults:

```yaml
force_base_10: false          # only try (10, 10) radix/digit_count pair
disabled_heuristics: []       # names of Python heuristic functions to bypass:
                              #   op_glyph_priority, sign_pattern_filter,
                              #   concat_optin, order_ops, tier_skip
tiers: [tier0, tier1, tier2, full]   # subset/order of tiers to search
input_parquet: data/solver_results_base.parquet
output_parquet: data/solver_results.parquet
```

---

## 8. Output Format — `solver_results.parquet`

One row per solution found (multi-row per puzzle when `solution_count > 1`). Columns:

| Column | Type | Description |
|---|---|---|
| `problem_id` | str | Sample identifier (from input parquet) |
| `solution_id` | int | Per-puzzle sequential index (0, 1, 2 …) |
| `prompt` | str | Original puzzle prompt |
| `answer` | str | Gold answer |
| `problem_category` | str\|null | High-level puzzle domain (from input parquet) |
| `solver_answer` | str\|null | Solver-produced answer |
| `solver_correct` | bool | `solver_answer == answer` |
| `solver_category` | str\|null | Solver-assigned family: `arithmetic` or `concatenation` |
| `solver_strategy` | str\|null | `pure_concat`, `standard`, or `little_endian` |
| `ops_mapping` | json\|null | `{op_char: op_name}` — first surviving candidate per operator |
| `digits_mapping` | json\|null | `{symbol: digit}` — content symbol to digit mapping |
| `solver_tier` | str\|null | Tier that yielded the solution: `tier0…full` |
| `solver_radix` | int\|null | Positional base used for arithmetic |
| `ops_candidates` | json\|null | `{op_char: [op_names]}` — all surviving candidates |
| `equations` | json\|null | Retained for post-processing; not in final CoT dataset |
| `solution_count` | int | Total distinct solutions found (`0` = no solution) |
| `standard_solution_count` | int | Solutions found under `standard` mode |
| `little_endian_solution_count` | int | Solutions found under `little_endian` mode |

No-solution rows have null for all `solver_*` columns except `solver_correct = false`.

---

## 9. Verification Baseline

When reimplementing, validate against the current results:

| Outcome | Count |
|---------|-------|
| Solved correctly | 800 |
| Wrong answer | 0 |
| No solution found | 23 |
| **Total** | **823** |

Zero wrong answers is the hard constraint — when the solver returns an answer it is verified
against the full program. Run:

---

## 10. Search-Trace CoT (`generate_cot`)

When `config.generate_cot = True`, the solver generates a prose narration of the search
process for each solved puzzle and saves it in the `search_cot` parquet column.

### Config flag

`generate_cot: bool = False` in `src/solver/config.py` / `config.yaml`. When false, no traced search
runs and `search_cot` is `None` for every row (zero overhead).

### Two instrumented entry points

**Python layer** — `_try_arithmetic` (`src/solver/solver_eq_symbolic.py`) accumulates `py_events`
as it makes decisions:
- `candidates` event: per operator glyph, which heuristics narrowed the candidate list and to
  what (heuristics #2 `sign_pattern_filter`, #3 `op_glyph_priority`, #4 `concat_optin`).
- `tier_skip` event: which tier was skipped because an operator group was empty (heuristic #9).
- `stage_enter` / `stage_result` events: which tier/mode/radix was tried and whether
  it produced results.

**Rust layer** — `search_traced` / `arithmetic_search_traced` (`lib.rs:620` / `lib.rs:889`) is
called instead of `arithmetic_search` for each stage until the first solution is found. It
emits an ordered event log alongside the solutions:

| kind | Name | Meaning | `aux` field |
|------|------|---------|-------------|
| 0 | TRY | About to test `digit` for `sym_idx` | — |
| 1 | PRUNE | Forward-check failed (`lib.rs:707`) | failing group index |
| 2 | RECURSE | Forward-check passed, descending (`lib.rs:715`) | — |
| 3 | BACKTRACK | Subtree returned no solution (`lib.rs:739`) | — |
| 4 | LEAF_REJECT | All symbols bound, a group has no valid op (`lib.rs:668`) | failing group index |
| 5 | SOLUTION | Full assignment accepted (`lib.rs:676`) | — |

Events are only recorded **up to and including the first SOLUTION**; after that `trace_done`
is set and the search continues silently to collect up to `max_solutions`. Always uses
canonical weight-priority variable ordering (`variable_order`, `lib.rs:454`) regardless of
`randomize_search`.

### Renderer

`src/solver/cot_search_trace.py` — pure functions, no side effects.

- `render_search_cot(py_events, rust_events, content_syms, sorted_ops, digits_mapping,
  ops_mapping, mode, radix, tier) -> str`
- `render_concat_cot(op_glyph, op_name, answer) -> str` for concat fast-path puzzles.

The output has three sections:
1. **[Candidate setup and search strategy]** — Python-layer events in prose.
2. **[Digit-assignment DFS]** — Rust events indented by depth; TRY events are emitted in
   Rust but skipped by the renderer (PRUNE/RECURSE carry the same `sym`/`digit` info).
3. **[Result]** — digit assignment and operator resolution summary.

### Column threading

`search_cot` is threaded from `_try_arithmetic` → `solve_sample` → `main` exactly as
`search_reached` is today. It appears in `_COLUMNS` (`src/solver/solver_eq_symbolic.py`) as a plain
string (no JSON encoding). `_no_solution` sets it to `None`.

