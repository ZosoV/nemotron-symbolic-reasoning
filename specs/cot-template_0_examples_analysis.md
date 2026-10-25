# Chain-of-Thought Template — Stage 0 (Analysis + Candidate Enumeration)

This document is the **authoritative template** for the analysis-style `cot` column, the
zero-example variant of `docs/cot-template_0_examples.md`. Instead of framing a lone equation
as a prediction task and committing to one fabricated mapping, this variant **shows the full
equation**, teaches the basics, narrows the hypothesis space from structural observations, and
**enumerates 2–3 real, verified candidate mappings** — closing with an explicit statement that
the equation is underdetermined. **There is no `solution:` line.**

> **Why a separate variant:** a single equation is underdetermined (mean ≈ 4.58 consistent
> mappings; `solution_count` up to the solver `k` cap). Committing to one arbitrary mapping and
> a single answer trains *recall of a non-deducible secret*. This variant instead teaches the
> ontology and the reasoning moves, and demonstrates the multiplicity directly by enumerating
> several mappings that all reproduce the shown right-hand side.

> **Voice rule:** The `cot` column is **pure natural-language reasoning**. No Rust, Python,
> function names, file paths, or "Heuristic #N" labels appear anywhere in the emitted text. The
> maintainer section is for maintainers only.

> **Diversity rule:** Variety comes from **content** — the real enumerated candidates, the
> per-equation structural observations, and mixed strategies/operations — **not** from
> paraphrasing connectives. The phrasing skeleton is fixed.

---

## 1. Prompt Format

The prompt **shows the full equation** (lhs *and* rhs) and asks for analysis, not a prediction:

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Here is
one such equation:
  <lhs_str> = <rhs_str>
A single equation does not uniquely pin down the hidden rules. Explain what can be inferred
about the operation mapping, digit mapping, and solving strategy from this equation alone, and
give a few example mappings that are each consistent with it.
```

Reconstruction helpers (validated 100% against source):

```python
def lhs_str(l):   return l["l0"] + l["l1"] + l["op"] + l["r0"] + l["r1"]
def rhs_str(rh):
    body = "".join(rh["actual_rhs"])
    if rh["op"] is None: return body
    return rh["op"] + body if rh["sign_pos"] == "prefix" else body + rh["op"]
```

There is no separate gold `answer` target — the rhs is shown in the prompt, and every
enumerated candidate is verified to reproduce it.

---

## 2. Full Annotated Template

```
{OPS_BLOCK}

{STRATEGIES_BLOCK}

Shown equation:
  {lhs_str} = {rhs_str}

{OBSERVATIONS}

{NARROWING}            ← concat ruling + sign-pattern op narrowing (space narrowing, not a
                         "wrong path I tried")

{CANDIDATES}           ← 2–3 worked candidate mappings (1 for pure_concat)

{UNDERDETERMINATION}   ← explicit close: how many mappings are consistent; examples narrow them
```

Removed vs the prediction template (`cot-template_0_examples.md`): the trailing
`solution: {answer}` line, and `FAKE_STEPS` as a distinct "plausible wrong path" block (its
rulings are folded into `{NARROWING}`).

### 2.1 `{OPS_BLOCK}` — always included (unchanged)

The numbered list of all 22 operations with natural-language formula and guard condition,
generated once from the operation registry (insertion order = frequency rank).

```
Let's assume the following set of available operations that could govern the secret equations:

 1. mul(a, b) = a × b
 2. add(a, b) = a + b
 3. absdiff(a, b) = |a − b|  (always non-negative)
 4. sub_signed(a, b) = a − b  (result may be negative; if so, the operator symbol is attached
    to the answer as a sign)
 5. rsub_signed(a, b) = b − a  (same sign convention as sub_signed)
 6. neg_absdiff(a, b) = −|a − b|  (always non-positive; the operator symbol is always
    attached to the answer)
 7. concat_fwd(a, b) = the symbols of a followed by the symbols of b  (structural, no
    arithmetic)
 8. concat_rev(a, b) = the symbols of b followed by the symbols of a  (structural, no
    arithmetic)
 9. add_m1(a, b) = a + b − 1  (invalid if result would be negative)
10. add_p1(a, b) = a + b + 1
11. add_p2(a, b) = a + b + 2
12. mul_m1(a, b) = a × b − 1  (invalid if result would be negative)
13. mul_p1(a, b) = a × b + 1
14. absdiff_m1(a, b) = |a − b| − 1  (invalid when a = b)
15. absdiff_p1(a, b) = |a − b| + 1  (always at least 1)
16. absdiff_m2(a, b) = |a − b| − 2  (invalid when |a − b| < 2)
17. gcd(a, b) = greatest common divisor of a and b  (always non-negative) ## NOTE: if the result is possitive then this operation is not feasible
18. lcm(a, b) = least common multiple of a and b  (0 if either operand is 0)
19. mod(a, b) = a mod b  (invalid when b = 0)
20. rmod(a, b) = b mod a  (invalid when a = 0)
21. fdiv(a, b) = ⌊a ÷ b⌋  (invalid when b = 0)
22. a2_plus_b(a, b) = a² + b
```

### 2.2 `{STRATEGIES_BLOCK}` — always included (unchanged)

```
There are three possible solving strategies:

- Pure concatenation: the operator simply joins the two operands. The result is either the
  symbols of the left operand followed by those of the right operand (forward), or the symbols
  of the right operand followed by those of the left (reverse). No digit mapping is needed.

- Standard arithmetic (base 10, most-significant digit first): each unique symbol maps to a
  unique decimal digit (0–9). The left operand is decoded as L = digit(l0) × 10 + digit(l1),
  and the right operand as R = digit(r0) × 10 + digit(r1). The chosen operation is applied:
  result = OP(L, R). The magnitude is then converted to decimal digits from most to least
  significant, zero-padded on the left if the answer needs more digits, and each digit is
  mapped back to its symbol.

- Little-endian arithmetic (base 10, least-significant digit first): the same as standard
  arithmetic, except each operand's two digits are swapped before the operation — the symbol
  in the first position carries the units digit (not the tens digit), so the raw two-digit
  value X is interpreted as (X mod 10) × 10 + (X ÷ 10). The result digit sequence is
  reversed after zero-padding before mapping back to symbols.

We always use base 10.

The left-hand side of any equation has exactly 5 symbols: l0 l1 op r0 r1, where op is the
operator symbol, l0 l1 is the left operand, and r0 r1 is the right operand.

The right-hand side can have 1 to 4 symbols:
  - mag           — all symbols represent the magnitude of the result.
  - op mag        — operator symbol first, then magnitude (result is negative; sign prefix).
  - mag op        — magnitude first, operator symbol last (result is negative; sign postfix).

The goal is to understand which digit mappings (symbol → digit), operation mappings (op →
function), and strategies are consistent with the equation.
```

### 2.3 `{OBSERVATIONS}` — always included (content varies by row)

A short bullet list of structural features of *this specific* equation:

- Always state whether the answer carries an operator symbol (a sign) or not.
- If `sign_pos == "prefix"`: note the sign prefix and that the result is negative.
- If `sign_pos == "postfix"`: note the sign postfix (little-endian signed form is plausible).
- If answer length == 4: note it matches the operand symbol count (concat candidate).
- If answer length < 4: note that concatenation is not possible.

### 2.4 `{NARROWING}` — always included (content varies by row)

Folds in the old fake-step *rulings*, but framed as narrowing the hypothesis space rather than
a wrong path that was tried:

- **Concat ruling.** Compute forward (l0 l1 r0 r1) and reverse (r0 r1 l0 l1). State whether
  either matches the shown rhs; if neither, concatenation is ruled out.
- **Sign-pattern narrowing** (when `sign_pos` is set): a sign character means the underlying
  result is negative, so a non-negative operation (absdiff and its variants) cannot apply; the
  operation must be a signed subtraction.
- **No-sign narrowing** (when `sign_pos is None`): a negated difference always attaches the
  operator symbol, so it is ruled out for an unsigned answer.

### 2.5 `{CANDIDATES}` — always included; 2–3 candidates (1 for pure_concat)

Each candidate is one compact worked derivation reusing the standard move set. Candidates are
introduced as **Candidate A / B / C** and each ends by confirming it reproduces the shown rhs.
Mix strategies/operations across candidates when the data offers them.

Per-candidate sub-structure:

```
**Candidate {A/B/C} — {strategy description}{, op → name}.**

{LITTLE_ENDIAN_INTRO}   ← only if candidate strategy == little_endian
Assign digits to the most frequent symbols first: {symbol → digit list}.
Left operand:  {decode}.  Right operand: {decode}.
{DIGIT_SWAP}            ← only if little_endian
Applying {formula}: {numeric expression} = {result}.
{OVERFLOW_CHECK}        ← result fits within N positions
{ZERO_PAD_NOTE}         ← only if magnitude digit count < res_len
{RESULT_REVERSE}        ← only if little_endian
Converting digits back to symbols: ... → {rhs}.
{SIGN_NOTE}             ← only if sign_pos is not null
This reproduces the shown right-hand side ✓
```

### 2.6 `{UNDERDETERMINATION}` — always included

Phrasing reflects `solution_count`:

- `solution_count == 1` → "This equation is fully determined — only one mapping is
  consistent." (typical for pure concatenation.)
- `solution_count > 1` → "A single equation does not pin down a unique mapping: this one
  admits {N} consistent mappings, of which the above are examples. In a full puzzle, the shown
  examples narrow this set."

---

## 3. Conditional Inclusion Table

| Element | Always | Condition for inclusion |
|---|---|---|
| OPS_BLOCK (22 ops) | ✓ | — |
| STRATEGIES_BLOCK | ✓ | — |
| Shown equation (lhs = rhs) | ✓ | — |
| Observations block | ✓ | — |
| Sign-prefix observation | | `sign_pos == "prefix"` |
| Sign-postfix observation | | `sign_pos == "postfix"` |
| "No sign" observation | | `sign_pos is None` |
| 4-symbol / concat-candidate observation | | answer length == 4 |
| NARROWING: concat ruling | ✓ | — |
| NARROWING: signed-op narrowing | | `sign_pos is not None` |
| NARROWING: neg_absdiff ruling | | `sign_pos is None` |
| CANDIDATES block | ✓ | 2–3 candidates; 1 if `category == "concatenation"` |
| Little-endian intro / digit-swap / result-reversal (per candidate) | | that candidate's `strategy == "little_endian"` |
| Zero-padding narration (per candidate) | | magnitude digit count < `res_len` |
| Sign prefix/postfix note (per candidate) | | `sign_pos == "prefix"` / `"postfix"` |
| UNDERDETERMINATION close | ✓ | phrasing varies by `solution_count` |
| `solution:` line | ✗ | **removed** |
| FAKE_STEPS section | ✗ | **removed** (rulings folded into NARROWING) |

---

## 4. Worked Examples

### 4.1 Standard + little-endian, unsigned — two real candidates (id: `00457d26_0`)

**Puzzle:** `` `!*[{ = '"[` ``  •  `solution_count: 20`
**Candidate A:** standard, `*` → mul, `` ` ``→6 `!`→3 `[`→9 `{`→2 `'`→5 `"`→7
**Candidate B:** little_endian, `*` → mul, `` ` ``→5 `!`→8 `[`→2 `{`→6 `'`→0 `"`→7

```
Let's assume the following set of available operations that could govern the secret equations:

 1. mul(a, b) = a × b
 2. add(a, b) = a + b
 3. absdiff(a, b) = |a − b|  (always non-negative)
 4. sub_signed(a, b) = a − b  (result may be negative; if so, the operator symbol is attached
    to the answer as a sign)
 5. rsub_signed(a, b) = b − a  (same sign convention as sub_signed)
 6. neg_absdiff(a, b) = −|a − b|  (always non-positive; the operator symbol is always
    attached to the answer)
 7. concat_fwd(a, b) = the symbols of a followed by the symbols of b  (structural, no
    arithmetic)
 8. concat_rev(a, b) = the symbols of b followed by the symbols of a  (structural, no
    arithmetic)
 9. add_m1(a, b) = a + b − 1  (invalid if result would be negative)
10. add_p1(a, b) = a + b + 1
11. add_p2(a, b) = a + b + 2
12. mul_m1(a, b) = a × b − 1  (invalid if result would be negative)
13. mul_p1(a, b) = a × b + 1
14. absdiff_m1(a, b) = |a − b| − 1  (invalid when a = b)
15. absdiff_p1(a, b) = |a − b| + 1  (always at least 1)
16. absdiff_m2(a, b) = |a − b| − 2  (invalid when |a − b| < 2)
17. gcd(a, b) = greatest common divisor of a and b  (always non-negative)
18. lcm(a, b) = least common multiple of a and b  (0 if either operand is 0)
19. mod(a, b) = a mod b  (invalid when b = 0)
20. rmod(a, b) = b mod a  (invalid when a = 0)
21. fdiv(a, b) = ⌊a ÷ b⌋  (invalid when b = 0)
22. a2_plus_b(a, b) = a² + b

There are three possible solving strategies:

- Pure concatenation: the operator simply joins the two operands. The result is either the
  symbols of the left operand followed by those of the right operand (forward), or the symbols
  of the right operand followed by those of the left (reverse). No digit mapping is needed.

- Standard arithmetic (base 10, most-significant digit first): each unique symbol maps to a
  unique decimal digit (0–9). The left operand is decoded as L = digit(l0) × 10 + digit(l1),
  and the right operand as R = digit(r0) × 10 + digit(r1). The chosen operation is applied:
  result = OP(L, R). The magnitude is then converted to decimal digits from most to least
  significant, zero-padded on the left if the answer needs more digits, and each digit is
  mapped back to its symbol.

- Little-endian arithmetic (base 10, least-significant digit first): the same as standard
  arithmetic, except each operand's two digits are swapped before the operation — the symbol
  in the first position carries the units digit (not the tens digit), so the raw two-digit
  value X is interpreted as (X mod 10) × 10 + (X ÷ 10). The result digit sequence is
  reversed after zero-padding before mapping back to symbols.

We always use base 10.

The left-hand side of any equation has exactly 5 symbols: l0 l1 op r0 r1, where op is the
operator symbol, l0 l1 is the left operand, and r0 r1 is the right operand.

The right-hand side can have 1 to 4 symbols:
  - mag           — all symbols represent the magnitude of the result.
  - op mag        — operator symbol first, then magnitude (result is negative; sign prefix).
  - mag op        — magnitude first, operator symbol last (result is negative; sign postfix).

The goal is to understand which digit mappings (symbol → digit), operation mappings (op →
function), and strategies are consistent with the equation.

Shown equation:
  `!*[{ = '"[`

Observations:
- The answer '"[` carries no operator symbol, so the result is non-negative and no signed
  operation is in play.
- The answer has 4 symbols, which matches the combined length of the two operands (2 + 2) —
  consistent with either a 4-digit arithmetic result or concatenation.

Checking concatenation: forward (l0 l1 r0 r1) gives `![{ and reverse (r0 r1 l0 l1) gives
[{`!. Neither matches '"[`, so concatenation is ruled out. The answer carries no sign
character, so a negated difference (which always signs its result) is also ruled out. That
leaves an arithmetic operation under either strategy.

**Candidate A — standard arithmetic, * → mul.**
Assign digits to the most frequent symbols first: ` → 6, [ → 9, ! → 3, { → 2, ' → 5, " → 7.
Left operand:  digit(`) × 10 + digit(!) = 6 × 10 + 3 = 63.
Right operand: digit([) × 10 + digit({) = 9 × 10 + 2 = 92.
Applying mul (a × b): 63 × 92 = 5796. It has 4 digits and the answer has 4 positions (5796 <
10000) ✓.
Converting digits back to symbols: 5 → ', 7 → ", 9 → [, 6 → ` → '"[`.
This reproduces the shown right-hand side ✓

**Candidate B — little-endian arithmetic, * → mul.**
In little-endian encoding each operand's two digits are read with the first symbol carrying the
units digit; the result digit sequence is reversed before mapping back.
Assign digits to the most frequent symbols first: ` → 5, [ → 2, ! → 8, { → 6, ' → 0, " → 7.
Raw left value: 5 × 10 + 8 = 58; raw right value: 2 × 10 + 6 = 26.
Applying the digit swap (units ↔ tens): 58 → 85, 26 → 62.
Applying mul (a × b): 85 × 62 = 5270. It has 4 digits and the answer has 4 positions (5270 <
10000) ✓.
Digits [5, 2, 7, 0] reversed for little-endian: [0, 7, 2, 5].
Converting digits back to symbols: 0 → ', 7 → ", 2 → [, 5 → ` → '"[`.
This reproduces the shown right-hand side ✓

A single equation does not pin down a unique mapping: this one admits 20 consistent mappings,
of which Candidates A and B are two examples — note they disagree on both the digit assignment
and the strategy, yet both satisfy the equation. In a full puzzle, the shown examples narrow
this set.
```

---

### 4.2 Standard arithmetic, signed prefix — two candidates (id: `0133bcec_4`)

**Puzzle:** `[^-[( = -^`  •  `solution_count: 10`
**Candidate A:** standard, `-` → sub_signed, `[`→0 `^`→1 `(`→2
**Candidate B:** standard, `-` → sub_signed, `[`→1 `^`→2 `(`→4

```
[OPS_BLOCK — omitted for brevity]

[STRATEGIES_BLOCK — omitted for brevity]

Shown equation:
  [^-[( = -^

Observations:
- The answer -^ starts with the operator symbol -, so the result is negative and the operator
  symbol is used as a sign prefix.
- The magnitude part (^) has 1 symbol, so the numeric result fits in 1 digit.

The answer begins with the operator symbol — concatenation cannot produce a leading sign
character, so concatenation is ruled out. Because the answer carries a sign, the underlying
result is negative; a non-negative operation such as an absolute difference can never need a
sign, so the operation must be a signed subtraction.

**Candidate A — standard arithmetic, - → sub_signed.**
Assign digits to the most frequent symbols first: [ → 0, ^ → 1, ( → 2.
Left operand:  digit([) × 10 + digit(^) = 0 × 10 + 1 = 1.
Right operand: digit([) × 10 + digit(() = 0 × 10 + 2 = 2.
Applying sub_signed (a − b): 1 − 2 = −1. The magnitude 1 has 1 digit and the answer magnitude
has 1 position — it fits ✓.
Converting the magnitude digit back to a symbol: 1 → ^. The result is negative, so the operator
symbol - is prepended → -^.
This reproduces the shown right-hand side ✓

**Candidate B — standard arithmetic, - → sub_signed.**
Assign digits: [ → 1, ^ → 2, ( → 4.
Left operand:  1 × 10 + 2 = 12.   Right operand: 1 × 10 + 4 = 14.
Applying sub_signed (a − b): 12 − 14 = −2. The magnitude 2 fits in 1 position ✓.
Converting the magnitude digit back to a symbol: 2 → ^. The result is negative, so - is
prepended → -^.
This reproduces the shown right-hand side ✓

A single equation does not pin down a unique mapping: this one admits 10 consistent mappings,
of which Candidates A and B are two examples — they share the operation but differ in the digit
assignment. In a full puzzle, the shown examples narrow this set.
```

---

### 4.3 Standard arithmetic, signed postfix (zero-padding) — two candidates (id: `1cce5949_4`)

**Puzzle:** `` `\}`] = [?} ``  •  `solution_count: 20`
**Candidate A:** standard, `}` → sub_signed, `` ` ``→8 `\`→2 `]`→3 `[`→0 `?`→1
**Candidate B:** standard, `}` → sub_signed, `` ` ``→5 `\`→1 `]`→4 `[`→0 `?`→3

```
[OPS_BLOCK — omitted for brevity]

[STRATEGIES_BLOCK — omitted for brevity]

Shown equation:
  `\}`] = [?}

Observations:
- The answer [?} ends with the operator symbol }, so the result is negative and the operator
  symbol is placed at the end (sign postfix form).
- The magnitude part [? has 2 symbols.

The answer ends with the operator symbol — concatenation produces only bare operand symbols
with no added sign character, so concatenation is ruled out. The sign means the underlying
result is negative; an absolute difference is always non-negative and can never need a sign, so
the operation must be a signed subtraction.

**Candidate A — standard arithmetic, } → sub_signed.**
Assign digits to the most frequent symbols first: ` → 8, \ → 2, ] → 3, [ → 0, ? → 1.
Left operand:  digit(`) × 10 + digit(\) = 8 × 10 + 2 = 82.
Right operand: digit(`) × 10 + digit(]) = 8 × 10 + 3 = 83.
Applying sub_signed (a − b): 82 − 83 = −1. The magnitude is 1, but the answer magnitude [? has
2 positions, so we zero-pad on the left: [0, 1].
Converting digits back to symbols: 0 → [, 1 → ? → [?. The result is negative, so the operator
symbol } is appended (sign postfix) → [?}.
This reproduces the shown right-hand side ✓

**Candidate B — standard arithmetic, } → sub_signed.**
Assign digits: ` → 5, \ → 1, ] → 4, [ → 0, ? → 3.
Left operand:  5 × 10 + 1 = 51.   Right operand: 5 × 10 + 4 = 54.
Applying sub_signed (a − b): 51 − 54 = −3. The magnitude 3 zero-pads to 2 positions: [0, 3].
Converting digits back to symbols: 0 → [, 3 → ? → [?. The result is negative, so } is appended
→ [?}.
This reproduces the shown right-hand side ✓

A single equation does not pin down a unique mapping: this one admits 20 consistent mappings,
of which Candidates A and B are two examples. Both rely on the symbol for zero appearing as the
leading magnitude symbol. In a full puzzle, the shown examples narrow this set.
```

---

### 4.4 Pure concatenation — single determined candidate (id: `0133bcec_0`)

**Puzzle:** `%|*"| = %|"|`  •  `solution_count: 1`
**Candidate:** pure_concat (forward)

Note: OPS_BLOCK and STRATEGIES_BLOCK are still emitted (identical to all other rows). There is
a single candidate, and the underdetermination close states the equation is determined.

```
[OPS_BLOCK — omitted for brevity]

[STRATEGIES_BLOCK — omitted for brevity]

Shown equation:
  %|*"| = %|"|

Observations:
- The answer %|"| carries no operator symbol, so the result is non-negative and no signed
  operation is in play.
- The answer has 4 symbols, matching the combined length of the two operands (2 + 2) —
  consistent with either a 4-digit arithmetic result or concatenation.

Checking concatenation: forward concatenation (left operand followed by right operand) gives
%|"|, which matches the answer exactly. No digit mapping is needed — the operator simply glues
the operands left-to-right.

**Candidate — pure concatenation (forward).**
The operator joins the operands: %| followed by "| gives %|"|.
This reproduces the shown right-hand side ✓

This equation is fully determined under concatenation — only the forward-concatenation rule is
consistent, and no digit mapping is required.
```

---

## 5. Maintainer Section — Generation Notes

> For maintainers only. Nothing here appears in the `cot` column.

### 5.1 Candidate sourcing

Candidates are **real verified solver solutions**, not fabricated. The solver retains up to `k`
distinct programs per equation, stored as **multiple rows per `id`** (long format), each row
carrying `digits_mapping`, `ops_mapping`, `solver_strategy` (plus `ops_candidates`,
`solver_radix`, `solver_tier`).

- **Source files:** `data/result_examples_0_result_k_6_balanced.parquet` (≤6 candidates/id) or
  `data/solver_examples_0_k_20.parquet` (≤20/id).
- **Assembly:** `groupby('id')`, filter `solver_correct == True`, take distinct
  `(digits_mapping, ops_mapping, solver_strategy)` tuples, then select **2–3** — prefer mixing
  strategy and operation across the selected candidates (e.g. one standard + one little-endian,
  or two different operations) to make the underdetermination vivid. For `pure_concat` /
  `solution_count == 1`, emit the single candidate.
- **Row collapse:** one analysis sample **per equation `id`** (≈4,165), replacing the previous
  one-row-per-candidate scheme (≈22.8k rows). The same equation no longer recurs with the same
  fabricated answer.

### 5.2 Reusable helpers

| Move | Helper | Location |
|---|---|---|
| Reconstruct lhs / rhs strings | `lhs_str`, `rhs_str` | `cot_dataset_0.ipynb` (cell 2) |
| Frequency-ordered digit assignment narrative | `symbol_freq_order(eq)` | `cot_dataset_0.ipynb` (cell 9) |
| Operand decode / digit swap / op-call formatting | `decode_lhs`, `swap_digits`, `format_op_call` | `cot_dataset_0.ipynb` (cell 9) |
| Verify a candidate reproduces the rhs | `encode_answer(...)` | `src/encoding.py:37` |
| 22-op registry (OPS_BLOCK ordering) | `OPERATIONS` | `src/operations.py:272` |

### 5.3 Reasoning-move → heuristic map (unchanged from prediction template)

| Natural-language reasoning move | Underlying heuristic |
|---|---|
| "Assign digits to the most frequent symbols first" | symbol variable ordering |
| Sign-pattern narrowing ("only signed subtraction applies") | sign-pattern op narrowing |
| Concat fast-check pass/fail | structural concat opt-in |
| "padding with the symbol for zero" | zero-padding strategy |
| Little-endian "standard fails here" reasoning | standard tried before little_endian |

### 5.4 Per-row guard

For each emitted candidate, assert `encode_answer(dm, om, op_char, eq["lhs"], mode, 10,
rhs_str) == rhs_str` before rendering — a candidate that does not reproduce the shown rhs is a
generation bug and must fail loudly. The voice-rule guard (no `.py`, `def `, function names,
"Heuristic #N", and no `solution:` line) also applies to the emitted text.
