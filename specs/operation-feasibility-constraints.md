# Operation Feasibility Constraints — Structural Pre-filter Spec

This is the authoritative spec for the structural feasibility filter implemented in
`src/solver/heuristic_constraints/operation_feasibility.py`.

Related code: `src/solver/heuristic_constraints/operation_feasibility.py` ·
`src/parsing.py` · `src/operations.py` · `src/op_index.py`

---

## 1. Motivation and Scope

The solver (see `specs/solver_eq_symbolic.md`) searches for arithmetic operations
that satisfy an encrypted equation.  Many candidate operations are structurally
*impossible* for a given equation before any digit value is known — purely from the
sign affix, the RHS magnitude length, and the symbolic pattern of the operands.

This module provides a single entry point, `filter_feasible_operations()`, that:

- Accepts one encrypted equation string and a list of candidate op names.
- Returns the feasible subset plus a human-readable reason for every discard.
- Makes **zero false prunes**: an op is discarded only when it is structurally
  impossible, regardless of what digits the symbols encode.

### Design assumptions

| Assumption | Value |
|---|---|
| Digit assignment | **Injective** — distinct symbols encode distinct digits.  Enforced by the Rust DFS core. |
| Leading zeros | **Allowed** (conservative).  Operand values range `[0, base²−1]`; the first symbol may map to 0. |
| Decode modes | Both **standard** (`L = l0·base + l1`) and **little_endian** (`L = l1·base + l0`) are evaluated.  An op is discarded only when infeasible under **both** modes, matching `solver_eq_symbolic.py:320`. |
| Numeric base | `base=10` by default; callers may pass `len(content_symbols(equations))` for per-puzzle base. |

---

## 2. Equation Structure

An encrypted equation is a 5-character LHS plus a RHS:

```
l0 l1 op r0 r1 = [sign] m0 m1 …
```

Parsed by `src/parsing.py:parse_equation()` into:

| Field | Meaning |
|---|---|
| `l0, l1` | Left-operand symbols.  `L = l0·base + l1` (standard) or `L = l1·base + l0` (little-endian). |
| `op_char` | Operator glyph at LHS position 2. |
| `r0, r1` | Right-operand symbols, decoded analogously to L. |
| `rhs_syms` | Magnitude symbols `(m0, m1, …)` — the `actual_rhs` list without the sign character. |
| `res_len` | `len(rhs_syms)`, the number of magnitude symbols. |
| `has_sign` | `True` when the RHS carries the operator glyph as a sign affix. |
| `sign_pos` | `'prefix'` (standard) / `'postfix'` (little-endian) / `None`. |

**Decode-mode symmetry:** standard and little-endian differ only in which symbol is
the units digit.  All v1 checks compare *glyph identity*, not *digit values*, so
their verdicts are identical under both modes.  The `mode` parameter is reserved for
the Deferred coefficient-propagation extension (§4.5).

---

## 3. Constraints

Each check has signature `(op, eq, base, mode) -> str | None`.
`None` = no objection; a non-empty string = human-readable discard reason.
Checks run in the order listed; the **first objection wins** for a given mode.

---

### 3.1 `check_sign` — Sign Affix vs. Op Output Sign

**Strength: STRONG.**  Fires on most equations that carry a sign affix.

#### Rule

- `has_sign = True` (negative result expected):
  Only `NEGATIVE_CAPABLE = {sub_signed, rsub_signed, neg_absdiff}` can produce
  values `< 0`.  All other ops are discarded.

- `has_sign = False` (non-negative result expected):
  `neg_absdiff(a, b) = −|a−b| ≤ 0`.  Its only non-negative value is 0, reached
  when `a = b`.  By digit injectivity, `a = b` iff `l0 == r0` **and** `l1 == r1`
  (L≡R).  Discard `neg_absdiff` unless L≡R.

#### Worked example — unsigned RHS

Equation `` `#+>@ = !> ``

```
l0=`  l1=#  op=+  r0=>  r1=@
rhs_syms=(!,>)  res_len=2  has_sign=False
L≢R  (l0=` ≠ r0=>)
```

`neg_absdiff`: `has_sign=False`, L≢R → **discard**
> "RHS is unsigned but neg_absdiff always produces ≤ 0 (only 0 when L≡R; L≢R here)"

All other ops: no objection from this check.

#### Worked example — signed RHS (prefix)

Equation `AB-CD = -EF`

```
l0=A  l1=B  op=-  r0=C  r1=D
rhs_syms=(E,F)  res_len=2  has_sign=True
```

`mul`, `add`, `absdiff`, `gcd`, `lcm`, `fdiv`, `mod`, `rmod`,
`add_m1`, `add_p1`, `mul_m1`, `mul_p1`, `add_p2`,
`absdiff_p1`, `absdiff_m1`, `absdiff_m2`, `a2_plus_b` → **all discarded**
> "RHS is signed (negative result) but this op always produces non-negative values"

`sub_signed`, `rsub_signed`, `neg_absdiff` → kept.

---

### 3.2 `check_concat_length` — Concat Output Length

**Strength: STRONG.**  Always fires when the RHS is shorter or longer than 4 symbols.

#### Rule

`concat_fwd(a, b) = str(a) + str(b)` and `concat_rev` always emit exactly
`len(str(a)) + len(str(b)) = 2 + 2 = 4` symbols (both operands are 2-symbol numbers).

Discard `concat_fwd` / `concat_rev` unless:
- `res_len == 4`, **and**
- `has_sign == False` (concatenation result is always unsigned).

#### Worked example

Equation `` `#+>@ = !> ``  (`res_len=2`, unsigned)

`concat_fwd` → **discarded**
> "concat result is always 4 symbols (2 + 2), but RHS magnitude has 2 symbol(s)"

`concat_rev` → **discarded** (same reason)

Equation `AB*CD = EFGH`  (`res_len=4`, unsigned)

Both concat ops → **kept** by this check.

---

### 3.3 `check_magnitude_range` — Op Output Range vs. RHS Capacity

**Strength: WEAK under conservative assumptions** (rarely fires with `base=10`).
Kept for correctness and for edge cases with small bases.

#### Rule

Compute `[lo, hi]` — the range of the op's result **magnitude** (i.e. `|result|`)
for operands in `[0, base²−1]`:

| Op | lo | hi |
|---|---|---|
| `mul` | 0 | `(base²−1)²` |
| `add` | 0 | `2(base²−1)` |
| `absdiff`, `sub_signed`, `rsub_signed`, `neg_absdiff` | 0 | `base²−1` |
| `add_p1` | 1 | `2(base²−1)+1` |
| `mul_p1` | 1 | `(base²−1)²+1` |
| `add_p2` | **2** | `2(base²−1)+2` |
| `absdiff_p1` | **1** | `base²` |
| … | … | … |

Representable magnitude range for `res_len` symbols: `[0, base^res_len − 1]`.

Discard if `lo > base^res_len − 1` (minimum exceeds RHS capacity).

#### When it fires

Under `base=10`, `res_len ≥ 1`: most ops have `lo=0` and survive.
`add_p2` (lo=2) fires only when `base^res_len ≤ 2`, e.g. `base=2, res_len=1`:

```
base=2, res_len=1  →  rhs_max = 2^1 − 1 = 1
add_p2 lo = 2 > 1  →  discard
```

#### Under conservative assumptions (`base=10`)

For equations with `res_len ≥ 1`, all 22 ops survive this check.
Its value is correctness (edge-case safety) and future extensibility.

---

### 3.4 `check_self_application` — Forced-Constant Result When L≡R

**Strength: STRONG when l0==r0 and l1==r1.**

#### Rule

When `l0 == r0` **and** `l1 == r1`, digit injectivity forces `L = R` numerically.
Several ops then collapse to a forced constant:

**`NONE_ON_EQUAL = {absdiff_m1, absdiff_m2}`** — guard always fails:
```
absdiff_m1(x, x) = |x−x| − 1 = −1  →  None  (guard discards)
absdiff_m2(x, x) = |x−x| − 2 = −2  →  None
```
Always discard when L≡R.

**`ZERO_ON_EQUAL = {sub_signed, rsub_signed, absdiff, neg_absdiff, mod, rmod}`**
— result always 0 (when guard succeeds):
```
sub_signed(x,x) = 0    rsub_signed(x,x) = 0    absdiff(x,x) = 0
neg_absdiff(x,x) = 0   mod(x,x) = 0 (x≠0)      rmod(x,x) = 0 (x≠0)
```
Discard if:
1. `has_sign = True` — result is 0, not negative.
2. `len(set(rhs_syms)) > 1` — the RHS magnitude needs ≥ 2 distinct digit-0s,
   impossible by injectivity.

**`ONE_ON_EQUAL = {fdiv}`** — result always 1 (when L ≠ 0):
```
fdiv(x, x) = x // x = 1  for x ≠ 0
```
Discard if `has_sign = True` (1 is not negative).

#### Worked example

Equation `AA+AA = BC`  (`l0=A, l1=A, r0=A, r1=A → L≡R; rhs_syms=(B,C)`)

```
set(rhs_syms) = {B, C}  →  2 distinct glyphs
```

`sub_signed` (ZERO_ON_EQUAL, unsigned, 2 distinct RHS glyphs) → **discarded**
> "L≡R: this op always produces 0, so all 2 RHS magnitude symbol(s) must represent
>  the zero-digit — but they contain 2 distinct glyphs (impossible by digit injectivity)"

`absdiff`, `rsub_signed` → **discarded** (same reason)

`absdiff_m1`, `absdiff_m2` (NONE_ON_EQUAL) → **discarded**
> "L≡R: this op's guard always fails when operands are equal (result is always None)"

`add(A·10+A, A·10+A) = 2·(A·10+A)` — not a fixed constant, so `add` is **kept**.
`mul(x, x) = x²` — also not a fixed constant, **kept**.

---

## 4. `filter_feasible_operations` — Contract

```python
def filter_feasible_operations(
    equation: str | dict,
    candidate_ops: list[str],
    base: int = 10,
    modes: tuple[str, ...] = ("standard", "little_endian"),
) -> FeasibilityReport
```

(`src/solver/heuristic_constraints/operation_feasibility.py`)

- `equation`: raw string **or** `parse_equation()` result dict.
- `candidate_ops`: op names to evaluate.  Unknown names → discarded with
  `"unknown operation"`.
- `base`: numeric base; pass `len(content_symbols(equations))` for per-puzzle base.
- `modes`: modes to try.  Op discarded only when infeasible under **all** modes.

### Return type: `FeasibilityReport`

```python
@dataclass
class FeasibilityReport:
    operator: str               # op glyph from LHS[2]
    feasible: list[str]         # op names that survive all checks
    discarded: dict[str, str]   # op name → aggregated discard reason
    verdicts: list[OpVerdict]   # one per candidate, in input order

@dataclass
class OpVerdict:
    op: str
    feasible: bool
    reason: str | None                  # None when kept
    per_mode: dict[str, str | None]     # mode → reason (None = feasible)
```

### Aggregation

For an op that fails all modes:
- If both modes give the **same reason**, `reason = that reason`.
- If they differ, `reason = "<standard reason> | <little_endian reason>"`.

Since v1 checks are mode-invariant, the two per-mode reasons are always identical.

---

## 5. Complete Example

Equation `` `#+>@ = !> ``, all 22 ops, `base=10`:

| Op | check_sign | check_concat_length | check_magnitude_range | check_self_application | Verdict |
|---|---|---|---|---|---|
| `mul` | — | — | — | — | **keep** |
| `add` | — | — | — | — | **keep** |
| `absdiff` | — | — | — | — | **keep** |
| `sub_signed` | — | — | — | — | **keep** |
| `neg_absdiff` | **discard** (unsigned, L≢R) | — | — | — | **discard** |
| `concat_fwd` | — | **discard** (res_len=2≠4) | — | — | **discard** |
| `concat_rev` | — | **discard** (res_len=2≠4) | — | — | **discard** |
| `rsub_signed` | — | — | — | — | **keep** |
| `gcd`, `lcm`, `fdiv`, `mod`, `rmod` | — | — | — | — | **keep** |
| `add_m1`, `add_p1`, `add_p2` | — | — | — | — | **keep** |
| `mul_m1`, `mul_p1` | — | — | — | — | **keep** |
| `absdiff_p1`, `absdiff_m1`, `absdiff_m2` | — | — | — | — | **keep** |
| `a2_plus_b` | — | — | — | — | **keep** |

**Result:** 3 ops discarded (`neg_absdiff`, `concat_fwd`, `concat_rev`), 19 kept.

---

## 6. Deferred Constraints

### 6.1 Coefficient-based shared-symbol propagation *(Deferred)*

When the two operands share a digit variable `dᵢ` (same glyph appearing in both
`l0 l1` and `r0 r1` positions), linear ops produce a result with a modified
coefficient for that variable.

Example: `a = 10·d₁ + d₂`, `b = 10·d₃ + d₂` (share `d₂`):
```
sub_signed(a, b) = 10·d₁ + d₂ − 10·d₃ − d₂ = 10·(d₁ − d₃)
```
`d₂` cancels.  If the RHS encoding requires `d₂` to be present, `sub_signed` can
be pruned **without knowing digit values**.

#### Why this is Deferred

Two conditions must hold for this to prune safely:

1. **No leading zeros** — the cancelation-derived tighter range produces a non-zero
   minimum only when operands cannot be 0.  Under the conservative (leading-zeros
   allowed) assumption the minimum is always 0 and the prune never fires.

2. **Cross-symbol injectivity** — to conclude that a symbol absent from the result
   expression cannot appear in the RHS, we need the full injectivity constraint
   (distinct symbols → distinct digits) over all symbols in the equation, not just
   the L/R comparison.

Implementing this extension requires:
- Tracking the digit-variable coefficient map for each operand (from the encoding).
- Symbolic simplification of `op(a, b)` to find which variables cancel.
- Comparing the resulting symbol support against the RHS symbol set.

This is feasible but increases coupling to the encoding representation.  It is
marked Deferred until the conservative checks prove insufficient in practice.
