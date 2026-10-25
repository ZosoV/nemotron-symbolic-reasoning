<digit_mapping_1>
Let's analyze the possible constraints for the digits given the operator candidates, example
equations, and structural position of the operand symbols across the system of equations:

- `'` leads `'` in eq[1] and eq[4]; `[` leads `[{` in eq[1]; `\` leads `\&` in eq[4]
  → each leads a multi-symbol operand with distinct characters; no operand carries a
  leading zero → 0 is excluded (per-equation constraint: {1..9}).
- In eq[1] (operator `*`, surviving candidates: {add, mul, absdiff, lcm}), the set of
  assignments consistent with any candidate constrains: `'` ∈ {0,1,2,3,4,5,6,7}.
- Symbols `!`, `"`, `>`, `@`, `&` appear in positions where no candidate operator
  restricts their value → unconstrained ({0..9}).

Each cell shows which digits are consistent with that equation; '*' means unconstrained,
'—' means the symbol does not appear.

  sym   eq[1]   eq[2]  →  allowed
  '     {0..7}    *    →  {0..7}
  `     {1..9}    —    →  {1..9}
  [     {1..9}    *    →  {1..9}
  \       —       *    →  {1..9}
  !       *       *    →  {0..9}
  "       *       —    →  {0..9}
  @       —       *    →  {0..9}

Picking from the initial pool {0,1,2,3,4,5,6,7,8,9}, starting with the symbol that has the
fewest allowed options:
  1. ' → allowed: {0..7} → pick randomly 2 → pool → {0,1,3,4,5,6,7,8,9}
  2. ` → reduce {2} from allowed {1..9} → pick randomly 5 → pool → {0,1,3,4,6,7,8,9}
  3. [ → reduce {2,5} from allowed {1..9} → pick randomly 4 → pool → {0,1,3,6,7,8,9}
  4. \ → reduce {2,4,5} from allowed {1..9} → pick randomly 3 → pool → {0,1,6,7,8,9}
  5. ! → reduce {2,3,4,5} from allowed {0..9} → pick randomly 6 → pool → {0,1,7,8,9}
  6. " → reduce {2,3,4,5,6} from allowed {0..9} → pick randomly 7 → pool → {0,1,8,9}
  7. @ → reduce {2,3,4,5,6,7} from allowed {0..9} → pick randomly 1 → pool → {0,8,9}

Resulting first mapping attempt:
  ' = 2
  ` = 5
  [ = 4
  \ = 3
  ! = 6
  " = 7
  @ = 1

</digit_mapping_1>
