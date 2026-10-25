Let's assume the following set of available operations that could govern the secret equations:

 1. mul(a, b) = a × b + offset, offset \in {−1,0,+1}
 2. add(a, b) = a + b + offset, offset \in {−1,0,+1,+2}
 3. absdiff(a, b) = |a − b| + offset, offset \in {−2,−1,0,+1}
 4. sub_signed(a, b) = a − b  (result may be negative; if so, the operator symbol is attached
    to the answer as a sign)
 5. rsub_signed(a, b) = b − a  (same sign convention as sub_signed)
 6. neg_absdiff(a, b) = −|a − b|  (always non-positive; the operator symbol is always
    attached to the answer)
 7. concat_fwd(a, b) = the symbols of a followed by the symbols of b  (structural, no
    arithmetic)
 8. concat_rev(a, b) = the symbols of b followed by the symbols of a  (structural, no
    arithmetic)
9. gcd(a, b) = greatest common divisor of a and b  (always non-negative)
10. lcm(a, b) = least common multiple of a and b  (0 if either operand is 0)
11. mod(a, b) = a mod b  (invalid when b = 0)
12. rmod(a, b) = b mod a  (invalid when a = 0)
13. fdiv(a, b) = ⌊a ÷ b⌋  (invalid when b = 0)


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

The goal is to find the digit mapping (symbol → digit) and operation mapping (op → function)
that satisfies the equation, then apply them to decode the query answer.
