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
