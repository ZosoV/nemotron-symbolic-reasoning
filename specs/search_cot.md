[Candidate setup and search strategy]
Operator ''': candidates prioritised by glyph → sign_pattern_filter (22→21) → 21 candidate(s) remaining: [add, absdiff, mul, sub_signed, rsub_signed…].
Operator '+': candidates prioritised by glyph → sign_pattern_filter (22→3) → 3 candidate(s) remaining: [sub_signed, rsub_signed, neg_absdiff].
Operator '-': candidates prioritised by glyph → sign_pattern_filter (22→21) → 21 candidate(s) remaining: [absdiff, sub_signed, rsub_signed, absdiff_m1, absdiff_p1…].
Operator ordering (MRV heuristic #5, fewest candidates first): '+' (3 cands / 2 eqs); ''' (20 cands / 2 eqs); '-' (20 cands / 1 eq).
Searching tier 'full', standard (big-endian), base 10:
  → 1 solution(s) found.

[Digit-assignment DFS]
  Symbol assignment order (most-constrained first, variable-ordering heuristic): | → } → < → % → ? → ( → @ → / → : → ^.

  Symbol '|' — digit trials:
    try '|'=0 → explored; subtree yielded no solution (backtrack).
    try '|'=1 → explored; subtree yielded no solution (backtrack).
    try '|'=2 → explored; subtree yielded no solution (backtrack).
    try '|'=3 → explored; subtree yielded no solution (backtrack).
    try '|'=4 → accepted ✓

    Symbol '}' — digit trials:
      try '}'=0 → explored; subtree yielded no solution (backtrack).
      try '}'=1 → accepted ✓

      Symbol '<' — digit trials:
        try '<'=0 → explored; subtree yielded no solution (backtrack).
        try '<'=2 → explored; subtree yielded no solution (backtrack).
        try '<'=3 → explored; subtree yielded no solution (backtrack).
        try '<'=5 → explored; subtree yielded no solution (backtrack).
        try '<'=6 → accepted ✓

        Symbol '%' — digit trials:
          try '%'=0 → explored; subtree yielded no solution (backtrack).
          try '%'=2 → explored; subtree yielded no solution (backtrack).
          try '%'=3 → explored; subtree yielded no solution (backtrack).
          try '%'=5 → explored; subtree yielded no solution (backtrack).
          try '%'=7 → accepted ✓

          Symbol '?' — digit trials:
            try '?'=0 → explored; subtree yielded no solution (backtrack).
            try '?'=2 → explored; subtree yielded no solution (backtrack).
            try '?'=3 → accepted ✓

            Symbol '(' — digit trials:
              try '('=0 → pruned: operator ''' has no surviving candidates (forward-checking).
              try '('=2 → pruned: operator ''' has no surviving candidates (forward-checking).
              try '('=5 → accepted ✓

              Symbol '@' — digit trials:
                try '@'=0 → accepted ✓

                Symbol '/' — digit trials:
                  try '/'=2 → accepted ✓

                  Symbol ':' — digit trials:
                    try ':'=8 → explored; subtree yielded no solution (backtrack).
                    try ':'=9 → accepted ✓

                    Symbol '^' — digit trials:
                      try '^'=8 → accepted ✓

  [All 10 symbols assigned — solution accepted.]

[Accepted candidates per operator]
  Operator '+' — 2 accepted candidate(s): rsub_signed (signed reverse subtraction (b − a, may be negative)), neg_absdiff (negated absolute difference (−|a − b|)).
    eq[1]: '|? + ?@ = −}?'  L=43, R=30, expected=-13 [signed, magnitude 13]
           rsub_signed=-13✓ ; neg_absdiff=-13✓
    eq[2]: '%< + (} = −/('  L=76, R=51, expected=-25 [signed, magnitude 25]
           rsub_signed=-25✓ ; neg_absdiff=-25✓

  Operator ''' — 2 accepted candidate(s): mul (multiplication (a × b)), lcm (least common multiple lcm(a, b)).
    eq[1]: '|? ' |< = }:%^'  L=43, R=46, expected=1978
           mul=1978✓ ; lcm=1978✓
    eq[2]: '(} ' }| = %}|'  L=51, R=14, expected=714
           mul=714✓ ; lcm=714✓

  Operator '-' — 1 accepted candidate(s): add (addition (a + b)).
    eq[1]: '<| - <@ = }/|'  L=64, R=60, expected=124
           add=124✓


[Result]
Solution: digits ['@' = 0, '}' = 1, '/' = 2, '?' = 3, '|' = 4, '(' = 5, '<' = 6, '%' = 7, '^' = 8, ':' = 9]; operators: ['+' → signed reverse subtraction (b − a, may be negative); ''' → multiplication (a × b); '-' → addition (a + b)].
(Strategy: standard (big-endian), base 10, tier 'full'.)