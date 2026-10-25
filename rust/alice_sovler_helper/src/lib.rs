use pyo3::prelude::*;

const MODE_STANDARD: u8 = 0;
const MODE_ALICE: u8 = 1;
const MODE_LITTLE_ENDIAN: u8 = 2;

// ════════════════════ Seeded PRNG (xorshift64) ════════════════════

/// Minimal xorshift64 PRNG — no external crate needed.
/// Produces a non-zero state; if seed is 0 we use a fixed non-zero constant.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift requires non-zero state; use a mixing constant as fallback.
        let state = if seed == 0 { 0x853C_49E6_748F_EA9B } else { seed };
        Self { state }
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Fisher–Yates in-place shuffle of a slice.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ════════════════════ Operations ════════════════════

#[inline(always)]
fn op_add(a: i32, b: i32) -> Option<i32> {
    Some(a + b)
}
#[inline(always)]
fn op_sub(a: i32, b: i32) -> Option<i32> {
    if a >= b { Some(a - b) } else { None }
}
#[inline(always)]
fn op_rsub(a: i32, b: i32) -> Option<i32> {
    if b >= a { Some(b - a) } else { None }
}
#[inline(always)]
fn op_absdiff(a: i32, b: i32) -> Option<i32> {
    Some((a - b).abs())
}
#[inline(always)]
fn op_mul(a: i32, b: i32) -> Option<i32> {
    Some(a * b)
}
#[inline(always)]
fn op_gcd(a: i32, b: i32) -> Option<i32> {
    Some(gcd(a, b))
}
#[inline(always)]
fn op_lcm(a: i32, b: i32) -> Option<i32> {
    if a == 0 || b == 0 {
        Some(0)
    } else {
        Some(a * b / gcd(a, b))
    }
}
#[inline(always)]
fn op_fdiv(a: i32, b: i32) -> Option<i32> {
    if b != 0 { Some(a / b) } else { None }
}
#[inline(always)]
fn op_rdiv(a: i32, b: i32) -> Option<i32> {
    if a != 0 { Some(b / a) } else { None }
}
#[inline(always)]
fn op_mod(a: i32, b: i32) -> Option<i32> {
    if b != 0 { Some(a % b) } else { None }
}
#[inline(always)]
fn op_rmod(a: i32, b: i32) -> Option<i32> {
    if a != 0 { Some(b % a) } else { None }
}
#[inline(always)]
fn op_min(a: i32, b: i32) -> Option<i32> {
    Some(if a < b { a } else { b })
}
#[inline(always)]
fn op_max(a: i32, b: i32) -> Option<i32> {
    Some(if a > b { a } else { b })
}
#[inline(always)]
fn op_xor(a: i32, b: i32) -> Option<i32> {
    Some(a ^ b)
}
#[inline(always)]
fn op_band(a: i32, b: i32) -> Option<i32> {
    Some(a & b)
}
#[inline(always)]
fn op_bor(a: i32, b: i32) -> Option<i32> {
    Some(a | b)
}

#[inline(always)]
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

#[inline(always)]
fn apply(op_idx: u8, a: i32, b: i32) -> Option<i32> {
    match op_idx {
        0 => op_add(a, b),
        1 => op_sub(a, b),
        2 => op_rsub(a, b),
        3 => op_absdiff(a, b),
        4 => op_absdiff(a, b), // neg_absdiff same math, sign handled outside
        5 => op_mul(a, b),
        6 => op_gcd(a, b),
        7 => op_lcm(a, b),
        8 => op_fdiv(a, b),
        9 => op_rdiv(a, b),
        10 => op_mod(a, b),
        11 => op_rmod(a, b),
        12 => op_min(a, b),
        13 => op_max(a, b),
        14 => nz(op_add(a, b)? - 1),
        15 => nz(op_add(a, b)? + 1),
        16 => nz(op_mul(a, b)? - 1),
        17 => nz(op_mul(a, b)? + 1),
        18 => nz(op_absdiff(a, b)? - 1),
        19 => nz(op_absdiff(a, b)? + 1),
        20 => nz(op_sub(a, b)? - 1),
        21 => nz(op_sub(a, b)? + 1),
        22 => nz(op_rsub(a, b)? - 1),
        23 => nz(op_rsub(a, b)? + 1),
        24 => nz(op_add(a, b)? - 2),
        25 => nz(op_add(a, b)? + 2),
        26 => nz(op_mul(a, b)? - 2),
        27 => nz(op_mul(a, b)? + 2),
        28 => nz(op_absdiff(a, b)? - 2),
        29 => nz(op_absdiff(a, b)? + 2),
        30 => op_mul(a, b).map(|v| v / 2),
        31 => op_mul(a, b).map(|v| v * 2),
        32 => op_sub(a, b).map(|v| v * v),
        33 => op_add(a, b).map(|v| v * v),
        34 => op_mul(a, b).map(|v| v + a),
        35 => op_mul(a, b).map(|v| v + b),
        36 => nz(op_mul(a, b)? - a),
        37 => nz(op_mul(a, b)? - b),
        38 => Some(a * a + b),
        39 => Some(a + b * b),
        40 => op_xor(a, b),
        41 => op_band(a, b),
        42 => op_bor(a, b),
        // Signed subtractions: return signed value; caller compares sign vs has_sign.
        43 => Some(a - b),
        44 => Some(b - a),
        _ => None,
    }
}

#[inline(always)]
fn is_signed_op(idx: u8) -> bool {
    idx == 43 || idx == 44
}

#[inline(always)]
fn is_concat_op(idx: u8) -> bool {
    idx == 45 || idx == 46
}

#[inline(always)]
fn nz(v: i32) -> Option<i32> {
    if v >= 0 { Some(v) } else { None }
}

// ════════════════════ Data structures ════════════════════

struct EqData {
    l0: u8,
    l1: u8,
    r0: u8,
    r1: u8,
    res_len: u8,
    has_sign: bool,
    res_syms: Vec<u8>,
}

struct OpGroup {
    candidates: Vec<u8>,
    eqs: Vec<EqData>,
}

// ════════════════════ Search ════════════════════

#[inline(always)]
fn pow_i32(base: i32, exp: usize) -> i32 {
    let mut out = 1i32;
    for _ in 0..exp {
        out *= base;
    }
    out
}

#[inline(always)]
fn eq_is_assigned(eq: &EqData, assigned: &[bool]) -> bool {
    if !assigned[eq.l0 as usize]
        || !assigned[eq.l1 as usize]
        || !assigned[eq.r0 as usize]
        || !assigned[eq.r1 as usize]
    {
        return false;
    }
    for &ri in &eq.res_syms {
        if !assigned[ri as usize] {
            return false;
        }
    }
    true
}

#[inline(always)]
fn candidate_matches_eq(cand: u8, eq: &EqData, perm: &[u8], mode: u8, base: i32) -> bool {
    if is_concat_op(cand) {
        if eq.has_sign || eq.res_syms.len() != 4 {
            return false;
        }
        if cand == 45 {
            return eq.res_syms[0] == eq.l0
                && eq.res_syms[1] == eq.l1
                && eq.res_syms[2] == eq.r0
                && eq.res_syms[3] == eq.r1;
        }
        return eq.res_syms[0] == eq.r0
            && eq.res_syms[1] == eq.r1
            && eq.res_syms[2] == eq.l0
            && eq.res_syms[3] == eq.l1;
    }
    let signed_op = is_signed_op(cand);
    let l = perm[eq.l0 as usize] as i32 * base + perm[eq.l1 as usize] as i32;
    let r = perm[eq.r0 as usize] as i32 * base + perm[eq.r1 as usize] as i32;
    let (mut lv, mut rv) = (l, r);
    if mode == MODE_ALICE || mode == MODE_LITTLE_ENDIAN {
        lv = (lv % base) * base + lv / base;
        rv = (rv % base) * base + rv / base;
    }
    let raw = match apply(cand, lv, rv) {
        Some(x) => x,
        None => return false,
    };
    let mut v = if signed_op {
        // Sign of raw result must match whether rhs has a prefix.
        if (raw < 0) != eq.has_sign {
            return false;
        }
        raw.abs()
    } else {
        if raw < 0 {
            return false;
        }
        raw
    };
    let rl = eq.res_len as usize;
    // len(base-N digits of v) > rl  <=>  v >= base^rl
    if v >= pow_i32(base, rl) {
        return false;
    }
    if mode == MODE_ALICE || mode == MODE_LITTLE_ENDIAN {
        // equivalent to: reverse base-N digits after zfill(rl)
        let mut rev = 0i32;
        let mut x = v;
        for _ in 0..rl {
            rev = rev * base + x % base;
            x /= base;
        }
        v = rev;
    }
    // Parse expected value from symbols as base-N digits.
    let mut exp = 0i32;
    for &ri in &eq.res_syms {
        exp = exp * base + perm[ri as usize] as i32;
    }
    v == exp
}

#[inline(always)]
fn group_has_candidate_for_assigned_eqs(
    group: &OpGroup,
    perm: &[u8],
    mode: u8,
    base: i32,
    assigned: &[bool],
) -> bool {
    'cand: for &cand in &group.candidates {
        for eq in &group.eqs {
            if eq_is_assigned(eq, assigned) && !candidate_matches_eq(cand, eq, perm, mode, base) {
                continue 'cand;
            }
        }
        return true;
    }
    false
}

fn search(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
    max_solutions: usize,
    randomize: bool,
    seed: u64,
) -> Vec<(Vec<u8>, Vec<Vec<u8>>)> {
    let mut perm = vec![0u8; n];
    let mut used = vec![false; digit_count];
    let mut assigned = vec![false; n];

    // Build symbol-assignment order and digit-iteration order.
    // When randomize=true, shuffle both using the seeded PRNG instead of the
    // default weight-priority symbol ordering.
    let mut rng = Xorshift64::new(seed);
    let var_order = if randomize {
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        order
    } else {
        variable_order(groups, n)
    };
    let digit_order: Vec<u8> = if randomize {
        let mut order: Vec<u8> = (0..digit_count as u8).collect();
        rng.shuffle(&mut order);
        order
    } else {
        (0..digit_count as u8).collect()
    };

    let base = radix as i32;
    let mut solutions: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();

    fn dfs(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        digit_order: &[u8],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        solutions: &mut Vec<(Vec<u8>, Vec<Vec<u8>>)>,
        max_solutions: usize,
    ) -> bool {
        if depth == n {
            // Verify all groups and collect valid ops.
            let mut ops_valid = vec![Vec::new(); groups.len()];
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    return false;
                }
                ops_valid[gi] = valid;
            }
            // Record this solution; signal quota-reached so the caller can stop early.
            solutions.push((perm.to_vec(), ops_valid));
            return solutions.len() >= max_solutions;
        }

        let sym_idx = var_order[depth];
        for &d in digit_order {
            if used[d as usize] {
                continue;
            }
            used[d as usize] = true;
            assigned[sym_idx] = true;
            perm[sym_idx] = d;
            let mut still_possible = true;
            for g in groups {
                if !group_has_candidate_for_assigned_eqs(g, perm, mode, base, assigned) {
                    still_possible = false;
                    break;
                }
            }
            if !still_possible {
                assigned[sym_idx] = false;
                used[d as usize] = false;
                continue;
            }
            // Propagate quota-reached signal upward to unwind early.
            if dfs(
                depth + 1,
                n,
                perm,
                used,
                assigned,
                var_order,
                digit_order,
                groups,
                mode,
                base,
                solutions,
                max_solutions,
            ) {
                assigned[sym_idx] = false;
                used[d as usize] = false;
                return true;
            }
            assigned[sym_idx] = false;
            used[d as usize] = false;
        }
        false
    }

    dfs(
        0,
        n,
        &mut perm,
        &mut used,
        &mut assigned,
        &var_order,
        &digit_order,
        groups,
        mode,
        base,
        &mut solutions,
        max_solutions,
    );
    solutions
}

fn variable_order(groups: &[OpGroup], n: usize) -> Vec<usize> {
    let mut score = vec![0usize; n];
    for g in groups {
        let group_weight = 1 + g.candidates.len();
        for eq in &g.eqs {
            let syms = [eq.l0, eq.l1, eq.r0, eq.r1];
            for &s in &syms {
                score[s as usize] += group_weight;
            }
            for &s in &eq.res_syms {
                score[s as usize] += group_weight;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(score[i]), i));
    order
}

/// variable_order_with_scores — same as variable_order but also returns per-symbol scores.
/// Returns (ordered symbol indices, score vector indexed by symbol slot).
fn variable_order_with_scores(groups: &[OpGroup], n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut score = vec![0usize; n];
    for g in groups {
        let group_weight = 1 + g.candidates.len();
        for eq in &g.eqs {
            for &s in &[eq.l0, eq.l1, eq.r0, eq.r1] {
                score[s as usize] += group_weight;
            }
            for &s in &eq.res_syms {
                score[s as usize] += group_weight;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(score[i]), i));
    (order, score)
}

/// group_surviving_candidates — candidate-list–returning counterpart to
/// group_has_candidate_for_assigned_eqs.  Returns the subset of candidate indices
/// that still satisfy every fully-assigned equation in the group.
fn group_surviving_candidates(
    group: &OpGroup,
    perm: &[u8],
    mode: u8,
    base: i32,
    assigned: &[bool],
) -> Vec<u8> {
    let mut survivors = Vec::new();
    'cand: for &cand in &group.candidates {
        for eq in &group.eqs {
            if eq_is_assigned(eq, assigned) && !candidate_matches_eq(cand, eq, perm, mode, base) {
                continue 'cand;
            }
        }
        survivors.push(cand);
    }
    survivors
}

// ════════════════════ Instrumented search ════════════════════

#[derive(Default, Clone)]
struct SearchStats {
    dfs_calls: usize,
    pruned: usize,           // digit filtered by still_possible (constraint propagation)
    recursed_into: usize,    // digit passed still_possible AND we entered recursion
    recursion_failed: usize, // recursion returned false (== backtrack count, including via leaf failure)
    leaf_failed: usize,      // depth==n verification failed
    max_branch_attempted: usize, // max children we attempted to recurse into at any single depth
}

fn search_with_stats(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
) -> (Option<(Vec<u8>, Vec<Vec<u8>>)>, SearchStats) {
    let mut perm = vec![0u8; n];
    let mut used = vec![false; digit_count];
    let mut assigned = vec![false; n];
    let mut ops_valid = vec![Vec::new(); groups.len()];
    let var_order = variable_order(groups, n);
    let base = radix as i32;
    let mut stats = SearchStats::default();

    fn dfs_t(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        ops_valid: &mut [Vec<u8>],
        stats: &mut SearchStats,
    ) -> bool {
        stats.dfs_calls += 1;
        if depth == n {
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    stats.leaf_failed += 1;
                    return false;
                }
                ops_valid[gi] = valid;
            }
            return true;
        }

        let sym_idx = var_order[depth];
        let mut attempted_here: usize = 0;
        for d in 0..used.len() as u8 {
            if used[d as usize] {
                continue;
            }
            used[d as usize] = true;
            assigned[sym_idx] = true;
            perm[sym_idx] = d;
            let mut still_possible = true;
            for g in groups {
                if !group_has_candidate_for_assigned_eqs(g, perm, mode, base, assigned) {
                    still_possible = false;
                    break;
                }
            }
            if !still_possible {
                stats.pruned += 1;
                assigned[sym_idx] = false;
                used[d as usize] = false;
                continue;
            }
            attempted_here += 1;
            stats.recursed_into += 1;
            if dfs_t(
                depth + 1,
                n,
                perm,
                used,
                assigned,
                var_order,
                groups,
                mode,
                base,
                ops_valid,
                stats,
            ) {
                if attempted_here > stats.max_branch_attempted {
                    stats.max_branch_attempted = attempted_here;
                }
                return true;
            }
            stats.recursion_failed += 1;
            assigned[sym_idx] = false;
            used[d as usize] = false;
        }
        false
    }

    let ok = dfs_t(
        0,
        n,
        &mut perm,
        &mut used,
        &mut assigned,
        &var_order,
        groups,
        mode,
        base,
        &mut ops_valid,
        &mut stats,
    );
    if ok {
        (Some((perm, ops_valid)), stats)
    } else {
        (None, stats)
    }
}

// ════════════════════ Traced search ════════════════════

/// Spine entry for one symbol in the accepted assignment chain.
/// `(sym_idx, accepted_digit, pruned, backtracked)`
///   pruned:      digits ruled out by forward-checking at this depth, with the first
///                group index that lost all candidates: `(digit, group_idx)`.
///   backtracked: digits that passed forward-checking but whose entire subtree failed.
type SpineEntry = (u8, u8, Vec<(u8, i32)>, Vec<u8>);

/// search_traced — returns all solutions (like search) PLUS a compact spine describing
/// the path the DFS took to reach the *first* solution.
///
/// The spine has one entry per symbol in assignment order (most-constrained first).
/// Each entry records which digits were rejected before the accepted one and why,
/// keeping the description to O(n × digit_count) entries regardless of tree size.
///
/// Always uses canonical weight-priority variable ordering (no randomise option).
fn search_traced(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
    max_solutions: usize,
) -> (Vec<(Vec<u8>, Vec<Vec<u8>>)>, Vec<SpineEntry>) {
    let var_order = variable_order(groups, n);
    let digit_order: Vec<u8> = (0..digit_count as u8).collect();
    let base = radix as i32;

    // --- Step 1: run DFS limited to 1 solution to build the spine ---
    let mut perm  = vec![0u8; n];
    let mut used  = vec![false; digit_count];
    let mut assigned = vec![false; n];
    let mut spine_sols: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();
    let mut spine: Vec<SpineEntry> = Vec::new();

    fn dfs_spine(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        digit_order: &[u8],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        solutions: &mut Vec<(Vec<u8>, Vec<Vec<u8>>)>,
        spine: &mut Vec<SpineEntry>,
    ) -> bool {
        if depth == n {
            // Leaf: verify all groups.
            let mut ops_valid = vec![Vec::new(); groups.len()];
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    let _ = gi; // suppress unused warning; not recorded in spine leaf
                    return false;
                }
                ops_valid[gi] = valid;
            }
            solutions.push((perm.to_vec(), ops_valid));
            return true; // stop after first solution
        }

        let sym_idx = var_order[depth];
        let mut pruned_here:      Vec<(u8, i32)> = Vec::new();
        let mut backtracked_here: Vec<u8>         = Vec::new();

        for &d in digit_order {
            if used[d as usize] { continue; }
            used[d as usize]  = true;
            assigned[sym_idx] = true;
            perm[sym_idx]     = d;

            let mut still_possible = true;
            let mut failed_group: i32 = -1;
            for (gi, g) in groups.iter().enumerate() {
                if !group_has_candidate_for_assigned_eqs(g, perm, mode, base, assigned) {
                    still_possible = false;
                    failed_group   = gi as i32;
                    break;
                }
            }

            if !still_possible {
                pruned_here.push((d, failed_group));
                assigned[sym_idx]  = false;
                used[d as usize]   = false;
                continue;
            }

            if dfs_spine(depth + 1, n, perm, used, assigned, var_order, digit_order,
                         groups, mode, base, solutions, spine) {
                // This digit leads to the first solution — record spine entry and unwind.
                spine.push((sym_idx as u8, d, pruned_here, backtracked_here));
                assigned[sym_idx] = false;
                used[d as usize]  = false;
                return true;
            }

            // Subtree failed — record as dead end.
            backtracked_here.push(d);
            assigned[sym_idx] = false;
            used[d as usize]  = false;
        }
        false
    }

    dfs_spine(0, n, &mut perm, &mut used, &mut assigned,
              &var_order, &digit_order, groups, mode, base,
              &mut spine_sols, &mut spine);

    // Spine accumulates deepest-first during unwind; reverse to assignment order.
    spine.reverse();

    if spine_sols.is_empty() {
        // No solution — return empty solutions and empty spine.
        return (Vec::new(), spine);
    }

    // --- Step 2: collect up to max_solutions via the regular (fast) search ---
    if max_solutions <= 1 {
        return (spine_sols, spine);
    }
    let all_solutions = search(groups, n, mode, radix, digit_count, max_solutions, false, 0);
    (all_solutions, spine)
}

// ════════════════════ Full-trace search ════════════════════

/// search_full_trace — DFS limited to the first solution, emitting one event per
/// (sym_idx, digit) trial in chronological (pre-order) order.
///
/// Each event: (depth, sym_idx, digit, passed, failed_group, surviving_per_group, on_solution_path)
///   depth              — DFS depth (0 = first symbol assigned).
///   sym_idx            — symbol index being assigned.
///   digit              — digit value tried.
///   passed             — true when the forward-check found a surviving candidate for every group.
///   failed_group       — index of the first group that lost all candidates when passed=false; -1 when passed=true.
///   surviving_per_group — per-group surviving candidate indices; groups 0..failed_group have real values,
///                         groups failed_group..len() have empty vecs when passed=false.
///   on_solution_path   — true for the unique node chain from root to the first solution leaf
///                        (set during DFS unwind).
///
/// Recording stops when events.len() reaches max_events (truncated=true), but the DFS
/// itself always runs to the first solution.  total_trials counts every (sym_idx, digit)
/// trial regardless of the recording cap.
///
/// Returns:
///   (solution_opt, var_order_scores, events, total_trials, truncated)
fn search_full_trace(
    groups: &[OpGroup],
    n: usize,
    mode: u8,
    radix: usize,
    digit_count: usize,
    max_events: usize,
) -> (
    Option<(Vec<u8>, Vec<Vec<u8>>)>,
    Vec<(u8, usize)>,
    Vec<(u8, u8, u8, bool, i32, Vec<Vec<u8>>, bool)>,
    usize,
    bool,
) {
    let (var_order, scores) = variable_order_with_scores(groups, n);
    let var_order_scores: Vec<(u8, usize)> = var_order.iter().map(|&i| (i as u8, scores[i])).collect();
    let digit_order: Vec<u8> = (0..digit_count as u8).collect();
    let base = radix as i32;

    let mut perm     = vec![0u8; n];
    let mut used     = vec![false; digit_count];
    let mut assigned = vec![false; n];
    let mut events: Vec<(u8, u8, u8, bool, i32, Vec<Vec<u8>>, bool)> = Vec::new();
    let mut truncated    = false;
    let mut total_trials = 0usize;

    fn dfs_full(
        depth: usize,
        n: usize,
        perm: &mut [u8],
        used: &mut [bool],
        assigned: &mut [bool],
        var_order: &[usize],
        digit_order: &[u8],
        groups: &[OpGroup],
        mode: u8,
        base: i32,
        events: &mut Vec<(u8, u8, u8, bool, i32, Vec<Vec<u8>>, bool)>,
        max_events: usize,
        truncated: &mut bool,
        total_trials: &mut usize,
    ) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
        if depth == n {
            // Leaf: verify all groups.
            let mut ops_valid = vec![Vec::new(); groups.len()];
            for (gi, g) in groups.iter().enumerate() {
                let mut valid = Vec::new();
                'cand: for &cand in &g.candidates {
                    for eq in &g.eqs {
                        if !candidate_matches_eq(cand, eq, perm, mode, base) {
                            continue 'cand;
                        }
                    }
                    valid.push(cand);
                }
                if valid.is_empty() {
                    return None;
                }
                ops_valid[gi] = valid;
            }
            return Some((perm.to_vec(), ops_valid));
        }

        let sym_idx = var_order[depth];

        for &d in digit_order {
            if used[d as usize] { continue; }

            used[d as usize]  = true;
            assigned[sym_idx] = true;
            perm[sym_idx]     = d;

            // Forward-check: collect surviving candidates per group.
            let mut still_possible = true;
            let mut failed_group: i32 = -1;
            let mut surviving_per_group: Vec<Vec<u8>> = Vec::with_capacity(groups.len());

            for (gi, g) in groups.iter().enumerate() {
                let survivors = group_surviving_candidates(g, perm, mode, base, assigned);
                if survivors.is_empty() {
                    still_possible = false;
                    failed_group   = gi as i32;
                    // Remaining groups are not evaluated; fill with empty vecs.
                    for _ in gi..groups.len() {
                        surviving_per_group.push(Vec::new());
                    }
                    break;
                }
                surviving_per_group.push(survivors);
            }

            // Always count this trial.
            *total_trials += 1;

            // Record event if under cap.
            let event_idx: Option<usize> = if events.len() < max_events {
                let idx = events.len();
                events.push((
                    depth as u8,
                    sym_idx as u8,
                    d,
                    still_possible,
                    failed_group,
                    surviving_per_group,
                    false, // on_solution_path — updated on unwind
                ));
                Some(idx)
            } else {
                *truncated = true;
                None
            };

            if still_possible {
                let result = dfs_full(
                    depth + 1, n, perm, used, assigned,
                    var_order, digit_order, groups, mode, base,
                    events, max_events, truncated, total_trials,
                );
                if let Some(sol) = result {
                    if let Some(idx) = event_idx {
                        events[idx].6 = true; // mark as on-solution-path
                    }
                    assigned[sym_idx] = false;
                    used[d as usize]  = false;
                    return Some(sol);
                }
            }

            assigned[sym_idx] = false;
            used[d as usize]  = false;
        }
        None
    }

    let solution = dfs_full(
        0, n, &mut perm, &mut used, &mut assigned,
        &var_order, &digit_order, groups, mode, base,
        &mut events, max_events, &mut truncated, &mut total_trials,
    );

    (solution, var_order_scores, events, total_trials, truncated)
}

// ════════════════════ Python binding ════════════════════

/// arithmetic_search(n, alice, groups, radix=None, digit_count=None, mode=None,
///                   max_solutions=1, randomize=False, seed=0)
///   -> list of (perm, ops_valid)   (empty list when no solution)
/// groups: list of (candidates, equations)
///   candidates: list of u8 op indices
///   equations: list of [l0, l1, r0, r1, res_len, r0, r1, ...]
/// max_solutions: stop after collecting this many distinct digit assignments (default 1).
/// randomize: when True, shuffle symbol-assignment order and digit-iteration order
///            using a seeded xorshift64 PRNG instead of the weight-priority default.
/// seed: seed for the PRNG (used only when randomize=True).
#[pyfunction]
#[pyo3(signature = (n, alice, groups, radix=None, digit_count=None, mode=None, max_solutions=None, randomize=None, seed=None))]
fn arithmetic_search(
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
    max_solutions: Option<usize>,
    randomize: Option<bool>,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<u8>, Vec<Vec<u8>>)>> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup {
                candidates: cands,
                eqs,
            }
        })
        .collect();

    Ok(search(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
        max_solutions.unwrap_or(1),
        randomize.unwrap_or(false),
        seed.unwrap_or(0),
    ))
}

/// arithmetic_search_with_stats — same as arithmetic_search but also returns DFS stats.
/// Returns (perm_or_none, stats_dict). On success perm is Some((perm, ops_valid)).
/// stats_dict keys: dfs_calls, pruned, recursed_into, recursion_failed, leaf_failed, max_branch_attempted
#[pyfunction]
fn arithmetic_search_with_stats<'py>(
    py: Python<'py>,
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
) -> PyResult<(Option<(Vec<u8>, Vec<Vec<u8>>)>, Bound<'py, pyo3::types::PyDict>)> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup {
                candidates: cands,
                eqs,
            }
        })
        .collect();

    let (result, stats) = search_with_stats(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
    );
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("dfs_calls", stats.dfs_calls)?;
    dict.set_item("pruned", stats.pruned)?;
    dict.set_item("recursed_into", stats.recursed_into)?;
    dict.set_item("recursion_failed", stats.recursion_failed)?;
    dict.set_item("leaf_failed", stats.leaf_failed)?;
    dict.set_item("max_branch_attempted", stats.max_branch_attempted)?;
    Ok((result, dict))
}

/// arithmetic_search_traced(n, alice, groups, radix=None, digit_count=None, mode=None,
///                          max_solutions=1)
///   -> (solutions, spine)
///   solutions: list of (perm, ops_valid)   — same format as arithmetic_search
///   spine:     list of (sym_idx, accepted_digit, pruned, backtracked) — one entry per
///              symbol in the accepted assignment chain (first solution only).
///              pruned      = [(digit, group_idx), ...]  — forward-check rejections
///              backtracked = [digit, ...]               — dead-end subtrees
///              Empty when no solution is found.
///
/// Always uses canonical weight-priority variable ordering (randomize not supported).
#[pyfunction]
#[pyo3(signature = (n, alice, groups, radix=None, digit_count=None, mode=None, max_solutions=None))]
fn arithmetic_search_traced(
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
    max_solutions: Option<usize>,
) -> PyResult<(Vec<(Vec<u8>, Vec<Vec<u8>>)>, Vec<(u8, u8, Vec<(u8, i32)>, Vec<u8>)>)> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup {
                candidates: cands,
                eqs,
            }
        })
        .collect();

    Ok(search_traced(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
        max_solutions.unwrap_or(1),
    ))
}

/// arithmetic_search_full_trace(n, alice, groups, radix=None, digit_count=None, mode=None,
///                              max_events=0)
///   -> (solution_opt, var_order_scores, events, digit_mappings_analyzed, truncated)
///
/// solution_opt: (perm, ops_valid) as in arithmetic_search, or None when no solution found.
/// var_order_scores: [(sym_idx, score)] in DFS assignment order (most-constrained first).
/// events: chronological list of (depth, sym_idx, digit, passed, failed_group,
///         surviving_per_group, on_solution_path):
///   depth               — DFS depth (0 = first symbol).
///   sym_idx             — symbol index assigned.
///   digit               — digit value tried.
///   passed              — forward-check passed (all groups still have survivors).
///   failed_group        — first group index that lost all candidates; -1 when passed=true.
///   surviving_per_group — per-group surviving candidate indices (i32 list of lists).
///                         Groups 0..failed_group have real values; the rest are empty
///                         when passed=false.
///   on_solution_path    — true for the accepted digit chain from root to first solution.
/// digit_mappings_analyzed: total (sym_idx, digit) trials including those not recorded.
/// truncated: true when event recording was capped at max_events.
///
/// max_events=0 (default) means unlimited recording.
/// Always uses canonical weight-priority variable ordering (randomize not supported).
#[pyfunction]
#[pyo3(signature = (n, alice, groups, radix=None, digit_count=None, mode=None, max_events=None))]
fn arithmetic_search_full_trace(
    n: usize,
    alice: bool,
    groups: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    radix: Option<usize>,
    digit_count: Option<usize>,
    mode: Option<u8>,
    max_events: Option<usize>,
) -> PyResult<(
    Option<(Vec<u8>, Vec<Vec<u8>>)>,
    Vec<(u8, usize)>,
    Vec<(u8, u8, u8, bool, i32, Vec<Vec<u8>>, bool)>,
    usize,
    bool,
)> {
    let rust_groups: Vec<OpGroup> = groups
        .into_iter()
        .map(|(cands, eqs)| {
            let eqs = eqs
                .into_iter()
                .map(|e| EqData {
                    l0: e[0],
                    l1: e[1],
                    r0: e[2],
                    r1: e[3],
                    res_len: e[4],
                    has_sign: e[5] != 0,
                    res_syms: e[6..].to_vec(),
                })
                .collect();
            OpGroup { candidates: cands, eqs }
        })
        .collect();

    // 0 means unlimited; map to usize::MAX internally.
    let cap = match max_events.unwrap_or(0) {
        0 => usize::MAX,
        v => v,
    };

    Ok(search_full_trace(
        &rust_groups,
        n,
        mode.unwrap_or(if alice { MODE_ALICE } else { MODE_STANDARD }),
        radix.unwrap_or(n),
        digit_count.unwrap_or(n),
        cap,
    ))
}

#[pymodule]
fn alice_sovler_helper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(arithmetic_search, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic_search_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic_search_traced, m)?)?;
    m.add_function(wrap_pyfunction!(arithmetic_search_full_trace, m)?)?;
    Ok(())
}
