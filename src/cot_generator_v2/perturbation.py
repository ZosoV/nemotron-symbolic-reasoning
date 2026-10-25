"""Backward perturbation chain for the digit-mapping attempts.

We already know the feasible digit mapping (the solution). To make the failed
attempts look like a solver homing in on the answer, we build them *backward*
from the solution by breaking it one small change (a perturbation) at a time:

  * swap     — exchange two symbols' digit values (the only option when the
               puzzle uses all `base` digits, a full permutation).
  * reassign — move one symbol onto a digit nobody currently uses (possible only
               when there are spare digits, i.e. fewer than `base` operand
               symbols).

The chain is `digit_mapping_1 -> ... -> digit_mapping_{R+1}` where the last is
the solution. Reading forward, each attempt fixes exactly one perturbation
(cumulative peeling). The break that turns mapping_k into the *more broken*
mapping is recorded so the follow-up reasoning (tells.py) can derive exactly the
fix that undoes it.

Construction is incremental (apply one break at a time to a working bijection),
so it is correct regardless of how many spare digits exist. Breaks touch
pairwise-disjoint symbol sets, so forward fixes are independent.

Run (self-test):
    uv run python src/cot_generator/perturbation.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cot_generator_v2.equations import Equation
from src.cot_generator_v2.families import derive_seed


@dataclass
class Perturb:
    """One break in the chain (and, read forward, the fix that undoes it)."""
    fix_index: int                 # mapping_k that this fix turns into mapping_{k+1}
    kind: str                      # 'swap' | 'reassign'
    symbols: tuple[str, ...]       # touched symbols (2 for swap, 1 for reassign)
    correct: dict[str, int]        # symbol -> solution digit (after the fix)
    broken: dict[str, int]         # symbol -> digit while the break is active
    coupling: int = 0              # # example equations the symbols appear in


@dataclass
class ChainStep:
    index: int                     # 1-based mapping index
    mapping: dict[str, int]
    unused: list[int]              # spare digits not assigned in this mapping
    fixes: list[Perturb]           # break(s) to undo to reach the next mapping
    active: list[Perturb]          # breaks still applied vs the solution
    fixes_into: Perturb | None = None  # the fix that produced THIS mapping (set by renderer)


@dataclass
class PerturbationChain:
    steps: list[ChainStep]         # steps[0]=mapping_1 ... steps[-1]=solution
    solution: dict[str, int]
    full_permutation: bool


def _eq_symbol_sets(examples: list[Equation]) -> list[set[str]]:
    return [set(eq.symbols) for eq in examples]


def _coupling(symbols, eq_sets) -> int:
    return sum(1 for s in eq_sets if s & set(symbols))


def _break_fails(symbols, broken, solution, examples, opfam, base, mode, verify) -> bool:
    """True iff applying just this break to the solution breaks some equation."""
    trial = dict(solution)
    trial.update(broken)
    return not verify(examples, opfam, trial, base, mode)


def build_chain(
    solution: dict[str, int],
    content_syms: list[str],
    examples: list[Equation],
    opfam: dict[str, tuple[str, int]],
    base: int,
    rounds: int,
    seed: int,
    problem_id: str,
    mode: str,
    break_order: list[int] | None = None,
) -> PerturbationChain:
    """Build the backward perturbation chain ending at `solution`.

    Aug 1: `break_order` is an optional permutation of indices into the
    candidate list (before coupling sort). When None (default), uses the
    standard coupling-sorted order. When provided, it selects and reorders
    the candidates according to the given permutation.
    """
    from src.cot_generator_v2.decode import verify_program

    rng = random.Random(derive_seed(seed, problem_id, "perturbation"))
    used = set(solution.values())
    spares = [d for d in range(base) if d not in used]
    full_perm = len(spares) == 0
    eq_sets = _eq_symbol_sets(examples)
    # eq_syms: symbols in at least one example equation.
    # Both 'a' and 'b' of a swap are restricted to eq_syms so neither member of a
    # swapped pair leaks a symbol invisible to the LLM (symbol-visibility rule).
    eq_syms = [s for s in content_syms if any(s in es for es in eq_sets)]

    candidates: list[Perturb] = []
    touched: set[str] = set()

    # 1) One reassign per spare digit (moves a symbol onto an unused digit).
    #    Reassigns require the symbol to be in an example equation (to produce a tell).
    spare_pool = list(spares)
    rng.shuffle(spare_pool)
    sym_pool = list(eq_syms)
    rng.shuffle(sym_pool)
    for spare in spare_pool:
        if len(candidates) >= rounds:
            break
        for s in sym_pool:
            if s in touched:
                continue
            broken = {s: spare}
            if _break_fails([s], broken, solution, examples, opfam, base, mode, verify_program):
                candidates.append(Perturb(
                    0, "reassign", (s,), {s: solution[s]}, broken,
                    _coupling([s], eq_sets),
                ))
                touched.add(s)
                break

    # 2) Swaps of disjoint symbol pairs (visibility-safe: both 'a' and 'b' from eq_syms).
    #    'a' must be in an example equation (so the swap breaks at least one eq).
    #    'b' is also restricted to example-visible symbols (symbol-visibility rule).
    #    _break_fails already checks that the swap actually breaks some example equation.
    pool_a = [s for s in eq_syms if s not in touched]
    pool_b_base = [s for s in eq_syms if s not in touched]  # visibility-safe: example-only
    rng.shuffle(pool_a)
    i = 0
    while len(candidates) < rounds and i < len(pool_a):
        a = pool_a[i]
        i += 1
        if a in touched:
            continue
        # Build pool_b excluding already-touched and a itself
        pool_b = [s for s in pool_b_base if s not in touched and s != a]
        rng.shuffle(pool_b)
        for b in pool_b:
            if solution[a] == solution[b]:
                continue
            broken = {a: solution[b], b: solution[a]}
            if _break_fails([a, b], broken, solution, examples, opfam, base, mode, verify_program):
                candidates.append(Perturb(
                    0, "swap", (a, b), {a: solution[a], b: solution[b]}, broken,
                    _coupling([a, b], eq_sets),
                ))
                touched.update((a, b))
                break

    # Aug 1: expose pre-sort candidates list, then apply ordering.
    # Default: coupling-sorted (messiest first, cleanest last).
    # When break_order is provided, reorder candidates accordingly.
    if break_order is not None:
        # break_order is a permutation of indices 0..len(candidates)-1
        valid = [i for i in break_order if i < len(candidates)]
        candidates = [candidates[i] for i in valid]
    else:
        candidates.sort(key=lambda p: p.coupling, reverse=True)
    R = len(candidates)
    for k, p in enumerate(candidates, start=1):
        p.fix_index = k

    # Build mappings backward: working starts at solution, apply breaks R..1.
    mappings: dict[int, dict[str, int]] = {R + 1: dict(solution)}
    working = dict(solution)
    for k in range(R, 0, -1):
        working = dict(working)
        working.update(candidates[k - 1].broken)
        mappings[k] = working

    steps: list[ChainStep] = []
    for k in range(1, R + 2):
        m = mappings[k]
        unused = sorted(set(range(base)) - set(m.values()))
        active = [p for p in candidates if p.fix_index >= k]   # still broken at mapping_k
        fixes = [p for p in candidates if p.fix_index == k]      # undo to reach k+1
        steps.append(ChainStep(index=k, mapping=m, unused=unused, fixes=fixes, active=active))

    return PerturbationChain(steps=steps, solution=dict(solution), full_permutation=full_perm)


def _selftest() -> None:
    import json
    import pandas as pd
    from src.cot_generator_v2.equations import parse_prompt, content_symbols
    from src.cot_generator_v2.decode import verify_program
    from src.cot_generator_v2.families import resolve_registry_op

    df = pd.read_parquet("inputs/data/result_examples_3_rand_k_6_max2.parquet")
    r = df[df["id"] == "00457d26_q0"].iloc[0]
    sol = json.loads(r["digits_mapping"])
    om = json.loads(r["ops_mapping"])
    opfam = {g: resolve_registry_op(n) for g, n in om.items()}
    base = int(round(float(r["solver_radix"])))
    mode = str(r["solver_strategy"]).strip().lower()
    examples, q = parse_prompt(r["prompt"], r["answer"])
    syms = content_symbols(examples)

    chain = build_chain(sol, syms, examples, opfam, base, rounds=4,
                        seed=0, problem_id="00457d26", mode=mode)
    # solution step passes; every earlier step is a valid bijection and fails.
    assert chain.steps[-1].mapping == sol
    assert verify_program(examples, opfam, chain.steps[-1].mapping, base, mode)
    for st in chain.steps[:-1]:
        assert len(set(st.mapping.values())) == len(st.mapping), "bijection broken"
        assert not verify_program(examples, opfam, st.mapping, base, mode), \
            f"mapping_{st.index} unexpectedly passes"
    # exactly one fix per non-final step (single-perturbation peeling)
    assert all(len(st.fixes) == 1 for st in chain.steps[:-1])
    print(f"perturbation self-test PASS  (chain length {len(chain.steps)}, spares={chain.steps[0].unused})")
    for st in chain.steps:
        fx = st.fixes[0] if st.fixes else None
        desc = f"{fx.kind} {fx.symbols}" if fx else "(solution)"
        print(f"  mapping_{st.index}: unused={st.unused} fix-> {desc}")


if __name__ == "__main__":
    _selftest()
