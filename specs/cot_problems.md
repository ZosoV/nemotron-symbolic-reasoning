# Generating Chain-of-Thought

## Summary

Building a chain-of-thought (CoT) trace directly from the solver is impractical:
it runs too many checks, producing traces too long and expensive for an LLM to
reproduce. Instead, we **work backwards from a solution (digit mapping, ops mapping, magnitude solver strategy - base 10) we already know** — treat the known solution as the endpoint and synthesize a short chain that arrives at it.

## 1. Solver output

For each instance the solver returns:

- operator mapping
- digit mapping
- magnitude solver strategy
- base

## 2. Approach: backward construction

We don't rediscover the solution; we reconstruct a few steps that lead into it.

- Start from the known solution and apply **perturbations** to roll it back into
  earlier "less solved" states.
- Replay those states forward to get a chain that ends at the correct answer.

This is **not** the solver's depth-first search. Each step instead:

1. runs a progressive check analysis,
2. extracts information from the result,
3. uses it to propose the next permutation of the digit mapping.

The chain converges in a **few steps (≈3–4)** — short enough for an LLM's budget,
unlike a full search trace.

## 3. Gap: random initial mapping

The chain currently starts from a **random** digit mapping. To the view point of the cot reasoning nothing explain that I pick in that way the first digit mapping. Chain length depends
entirely on that start: a good one solves in 3–4 steps, a bad one doesn't. The
choice is currently unprincipled.

## 4. Strategies for choosing the start


**4.2 Distance-controlled starts (recommended).** Since the solution is known, the
start needn't be random. Sample a start at a fixed perturbation distance *k* from
the solution (e.g. exactly *k* digit swaps). This **sets** the chain length to ≈*k*
and **guarantees** convergence by construction.

**4.3 Diversity at fixed distance.** Draw several distinct starts at the same
distance so that the llm can understand how to proceed under different situations.

IMPORTANT: In theory, e.g. if there are 10 digit symbols, it is possible to get n! = 10! permutations. From those permutations, only some of them can arrive to a feasible solution in a reasonable progressive check analysis. I would like to understand 1)

- if it is recommended, extract different permutations that leads to the same feasible solution.
- if it is recommended, extract different permutations that leads to different feasible solutions.
- teach how to choose the initial permutation.

I want to understand if this cot makes sense to reason about the solution in a shor time.


### Some clues for taking the first digit mapping decision

- Symbol eligibility — only symbols appearing in at least one example equation (eq_syms) are of main interest because those are more informative than the ones in the question, as they can be evaluated to a value and check the result. 
