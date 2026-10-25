"""Augmentation 1 — Perturbation-order variants as extra training rows.

Given k≥2 candidate perturbations, there are k! distinct orderings. Each
ordering produces a genuinely different reasoning chain (different intermediate
mappings, failing equations, and follow-up reasoning) while sharing the same
starting mapping_1 and converging to the same solution.

This module generates alternative orderings (sampled from the k! permutations)
and returns the corresponding `break_order` lists that can be passed to
`build_chain(break_order=...)`.

Public API:
  generate_order_variants(num_candidates, num_rounds, seed, max_variants)
      → list of break_order lists (variant 0 = default coupling-sorted order)
"""

from __future__ import annotations

import random
from itertools import permutations


def generate_order_variants(
    num_candidates: int,
    num_rounds: int,
    seed: int,
    max_variants: int,
) -> list[list[int]]:
    """Return up to `max_variants` distinct break_order lists for `num_candidates` candidates.

    The first returned order (variant 0) is always None — the caller uses
    `break_order=None` to trigger the default coupling-sorted ordering.

    Additional orders are sampled from the k! permutations of 0..num_candidates-1,
    excluding the coupling-sorted order (which is not representable here; it's just
    None). Returns a list of `break_order` values where None = default, list = custom.

    Note: the effective number of candidates used is min(num_candidates, num_rounds).
    For k=1 there are no alternative orderings; returns [None].
    """
    effective_k = min(num_candidates, num_rounds)
    if effective_k <= 1 or max_variants <= 1:
        return [None]

    rng = random.Random(seed)
    base_indices = list(range(effective_k))

    # All permutations (capped to avoid memory issues for large k)
    if effective_k <= 6:
        all_perms = list(permutations(base_indices))
    else:
        # Too many permutations — just sample
        all_perms = [base_indices]  # start with default, add random ones
        for _ in range(max_variants * 3):
            p = list(base_indices)
            rng.shuffle(p)
            all_perms.append(p)

    # Shuffle and pick up to max_variants - 1 alternatives (variant 0 = default None)
    rng.shuffle(all_perms)
    orders: list[list[int] | None] = [None]  # variant 0 = default
    seen = set()
    for perm in all_perms:
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        orders.append(list(perm))
        if len(orders) >= max_variants:
            break

    return orders
