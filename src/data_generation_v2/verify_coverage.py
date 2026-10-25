"""Simulate the curriculum schedule and report per-bucket coverage.

Functions copied verbatim from
notebooks/training_eval/end-to-end-curriculum_learning.ipynb
(cells curric_config / curric_helpers) so they can be imported without
running the notebook.
"""

from __future__ import annotations

import math
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Schedule simulation (copied verbatim from notebook)
# ---------------------------------------------------------------------------

def competence_value(t: int, c0: float, T: int, kind: str) -> float:
    if kind == "sqrt":
        return min(1.0, math.sqrt(t * (1 - c0 ** 2) / T + c0 ** 2))
    return min(1.0, t * (1 - c0) / T + c0)


class _Cycler:
    def __init__(self, items, rng):
        self.items = list(items)
        self.rng = rng
        self._refill()

    def _refill(self):
        self.pool = self.items[:]
        self.rng.shuffle(self.pool)
        self.idx = 0

    def pop(self):
        if self.idx >= len(self.pool):
            self._refill()
        x = self.pool[self.idx]
        self.idx += 1
        return x


def build_schedule(
    crypt_ids_by_bucket: dict[int, list[str]],
    replay_ids: list[str],
    *,
    num_steps: int,
    batch_size: int,
    replay_ratio: float,
    c0: float,
    T: int,
    competence: str,
    target_buckets: set[int],
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    buckets = sorted(crypt_ids_by_bucket)
    total = sum(len(v) for v in crypt_ids_by_bucket.values())
    cdf, cum = {}, 0
    for n in buckets:
        cum += len(crypt_ids_by_bucket[n])
        cdf[n] = cum / total
    cyclers = {n: _Cycler(crypt_ids_by_bucket[n], rng) for n in buckets}
    replay_cyc = _Cycler(replay_ids, rng) if replay_ids else None

    n_replay = round(batch_size * replay_ratio) if replay_cyc else 0
    n_crypt = batch_size - n_replay

    schedule = []
    for t in range(num_steps):
        if t >= T:
            unlocked = [n for n in buckets if n in target_buckets]
        else:
            c = competence_value(t, c0, T, competence)
            unlocked = [n for n in buckets if cdf[n] <= c] or [buckets[0]]
        ids = [cyclers[rng.choice(unlocked)].pop() for _ in range(n_crypt)]
        ids += [replay_cyc.pop() for _ in range(n_replay)]
        rng.shuffle(ids)
        schedule.append(ids)
    return schedule


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def verify_coverage(
    parquet_paths: list[str | Path],
    num_steps: int = 250,
    batch_size: int = 32,
    replay_ratio: float = 0.25,
    c0: float = 0.01,
    t_frac: float = 0.70,
    competence: str = "sqrt",
    target_buckets: tuple[int, ...] = (3, 4, 5),
    seed: int = 0,
) -> None:
    """Load parquets, simulate the curriculum schedule, and print coverage table."""
    import pandas as pd

    dfs = [pd.read_parquet(p) for p in parquet_paths]
    df = pd.concat(dfs, ignore_index=True)

    if "dataset_category" not in df.columns:
        print("ERROR: no dataset_category column found.")
        return

    crypt_ids_by_bucket: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        cat = str(row.get("dataset_category") or "")
        if not cat.startswith("examples_"):
            continue
        try:
            n = int(cat.split("_")[1])
        except (IndexError, ValueError):
            continue
        pid = str(row.get("problem_id") or row.get("id") or "")
        crypt_ids_by_bucket.setdefault(n, []).append(pid)

    T = max(1, int(t_frac * num_steps))
    n_replay = round(batch_size * replay_ratio)
    n_crypt = batch_size - n_replay

    schedule = build_schedule(
        crypt_ids_by_bucket,
        replay_ids=[],  # replay not simulated (separate corpus)
        num_steps=num_steps,
        batch_size=n_crypt,  # use only crypt slots for bucket draws
        replay_ratio=0.0,
        c0=c0,
        T=T,
        competence=competence,
        target_buckets=set(target_buckets),
        seed=seed,
    )

    # Count draws per bucket
    from collections import Counter
    draw_counter: Counter[str] = Counter()
    for step_ids in schedule:
        for pid in step_ids:
            draw_counter[pid] += 1

    # Compute bucket-level coverage
    print(f"\n{'Bucket':>8} {'Pool':>8} {'Draws':>8} {'Coverage':>10}")
    print("-" * 38)
    total_draws = 0
    for n in sorted(crypt_ids_by_bucket):
        pool = crypt_ids_by_bucket[n]
        draws = sum(draw_counter[pid] for pid in pool)
        cov = draws / len(pool) if pool else 0.0
        total_draws += draws
        print(f"{'n='+str(n):>8} {len(pool):>8,} {draws:>8,} {cov:>10.2f}×")
    print(f"\nTotal crypt draws in schedule: {total_draws:,} "
          f"(expected {num_steps * n_crypt:,})")
    print(f"T={T} (phase 2 starts step {T}), "
          f"target_buckets={set(target_buckets)}")
