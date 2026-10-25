"""SolverConfig dataclass and YAML loader.

YAML keys:
  force_base_10: bool          — only try (10, 10) radix/digit_count pair
  disabled_heuristics: list    — Python heuristic function names to bypass;
                                 choose from: op_glyph_priority, sign_pattern_filter,
                                 concat_optin, order_ops, tier_skip
  tiers: list[str]             — subset/order of tiers to search;
                                 choices: tier0, tier1, tier2, full
  input_parquet: str           — path to input .parquet (columns: id, prompt, answer)
  output_parquet: str          — path to write solver_results.parquet
  max_solutions: int           — collect up to this many distinct programs per puzzle;
                                 search escalates: digit mappings → little_endian → alt base
  max_digit_trials_standard: int|null   — cap on digit mappings for the standard (big-endian)
                                         search stage per radix. null = unlimited (default).
  max_digit_trials_little_endian: int|null — cap on digit mappings for the little-endian
                                             search stage per radix. null = unlimited (default).
                                 Example: max_digit_trials_standard=10, max_digit_trials_little_endian=10
                                 with max_solutions=20 yields ≈10 standard + ≈10 little_endian
                                 solutions per puzzle.
  randomize_search: bool       — shuffle symbol-assignment order and digit-iteration order
                                 inside the Rust DFS using a seeded PRNG, replacing the
                                 default weight-priority symbol ordering. false = current
                                 deterministic weighted ordering (default).
  randomize_candidates: bool   — shuffle the per-operator candidate operation-name lists
                                 (produced by op_glyph_priority) before each puzzle's
                                 search, so that when multiple operations satisfy the
                                 constraints the chosen op is random rather than always
                                 the highest-priority one. Independent of randomize_search.
                                 Reuses random_seed for reproducibility. Default false.
  random_seed: int             — base seed for the seeded PRNG when randomize_search or
                                 randomize_candidates is true. Different puzzles and
                                 operators get derived seeds. Default 0.
  filter_out_operations: list  — operation names to exclude from the arithmetic search
                                 candidate lists. Names must match keys in OPERATIONS
                                 (src/operations.py). Unknown names are silently ignored.
                                 Does not affect the concat fast-path (_try_concat).
                                 Example: [concat_fwd, concat_rev]. Default [].
  generate_cot: bool           — when true, run the traced DFS (arithmetic_search_traced)
                                 and render a search-trace chain-of-thought for each solved
                                 puzzle, saved in the `search_cot` column of the output
                                 parquet. Always uses canonical weight-priority variable
                                 ordering regardless of randomize_search. Default false.
  disable_concat_fastpath: bool — when true, skip _try_concat entirely and route every
                                 puzzle through the arithmetic search. Concat ops
                                 (concat_fwd, concat_rev) remain in the candidate lists
                                 unless excluded via filter_out_operations. Useful for
                                 generating a uniform arithmetic CoT corpus. Default false.
  generate_search_trace: bool  — when true, run arithmetic_search_full_trace for the first
                                 solution-producing stage and write a per-puzzle JSON file
                                 (under search_trace_dir) capturing the full branch-by-branch
                                 DFS tree up to the first solution. The parquet gains two new
                                 columns: digit_mappings_analyzed (int) and search_trace_path
                                 (str, relative to inputs/). Default false.
  search_trace_dir: str        — directory to write per-puzzle search-trace JSON files into.
                                 Must be under inputs/ so the Vite dev server can serve them.
                                 Default "inputs/data/search_traces".
  search_trace_max_nodes: int|null — cap on the number of events recorded in each trace file.
                                 When the cap is reached the file is written with truncated=true.
                                 The DFS itself always runs to the first solution; only recording
                                 is capped. 0 or null = unlimited. Default null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    force_base_10: bool = False
    disabled_heuristics: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=lambda: ["tier0", "tier1", "tier2", "full"])
    input_parquet: str = "data/solver_results_base.parquet"
    output_parquet: str = "data/solver_results.parquet"
    max_solutions: int = 5
    max_digit_trials_standard: int | None = None
    max_digit_trials_little_endian: int | None = None
    randomize_search: bool = False
    randomize_candidates: bool = False
    random_seed: int = 0
    filter_out_operations: list[str] = field(default_factory=list)
    generate_cot: bool = False
    disable_concat_fastpath: bool = False
    generate_search_trace: bool = False
    search_trace_dir: str = "inputs/data/search_traces"
    search_trace_max_nodes: int | None = None


def load_config(path: str | Path | None = None) -> SolverConfig:
    """Load a SolverConfig from a YAML file, or return defaults if path is None."""
    if path is None:
        return SolverConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    import yaml  # pyyaml; imported lazily so the module loads without it

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    raw_mdt_std = data.get("max_digit_trials_standard", None)
    raw_mdt_le = data.get("max_digit_trials_little_endian", None)
    return SolverConfig(
        force_base_10=bool(data.get("force_base_10", False)),
        disabled_heuristics=list(data.get("disabled_heuristics", [])),
        tiers=[t.lower() for t in data.get("tiers", ["tier0", "tier1", "tier2", "full"])],
        input_parquet=str(data.get("input_parquet", "data/solver_results_base.parquet")),
        output_parquet=str(data.get("output_parquet", "data/solver_results.parquet")),
        max_solutions=int(data.get("max_solutions", 5)),
        max_digit_trials_standard=int(raw_mdt_std) if raw_mdt_std is not None else None,
        max_digit_trials_little_endian=int(raw_mdt_le) if raw_mdt_le is not None else None,
        randomize_search=bool(data.get("randomize_search", False)),
        randomize_candidates=bool(data.get("randomize_candidates", False)),
        random_seed=int(data.get("random_seed", 0)),
        filter_out_operations=list(data.get("filter_out_operations", [])),
        generate_cot=bool(data.get("generate_cot", False)),
        disable_concat_fastpath=bool(data.get("disable_concat_fastpath", False)),
        generate_search_trace=bool(data.get("generate_search_trace", False)),
        search_trace_dir=str(data.get("search_trace_dir", "inputs/data/search_traces")),
        search_trace_max_nodes=int(data["search_trace_max_nodes"])
            if data.get("search_trace_max_nodes") is not None else None,
    )
