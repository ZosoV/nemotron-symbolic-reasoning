# Search Trace Schema

Spec for the full branch-by-branch DFS trace produced when `generate_search_trace: true`
in `config.yaml`. One JSON file per puzzle, plus two new parquet columns.

## Related code

- Rust: `search_full_trace` / `arithmetic_search_full_trace` in `rust/alice_sovler_helper/src/lib.rs`
- Python: `_search_full_trace`, `_build_search_trace_json`, `_spine_from_full_trace` in `src/solver/solver_eq_symbolic.py`
- Config: `generate_search_trace`, `search_trace_dir`, `search_trace_max_nodes` in `src/solver/config.py`
- UI: `SearchTraceView` in `src/web_ui_inspector/src/components/SearchTraceView.tsx`

## New parquet columns

| Column | Type | Meaning |
|--------|------|---------|
| `digit_mappings_analyzed` | int \| null | Total `(symbol, digit)` trials explored by the DFS. Includes all partial and complete assignments; always equal to `trace.count`. `null` for concat puzzles and no-solution cases. |
| `search_trace_path` | string \| null | Path to the trace JSON file relative to `inputs/` (e.g. `data/search_traces/abc123.json`). Served by the Vite dev-server via `/files/<path>`. |

## JSON file schema

```
{
  "problem_id": string | null,
  "var_order": [{"glyph": string, "score": int}, ...],
  "initial_candidates": {<op_glyph>: [<op_name>, ...], ...},
  "count": int,
  "truncated": bool,
  "nodes": [TraceNode, ...]
}
```

### `var_order`

Symbols in DFS assignment order (most-constrained first). The weight `score` is computed
by `variable_order_with_scores` (lib.rs): for each group, `group_weight = 1 + |candidates|`,
summed over all equations the symbol appears in. Higher score → assigned earlier.

### `initial_candidates`

Per-operator-glyph candidate operation names **before any digit is assigned**, in the
order produced by the solver's Python heuristics (#3/#2/#4) plus any `filter_out_operations`
removal. These are the candidates that enter the Rust DFS.

### `count` and `truncated`

`count` = total `(symbol, digit)` trials, including both recorded and un-recorded (when cap
is active). The Rust DFS always runs to the first solution. When `search_trace_max_nodes`
is reached, recording stops (`truncated=true`) but `count` still reflects all trials.

### TraceNode

```
{
  "depth": int,                              // DFS depth (0 = first symbol)
  "glyph": string,                           // symbol glyph being assigned
  "digit": int,                              // digit value tried
  "partial_mapping": {<glyph>: int, ...},   // all assigned symbols up to this point
  "passed": bool,                            // forward-check passed?
  "pruned_glyph": string | null,            // op glyph of first group that lost all candidates; null when passed=true
  "surviving": {<op_glyph>: [names], ...},  // surviving candidate names per group after this assignment
  "eliminated": {<op_glyph>: [names], ...}, // candidates removed by this trial (vs. parent node's surviving)
  "on_solution_path": bool                  // this node is on the accepted chain to the first solution
}
```

#### Notes on `surviving` and `eliminated`

- When `passed=false`, `surviving` lists are empty for the failing group and all subsequent groups
  (they were not evaluated). Groups before the failing group have real survivor lists.
- `eliminated` is computed in Python by diffing the current node's `surviving` against the
  parent node's `surviving`. For depth-0 nodes the parent surviving is `initial_candidates`.
- The diff is approximate: it only tracks candidates that were explicitly in the parent's list.

#### `on_solution_path` invariants

- Exactly `len(var_order)` nodes have `on_solution_path=true` (one per DFS depth).
- Their `digit` values form the accepted digit mapping (same as `digits_mapping` in parquet).
- They appear scattered throughout `nodes`; sort by `depth` to reconstruct the accepted chain.

## Config keys

| Key | Default | Meaning |
|-----|---------|---------|
| `generate_search_trace` | `false` | Enable/disable trace generation |
| `search_trace_dir` | `inputs/data/search_traces` | Directory for JSON files (must be under `inputs/`) |
| `search_trace_max_nodes` | `null` | Cap on recorded nodes per puzzle; 0 or null = unlimited |
