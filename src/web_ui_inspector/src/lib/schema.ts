export type ColKind = 'id' | 'text' | 'longtext' | 'cot-trace' | 'json' | 'bool' | 'number' | 'trace-file';

export interface ColDef {
  key: string;
  label: string;
  kind: ColKind;
}

// Dataset parquet: solver output (multi-row-per-puzzle, one row per solution found)
export const DATASET_COLUMNS: ColDef[] = [
  { key: 'problem_id',                  label: 'Problem ID',     kind: 'id'       },
  { key: 'solution_id',                 label: 'Sol #',          kind: 'number'   },
  { key: 'problem_category',            label: 'Category',       kind: 'text'     },
  { key: 'dataset_category',            label: 'Dataset Cat',    kind: 'text'     },
  { key: 'solver_category',             label: 'Solver Cat',     kind: 'text'     },
  { key: 'solver_strategy',             label: 'Strategy',       kind: 'text'     },
  { key: 'solver_tier',                 label: 'Tier',           kind: 'text'     },
  { key: 'solver_radix',                label: 'Radix',          kind: 'number'   },
  { key: 'solution_count',              label: 'Sol Count',      kind: 'number'   },
  { key: 'standard_solution_count',     label: 'Std Sols',       kind: 'number'   },
  { key: 'little_endian_solution_count',label: 'LE Sols',        kind: 'number'   },
  { key: 'answer',                      label: 'Answer',         kind: 'text'     },
  { key: 'solver_answer',               label: 'Solver Ans',     kind: 'text'     },
  { key: 'solver_correct',              label: 'Correct',        kind: 'bool'     },
  { key: 'ops_mapping',                 label: 'Ops Map',        kind: 'json'     },
  { key: 'digits_mapping',              label: 'Digits Map',     kind: 'json'     },
  { key: 'ops_candidates',              label: 'Ops Candidates', kind: 'json'     },
  { key: 'equations',                   label: 'Equations',      kind: 'json'     },
  { key: 'prompt',                      label: 'Prompt',         kind: 'longtext'  },
  { key: 'cot',                         label: 'CoT',            kind: 'longtext'  },
  { key: 'cot_1_5',                     label: 'CoT 1-5',        kind: 'longtext'  },
  { key: 'search_cot',                  label: 'Search CoT',     kind: 'cot-trace'   },
  { key: 'cot_num_perturbations',       label: 'Perturbations',  kind: 'number'      },
  { key: 'digit_mappings_analyzed',     label: 'Mappings',       kind: 'number'      },
  { key: 'search_trace_path',           label: 'Search Trace',   kind: 'trace-file'  },
];

// Results parquet: model evaluation output
export const RESULTS_COLUMNS: ColDef[] = [
  { key: 'problem_id',      label: 'Problem ID',     kind: 'id'       },
  { key: 'correct',         label: 'Correct',        kind: 'bool'     },
  { key: 'has_boxed',       label: 'Has \\boxed{}',  kind: 'bool'     },
  { key: 'answer',          label: 'Answer',         kind: 'text'     },
  { key: 'solver_answer',   label: 'Model Answer',   kind: 'text'     },
  { key: 'problem_category',label: 'Category',       kind: 'text'     },
  { key: 'solver_category', label: 'Solver Cat',     kind: 'text'     },
  { key: 'solver_strategy', label: 'Strategy',       kind: 'text'     },
  { key: 'solver_tier',     label: 'Tier',           kind: 'text'     },
  { key: 'solver_radix',    label: 'Radix',          kind: 'number'   },
  { key: 'ops_mapping',     label: 'Ops Map',        kind: 'json'     },
  { key: 'digits_mapping',  label: 'Digits Map',     kind: 'json'     },
  { key: 'ops_candidates',  label: 'Ops Candidates', kind: 'json'     },
  { key: 'prompt',          label: 'Prompt',         kind: 'longtext' },
  { key: 'raw_output',      label: 'Raw Output',     kind: 'longtext' },
  { key: 'cot',             label: 'CoT',            kind: 'longtext' },
  { key: 'cot_1_5',        label: 'CoT 1-5',        kind: 'longtext' },
];
