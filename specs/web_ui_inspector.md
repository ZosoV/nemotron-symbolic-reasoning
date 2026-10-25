# Web UI Inspector Spec

A TypeScript + Vite + React inspector for browsing `.parquet` dataset and evaluation-results files.

## How to run

```bash
cd src/web_ui_inspector
npm install          # first time only
npm run dev          # opens http://localhost:5173
```

## Pages

| Page | URL | Source dir | Schema |
|---|---|---|---|
| **Dataset** | `/data` | `inputs/data/*.parquet` | `DATASET_COLUMNS` |
| **Results** | `/results` | `inputs/results/*.parquet` | `RESULTS_COLUMNS` |

---

## Architecture

Pure client-side React SPA served by the Vite dev server.  
A **Vite dev-server middleware plugin** (`vite.config.ts`) exposes two endpoints so the browser can enumerate and fetch files from disk:

| Endpoint | Purpose |
|---|---|
| `GET /api/files?dir=data\|results` | JSON list of `*.parquet` under `inputs/<dir>/` (recursive; returns `{ name, size, mtime }[]`) |
| `GET /files/<dir>/<relpath>` | Raw parquet bytes for the chosen file |

Parquet parsing is done **client-side** with `hyparquet` + `hyparquet-compressors` (required for snappy, which pandas uses by default).

---

## Dataset columns

From `specs/solver-results-schema.md` (multi-row-per-puzzle layout).

| Column | Kind | Notes |
|---|---|---|
| `problem_id` | id | puzzle identifier |
| `solution_id` | number | per-puzzle sequential index |
| `problem_category` | text | Numeral, Unit conversion, Cryptarithm, … |
| `solver_category` | text | `arithmetic` / `concatenation` |
| `solver_strategy` | text | `standard` / `little_endian` / `pure_concat` |
| `solver_tier` | text | tier0 / tier1 / tier2 / full |
| `solver_radix` | number | usually 10 |
| `solution_count` | number | |
| `standard_solution_count` | number | |
| `little_endian_solution_count` | number | |
| `answer` | text | ground-truth answer |
| `solver_answer` | text | extracted from `\boxed{}` or generated |
| `solver_correct` | bool | |
| `ops_mapping` | json | JSON string → `{op: fn_name}` |
| `digits_mapping` | json | JSON string → `{sym: digit}` |
| `ops_candidates` | json | JSON string → `{op: [candidate]}` |
| `equations` | json | JSON string (list of equation dicts) |
| `prompt` | longtext | |
| `cot` | longtext | added by cot_generation notebooks |
| `cot_1_5` | longtext | CoT with 1–5 worked-example chain |

## Results columns

| Column | Kind | Notes |
|---|---|---|
| `problem_id` | id | |
| `correct` | bool | via `verify()` |
| `has_boxed` | bool | |
| `answer` | text | ground truth |
| `solver_answer` | text | extracted from `\boxed{}` or fallback |
| `problem_category` | text | optional pass-through |
| `solver_category` | text | optional pass-through |
| `solver_strategy` | text | optional pass-through |
| `solver_tier` | text | optional pass-through |
| `solver_radix` | number | optional pass-through |
| `ops_mapping` | json | optional pass-through |
| `digits_mapping` | json | optional pass-through |
| `ops_candidates` | json | optional pass-through |
| `prompt` | longtext | |
| `raw_output` | longtext | full model generation |
| `cot` | longtext | optional pass-through |
| `cot_1_5` | longtext | optional pass-through |

---

## Project layout

```
src/web_ui_inspector/
├── package.json            # deps: react, react-router-dom, hyparquet, hyparquet-compressors
├── tsconfig.json
├── vite.config.ts          # React plugin + parquet-file-server middleware
├── index.html
├── .gitignore              # node_modules/, dist/
└── src/
    ├── main.tsx            # ReactDOM.createRoot + BrowserRouter
    ├── App.tsx             # nav header + Routes (/data, /results)
    ├── styles.css          # dark-theme CSS variables, all component styles
    ├── lib/
    │   ├── schema.ts       # DATASET_COLUMNS, RESULTS_COLUMNS (ColDef[])
    │   ├── format.ts       # safeParse, jsonSummary, truncate, prettyJson
    │   ├── files.ts        # listFiles(), fetchParquetFile()
    │   └── parquet.ts      # readParquet() → { columns, rows }
    ├── components/
    │   ├── InspectorView.tsx  # shared view: state management, layout
    │   ├── FilePicker.tsx     # file dropdown + size/mtime/row-count metadata
    │   ├── Toolbar.tsx        # filter input, column show/hide, group-by-problem
    │   ├── DataTable.tsx      # sortable table; click row → RowDetail
    │   ├── RowDetail.tsx      # slide-over drawer with full field values
    │   ├── LongText.tsx       # scrollable pre-wrap with copy button
    │   └── JsonView.tsx       # syntax-highlighted collapsible JSON
    └── pages/
        ├── DataPage.tsx        # <InspectorView dir="data" schema={DATASET_COLUMNS} />
        └── ResultsPage.tsx     # <InspectorView dir="results" schema={RESULTS_COLUMNS} />
```

---

## UX features

- **File picker** — dropdown auto-discovers all `.parquet` under `inputs/<dir>/`; shows size, mtime, and row count.
- **Table** — compact cells: long text truncated, JSON shown as `{…} N keys` badge, bools as ✓/✗.
  Click any column header to sort. `correct`/`solver_correct` color-coded (green/red).
- **Row detail** — click a row → slide-over drawer shows every field full-size.
  `longtext` fields (cot, raw_output, prompt) render in a scrollable monospace pane with a **Copy** button.
  `json` fields render as syntax-highlighted collapsible JSON.
- **Toolbar** — free-text filter across all columns, column show/hide popover.
  Dataset page adds **Group by problem** toggle (inserts a header row between different `problem_id` groups).
- **Graceful degradation** — columns absent from a file are silently skipped; extra columns not in the schema appear under "Other columns" in the detail drawer.

---

## Dependencies

| Package | Role |
|---|---|
| `react` + `react-dom` | UI framework |
| `react-router-dom` | client-side routing (`/data`, `/results`) |
| `hyparquet` | pure-JS in-browser parquet parser |
| `hyparquet-compressors` | snappy decompression (needed for pandas-written parquet) |
| `vite` + `@vitejs/plugin-react` | dev server + bundler |
