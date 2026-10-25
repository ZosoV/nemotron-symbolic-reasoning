export interface FileEntry {
  /** Path relative to inputs/<dir>/ */
  name: string;
  size: number;
  mtime: number;
}

/** List all .parquet files under inputs/<dir>/. */
export async function listFiles(dir: 'data' | 'results'): Promise<FileEntry[]> {
  const res = await fetch(`/api/files?dir=${dir}`);
  if (!res.ok) throw new Error(`Failed to list files (${res.status}): ${res.statusText}`);
  return (await res.json()) as FileEntry[];
}

/**
 * Fetch the raw bytes of a parquet file.
 * @param dir  'data' or 'results'
 * @param name relative path returned by listFiles (e.g. "train.parquet")
 */
export async function fetchParquetFile(dir: 'data' | 'results', name: string): Promise<ArrayBuffer> {
  // Encode each path segment separately to preserve '/'
  const encoded = [dir, ...name.split('/')]
    .map(encodeURIComponent)
    .join('/');
  const res = await fetch(`/files/${encoded}`);
  if (!res.ok) throw new Error(`Failed to fetch "${name}" (${res.status}): ${res.statusText}`);
  return res.arrayBuffer();
}

/**
 * Fetch and parse a search-trace JSON file.
 * @param relPath  path relative to inputs/ (as stored in search_trace_path column)
 */
export async function fetchTraceFile(relPath: string): Promise<SearchTrace> {
  const encoded = relPath.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`/files/${encoded}`);
  if (!res.ok) throw new Error(`Failed to fetch trace "${relPath}" (${res.status}): ${res.statusText}`);
  return res.json();
}

// ── Search-trace JSON schema ────────────────────────────────────────────────

export interface TraceVarEntry {
  glyph: string;
  score: number;
}

export interface TraceNode {
  depth: number;
  glyph: string;
  digit: number;
  partial_mapping: Record<string, number>;
  passed: boolean;
  pruned_glyph: string | null;
  surviving: Record<string, string[]>;
  eliminated: Record<string, string[]>;
  on_solution_path: boolean;
}

export interface SearchTrace {
  problem_id: string | null;
  var_order: TraceVarEntry[];
  initial_candidates: Record<string, string[]>;
  count: number;
  truncated: boolean;
  nodes: TraceNode[];
}
