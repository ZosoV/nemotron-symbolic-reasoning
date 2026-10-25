/** Try to JSON-parse a value; return the raw value on failure. */
export function safeParse(val: unknown): unknown {
  if (typeof val === 'string' && (val.startsWith('{') || val.startsWith('['))) {
    try { return JSON.parse(val); } catch { /* fall through */ }
  }
  return val;
}

/** One-line summary badge for a JSON column value. */
export function jsonSummary(val: unknown): string {
  const parsed = safeParse(val);
  if (parsed === null || parsed === undefined) return 'null';
  if (Array.isArray(parsed)) return `[…] ${parsed.length} items`;
  if (typeof parsed === 'object') return `{…} ${Object.keys(parsed as object).length} keys`;
  return String(parsed);
}

/** Truncate a string for table cell display. */
export function truncate(text: string, maxLen = 100): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length <= maxLen ? flat : flat.slice(0, maxLen) + '…';
}

/** Human-readable file size. */
export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Human-readable timestamp. */
export function formatDate(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

/** Pretty-print any value as JSON (parses JSON strings first). */
export function prettyJson(val: unknown): string {
  const parsed = safeParse(val);
  try {
    return JSON.stringify(parsed, null, 2);
  } catch {
    return String(val);
  }
}

/** Returns true if a value looks like a JSON string. */
export function isJsonString(val: unknown): val is string {
  if (typeof val !== 'string') return false;
  const t = val.trimStart();
  return t.startsWith('{') || t.startsWith('[');
}
