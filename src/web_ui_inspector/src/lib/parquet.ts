// @ts-ignore — hyparquet ships its own types; cast as needed below
import { parquetRead } from 'hyparquet';
// @ts-ignore
import { compressors } from 'hyparquet-compressors';

export interface ParquetResult {
  columns: string[];
  rows: Record<string, unknown>[];
}

/**
 * Parse a parquet ArrayBuffer into column names + rows.
 * Uses hyparquet for pure-JS in-browser parsing (supports snappy via hyparquet-compressors).
 */
export async function readParquet(buf: ArrayBuffer): Promise<ParquetResult> {
  // hyparquet expects an AsyncBuffer: { byteLength, slice(start, end?) -> Promise<ArrayBuffer> }
  const file = {
    byteLength: buf.byteLength,
    slice: (start: number, end?: number): Promise<ArrayBuffer> =>
      Promise.resolve(buf.slice(start, end)),
  };

  const allRows: Record<string, unknown>[] = [];

  await parquetRead({
    file,
    compressors,
    rowFormat: 'object',
    // onComplete is called once per row-group with an array of row objects
    onComplete: (rows: Record<string, unknown>[]) => {
      allRows.push(...rows);
    },
  });

  const columns = allRows.length > 0 ? Object.keys(allRows[0]) : [];
  return { columns, rows: allRows };
}
