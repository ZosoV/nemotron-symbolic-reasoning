import { useState, useEffect, useMemo } from 'react';
import { ColDef } from '../lib/schema';
import { listFiles, fetchParquetFile, FileEntry } from '../lib/files';
import { readParquet, ParquetResult } from '../lib/parquet';
import FilePicker from './FilePicker';
import Toolbar from './Toolbar';
import DataTable from './DataTable';
import RowDetail from './RowDetail';

interface Props {
  dir: 'data' | 'results';
  schema: ColDef[];
}

export default function InspectorView({ dir, schema }: Props) {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileEntry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ParquetResult | null>(null);

  const [filter, setFilter] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [perturbationsFilter, setPerturbationsFilter] = useState('');
  const [visibleCols, setVisibleCols] = useState<Set<string>>(new Set());
  const [groupByProblem, setGroupByProblem] = useState(false);
  const [selectedRow, setSelectedRow] = useState<Record<string, unknown> | null>(null);

  // Load file list on mount / when dir changes
  useEffect(() => {
    setFiles([]);
    setSelectedFile(null);
    setResult(null);
    setError(null);
    listFiles(dir)
      .then(f => {
        setFiles(f);
        if (f.length > 0) setSelectedFile(f[0]);
      })
      .catch(e => setError(String(e)));
  }, [dir]);

  // Load selected file
  useEffect(() => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedRow(null);
    setFilter('');
    setCategoryFilter('');
    setPerturbationsFilter('');

    fetchParquetFile(dir, selectedFile.name)
      .then(buf => readParquet(buf))
      .then(r => {
        setResult(r);
        const fileColSet = new Set(r.columns);
        setVisibleCols(new Set(schema.map(c => c.key).filter(k => fileColSet.has(k))));
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedFile, dir, schema]);

  // Unique problem_category values for the results-page filter dropdown
  const categoryOptions = useMemo(() => {
    if (dir !== 'results' || !result) return [];
    const cats = new Set<string>();
    for (const row of result.rows) {
      const cat = row['problem_category'];
      if (cat != null && String(cat).trim() !== '') cats.add(String(cat));
    }
    return Array.from(cats).sort();
  }, [result, dir]);

  // Unique cot_num_perturbations values for the data-page filter dropdown
  const perturbationsOptions = useMemo(() => {
    if (!result || !result.columns.includes('cot_num_perturbations')) return [];
    const vals = new Set<number>();
    for (const row of result.rows) {
      const v = row['cot_num_perturbations'];
      if (v != null && !Number.isNaN(Number(v))) vals.add(Number(v));
    }
    return Array.from(vals).sort((a, b) => a - b).map(String);
  }, [result]);

  const filteredRows = useMemo(() => {
    if (!result) return [];
    let rows = result.rows;

    // Category filter (results page)
    if (categoryFilter) {
      rows = rows.filter(row => String(row['problem_category'] ?? '') === categoryFilter);
    }

    // Perturbations filter (data page)
    if (perturbationsFilter) {
      rows = rows.filter(row => String(row['cot_num_perturbations'] ?? '') === perturbationsFilter);
    }

    // Free-text filter
    if (filter.trim()) {
      const q = filter.toLowerCase();
      rows = rows.filter(row =>
        Object.values(row).some(v => v != null && String(v).toLowerCase().includes(q))
      );
    }

    return rows;
  }, [result, filter, categoryFilter, perturbationsFilter]);

  // Schema columns present in this file, in schema order
  const availableCols = useMemo(() => {
    if (!result) return [];
    const fileColSet = new Set(result.columns);
    return schema.filter(c => fileColSet.has(c.key));
  }, [result, schema]);

  // Intersection of available and currently-visible
  const activeCols = useMemo(
    () => availableCols.filter(c => visibleCols.has(c.key)),
    [availableCols, visibleCols]
  );

  return (
    <div className="inspector">
      {/* Top bar: file picker + toolbar */}
      <div className="inspector-top">
        <FilePicker
          files={files}
          selected={selectedFile}
          onSelect={f => { setSelectedFile(f); }}
          dir={dir}
          rowCount={result ? result.rows.length : undefined}
        />
        {result && (
          <Toolbar
            filter={filter}
            onFilterChange={setFilter}
            rowCount={filteredRows.length}
            totalCount={result.rows.length}
            availableCols={availableCols}
            visibleCols={visibleCols}
            onVisibleColsChange={setVisibleCols}
            showGroupBy={dir === 'data'}
            groupByProblem={groupByProblem}
            onGroupByChange={setGroupByProblem}
            categoryOptions={categoryOptions}
            categoryFilter={categoryFilter}
            onCategoryFilterChange={setCategoryFilter}
            perturbationsOptions={perturbationsOptions}
            perturbationsFilter={perturbationsFilter}
            onPerturbationsFilterChange={setPerturbationsFilter}
          />
        )}
      </div>

      {/* Table area */}
      <div className="inspector-body">
        <div className="inspector-table">
          {loading && <div className="state-message">⏳ Loading parquet…</div>}
          {!loading && error && (
            <div className="state-message state-error">⚠ {error}</div>
          )}
          {!loading && !error && !result && files.length === 0 && (
            <div className="state-message">No .parquet files found in <code>inputs/{dir}/</code>.</div>
          )}
          {!loading && !error && !result && files.length > 0 && (
            <div className="state-message">Select a file above to inspect.</div>
          )}
          {!loading && !error && result && (
            <DataTable
              columns={activeCols}
              rows={filteredRows}
              selectedRow={selectedRow}
              onRowClick={setSelectedRow}
              groupByProblem={groupByProblem}
            />
          )}
        </div>
      </div>

      {/* Row detail drawer */}
      {selectedRow && (
        <RowDetail
          row={selectedRow}
          schema={schema}
          onClose={() => setSelectedRow(null)}
        />
      )}
    </div>
  );
}
