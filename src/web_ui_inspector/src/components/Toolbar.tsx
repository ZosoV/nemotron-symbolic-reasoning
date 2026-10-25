import { useState, useRef, useEffect } from 'react';
import { ColDef } from '../lib/schema';

interface Props {
  filter: string;
  onFilterChange: (v: string) => void;
  rowCount: number;
  totalCount: number;
  availableCols: ColDef[];
  visibleCols: Set<string>;
  onVisibleColsChange: (s: Set<string>) => void;
  showGroupBy: boolean;
  groupByProblem: boolean;
  onGroupByChange: (v: boolean) => void;
  // Results-page category filter
  categoryOptions?: string[];
  categoryFilter?: string;
  onCategoryFilterChange?: (v: string) => void;
  // Data-page perturbations filter
  perturbationsOptions?: string[];
  perturbationsFilter?: string;
  onPerturbationsFilterChange?: (v: string) => void;
}

export default function Toolbar({
  filter, onFilterChange,
  rowCount, totalCount,
  availableCols, visibleCols, onVisibleColsChange,
  showGroupBy, groupByProblem, onGroupByChange,
  categoryOptions, categoryFilter, onCategoryFilterChange,
  perturbationsOptions, perturbationsFilter, onPerturbationsFilterChange,
}: Props) {
  const [colPopoverOpen, setColPopoverOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close popover on outside click
  useEffect(() => {
    if (!colPopoverOpen) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setColPopoverOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [colPopoverOpen]);

  const toggleCol = (key: string) => {
    const next = new Set(visibleCols);
    if (next.has(key)) {
      // Keep at least one column visible
      if (next.size > 1) next.delete(key);
    } else {
      next.add(key);
    }
    onVisibleColsChange(next);
  };

  const showAll = () => onVisibleColsChange(new Set(availableCols.map(c => c.key)));
  const hideAll = () => {
    const first = availableCols[0];
    onVisibleColsChange(first ? new Set([first.key]) : new Set());
  };

  return (
    <div className="toolbar">
      <input
        className="search-input"
        type="search"
        placeholder="Filter rows…"
        value={filter}
        onChange={e => onFilterChange(e.target.value)}
      />

      {/* Category filter — results page only */}
      {categoryOptions && categoryOptions.length > 0 && onCategoryFilterChange && (
        <>
          <span className="filter-label">Category</span>
          <select
            className={`filter-select${categoryFilter ? ' has-value' : ''}`}
            value={categoryFilter ?? ''}
            onChange={e => onCategoryFilterChange(e.target.value)}
          >
            <option value="">All categories</option>
            {categoryOptions.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </>
      )}

      {/* Perturbations filter — data page only */}
      {perturbationsOptions && perturbationsOptions.length > 0 && onPerturbationsFilterChange && (
        <>
          <span className="filter-label">Perturbations</span>
          <select
            className={`filter-select${perturbationsFilter ? ' has-value' : ''}`}
            value={perturbationsFilter ?? ''}
            onChange={e => onPerturbationsFilterChange(e.target.value)}
          >
            <option value="">All</option>
            {perturbationsOptions.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </>
      )}

      <span className="row-count">
        {filter
          ? `${rowCount.toLocaleString()} / ${totalCount.toLocaleString()} rows`
          : `${totalCount.toLocaleString()} rows`}
      </span>

      <div className="toolbar-right">
        {showGroupBy && (
          <button
            className={`btn${groupByProblem ? ' active' : ''}`}
            onClick={() => onGroupByChange(!groupByProblem)}
            title="Group rows by problem_id"
          >
            Group by problem
          </button>
        )}

        <div className="col-toggle-wrap" ref={popoverRef}>
          <button
            className="btn"
            onClick={() => setColPopoverOpen(v => !v)}
            title="Show/hide columns"
          >
            Columns ▾
          </button>
          {colPopoverOpen && (
            <div className="col-toggle-popover">
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <button className="btn" style={{ fontSize: '0.7rem', padding: '2px 8px' }} onClick={showAll}>All</button>
                <button className="btn" style={{ fontSize: '0.7rem', padding: '2px 8px' }} onClick={hideAll}>None</button>
              </div>
              {availableCols.map(col => (
                <label key={col.key} className="col-toggle-item">
                  <input
                    type="checkbox"
                    checked={visibleCols.has(col.key)}
                    onChange={() => toggleCol(col.key)}
                  />
                  {col.label}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
