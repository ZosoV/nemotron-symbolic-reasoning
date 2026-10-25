import React, { useState, useMemo } from 'react';
import { ColDef } from '../lib/schema';
import { truncate, jsonSummary } from '../lib/format';

interface Props {
  columns: ColDef[];
  rows: Record<string, unknown>[];
  selectedRow: Record<string, unknown> | null;
  onRowClick: (row: Record<string, unknown>) => void;
  groupByProblem: boolean;
}

type SortDir = 'asc' | 'desc';

function CellValue({ col, val }: { col: ColDef; val: unknown }) {
  if (val === null || val === undefined) return <span className="cell-null">—</span>;

  switch (col.kind) {
    case 'id':
      return <span className="cell-id">{String(val)}</span>;

    case 'bool': {
      const isTrue = val === true || val === 1 || val === 'true';
      return <span className={isTrue ? 'cell-bool-true' : 'cell-bool-false'}>{isTrue ? '✓' : '✗'}</span>;
    }

    case 'json':
      return <span className="cell-json">{jsonSummary(val)}</span>;

    case 'longtext': {
      const text = truncate(String(val), 80);
      return <span className="cell-longtext">{text || '(empty)'}</span>;
    }

    case 'number':
      return <span className="cell-number">{String(val)}</span>;

    default:
      return <span>{truncate(String(val), 60)}</span>;
  }
}

export default function DataTable({ columns, rows, selectedRow, onRowClick, groupByProblem }: Props) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleHeaderClick = (key: string) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortedRows = useMemo(() => {
    if (!sortKey) return rows;
    return [...rows].sort((a, b) => {
      const av = String(a[sortKey] ?? '');
      const bv = String(b[sortKey] ?? '');
      return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    });
  }, [rows, sortKey, sortDir]);

  if (columns.length === 0) {
    return <div className="state-message">No columns to display. Check the column selector.</div>;
  }
  if (rows.length === 0) {
    return <div className="state-message">No rows match the current filter.</div>;
  }

  let lastProblemId: unknown = Symbol('sentinel');

  return (
    <table className="data-table">
      <thead>
        <tr>
          {columns.map(col => (
            <th key={col.key} onClick={() => handleHeaderClick(col.key)}>
              {col.label}
              {sortKey === col.key && (
                <span style={{ marginLeft: 4 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sortedRows.map((row, idx) => {
          const isSelected = row === selectedRow;
          const problemId = row['problem_id'];
          const showGroupHeader = groupByProblem && problemId !== lastProblemId;
          if (groupByProblem) lastProblemId = problemId;

          return (
            <React.Fragment key={idx}>
              {showGroupHeader && (
                <tr style={{ pointerEvents: 'none' }}>
                  <td
                    colSpan={columns.length}
                    style={{
                      background: 'var(--bg)',
                      padding: '4px 12px',
                      borderBottom: '1px solid var(--border)',
                    }}
                  >
                    <span className="group-badge">{String(problemId ?? '(no id)')}</span>
                  </td>
                </tr>
              )}
              <tr
                className={isSelected ? 'selected' : ''}
                onClick={() => onRowClick(row)}
              >
                {columns.map(col => (
                  <td key={col.key}>
                    <CellValue col={col} val={row[col.key]} />
                  </td>
                ))}
              </tr>
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
  );
}
