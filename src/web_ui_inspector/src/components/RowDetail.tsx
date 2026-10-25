import { useEffect } from 'react';
import { ColDef } from '../lib/schema';
import { isJsonString } from '../lib/format';
import LongText from './LongText';
import JsonView from './JsonView';
import CotTrace from './CotTrace';
import SearchTraceView from './SearchTraceView';

interface Props {
  row: Record<string, unknown>;
  schema: ColDef[];
  onClose: () => void;
}

function BoolChip({ value }: { value: unknown }) {
  const isTrue = value === true || value === 1 || value === 'true';
  return (
    <span className={isTrue ? 'cell-bool-true' : 'cell-bool-false'} style={{ fontSize: '1rem' }}>
      {isTrue ? '✓ true' : '✗ false'}
    </span>
  );
}

export default function RowDetail({ row, schema, onClose }: Props) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // All keys in the row, ordered: schema-defined first, then extras
  const schemaKeys = new Set(schema.map(c => c.key));
  const schemaColsInRow = schema.filter(c => c.key in row);
  const extraKeys = Object.keys(row).filter(k => !schemaKeys.has(k));

  const renderValue = (col: ColDef, val: unknown) => {
    if (val === null || val === undefined) {
      return <span className="cell-null">—</span>;
    }
    switch (col.kind) {
      case 'bool':
        return <BoolChip value={val} />;
      case 'json':
        return <JsonView value={val} />;
      case 'longtext': {
        const text = String(val);
        return text.length > 0
          ? <LongText value={text} label={col.label} />
          : <span className="cell-null">(empty)</span>;
      }
      case 'cot-trace': {
        const text = String(val);
        return text.length > 0
          ? <CotTrace value={text} />
          : <span className="cell-null">(empty)</span>;
      }
      case 'trace-file': {
        const tracePath = String(val);
        return tracePath.length > 0
          ? <SearchTraceView path={tracePath} />
          : <span className="cell-null">(no trace)</span>;
      }
      case 'id':
        return <code style={{ fontFamily: 'var(--mono)', color: 'var(--accent)' }}>{String(val)}</code>;
      default: {
        const s = String(val);
        // Auto-detect JSON strings for non-json-typed cols
        if (isJsonString(s)) return <JsonView value={s} />;
        return <span>{s}</span>;
      }
    }
  };

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true">
        <div className="drawer-header">
          <h2>
            Row detail
            {row.problem_id !== undefined && (
              <> — <code style={{ fontFamily: 'var(--mono)', color: 'var(--accent)', fontWeight: 400 }}>{String(row.problem_id)}</code></>
            )}
          </h2>
          <button className="drawer-close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="drawer-body">
          {schemaColsInRow.map(col => (
            <div key={col.key} className="field-group">
              <div className="field-label">{col.label}</div>
              <div className="field-value">
                {renderValue(col, row[col.key])}
              </div>
            </div>
          ))}

          {/* Extra columns not in schema */}
          {extraKeys.length > 0 && (
            <div className="field-section">
              <div className="field-label" style={{ marginBottom: '0.75rem' }}>Other columns</div>
              {extraKeys.map(key => (
                <div key={key} className="field-group">
                  <div className="field-label">{key}</div>
                  <div className="field-value">
                    {renderValue({ key, label: key, kind: isJsonString(String(row[key] ?? '')) ? 'json' : 'text' }, row[key])}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
