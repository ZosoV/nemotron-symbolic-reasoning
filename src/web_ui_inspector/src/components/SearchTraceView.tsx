import { useState, useEffect } from 'react';
import { fetchTraceFile, SearchTrace, TraceNode } from '../lib/files';

interface Props {
  path: string; // relative to inputs/
}

// ── Small helpers ────────────────────────────────────────────────────────────

function Badge({ label, variant }: { label: string; variant: 'accepted' | 'pruned' | 'backtracked' | 'neutral' }) {
  const cls = `trace-badge trace-badge-${variant}`;
  return <span className={cls}>{label}</span>;
}

function MappingChips({ mapping }: { mapping: Record<string, number> }) {
  return (
    <span className="trace-mapping">
      {Object.entries(mapping).map(([g, d]) => (
        <span key={g} className="trace-chip">{g}={d}</span>
      ))}
    </span>
  );
}

// ── DFS tree node row ────────────────────────────────────────────────────────

function NodeRow({ node }: { node: TraceNode }) {
  const [open, setOpen] = useState(false);
  const indent = node.depth * 16;

  const statusBadge = node.on_solution_path
    ? <Badge label="✓ accepted" variant="accepted" />
    : !node.passed
    ? <Badge label="✗ pruned" variant="pruned" />
    : <Badge label="↩ backtracked" variant="backtracked" />;

  const hasDetail = Object.values(node.surviving).some(arr => arr.length > 0)
    || Object.values(node.eliminated).some(arr => arr.length > 0);

  return (
    <>
      <tr
        className={`trace-node-row${node.on_solution_path ? ' trace-on-path' : ''}`}
        onClick={() => hasDetail && setOpen(o => !o)}
        style={{ cursor: hasDetail ? 'pointer' : 'default' }}
      >
        <td className="trace-cell-depth" style={{ paddingLeft: indent + 8 }}>
          {hasDetail ? (open ? '▾' : '▸') : '·'}
        </td>
        <td className="trace-cell-sym"><code>{node.glyph}</code></td>
        <td className="trace-cell-digit"><code>{node.digit}</code></td>
        <td className="trace-cell-status">{statusBadge}</td>
        <td className="trace-cell-pruned-glyph">
          {node.pruned_glyph && <span className="trace-pruned-by">op '{node.pruned_glyph}' lost all candidates</span>}
        </td>
        <td className="trace-cell-map">
          <MappingChips mapping={node.partial_mapping} />
        </td>
      </tr>
      {open && hasDetail && (
        <tr className="trace-detail-row">
          <td colSpan={6} style={{ paddingLeft: indent + 32, paddingBottom: 6 }}>
            <div className="trace-detail">
              {Object.entries(node.surviving).map(([op, names]) => names.length > 0 && (
                <div key={op}>
                  <span className="trace-op-glyph">'{op}'</span> surviving:&nbsp;
                  <span className="trace-cands">{names.join(', ')}</span>
                  {(node.eliminated[op]?.length ?? 0) > 0 && (
                    <span className="trace-elim"> — eliminated: {node.eliminated[op].join(', ')}</span>
                  )}
                </div>
              ))}
              {Object.entries(node.surviving).every(([, n]) => n.length === 0) &&
                Object.entries(node.eliminated).map(([op, names]) => names.length > 0 && (
                  <div key={op}>
                    <span className="trace-op-glyph">'{op}'</span> eliminated: <span className="trace-elim">{names.join(', ')}</span>
                  </div>
                ))
              }
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export default function SearchTraceView({ path }: Props) {
  const [trace, setTrace] = useState<SearchTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [filterPath, setFilterPath] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setTrace(null);
    fetchTraceFile(path)
      .then(t => setTrace(t))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, [path]);

  if (loading) return <div className="trace-state">⏳ Loading search trace…</div>;
  if (error) return <div className="trace-state trace-error">⚠ {error}</div>;
  if (!trace) return null;

  const displayNodes: TraceNode[] = filterPath
    ? trace.nodes.filter(n => n.on_solution_path)
    : trace.nodes;

  // Cap display at 500 nodes unless the user explicitly asked to show all.
  const MAX_DISPLAY = 500;
  const capped = !showAll && displayNodes.length > MAX_DISPLAY;
  const shownNodes = capped ? displayNodes.slice(0, MAX_DISPLAY) : displayNodes;

  return (
    <div className="trace-wrap">
      {/* ── Header ── */}
      <div className="trace-header">
        <div className="trace-header-row">
          <span className="trace-title">Search Trace</span>
          {trace.truncated && <Badge label="truncated" variant="neutral" />}
          <span className="trace-stat">{trace.count.toLocaleString()} trials</span>
          <span className="trace-stat">{trace.nodes.length.toLocaleString()} nodes recorded</span>
        </div>

        {/* Variable ordering */}
        <div className="trace-var-order">
          <span className="trace-section-label">Symbol order (weight):</span>
          {trace.var_order.map((v, i) => (
            <span key={i} className="trace-chip">
              {i + 1}. '{v.glyph}' ({v.score})
            </span>
          ))}
        </div>

        {/* Initial candidates per operator */}
        <div className="trace-init-cands">
          <span className="trace-section-label">Initial candidates:</span>
          {Object.entries(trace.initial_candidates).map(([op, names]) => (
            <div key={op} className="trace-init-cand-row">
              <span className="trace-op-glyph">'{op}'</span>
              <span className="trace-cands">{names.slice(0, 8).join(', ')}{names.length > 8 ? ` … (+${names.length - 8})` : ''}</span>
            </div>
          ))}
        </div>

        {/* Controls */}
        <div className="trace-controls">
          <label className="trace-toggle">
            <input type="checkbox" checked={filterPath} onChange={e => setFilterPath(e.target.checked)} />
            &nbsp;Show accepted path only ({trace.nodes.filter(n => n.on_solution_path).length} nodes)
          </label>
          {capped && (
            <span className="trace-cap-notice">
              Showing {MAX_DISPLAY.toLocaleString()} of {displayNodes.length.toLocaleString()} nodes.&nbsp;
              <button className="trace-show-all-btn" onClick={() => setShowAll(true)}>Show all</button>
            </span>
          )}
        </div>
      </div>

      {/* ── DFS tree table ── */}
      <div className="trace-table-wrap">
        <table className="trace-table">
          <thead>
            <tr>
              <th></th>
              <th>Symbol</th>
              <th>Digit</th>
              <th>Status</th>
              <th>Prune reason</th>
              <th>Partial mapping</th>
            </tr>
          </thead>
          <tbody>
            {shownNodes.map((node, i) => (
              <NodeRow key={i} node={node} />
            ))}
            {capped && (
              <tr>
                <td colSpan={6} className="trace-cap-row">
                  … {(displayNodes.length - MAX_DISPLAY).toLocaleString()} more nodes hidden.
                  <button className="trace-show-all-btn" onClick={() => setShowAll(true)}>Show all</button>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
