import { useState } from 'react';
import { safeParse, prettyJson } from '../lib/format';

interface Props {
  value: unknown;
}

/** Syntax-highlight a JSON string with spans. */
function highlight(json: string): React.ReactNode[] {
  const tokens = json.split(/("(?:[^"\\]|\\.)*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g);
  return tokens.map((tok, i) => {
    if (!tok) return null;
    if (tok.startsWith('"')) {
      // Distinguish keys (followed by ':') from string values
      const isKey = /^"[^"]*"\s*:/.test(tok + (tokens[i + 1] ?? ''));
      return <span key={i} className={isKey ? 'json-key' : 'json-string'}>{tok}</span>;
    }
    if (tok === 'true' || tok === 'false') return <span key={i} className="json-bool">{tok}</span>;
    if (tok === 'null') return <span key={i} className="json-null">{tok}</span>;
    if (/^-?\d/.test(tok)) return <span key={i} className="json-number">{tok}</span>;
    return tok;
  });
}

export default function JsonView({ value }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const parsed = safeParse(value);
  const pretty = prettyJson(parsed);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.375rem' }}>
        <button
          className="btn"
          style={{ fontSize: '0.7rem', padding: '2px 8px' }}
          onClick={() => setCollapsed(v => !v)}
        >
          {collapsed ? '▶ Show' : '▼ Hide'}
        </button>
      </div>
      {!collapsed && (
        <div className="json-view">
          {highlight(pretty)}
        </div>
      )}
    </div>
  );
}
