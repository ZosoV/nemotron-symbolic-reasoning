import { useState } from 'react';

interface Props {
  value: string;
}

export default function CotTrace({ value }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard not available */ }
  };

  const lines = value.split('\n');

  return (
    <div className="long-text-wrap">
      <div className="long-text-toolbar">
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginRight: 'auto' }}>
          Search CoT trace
        </span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {value.length.toLocaleString()} chars
        </span>
        <button className={`copy-btn${copied ? ' copied' : ''}`} onClick={handleCopy}>
          {copied ? '✓ Copied' : 'Copy'}
        </button>
      </div>
      <div className="long-text-content cot-trace-content">
        {lines.map((line, i) => {
          const isSectionHeader = /^\[.+\]$/.test(line.trim()) && line.trim() === line;
          const isLeafAccepted = line.includes('[All') && line.includes('solution accepted');
          if (isSectionHeader || isLeafAccepted) {
            return (
              <div key={i} className={`cot-section-header${isLeafAccepted ? ' cot-section-accepted' : ''}`}>
                {line}
              </div>
            );
          }
          return <div key={i}>{line || ' '}</div>;
        })}
      </div>
    </div>
  );
}
