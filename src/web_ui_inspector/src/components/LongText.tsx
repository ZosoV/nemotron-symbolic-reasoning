import { useState } from 'react';

interface Props {
  value: string;
  label?: string;
}

export default function LongText({ value, label }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard not available */
    }
  };

  return (
    <div className="long-text-wrap">
      <div className="long-text-toolbar">
        {label && <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginRight: 'auto' }}>{label}</span>}
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {value.length.toLocaleString()} chars
        </span>
        <button className={`copy-btn${copied ? ' copied' : ''}`} onClick={handleCopy}>
          {copied ? '✓ Copied' : 'Copy'}
        </button>
      </div>
      <div className="long-text-content">{value}</div>
    </div>
  );
}
