import { FileEntry } from '../lib/files';
import { formatSize, formatDate } from '../lib/format';

interface Props {
  files: FileEntry[];
  selected: FileEntry | null;
  onSelect: (f: FileEntry) => void;
  dir: 'data' | 'results';
  rowCount?: number;
}

export default function FilePicker({ files, selected, onSelect, dir, rowCount }: Props) {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const file = files.find(f => f.name === e.target.value);
    if (file) onSelect(file);
  };

  return (
    <div className="file-picker-wrap">
      <label htmlFor="file-select">
        {dir === 'data' ? '📁 Dataset' : '📊 Results'}
      </label>
      <select
        id="file-select"
        value={selected?.name ?? ''}
        onChange={handleChange}
        disabled={files.length === 0}
      >
        {files.length === 0 && (
          <option value="">No .parquet files found in inputs/{dir}/</option>
        )}
        {files.map(f => (
          <option key={f.name} value={f.name}>
            {f.name}
          </option>
        ))}
      </select>
      {selected && (
        <span className="file-meta">
          {formatSize(selected.size)} · {formatDate(selected.mtime)}
          {rowCount !== undefined && ` · ${rowCount.toLocaleString()} rows`}
        </span>
      )}
    </div>
  );
}
