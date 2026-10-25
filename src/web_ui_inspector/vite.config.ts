import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import type { Connect } from 'vite';
import type { IncomingMessage, ServerResponse } from 'http';

// ESM-compatible __dirname
const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Repo root is two levels up from src/web_ui_inspector/
const REPO_ROOT = path.resolve(__dirname, '../..');

interface FileEntry {
  name: string;
  size: number;
  mtime: number;
}

function listParquets(dir: string): FileEntry[] {
  const results: FileEntry[] = [];
  function walk(current: string, rel: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      const relPath = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        walk(fullPath, relPath);
      } else if (entry.name.endsWith('.parquet')) {
        const stat = fs.statSync(fullPath);
        results.push({ name: relPath, size: stat.size, mtime: stat.mtimeMs });
      }
    }
  }
  walk(dir, '');
  return results;
}

function parquetMiddleware(): Connect.HandleFunction {
  return (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
    if (!req.url) return next();
    const url = new URL(req.url, 'http://localhost');

    // GET /api/files?dir=data|results
    if (url.pathname === '/api/files') {
      const dir = url.searchParams.get('dir') || 'data';
      const targetDir = path.join(REPO_ROOT, 'inputs', dir);
      try {
        const files = listParquets(targetDir);
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.end(JSON.stringify(files));
      } catch (e) {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: String(e) }));
      }
      return;
    }

    // GET /files/<dir>/<relpath>  — serve raw parquet bytes
    if (url.pathname.startsWith('/files/')) {
      const relPath = decodeURIComponent(url.pathname.slice('/files/'.length));
      // Guard against path traversal
      const normalized = path.normalize(relPath);
      if (normalized.startsWith('..') || path.isAbsolute(normalized)) {
        res.statusCode = 400;
        res.end('Bad path');
        return;
      }
      const filePath = path.join(REPO_ROOT, 'inputs', normalized);
      if (!fs.existsSync(filePath)) {
        res.statusCode = 404;
        res.end('Not found');
        return;
      }
      try {
        const data = fs.readFileSync(filePath);
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', data.length);
        res.end(data);
      } catch (e) {
        res.statusCode = 500;
        res.end(String(e));
      }
      return;
    }

    next();
  };
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'parquet-file-server',
      configureServer(server) {
        server.middlewares.use(parquetMiddleware());
      },
    },
  ],
  server: {
    fs: {
      allow: ['.', path.join(REPO_ROOT, 'inputs')],
    },
  },
});
