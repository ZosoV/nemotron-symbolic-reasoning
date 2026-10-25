import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import DataPage from './pages/DataPage';
import ResultsPage from './pages/ResultsPage';

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🔍 Parquet Inspector</h1>
        <nav>
          <NavLink
            to="/data"
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            📁 Dataset
          </NavLink>
          <NavLink
            to="/results"
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            📊 Results
          </NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/data" replace />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </main>
    </div>
  );
}
