import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DashboardPage } from "./pages/Dashboard/DashboardPage";
import { BacktestPage } from "./pages/Backtest/BacktestPage";
import { ResultsPage } from "./pages/Results/ResultsPage";
import { ComparePage } from "./pages/Compare/ComparePage";
import { NewsPage } from "./pages/News/NewsPage";
import { SettingsPage } from "./pages/Settings/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="backtest" element={<BacktestPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/:jobId" element={<ResultsPage />} />
        <Route path="compare" element={<ComparePage />} />
        <Route path="news" element={<NewsPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
