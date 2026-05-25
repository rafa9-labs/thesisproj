import { lazy, Suspense, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DashboardPage } from "./pages/Dashboard/DashboardPage";
import { BacktestPage } from "./pages/Backtest/BacktestPage";
import { MonitorPage } from "./pages/Monitor/MonitorPage";
import { WelcomePage } from "./pages/Welcome/WelcomePage";

const ResultsPage = lazy(() =>
  import("./pages/Results/ResultsPage").then((m) => ({ default: m.ResultsPage })),
);
const ResultsHistoryPage = lazy(() =>
  import("./pages/Results/ResultsHistoryPage").then((m) => ({ default: m.ResultsHistoryPage })),
);
const NewsPage = lazy(() =>
  import("./pages/News/NewsPage").then((m) => ({ default: m.NewsPage })),
);
const SettingsPage = lazy(() =>
  import("./pages/Settings/SettingsPage").then((m) => ({ default: m.SettingsPage })),
);
const LiveTradingPage = lazy(() =>
  import("./pages/LiveTrading/LiveTradingPage").then((m) => ({ default: m.LiveTradingPage })),
);
const DeployedModelsPage = lazy(() =>
  import("./pages/Models/DeployedModelsPage").then((m) => ({ default: m.DeployedModelsPage })),
);

function PageSpinner() {
  return (
    <div
      className="flex h-full items-center justify-center"
      style={{ backgroundColor: "var(--color-app)" }}
    >
      <div className="flex flex-col items-center gap-3">
        <div
          className="h-6 w-6 animate-spin rounded-full border-2"
          style={{ borderColor: "var(--color-border)", borderTopColor: "var(--color-accent)" }}
        />
        <span className="text-xs" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          Loading…
        </span>
      </div>
    </div>
  );
}

const WELCOME_KEY = "kodaquant-welcome-done";

export default function App() {
  const [showWelcome, setShowWelcome] = useState(() => !localStorage.getItem(WELCOME_KEY));

  if (showWelcome) {
    return <WelcomePage onComplete={() => setShowWelcome(false)} />;
  }

  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="backtest" element={<BacktestPage />} />
        <Route path="monitor" element={<MonitorPage />} />
        <Route
          path="results"
          element={<Suspense fallback={<PageSpinner />}><ResultsHistoryPage /></Suspense>}
        />
        <Route
          path="results/:jobId"
          element={<Suspense fallback={<PageSpinner />}><ResultsPage /></Suspense>}
        />
        <Route
          path="news"
          element={<Suspense fallback={<PageSpinner />}><NewsPage /></Suspense>}
        />
        <Route
          path="settings"
          element={<Suspense fallback={<PageSpinner />}><SettingsPage /></Suspense>}
        />
        <Route
          path="live-trading"
          element={<Suspense fallback={<PageSpinner />}><LiveTradingPage /></Suspense>}
        />
        <Route
          path="models"
          element={<Suspense fallback={<PageSpinner />}><DeployedModelsPage /></Suspense>}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}