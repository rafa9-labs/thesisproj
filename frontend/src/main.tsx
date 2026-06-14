import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { ErrorBoundary } from "./components/shared/ErrorBoundary";
import "./index.css";
import App from "./App";

const NEWS_CACHE_KEY = "kodaquant_news_cache_v1";

function persistNewsCache(queryClient: QueryClient) {
  try {
    const data = queryClient.getQueryData(["live-sentiment", "EURUSD"]);
    if (data) {
      localStorage.setItem(
        NEWS_CACHE_KEY,
        JSON.stringify({
          timestamp: Date.now(),
          sentiment: data,
        }),
      );
    }
  } catch {}
}

function restoreNewsCache(queryClient: QueryClient) {
  try {
    const raw = localStorage.getItem(NEWS_CACHE_KEY);
    if (!raw) return;
    const cached = JSON.parse(raw);
    if (Date.now() - cached.timestamp < 30 * 60_000) {
      queryClient.setQueryData(["live-sentiment", "EURUSD"], cached.sentiment);
    }
  } catch {}
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 30 * 60_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

restoreNewsCache(queryClient);

queryClient.prefetchQuery({
  queryKey: ["live-sentiment", "EURUSD"],
  queryFn: async () => {
    const { default: apiClient } = await import("./api/client");
    const { data } = await apiClient.get("/news/sentiment/live", { params: { pair: "EURUSD" } });
    return data;
  },
  staleTime: 60_000,
});

queryClient.prefetchQuery({
  queryKey: ["news-articles", undefined, 30],
  queryFn: async () => {
    const { default: apiClient } = await import("./api/client");
    const { data } = await apiClient.get("/news/articles", { params: { pair: "", days: 30 } });
    return data;
  },
  staleTime: 60_000,
});

queryClient.getQueryCache().subscribe((event) => {
  if (event.type === "updated" && event.query.queryKey[0] === "live-sentiment") {
    persistNewsCache(queryClient);
  }
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
);
