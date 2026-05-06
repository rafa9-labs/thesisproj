import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiClient from "./client";
import type {
  PairInfo,
  ModelInfo,
  ModelHyperparamsResponse,
  JobSummary,
  JobStatus,
  JobResults,
  BacktestRequest,
  HealthResponse,
  QuickTestPreset,
  DateRangeResponse,
  RuntimeEstimateRequest,
  RuntimeEstimateResponse,
  HpoIntensity,
  HeatmapResponse,
  NewsEvent,
  LicenseStatusResponse,
} from "./schemas";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data } = await apiClient.get<HealthResponse>("/health");
      return data;
    },
    staleTime: 10_000,
    refetchInterval: 30_000,
  });
}

export function usePairs() {
  return useQuery({
    queryKey: ["pairs"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ pairs: PairInfo[] }>("/pairs");
      return data.pairs;
    },
    staleTime: 5 * 60_000,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: ModelInfo[] }>("/models");
      return data.models;
    },
    staleTime: 5 * 60_000,
  });
}

export function useModelHyperparams() {
  return useQuery({
    queryKey: ["model-hyperparams"],
    queryFn: async () => {
      const { data } = await apiClient.get<ModelHyperparamsResponse>("/models/hyperparams");
      return data.models;
    },
    staleTime: 5 * 60_000,
  });
}

export function useJobHistory(limit = 50) {
  return useQuery({
    queryKey: ["jobs", limit],
    queryFn: async () => {
      const { data } = await apiClient.get<{ jobs: JobSummary[] }>("/backtest", {
        params: { limit },
      });
      return data.jobs;
    },
    staleTime: 10_000,
  });
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<JobStatus>(`/backtest/${jobId}`);
      return data;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "pending" || status === "running") return 2_000;
      return false;
    },
  });
}

export function useJobResults(jobId: string | null) {
  const { data: statusData } = useJobStatus(jobId);
  const isDone = statusData?.status === "completed" || statusData?.status === "failed";

  return useQuery({
    queryKey: ["job-results", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<JobResults>(`/backtest/${jobId}/results`);
      return data;
    },
    enabled: !!jobId && isDone,
    refetchInterval: isDone ? false : 3_000,
  });
}

export function useSubmitBacktest() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: BacktestRequest) => {
      const { data } = await apiClient.post<{
        job_id: string;
        status: string;
        pair: string;
        models: string[];
      }>("/backtest", payload);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useDeleteJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      await apiClient.delete(`/backtest/${jobId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useConfig() {
  return useQuery({
    queryKey: ["config"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ settings: Record<string, unknown> }>("/config");
      return data.settings;
    },
    staleTime: 60_000,
  });
}

export function useSaveConfig() {
  return useMutation({
    mutationFn: async (settings: Record<string, unknown>) => {
      await apiClient.put("/config", { settings });
    },
  });
}

export interface NewsStatus {
  sentiment_backend: string;
  cached_articles: number;
  event_types: string[];
  features: {
    vader_compound: boolean;
    event_flags: boolean;
    news_volume_windows: number[];
  };
  finbert_available: boolean;
}

export function useNewsStatus() {
  return useQuery({
    queryKey: ["news-status"],
    queryFn: async () => {
      const { data } = await apiClient.get<NewsStatus>("/news/status");
      return data;
    },
    staleTime: 60_000,
  });
}

export function useQuickTestPresets() {
  return useQuery({
    queryKey: ["quick-test-presets"],
    queryFn: async () => {
      const { data } = await apiClient.get<QuickTestPreset[]>("/backtest/presets");
      return data;
    },
    staleTime: 5 * 60_000,
  });
}

export function useDateRanges(pair: string, timeframe: string) {
  return useQuery({
    queryKey: ["date-ranges", pair, timeframe],
    queryFn: async () => {
      const { data } = await apiClient.get<DateRangeResponse>("/backtest/date-ranges", {
        params: { pair, timeframe },
      });
      return data;
    },
    enabled: !!pair && !!timeframe,
    staleTime: 5 * 60_000,
  });
}

export function useRuntimeEstimate(models: string[], months: number, hpoIntensity: HpoIntensity) {
  return useQuery({
    queryKey: ["runtime-estimate", models, months, hpoIntensity],
    queryFn: async () => {
      const { data } = await apiClient.post<RuntimeEstimateResponse>(
        "/backtest/estimate-runtime",
        { models, months, hpo_intensity: hpoIntensity } as RuntimeEstimateRequest,
      );
      return data;
    },
    enabled: models.length > 0 && months > 0,
    staleTime: 30_000,
  });
}

export function useHeatmap() {
  return useQuery({
    queryKey: ["heatmap"],
    queryFn: async () => {
      const { data } = await apiClient.get<HeatmapResponse>("/backtest/heatmap");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useNewsEvents(start: number | null, end: number | null, impact?: string) {
  return useQuery({
    queryKey: ["news-events", start, end, impact],
    queryFn: async () => {
      const { data } = await apiClient.get<NewsEvent[]>("/news/events", {
        params: { start, end, impact },
      });
      return data;
    },
    enabled: start !== null && end !== null,
    staleTime: 5 * 60_000,
  });
}

export function useLiveSentiment(pair: string = "EURUSD") {
  return useQuery({
    queryKey: ["live-sentiment", pair],
    queryFn: async () => {
      const { data } = await apiClient.get<Record<string, any>>("/news/sentiment/live", {
        params: { pair },
      });
      return data;
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });
}

export function useLicenseStatus() {
  return useQuery({
    queryKey: ["license-status"],
    queryFn: async () => {
      const { data } = await apiClient.get<LicenseStatusResponse>("/license/status");
      return data;
    },
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

export function useActivateLicense() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (licenseKey: string) => {
      const { data } = await apiClient.post<LicenseStatusResponse>("/license/activate", {
        license_key: licenseKey,
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["license-status"] });
    },
  });
}

export function useDeactivateLicense() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      await apiClient.post("/license/deactivate");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["license-status"] });
    },
  });
}

export function useStartTrial() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post("/license/trial");
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["license-status"] });
    },
  });
}

export function useResultsHistory(params: { limit?: number; offset?: number; pair?: string; model?: string; sort_by?: string; sort_order?: string } = {}) {
  return useQuery({
    queryKey: ["results-history", params],
    queryFn: async () => {
      const { data } = await apiClient.get<{ results: import("./schemas").BacktestSummaryItem[]; total: number }>("/backtest/results/summary", { params });
      return data;
    },
    staleTime: 10_000,
  });
}

export function useLivePrices(pairs: string[], lookbackBars = 50) {
  return useQuery({
    queryKey: ["live-prices", ...pairs, lookbackBars],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").LivePricesResponse>("/prices/live", {
        params: { pairs: pairs.join(","), lookback_bars: lookbackBars },
      });
      return data;
    },
    enabled: pairs.length > 0,
    refetchInterval: 3_000,
    staleTime: 2_000,
    retry: 1,
  });
}

export function useCandles(pair: string, timeframe: string, limit = 200) {
  return useQuery({
    queryKey: ["candles", pair, timeframe, limit],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").CandlesResponse>(`/candles/${pair}/${timeframe}`, {
        params: { limit },
      });
      return data;
    },
    staleTime: 15_000,
    enabled: !!pair && !!timeframe,
  });
}

export function useTradeChartData(jobId: string, model: string) {
  return useQuery({
    queryKey: ["trade-chart-data", jobId, model],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").TradeChartData>(`/backtest/${jobId}/trades/chart-data`, {
        params: { model },
      });
      return data;
    },
    enabled: !!jobId && !!model,
    staleTime: Infinity,
  });
}
