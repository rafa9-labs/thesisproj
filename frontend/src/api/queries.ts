import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiClient from "./client";
import { useJobStore } from "@/stores/useJobStore";
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
  DataStatusResponse,
  DefinePairRequest,
  DefinePairResponse,
  WsEvent,
  LlmAnalysisResponse,
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
    retry: 1,
    refetchInterval: (query) => {
      const error = query.state.error;
      if (error && (error as { response?: { status?: number } })?.response?.status === 404) {
        return false;
      }
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

export function useActiveBacktests() {
  return useQuery({
    queryKey: ["active-backtests"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ jobs: JobSummary[]; total: number }>("/backtest/active");
      return data;
    },
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useForceStopJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const { data } = await apiClient.post<{ job_id: string; status: string }>(`/backtest/${jobId}/force-stop`);
      return data;
    },
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["active-backtests"] });
      queryClient.invalidateQueries({ queryKey: ["job", jobId] });
      // Allow DB update to propagate, then refetch to clear any stale 409 state
      setTimeout(() => {
        queryClient.refetchQueries({ queryKey: ["active-backtests"] });
      }, 800);
    },
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
      queryClient.invalidateQueries({ queryKey: ["active-backtests"] });
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

export function useStoreApiKey() {
  return useMutation({
    mutationFn: async ({ name, value }: { name: string; value: string }) => {
      await apiClient.post("/config/api-key", { name, value });
    },
  });
}

export function useStoreKv() {
  return useMutation({
    mutationFn: async ({ key, value }: { key: string; value: string }) => {
      await apiClient.post("/config/kv", { key, value });
    },
  });
}

export function useCredentialStatus() {
  return useQuery({
    queryKey: ["credential-status"],
    queryFn: async () => {
      const { data } = await apiClient.get<{
        oanda_token_configured: boolean;
        oanda_account_id_configured: boolean;
      }>("/config/credential-status");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useUploadCsv() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (formData: FormData) => {
      const { data } = await apiClient.post<{
        status: string;
        filename: string;
        pair: string;
        timeframe: string;
      }>("/data/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pairs"] });
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
      const { data } = await apiClient.get<import("./schemas").LiveSentimentResponse>("/news/sentiment/live", {
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
  const validPairs = pairs.filter((p) => p && p.trim() !== "");
  return useQuery({
    queryKey: ["live-prices", ...validPairs, lookbackBars],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").LivePricesResponse>("/prices/live", {
        params: { pairs: validPairs.join(","), lookback_bars: lookbackBars },
      });
      return data;
    },
    enabled: validPairs.length > 0,
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

export function useDeployLiveSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: { pair: string; model: string; timeframe: string; initial_equity?: number }) => {
      const { data } = await apiClient.post<import("./schemas").LiveSessionInfo>("/live/deploy", payload);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["live-sessions"] });
    },
  });
}

export function useStopLiveSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionId: string) => {
      const { data } = await apiClient.post<{
        session_id: string;
        status: string;
        equity: number;
        signal_count: number;
      }>(`/live/${sessionId}/stop`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["live-sessions"] });
    },
  });
}

export function useLiveSessionStatus(sessionId: string | null) {
  return useQuery({
    queryKey: ["live-session", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").LiveSessionInfo>(`/live/${sessionId}/status`);
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 5_000,
  });
}

export function useLiveSessions() {
  return useQuery({
    queryKey: ["live-sessions"],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").LiveSessionInfo[]>("/live/sessions");
      return data;
    },
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useDataStatus(pair: string) {
  return useQuery({
    queryKey: ["data-status", pair],
    queryFn: async () => {
      const { data } = await apiClient.get<DataStatusResponse>(`/pairs/${pair}/data-status`);
      return data;
    },
    enabled: !!pair && pair.length === 6,
    staleTime: 30_000,
  });
}

export function useDefinePair() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (req: DefinePairRequest) => {
      const { data } = await apiClient.post<DefinePairResponse>("/pairs/define", req);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pairs"] });
      queryClient.invalidateQueries({ queryKey: ["data-status"] });
    },
  });
}

export function useDownloadData() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ pair, years = 10 }: { pair: string; years?: number }) => {
      const { data } = await apiClient.post<{
        job_id: string;
        pair: string;
        status: string;
      }>("/data/download", { pair, years });
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pairs"] });
      queryClient.invalidateQueries({ queryKey: ["data-status"] });
    },
  });
}

export function useDownloadJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["download-job", jobId],
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

const _progressCursors = new Map<string, number>();

export function useBacktestProgress(jobId: string | null) {
  const handleWsEvent = useJobStore((s) => s.handleWsEvent);
  const activeJobs = useJobStore((s) => s.activeJobs);

  return useQuery({
    queryKey: ["backtest-progress", jobId],
    queryFn: async () => {
      if (!jobId) return null;
      const cursor = _progressCursors.get(jobId) ?? 0;
      const { data } = await apiClient.get<{ events: WsEvent[]; total: number }>(
        `/backtest/${jobId}/events?after=${cursor}`,
      );
      if (data.events && data.events.length > 0) {
        if (import.meta.env.DEV) {
          console.log("[POLL] events:", data.events.length, "total:", data.total, "cursor:", cursor);
        }
        for (const evt of data.events) {
          handleWsEvent(evt as WsEvent);
        }
        _progressCursors.set(jobId, data.total);
      }
      return data;
    },
    enabled:
      !!jobId &&
      activeJobs.has(jobId) &&
      (activeJobs.get(jobId)?.status === "pending" || activeJobs.get(jobId)?.status === "running"),
    refetchInterval: 2_000,
    staleTime: 0,
    refetchOnMount: true,
  });
}

export function useLlmAnalysis(jobId: string | null, model: string | null) {
  return useQuery({
    queryKey: ["llm-analysis", jobId, model],
    queryFn: async () => {
      const { data } = await apiClient.post<LlmAnalysisResponse>(
        `/backtest/${jobId}/analyze?model=${encodeURIComponent(model ?? "")}`
      );
      return data;
    },
    enabled: false,
    staleTime: 60_000,
    retry: 1,
  });
}

export function useSaveModelFromJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ jobId, modelName }: { jobId: string; modelName?: string }) => {
      const params = modelName ? `?model_name=${encodeURIComponent(modelName)}` : "";
      const { data } = await apiClient.post<{ status: string; model_id: string; snapshot_path: string }>(
        `/models/save-from-job/${jobId}${params}`
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployed-models"] });
    },
  });
}
