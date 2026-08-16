import { useMemo } from "react";
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
  BacktestSummaryItem,
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
  SeedDemoResponse,
  CommitteeConfigSchema,
  RegimeMatrixResponse,
  RegimeLabelsResponse,
  CommitteeSnapshotListResponse,
  SavedCommitteeListResponse,
  FullCycleRequest,
  FullCycleStatusResponse,
  FullCycleResultsResponse,
  FullCycleHistoryResponse,
  CancelFullCycleResponse,
  LogsResponse,
  RetrainRequest,
  RetrainStartedResponse,
  RetrainStatus,
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
  const isDone = statusData?.status === "completed";
  const isFailed = statusData?.status === "failed";

  return useQuery({
    queryKey: ["job-results", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<JobResults>(`/backtest/${jobId}/results`);
      return data;
    },
    enabled: !!jobId && isDone,
    refetchInterval: isDone || isFailed ? false : 3_000,
  });
}

export function useActiveBacktests() {
  return useQuery({
    queryKey: ["active-backtests"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ jobs: JobSummary[]; total: number }>(
        "/backtest/active",
      );
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
      const { data } = await apiClient.post<{ job_id: string; status: string }>(
        `/backtest/${jobId}/force-stop`,
      );
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

export function useRerunBacktest() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (row: BacktestSummaryItem) => {
      const payload: BacktestRequest = {
        pair: row.pair || "EURUSD",
        models: row.models || [],
        seed: Math.floor(Math.random() * 100000),
        config_overrides: {},
      };
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
      queryClient.invalidateQueries({ queryKey: ["results-history"] });
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
      queryClient.invalidateQueries({ queryKey: ["results-history"] });
    },
  });
}

export function useBatchDeleteJobs() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobIds: string[]) => {
      const { data } = await apiClient.post<{ deleted: string[]; not_found: string[] }>(
        "/backtest/batch-delete",
        { job_ids: jobIds },
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["results-history"] });
    },
  });
}

export function useBatchDeleteCommittees() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobIds: string[]) => {
      const { data } = await apiClient.post<{ deleted: string[]; not_found: string[] }>(
        "/committee/full-cycle/batch-delete",
        { job_ids: jobIds },
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["full-cycle", "history"] });
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

export interface ExecutionSettings {
  max_concurrent_backtests: number;
  gpu_enabled: boolean;
  max_concurrent_gpu: number;
}

export function useExecutionSettings() {
  return useQuery({
    queryKey: ["config", "execution"],
    queryFn: async () => {
      const { data } = await apiClient.get<ExecutionSettings>("/config/execution");
      return data;
    },
    staleTime: 60_000,
  });
}

export function useSaveExecutionSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (settings: Partial<ExecutionSettings>) => {
      const { data } = await apiClient.put<{ status: string; message: string }>(
        "/config/execution",
        settings,
      );
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["config", "execution"] });
    },
  });
}

export interface HardwareInfo {
  cpu: {
    model: string;
    physical_cores: number;
    logical_cores: number;
    is_hybrid: boolean;
    p_cores: number;
    e_cores: number;
    ram_total_gb: number;
  };
  gpu: {
    available: boolean;
    name: string;
    vram_mb: number;
    compute_capability: string;
    tensor_cores: boolean;
  };
  budget: {
    blas_threads: number;
    cv_n_jobs: number;
    batch_size: number;
    xla_enabled: boolean;
    vram_limit_mb: number;
    ram_limit_gb: number;
  };
}

export function useHardware() {
  return useQuery({
    queryKey: ["hardware"],
    queryFn: async () => {
      const { data } = await apiClient.get<HardwareInfo>("/system/hardware");
      return data;
    },
    staleTime: 300_000,
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
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file);
      const { data } = await apiClient.post("/data/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data as { status: string; pair: string; timeframe: string; rows: number };
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

export function useRuntimeEstimate(models: string[], months: number, hpoIntensity: HpoIntensity, nTrials?: number) {
  return useQuery({
    queryKey: ["runtime-estimate", models, months, hpoIntensity, nTrials],
    queryFn: async () => {
      const { data } = await apiClient.post<RuntimeEstimateResponse>("/backtest/estimate-runtime", {
        models,
        months,
        hpo_intensity: hpoIntensity,
        ...(nTrials !== undefined && { n_trials: nTrials }),
      } as RuntimeEstimateRequest);
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

export function useNewsArticles(pair?: string, days?: number) {
  return useQuery({
    queryKey: ["news-articles", pair, days],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").NewsArticlesResponse>(
        "/news/articles",
        {
          params: { pair: pair ?? "", days: days ?? 30 },
        },
      );
      return data;
    },
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });
}

export function useLiveSentiment(pair: string = "EURUSD", pollingEnabled = true) {
  return useQuery({
    queryKey: ["live-sentiment", pair],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").LiveSentimentResponse>(
        "/news/sentiment/live",
        {
          params: { pair },
        },
      );
      return data;
    },
    staleTime: pollingEnabled ? 60_000 : Infinity,
    refetchInterval: pollingEnabled ? 5 * 60_000 : false,
    enabled: !!pair,
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

export function useResultsHistory(
  params: {
    limit?: number;
    offset?: number;
    pair?: string;
    model?: string;
    sort_by?: string;
    sort_order?: string;
    status?: string;
  } = {},
) {
  return useQuery({
    queryKey: ["results-history", params],
    queryFn: async () => {
      const { data } = await apiClient.get<{
        results: import("./schemas").BacktestSummaryItem[];
        total: number;
      }>("/backtest/results/summary", { params });
      return data;
    },
    staleTime: 10_000,
  });
}

export function useLivePrices(pairs: string[], lookbackBars = 50, pollingEnabled = true) {
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
    refetchInterval: pollingEnabled ? 3_000 : false,
    staleTime: pollingEnabled ? 5_000 : Infinity,
    retry: 1,
  });
}

export function useCandles(
  pair: string,
  timeframe: string,
  limit = 200,
  refetchInterval?: number | false,
) {
  const refetchMs = useMemo(() => {
    if (refetchInterval !== undefined) return refetchInterval;
    const t = timeframe.toUpperCase();
    if (t === "M1") return 15_000;
    if (t === "M5") return 30_000;
    if (t === "M15" || t === "M30") return 60_000;
    if (t === "H1") return 120_000;
    if (t === "H4") return 300_000;
    return 300_000;
  }, [timeframe, refetchInterval]);

  return useQuery({
    queryKey: ["candles", pair, timeframe, limit],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").CandlesResponse>(
        `/candles/${pair}/${timeframe}`,
        {
          params: { limit },
        },
      );
      return data;
    },
    staleTime: 15_000,
    refetchInterval: refetchMs,
    enabled: !!pair && !!timeframe,
  });
}

export function useLiveCandles(
  pair: string,
  timeframe: string,
  limit = 1000,
) {
  return useQuery({
    queryKey: ["live-candles", pair, timeframe, limit],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").CandlesResponse>(
        `/candles/${pair}/${timeframe}/live`,
        { params: { limit } },
      );
      return data;
    },
    staleTime: Infinity,
    gcTime: 24 * 60 * 60 * 1000,
    enabled: !!pair && !!timeframe,
  });
}

export function useTradeChartData(jobId: string, model: string) {
  return useQuery({
    queryKey: ["trade-chart-data", jobId, model],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").TradeChartData>(
        `/backtest/${jobId}/trades/chart-data`,
        {
          params: { model },
        },
      );
      return data;
    },
    enabled: !!jobId && !!model,
    staleTime: Infinity,
  });
}

export function useDeployPaperSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: import("./schemas").DeployPaperRequest) => {
      const { data } = await apiClient.post<import("./schemas").PaperSessionInfo>(
        "/trading/paper/start",
        payload,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["paper-sessions"] });
    },
  });
}

export function useStopPaperSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionId: string) => {
      const { data } = await apiClient.post<import("./schemas").PaperStopResult>(
        `/trading/paper/${sessionId}/stop`,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["paper-sessions"] });
    },
  });
}

export function usePaperSessionStatus(sessionId: string | null) {
  return useQuery({
    queryKey: ["paper-session", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").PaperSessionInfo>(
        `/trading/paper/${sessionId}/status`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 5_000,
  });
}

export function usePaperSessionTrades(sessionId: string | null) {
  return useQuery({
    queryKey: ["paper-session-trades", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").PaperTradesResponse>(
        `/trading/paper/${sessionId}/trades`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 10_000,
  });
}

export function usePaperSessionSummary(sessionId: string | null) {
  return useQuery({
    queryKey: ["paper-session-summary", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<import("./schemas").PaperSummaryResponse>(
        `/trading/paper/${sessionId}/summary`,
      );
      return data;
    },
    enabled: !!sessionId && false,
  });
}

export function usePaperSessions() {
  return useQuery({
    queryKey: ["paper-sessions"],
    queryFn: async () => {
      const { data } = await apiClient.get<
        Array<{
          session_id: string;
          pair: string;
          model_type: string;
          timeframe: string;
          status: string;
          created_at: string;
        }>
      >("/trading/paper/sessions");
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

export const _progressCursors = new Map<string, number>();

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
          console.log(
            "[POLL] events:",
            data.events.length,
            "total:",
            data.total,
            "cursor:",
            cursor,
          );
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
      activeJobs instanceof Map &&
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
        `/backtest/${jobId}/analyze?model=${encodeURIComponent(model ?? "")}`,
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
      const { data } = await apiClient.post<{
        status: string;
        model_id: string;
        snapshot_path: string;
      }>(`/models/save-from-job/${jobId}${params}`);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployed-models"] });
      queryClient.invalidateQueries({ queryKey: ["deployed-models-for-live"] });
    },
  });
}

export function useBulkDeleteModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (modelIds: string[]) => {
      const { data } = await apiClient.post<{ status: string; deleted: number }>(
        "/models/deployed/bulk/delete",
        { model_ids: modelIds },
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployed-models"] });
      queryClient.invalidateQueries({ queryKey: ["deployed-models-for-live"] });
    },
  });
}

export function useBulkActivateModels() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (modelIds: string[]) => {
      const { data } = await apiClient.post<{ status: string; activated: number }>(
        "/models/deployed/bulk/activate",
        { model_ids: modelIds },
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["deployed-models"] });
      queryClient.invalidateQueries({ queryKey: ["deployed-models-for-live"] });
    },
  });
}

export function useDeployLiveSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: import("./schemas").DeployLiveRequest) => {
      const { data } = await apiClient.post<import("./schemas").LiveSessionInfo>(
        "/trading/live/start",
        payload,
      );
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
      const { data } = await apiClient.post<import("./schemas").LiveStopResult>(
        `/trading/live/${sessionId}/stop`,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["live-sessions"] });
    },
  });
}

export function useEmergencyKillSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionId: string) => {
      const { data } = await apiClient.post<import("./schemas").LiveEmergencyResult>(
        `/trading/live/${sessionId}/emergency`,
      );
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
      const { data } = await apiClient.get<import("./schemas").LiveSessionInfo>(
        `/trading/live/${sessionId}/status`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 5_000,
  });
}

export function useLiveJournal(sessionId: string | null) {
  return useQuery({
    queryKey: ["live-journal", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<{ journal: import("./schemas").LiveJournalItem[] }>(
        `/trading/live/${sessionId}/journal`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 10_000,
  });
}

export function useLiveRiskState(sessionId: string | null) {
  return useQuery({
    queryKey: ["live-risk", sessionId],
    queryFn: async () => {
      const { data } = await apiClient.get<{ risk: import("./schemas").LiveRiskState }>(
        `/trading/live/${sessionId}/risk`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: 15_000,
  });
}

export function useLiveSessionsList() {
  return useQuery({
    queryKey: ["live-sessions"],
    queryFn: async () => {
      const { data } =
        await apiClient.get<Array<import("./schemas").LiveSessionInfo>>("/trading/live/sessions");
      return data;
    },
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useDemoSeed() {
  return useMutation({
    mutationFn: async (payload?: { pairs?: string[]; timeframes?: string[] }) => {
      const { data } = await apiClient.post<SeedDemoResponse>("/data/seed-demo", payload ?? {});
      return data;
    },
  });
}

// ════════════════════════════════════════════════════════════════════
// Committee (Racecar Phases A-E)
// ════════════════════════════════════════════════════════════════════

export function useCommitteeConfig() {
  return useQuery({
    queryKey: ["committee", "config"],
    queryFn: async () => {
      const { data } = await apiClient.get<CommitteeConfigSchema>("/committee/config");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useSaveCommitteeConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (config: CommitteeConfigSchema) => {
      const { data } = await apiClient.post<{ status: string }>("/committee/config", config);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["committee", "config"] });
    },
  });
}

export function useRegimeMatrix() {
  return useQuery({
    queryKey: ["committee", "regime-matrix"],
    queryFn: async () => {
      const { data } = await apiClient.get<RegimeMatrixResponse>("/committee/regime-matrix");
      return data;
    },
    staleTime: 60_000,
  });
}

export function useRegimeLabels(pair: string, timeframe: string, bars: number = 500) {
  return useQuery({
    queryKey: ["committee", "regime-labels", pair, timeframe, bars],
    queryFn: async () => {
      const { data } = await apiClient.get<RegimeLabelsResponse>(
        `/committee/regime-labels/${pair}/${timeframe}?bars=${bars}`,
      );
      return data;
    },
    staleTime: 60_000,
    enabled: !!pair && !!timeframe,
  });
}
export function useCommitteeSnapshots() {
  return useQuery({
    queryKey: ["committee", "snapshots"],
    queryFn: async () => {
      const { data } = await apiClient.get<CommitteeSnapshotListResponse>("/committee/snapshots");
      return data;
    },
    staleTime: 30_000,
  });
}

export function useSavedCommittees() {
  return useQuery({
    queryKey: ["committee", "saved"],
    queryFn: async () => {
      const { data } = await apiClient.get<SavedCommitteeListResponse>("/committee/saved");
      return data;
    },
    staleTime: 15_000,
  });
}

export function useSaveCommittee() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (req: {
      name: string;
      full_cycle_job_id?: string;
      pair?: string;
      timeframe?: string;
      config_json: Record<string, unknown>;
      trust_score?: number;
      avg_sharpe?: number;
      tags?: string[];
    }) => {
      const { data } = await apiClient.post<{ status: string; id: string }>(
        "/committee/saved",
        req,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["committee", "saved"] });
    },
  });
}

export function useDeleteSavedCommittee() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await apiClient.delete(`/committee/saved/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["committee", "saved"] });
    },
  });
}

export function useActivateSavedCommittee() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await apiClient.post(`/committee/saved/${id}/activate`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["committee", "saved"] });
      queryClient.invalidateQueries({ queryKey: ["committee", "config"] });
    },
  });
}

// ════════════════════════════════════════════════════════════════════
// Live Metrics (committee sessions)
// ════════════════════════════════════════════════════════════════════

export interface CommitteeMetricsResponse {
  session_id: string;
  uptime_seconds: number;
  bar_count: number;
  signal_count: number;
  non_zero_signals: number;
  committee_healthy: boolean;
  current_regime: string;
  regime_distribution: Record<string, number>;
  trust_score: number | null;
  trust_multiplier: number;
  effective_multiplier: number;
  throttle_summary: Record<string, { multiplier: number; level: string }>;
  per_model_health: Record<
    string,
    {
      rolling_sharpe: number | null;
      rolling_hit_rate: number | null;
      total_signals: number;
      wins: number;
      losses: number;
      status: "healthy" | "unhealthy" | "insufficient_data";
    }
  >;
  recent_signals: Array<{
    timestamp: string;
    signal: number;
    confidence: number;
    regime: string;
    is_healthy: boolean;
    meta_override: boolean;
    throttle_level: string;
    active_models: string[];
  }>;
  error?: string;
}

export function useCommitteeMetrics(sessionId: string | null) {
  return useQuery({
    queryKey: ["live", "committee", "metrics", sessionId],
    queryFn: async () => {
      if (!sessionId) return null;
      const { data } = await apiClient.get<CommitteeMetricsResponse>(
        `/live/committee/${sessionId}/metrics`,
      );
      return data;
    },
    refetchInterval: 15_000,
    enabled: !!sessionId,
  });
}

// ── Fast Loop Retrain ──────────────────────────────────────────

export function useRetrainCommittee() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ sessionId, ...req }: { sessionId: string } & RetrainRequest) => {
      const { data } = await apiClient.post<RetrainStartedResponse>(
        `/trading/live/committee/${sessionId}/retrain`,
        req,
      );
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["live", "committee", "metrics"] });
    },
  });
}

export function useRetrainStatus(sessionId: string | null) {
  return useQuery({
    queryKey: ["live", "committee", "retrain", sessionId],
    queryFn: async () => {
      if (!sessionId) return null;
      const { data } = await apiClient.get<RetrainStatus>(
        `/trading/live/committee/${sessionId}/retrain/status`,
      );
      return data;
    },
    enabled: !!sessionId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed" || status === "idle") return false;
      return 2_000;
    },
  });
}

// ════════════════════════════════════════════════════════════════════
// Full Cycle — Racecar (B→C→D) + Factory (optimization) in one shot
// ════════════════════════════════════════════════════════════════════

export function useStartFullCycle() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (req: FullCycleRequest) => {
      const { data } = await apiClient.post<FullCycleStatusResponse>("/committee/full-cycle", req);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["full-cycle"] });
    },
  });
}

export function useFullCycleStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["full-cycle", "status", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<FullCycleStatusResponse>(
        `/committee/full-cycle/${jobId}/status`,
      );
      return data;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const phase = query.state.data?.phase;
      if (
        !phase ||
        phase === "completed" ||
        phase === "failed" ||
        phase === "validation_failed" ||
        phase === "cancelled" ||
        phase === "orphaned"
      )
        return false;
      return 2_000;
    },
  });
}

export function useFullCycleResults(jobId: string | null) {
  return useQuery({
    queryKey: ["full-cycle", "results", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<FullCycleResultsResponse>(
        `/committee/full-cycle/${jobId}/results`,
      );
      return data;
    },
    enabled: !!jobId,
    staleTime: 30_000,
  });
}

export function useFullCycleHistory() {
  return useQuery({
    queryKey: ["full-cycle", "history"],
    queryFn: async () => {
      const { data } = await apiClient.get<FullCycleHistoryResponse>(
        "/committee/full-cycle/history",
      );
      return data;
    },
    staleTime: 10_000,
  });
}

export function useFullCycleLogs(jobId: string | null, since: number) {
  return useQuery({
    queryKey: ["full-cycle", "logs", jobId, since],
    queryFn: async () => {
      const { data } = await apiClient.get<LogsResponse>(`/committee/full-cycle/${jobId}/logs`, {
        params: { since },
      });
      return data;
    },
    enabled: !!jobId,
    refetchInterval: () => {
      return 1_500;
    },
  });
}

export function useCancelFullCycle() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const { data } = await apiClient.post<CancelFullCycleResponse>(
        `/committee/full-cycle/${jobId}/cancel`,
      );
      return data;
    },
    onSuccess: (_data, jobId) => {
      queryClient.invalidateQueries({ queryKey: ["full-cycle", "status", jobId] });
      queryClient.invalidateQueries({ queryKey: ["full-cycle", "history"] });
    },
  });
}

// ── Vast.ai GPU rental ──────────────────────────────────────────────

export interface VastSettings {
  vast_enabled: boolean;
  vast_min_gpu_class: string;
  vast_min_vram_gb: number;
  vast_max_dph: number;
  vast_disk_gb: number;
  vast_image: string;
  vast_repo_url: string;
  vast_remote_api_url: string;
  has_api_key: boolean;
}

export interface VastOffer {
  ask_id: number;
  machine_id: number | null;
  gpu_name: string;
  gpu_ram_gb: number;
  dph_total: number;
  dlperf: number | null;
  num_gpus: number;
  cpu_cores: number | null;
  reliability: number | null;
}

export interface VastInstance {
  id: number;
  actual_status: string;
  status_msg: string | null;
  gpu_name: string;
  dph_total: number;
  ssh_host: string | null;
  ssh_port: number | null;
  public_ipaddr: string | null;
  remote_api_url: string | null;
}

export function useVastSettings() {
  return useQuery({
    queryKey: ["vast", "settings"],
    queryFn: async () => {
      const { data } = await apiClient.get<VastSettings>("/vast/settings");
      return data;
    },
    staleTime: 60_000,
  });
}

export function useSaveVastSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (settings: Partial<VastSettings>) => {
      const { data } = await apiClient.put("/vast/settings", settings);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vast", "settings"] });
    },
  });
}

export function useStoreVastApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (value: string) => {
      const { data } = await apiClient.post("/vast/api-key", { value });
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vast", "settings"] });
    },
  });
}

export function useVastOffers(filters: {
  gpu_class?: string;
  min_vram_gb?: number;
  max_dph?: number;
} | null) {
  return useQuery({
    queryKey: ["vast", "offers", filters],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (filters?.gpu_class) params.set("gpu_class", filters.gpu_class);
      if (filters?.min_vram_gb != null) params.set("min_vram_gb", String(filters.min_vram_gb));
      if (filters?.max_dph != null) params.set("max_dph", String(filters.max_dph));
      const qs = params.toString();
      const { data } = await apiClient.get<{ offers: VastOffer[] }>(
        `/vast/offers${qs ? `?${qs}` : ""}`,
      );
      return data.offers;
    },
    enabled: !!filters,
    staleTime: 30_000,
  });
}

export function useLaunchVastInstance() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (payload: {
      ask_id?: number;
      image?: string;
      disk_gb?: number;
      gpu_class?: string;
    }) => {
      const { data } = await apiClient.post<{
        success: boolean;
        instance_id: number;
        ask_id: number;
      }>("/vast/instances", payload);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vast", "instances"] });
    },
  });
}

export function useVastInstances() {
  return useQuery({
    queryKey: ["vast", "instances"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ instances: VastInstance[] }>(
        "/vast/instances",
      );
      return data.instances;
    },
    refetchInterval: 20_000,
  });
}

export function useDestroyVastInstance() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (instanceId: number) => {
      const { data } = await apiClient.delete(`/vast/instances/${instanceId}`);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vast", "instances"] });
    },
  });
}
