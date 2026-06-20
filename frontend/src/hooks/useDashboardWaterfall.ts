import { useMemo, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { usePairs } from "@/api/queries";
import { computeDashboardKPIs, computeCommitteeKPIs } from "@/pages/Dashboard/DashboardKPIs";
import type { DashboardKPIValues, CommitteeKPIValues } from "@/pages/Dashboard/DashboardKPIs";
import type {
  LivePricesResponse,
  CandlesResponse,
  LiveSentimentResponse,
  NewsArticlesResponse,
  JobSummary,
  JobResults,
  FullCycleHistoryEntry,
  FullCycleHistoryResponse,
} from "@/api/schemas";

export type { DashboardKPIValues, CommitteeKPIValues };

export function useDashboardWaterfall(activePair: string, pollingEnabled = true) {
  const queryClient = useQueryClient();

  const pairsQuery = usePairs();

  const top3Pairs = useMemo(() => {
    const all = (pairsQuery.data ?? [])
      .map((p) => p.pair?.symbol ?? "")
      .filter((s) => s !== "");
    return all.length > 0 ? all.slice(0, 3) : ["EURUSD", "GBPUSD", "USDJPY"];
  }, [pairsQuery.data]);

  const dataGate = pairsQuery.isSuccess;

  const pricesQuery = useQuery({
    queryKey: ["live-prices", ...top3Pairs, 50],
    queryFn: async () => {
      const { data } = await apiClient.get<LivePricesResponse>("/prices/live", {
        params: { pairs: top3Pairs.join(","), lookback_bars: 50 },
      });
      return data;
    },
    enabled: dataGate && top3Pairs.length > 0,
    refetchInterval: pollingEnabled ? 3_000 : false,
    staleTime: pollingEnabled ? 5_000 : Infinity,
    retry: 1,
  });

  const candlesQuery = useQuery({
    queryKey: ["candles", activePair, "M30", 150],
    queryFn: async () => {
      const { data } = await apiClient.get<CandlesResponse>(
        `/candles/${activePair}/M30`,
        { params: { limit: 150 } },
      );
      return data;
    },
    enabled: dataGate && !!activePair,
    staleTime: 15_000,
  });

  const sentimentQuery = useQuery({
    queryKey: ["live-sentiment", activePair],
    queryFn: async () => {
      const { data } = await apiClient.get<LiveSentimentResponse>(
        "/news/sentiment/live",
        { params: { pair: activePair } },
      );
      return data;
    },
    enabled: dataGate && !!activePair,
    staleTime: pollingEnabled ? 60_000 : Infinity,
    refetchInterval: pollingEnabled ? 5 * 60_000 : false,
  });

  const articlesQuery = useQuery({
    queryKey: ["news-articles", activePair, 7],
    queryFn: async () => {
      const { data } = await apiClient.get<NewsArticlesResponse>(
        "/news/articles",
        { params: { pair: activePair, days: 7 } },
      );
      return data;
    },
    enabled: dataGate && !!activePair,
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });

  const jobsQuery = useQuery({
    queryKey: ["jobs", 50],
    queryFn: async () => {
      const { data } = await apiClient.get<{ jobs: JobSummary[] }>("/backtest", {
        params: { limit: 50 },
      });
      return data.jobs;
    },
    enabled: dataGate,
    staleTime: 10_000,
  });

  const completedIds = useMemo(() => {
    const completed = (jobsQuery.data ?? [])
      .filter((j) => j.status === "completed")
      .slice(0, 5);
    return completed.map((j) => j.job_id);
  }, [jobsQuery.data]);

  const resultsQuery = useQuery({
    queryKey: ["dashboard-aggregate", completedIds],
    queryFn: async (): Promise<JobResults[]> => {
      const results = await Promise.allSettled(
        completedIds.map((id) =>
          apiClient.get<JobResults>(`/backtest/${id}/results`).then((r) => r.data),
        ),
      );
      const successful: JobResults[] = [];
      for (const r of results) {
        if (r.status === "fulfilled") successful.push(r.value);
      }
      return successful;
    },
    enabled: completedIds.length > 0,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (resultsQuery.isFetchedAfterMount && resultsQuery.isSuccess) {
      queryClient.prefetchQuery({
        queryKey: ["models"],
        queryFn: async () => {
          const { data } = await apiClient.get<{ models: import("@/api/schemas").ModelInfo[] }>(
            "/models",
          );
          return data.models;
        },
        staleTime: 5 * 60_000,
      });
      queryClient.prefetchQuery({
        queryKey: ["model-hyperparams"],
        queryFn: async () => {
          const { data } =
            await apiClient.get<import("@/api/schemas").ModelHyperparamsResponse>(
              "/models/hyperparams",
            );
          return data.models;
        },
        staleTime: 5 * 60_000,
      });
    }
  }, [resultsQuery.isFetchedAfterMount, resultsQuery.isSuccess, queryClient]);

  const kpis = useMemo(
    () => computeDashboardKPIs(resultsQuery.data ?? []),
    [resultsQuery.data],
  );

  const equityDataMap = useMemo(() => {
    const map: Record<string, import("@/api/schemas").EquityPoint[] | null> = {};
    if (resultsQuery.data) {
      for (const r of resultsQuery.data) {
        const curves =
          r.metrics?.flatMap((m) => (m.equity_curve ? [m.equity_curve] : [])) ?? [];
        map[r.job_id] = curves.length > 0 ? curves[0] : null;
      }
    }
    return map;
  }, [resultsQuery.data]);

  const allCompletedJobs = useMemo(
    () => (jobsQuery.data ?? []).filter((j) => j.status === "completed"),
    [jobsQuery.data],
  );

  const completedJobs = useMemo(() => allCompletedJobs.slice(0, 5), [allCompletedJobs]);

  const committeeQuery = useQuery({
    queryKey: ["full-cycle", "history"],
    queryFn: async () => {
      const { data } = await apiClient.get<FullCycleHistoryResponse>(
        "/committee/full-cycle/history",
      );
      return data;
    },
    staleTime: 10_000,
  });

  const committeeEntries: FullCycleHistoryEntry[] = useMemo(
    () => committeeQuery.data?.entries ?? [],
    [committeeQuery.data],
  );

  const allCompletedCommittees = useMemo(
    () => committeeEntries.filter((e) => e.status === "completed"),
    [committeeEntries],
  );

  const completedCommittees = useMemo(
    () => allCompletedCommittees.slice(0, 5),
    [allCompletedCommittees],
  );

  const committeeKpis = useMemo(
    () => computeCommitteeKPIs(committeeEntries),
    [committeeEntries],
  );

  return {
    isPricesReady: pricesQuery.isSuccess,
    isCandlesReady: candlesQuery.isSuccess,
    isSentimentReady: sentimentQuery.isSuccess,
    isPerformanceReady:
      resultsQuery.isSuccess ||
      committeeQuery.isSuccess ||
      (completedJobs.length === 0 && allCompletedCommittees.length === 0),
    top3Pairs,
    pairs: pairsQuery.data ?? [],
    prices: pricesQuery.data,
    candles: candlesQuery.data,
    sentiment: sentimentQuery.data,
    articles: articlesQuery.data,
    completedJobs,
    totalJobCount: allCompletedJobs.length,
    allResults: resultsQuery.data ?? [],
    kpis,
    equityDataMap,
    isPricesLoading: pricesQuery.isLoading || pairsQuery.isLoading,
    isCandlesLoading: candlesQuery.isLoading,
    isSentimentLoading: sentimentQuery.isLoading,
    isJobsLoading: jobsQuery.isLoading,
    completedCommittees,
    totalCommitteeCount: allCompletedCommittees.length,
    committeeKpis,
    isCommitteesLoading: committeeQuery.isLoading,
  };
}
