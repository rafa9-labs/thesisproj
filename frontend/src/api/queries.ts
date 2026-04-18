import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiClient from "./client";
import type {
  PairInfo,
  ModelInfo,
  JobSummary,
  JobStatus,
  JobResults,
  BacktestRequest,
  HealthResponse,
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
  return useQuery({
    queryKey: ["job-results", jobId],
    queryFn: async () => {
      const { data } = await apiClient.get<JobResults>(`/backtest/${jobId}/results`);
      return data;
    },
    enabled: !!jobId,
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
