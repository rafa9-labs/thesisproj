import { create } from "zustand";
import type { Metrics, WsEvent } from "@/api/schemas";

interface JobState {
  jobId: string;
  pair: string;
  models: string[];
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  progressText: string;
  completedModels: string[];
  currentModel: string | null;
  metrics: Map<string, Partial<Metrics>>;
  error: string | null;
  createdAt: Date;
  completedAt: Date | null;
}

interface JobStore {
  activeJobs: Map<string, JobState>;
  selectedJobId: string | null;

  startJob: (jobId: string, pair: string, models: string[]) => void;
  handleWsEvent: (event: WsEvent) => void;
  selectJob: (jobId: string | null) => void;
  removeJob: (jobId: string) => void;
  getJob: (jobId: string) => JobState | undefined;
}

export const useJobStore = create<JobStore>()((set, get) => ({
  activeJobs: new Map(),
  selectedJobId: null,

  startJob: (jobId, pair, models) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.set(jobId, {
        jobId,
        pair,
        models,
        status: "pending",
        progress: 0,
        progressText: "Starting...",
        completedModels: [],
        currentModel: null,
        metrics: new Map(),
        error: null,
        createdAt: new Date(),
        completedAt: null,
      });
      return { activeJobs: next, selectedJobId: jobId };
    }),

  handleWsEvent: (event) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      const jobId = (event as { job_id?: string }).job_id;
      if (!jobId) return state;

      const job = next.get(jobId);
      if (!job) return state;

      const updated = { ...job };

      if (event.event === "job_started") {
        updated.status = "running";
        updated.progressText = "Backtest running...";
      }

      if (event.event === "model_training") {
        if (event.status === "starting") {
          updated.currentModel = event.model;
          updated.progressText = `Training ${event.model}...`;
        }
        if (event.status === "complete") {
          updated.completedModels = [...updated.completedModels, event.model];
          const nextMetrics = new Map(updated.metrics);
          nextMetrics.set(event.model, event.metrics ?? {});
          updated.metrics = nextMetrics;
          const progress = Math.round(
            (updated.completedModels.length / updated.models.length) * 100,
          );
          updated.progress = progress;
          updated.progressText = `${event.model} complete (${updated.completedModels.length}/${updated.models.length})`;
        }
      }

      if (event.event === "job_complete") {
        updated.status = "completed";
        updated.progress = 100;
        updated.progressText = "Complete";
        updated.completedAt = new Date();
      }

      if (event.event === "job_failed") {
        updated.status = "failed";
        updated.error = event.error;
        updated.progressText = `Failed: ${event.error}`;
      }

      next.set(jobId, updated);
      return { activeJobs: next };
    }),

  selectJob: (jobId) => set({ selectedJobId: jobId }),

  removeJob: (jobId) =>
    set((state) => {
      const next = new Map(state.activeJobs);
      next.delete(jobId);
      return {
        activeJobs: next,
        selectedJobId: state.selectedJobId === jobId ? null : state.selectedJobId,
      };
    }),

  getJob: (jobId) => get().activeJobs.get(jobId),
}));
