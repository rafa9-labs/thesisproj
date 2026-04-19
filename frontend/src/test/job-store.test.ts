import { describe, it, expect, beforeEach } from "vitest";
import { useJobStore } from "@/stores/useJobStore";
import type { WsEvent } from "@/api/schemas";

describe("useJobStore", () => {
  beforeEach(() => {
    const { activeJobs } = useJobStore.getState();
    const next = new Map(activeJobs);
    next.clear();
    useJobStore.setState({ activeJobs: next, selectedJobId: null });
  });

  it("starts a job with correct initial state", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost", "lstm"]);
    const job = useJobStore.getState().getJob("job-1");
    expect(job).toBeDefined();
    expect(job!.status).toBe("pending");
    expect(job!.pair).toBe("EURUSD");
    expect(job!.models).toEqual(["xgboost", "lstm"]);
    expect(job!.progress).toBe(0);
    expect(job!.completedModels).toEqual([]);
  });

  it("selects job on start", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    expect(useJobStore.getState().selectedJobId).toBe("job-1");
  });

  it("handles job_started event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    const event: WsEvent = { event: "job_started", job_id: "job-1", pair: "EURUSD", models: ["xgboost"] };
    useJobStore.getState().handleWsEvent(event);
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.status).toBe("running");
  });

  it("handles model_training starting event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost", "lstm"]);
    useJobStore.getState().handleWsEvent({ event: "job_started", job_id: "job-1", pair: "EURUSD", models: ["xgboost", "lstm"] });
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "xgboost", status: "starting" });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.currentModel).toBe("xgboost");
    expect(job!.progressText).toContain("xgboost");
  });

  it("handles model_training complete event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost", "lstm"]);
    useJobStore.getState().handleWsEvent({ event: "job_started", job_id: "job-1", pair: "EURUSD", models: ["xgboost", "lstm"] });
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "xgboost", status: "complete", metrics: { sharpe: 1.2 } });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.completedModels).toEqual(["xgboost"]);
    expect(job!.progress).toBe(50);
    expect(job!.metrics.get("xgboost")).toEqual({ sharpe: 1.2 });
  });

  it("handles job_complete event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({ event: "job_started", job_id: "job-1", pair: "EURUSD", models: ["xgboost"] });
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "xgboost", status: "complete", metrics: { sharpe: 1.2 } });
    useJobStore.getState().handleWsEvent({ event: "job_complete", job_id: "job-1", metrics: [{ model: "xgboost", sharpe: 1.2 }] });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.status).toBe("completed");
    expect(job!.progress).toBe(100);
    expect(job!.completedAt).not.toBeNull();
  });

  it("handles job_failed event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({ event: "job_failed", job_id: "job-1", error: "OOM" });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.status).toBe("failed");
    expect(job!.error).toBe("OOM");
  });

  it("ignores events for unknown jobs", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({ event: "job_started", job_id: "unknown-job", pair: "EURUSD", models: ["xgboost"] });
    expect(useJobStore.getState().getJob("unknown-job")).toBeUndefined();
  });

  it("removes a job", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().removeJob("job-1");
    expect(useJobStore.getState().getJob("job-1")).toBeUndefined();
    expect(useJobStore.getState().selectedJobId).toBeNull();
  });

  it("tracks progress correctly for multi-model job", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["a", "b", "c"]);
    useJobStore.getState().handleWsEvent({ event: "job_started", job_id: "job-1", pair: "EURUSD", models: ["a", "b", "c"] });
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "a", status: "complete", metrics: {} });
    expect(useJobStore.getState().getJob("job-1")!.progress).toBe(33);
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "b", status: "complete", metrics: {} });
    expect(useJobStore.getState().getJob("job-1")!.progress).toBe(67);
    useJobStore.getState().handleWsEvent({ event: "model_training", job_id: "job-1", model: "c", status: "complete", metrics: {} });
    expect(useJobStore.getState().getJob("job-1")!.progress).toBe(100);
  });
});
