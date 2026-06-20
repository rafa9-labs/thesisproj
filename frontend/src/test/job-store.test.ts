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
    expect(job!.totalWork).toBe(0);
    expect(job!.completedWork).toBe(0);
  });

  it("selects job on start", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    expect(useJobStore.getState().selectedJobId).toBe("job-1");
  });

  it("handles job_started event with total_work", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    const event: WsEvent = {
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 66,
    };
    useJobStore.getState().handleWsEvent(event);
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.status).toBe("running");
    expect(job!.totalWork).toBe(66);
  });

  it("handles model_training starting event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost", "lstm"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost", "lstm"],
      total_work: 0,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "starting",
    });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.currentModel).toBe("xgboost");
    expect(job!.progressText).toContain("xgboost");
  });

  it("handles model_phase event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 33,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "starting",
    });
    useJobStore
      .getState()
      .handleWsEvent({ event: "model_phase", job_id: "job-1", model: "xgboost", phase: "hpo" });
    const job = useJobStore.getState().getJob("job-1");
    const mp = job!.modelPhases.get("xgboost");
    expect(mp).toBeDefined();
    expect(mp!.phase).toBe("hpo");
  });

  it("handles hpo_progress event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 33,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "starting",
    });
    useJobStore
      .getState()
      .handleWsEvent({ event: "model_phase", job_id: "job-1", model: "xgboost", phase: "hpo" });
    useJobStore.getState().handleWsEvent({
      event: "hpo_progress",
      job_id: "job-1",
      model: "xgboost",
      trial: 3,
      total_trials: 6,
      cv_blocks: 5,
      completed_work: 15,
      total_work: 33,
      progress_pct: 45.5,
    });
    const job = useJobStore.getState().getJob("job-1");
    const mp = job!.modelPhases.get("xgboost");
    expect(mp!.hpoTrial).toBe(3);
    expect(mp!.hpoTotalTrials).toBe(6);
    expect(job!.progress).toBe(45.5);
    expect(job!.completedWork).toBe(15);
    expect(job!.progressText).toContain("3/6");
  });

  it("handles month_progress event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 33,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "starting",
    });
    useJobStore.getState().handleWsEvent({
      event: "model_phase",
      job_id: "job-1",
      model: "xgboost",
      phase: "simulation",
    });
    useJobStore.getState().handleWsEvent({
      event: "month_progress",
      job_id: "job-1",
      model: "xgboost",
      month: 2,
      total_months: 3,
      sharpe: 0.8,
      trades: 15,
      completed_work: 32,
      total_work: 33,
      progress_pct: 97.0,
    });
    const job = useJobStore.getState().getJob("job-1");
    const mp = job!.modelPhases.get("xgboost");
    expect(mp!.simMonth).toBe(2);
    expect(mp!.simTotalMonths).toBe(3);
    expect(job!.progress).toBe(97.0);
    expect(job!.progressText).toContain("2/3");
  });

  it("handles model_training complete event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost", "lstm"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost", "lstm"],
      total_work: 66,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "complete",
      metrics: { sharpe: 1.2 },
    });
    const job = useJobStore.getState().getJob("job-1");
    expect(job!.completedModels).toEqual(["xgboost"]);
    expect(job!.metrics.get("xgboost")).toEqual({ sharpe: 1.2 });
  });

  it("handles job_complete event", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 33,
    });
    useJobStore.getState().handleWsEvent({
      event: "model_training",
      job_id: "job-1",
      model: "xgboost",
      status: "complete",
      metrics: { sharpe: 1.2 },
    });
    useJobStore.getState().handleWsEvent({
      event: "job_complete",
      job_id: "job-1",
      metrics: [{ model: "xgboost", sharpe: 1.2 }],
    });
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

  it("auto-creates job from events for unknown jobs", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "unknown-job",
      pair: "EURUSD",
      models: ["xgboost"],
      total_work: 0,
    });
    const created = useJobStore.getState().getJob("unknown-job");
    expect(created).toBeDefined();
    expect(created!.status).toBe("running");
    expect(created!.pair).toBe("");
    expect(created!.models).toEqual([]);
  });

  it("removes a job", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["xgboost"]);
    useJobStore.getState().removeJob("job-1");
    expect(useJobStore.getState().getJob("job-1")).toBeUndefined();
    expect(useJobStore.getState().selectedJobId).toBeNull();
  });

  it("computes proportional progress from work units", () => {
    useJobStore.getState().startJob("job-1", "EURUSD", ["a", "b"]);
    useJobStore.getState().handleWsEvent({
      event: "job_started",
      job_id: "job-1",
      pair: "EURUSD",
      models: ["a", "b"],
      total_work: 66,
    });
    useJobStore.getState().handleWsEvent({
      event: "hpo_progress",
      job_id: "job-1",
      model: "a",
      trial: 3,
      total_trials: 6,
      cv_blocks: 5,
      completed_work: 15,
      total_work: 66,
      progress_pct: 22.7,
    });
    expect(useJobStore.getState().getJob("job-1")!.progress).toBe(22.7);
    useJobStore.getState().handleWsEvent({
      event: "hpo_progress",
      job_id: "job-1",
      model: "a",
      trial: 6,
      total_trials: 6,
      cv_blocks: 5,
      completed_work: 30,
      total_work: 66,
      progress_pct: 45.5,
    });
    expect(useJobStore.getState().getJob("job-1")!.progress).toBe(45.5);
  });
});
