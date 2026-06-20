import { useCommitteeConfig, useCommitteeSnapshots, useCommitteeMetrics } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import apiClient from "@/api/client";
import { useState } from "react";

const healthCardClasses = "bg-(--color-elevated) border border-(--color-glass-border)";

interface Props {
  sessionId?: string | null;
}

export function LiveCommitteePanel({ sessionId }: Props) {
  const store = useFullCycleStore();
  const effectiveSessionId = sessionId ?? store.deployedSessionId;
  const { data: config } = useCommitteeConfig();
  const { data: snapshots } = useCommitteeSnapshots();
  const { data: metrics } = useCommitteeMetrics(effectiveSessionId ?? null);
  const [deploying, setDeploying] = useState(false);
  const [deployError, setDeployError] = useState<string | null>(null);

  const handleDeploy = async () => {
    setDeploying(true);
    setDeployError(null);
    try {
      const payload: Record<string, unknown> = {
        pair: store.deployedPair,
        timeframe: store.deployedTimeframe,
        initial_equity: 10000.0,
        confidence_threshold: 0.55,
        mode: store.executionMode,
      };
      if (store.deployedJobId) {
        payload.full_cycle_job_id = store.deployedJobId;
      }
      const { data } = await apiClient.post<{
        session_id: string;
        pair: string;
        timeframe: string;
        models: string[];
        features: string[];
        snapshot_loaded: boolean;
        feature_count: number;
      }>("/trading/live/committee/start", payload);
      store.setDeployedSession(data.session_id, data.pair, data.timeframe);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data
          ?.detail ??
        (err as { message?: string })?.message ??
        "Deploy failed";
      setDeployError(msg);
    } finally {
      setDeploying(false);
    }
  };

  const handleStop = async () => {
    if (!effectiveSessionId) return;
    try {
      await apiClient.post(`/live/${effectiveSessionId}/stop`);
      store.clearDeployedSession();
    } catch {
      // session may already be stopped
      store.clearDeployedSession();
    }
  };

  return (
    <div className="flex flex-col gap-[24px]">
      {/* Status */}
      <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[20px]">
        <h2 className="mb-[16px] text-[13px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Live Committee Status
          <span className="ml-[12px] inline-flex items-center gap-[6px] text-[10px] font-normal text-(--color-text-muted)">
            <span
              className="inline-block h-[6px] w-[6px] rounded-full"
              style={{
                background: metrics
                  ? metrics.committee_healthy
                    ? "var(--color-accent-success)"
                    : "#F23645"
                  : deploying
                    ? "#F2B436"
                    : "var(--color-text-dim)",
              }}
            />
            {metrics
              ? `${metrics.committee_healthy ? "Healthy" : "Unhealthy"} \u00b7 ${(metrics.uptime_seconds / 3600).toFixed(1)}h`
              : deploying
                ? "Deploying..."
                : effectiveSessionId
                  ? "Connecting..."
                  : "Awaiting deployment"}
          </span>
        </h2>

        <div className="mb-[12px] flex flex-wrap items-center gap-[10px]">
          {!effectiveSessionId && (
            <>
              <button
                onClick={handleDeploy}
                disabled={deploying}
                className="rounded-[4px] border-none px-[20px] py-[8px] text-[10px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase"
                style={{
                  background: deploying ? "var(--color-text-dim)" : "var(--color-accent-success)",
                  cursor: deploying ? "not-allowed" : "pointer",
                }}
              >
                {deploying ? "Deploying..." : "Deploy Live"}
              </button>
              <select
                value={store.executionMode}
                onChange={(e) => store.setExecutionMode(e.target.value)}
                className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
                style={{
                  backgroundColor: "var(--color-glass)",
                  borderColor: "var(--color-glass-border)",
                  color: "var(--color-text-primary)",
                }}
              >
                <option value="paper">Paper Sim</option>
                <option value="oanda">OANDA Direct</option>
                <option value="lean">LEAN + IB</option>
              </select>
            </>
          )}
          {effectiveSessionId && (
            <>
              <span className="rounded-[3px] border border-[rgba(0,229,255,0.2)] bg-[rgba(0,229,255,0.08)] px-[10px] py-[4px] font-mono text-[10px] text-(--color-brand)">
                Session: {effectiveSessionId}
              </span>
              <button
                onClick={handleStop}
                className="cursor-pointer rounded-[4px] border border-[rgba(242,54,69,0.25)] bg-[rgba(242,54,69,0.12)] px-[14px] py-[6px] text-[10px] font-semibold tracking-[0.06em] text-(--color-accent-danger) uppercase"
              >
                Stop
              </button>
            </>
          )}
        </div>

        {deployError && (
          <div className="mb-[12px] rounded-[4px] border border-[rgba(242,54,69,0.2)] bg-[rgba(242,54,69,0.08)] p-[10px] font-mono text-[10px] text-(--color-accent-danger)">
            {deployError}
          </div>
        )}

        <p className="m-0 text-[12px] leading-[1.6] text-(--color-text-secondary)">
          The live committee runner processes streaming OHLC bars, classifies the current market
          regime, routes predictions through the best models for that regime, and blends signals.
          Deploy a committee configuration to start live trading.
        </p>
      </div>

      {/* Active Config */}
      {config && (
        <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[20px]">
          <h3 className="mb-[12px] text-[12px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
            Active Configuration
          </h3>
          <div className="flex flex-col gap-[8px]">
            {Object.entries(config.regimes).map(([regime, assignment]) => (
              <div key={regime} className="flex items-center gap-[12px] text-[11px]">
                <span className="w-[120px] shrink-0 font-medium tracking-[0.06em] text-(--color-brand) uppercase">
                  {regime.replace(/_/g, " ")}
                </span>
                <div className="flex flex-wrap gap-[4px]">
                  {assignment.models.map((model, idx) => (
                    <span
                      key={`${model}-${idx}`}
                      className="rounded-[3px] bg-(--color-elevated) px-[8px] py-[2px] font-mono text-[10px] text-(--color-text-secondary)"
                    >
                      {model} {(assignment.weights[idx] * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Store Snapshots */}
      {snapshots && snapshots.snapshots.length > 0 && (
        <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[20px]">
          <h3 className="mb-[12px] text-[12px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
            Model Store Snapshots
          </h3>
          {snapshots.snapshots.map((snap) => (
            <div
              key={snap.version}
              className="flex items-center justify-between border-b border-(--color-glass-border) py-[8px]"
            >
              <div>
                <div className="font-mono text-[11px] text-(--color-text-primary)">
                  {snap.version}
                </div>
                <div className="mt-[2px] text-[10px] text-(--color-text-muted)">
                  {snap.created_at}
                </div>
              </div>
              <div className="flex flex-wrap gap-[4px]">
                {snap.models.map((m) => (
                  <span
                    key={m}
                    className="rounded-[3px] bg-(--color-elevated) px-[6px] py-[1px] font-mono text-[9px] text-(--color-text-secondary)"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Health Monitoring */}
      <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[20px]">
        <h3 className="mb-[12px] text-[12px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Health Monitoring
        </h3>

        {metrics ? (
          <>
            {/* Summary cards */}
            <div className="mb-[16px] grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-[8px]">
              <div className={`rounded px-3 py-2.5 text-center ${healthCardClasses}`}>
                <div
                  className="font-mono text-[18px] font-semibold"
                  style={{
                    color: metrics.committee_healthy ? "#089981" : "#F23645",
                  }}
                >
                  {metrics.committee_healthy ? "HEALTHY" : "UNHEALTHY"}
                </div>
                <div className="text-[9px] tracking-[0.06em] text-(--color-text-dim) uppercase">
                  Status
                </div>
              </div>
              <div className={`rounded px-3 py-2.5 text-center ${healthCardClasses}`}>
                <div className="font-mono text-[18px] font-semibold text-(--color-text-secondary)">
                  {metrics.bar_count}
                </div>
                <div className="text-[9px] tracking-[0.06em] text-(--color-text-dim) uppercase">
                  Bars
                </div>
              </div>
              <div className={`rounded px-3 py-2.5 text-center ${healthCardClasses}`}>
                <div className="font-mono text-[18px] font-semibold text-(--color-text-secondary)">
                  {metrics.non_zero_signals}
                </div>
                <div className="text-[9px] tracking-[0.06em] text-(--color-text-dim) uppercase">
                  Signals
                </div>
              </div>
              <div className={`rounded px-3 py-2.5 text-center ${healthCardClasses}`}>
                <div className="font-mono text-[18px] font-semibold text-(--color-text-secondary)">
                  {metrics.current_regime.replace(/_/g, " ")}
                </div>
                <div className="text-[9px] tracking-[0.06em] text-(--color-text-dim) uppercase">
                  Regime
                </div>
              </div>
            </div>

            {/* Per-model health table */}
            <div className="mb-[8px] text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
              Model Health
            </div>
            <table className="w-full border-collapse text-[10px]">
              <thead>
                <tr>
                  <th className="border-b border-(--color-glass-border) px-2 py-1 text-left text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Model
                  </th>
                  <th className="border-b border-(--color-glass-border) px-2 py-1 text-right text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Sharpe
                  </th>
                  <th className="border-b border-(--color-glass-border) px-2 py-1 text-right text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Hit Rate
                  </th>
                  <th className="border-b border-(--color-glass-border) px-2 py-1 text-right text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Signals
                  </th>
                  <th className="border-b border-(--color-glass-border) px-2 py-1 text-left text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_model_health).map(([model, health]) => (
                  <tr key={model} className="border-b border-(--color-glass-border)">
                    <td className="font-mono">{model}</td>
                    <td
                      className="text-right font-mono"
                      style={{
                        color: (health.rolling_sharpe ?? 0) >= 0 ? "#089981" : "#F23645",
                      }}
                    >
                      {health.rolling_sharpe !== null ? health.rolling_sharpe.toFixed(2) : "—"}
                    </td>
                    <td
                      className="text-right font-mono"
                      style={{
                        color:
                          (health.rolling_hit_rate ?? 0) >= 0.5
                            ? "#089981"
                            : (health.rolling_hit_rate ?? 0) >= 0.35
                              ? "#F2B436"
                              : "#F23645",
                      }}
                    >
                      {health.rolling_hit_rate !== null
                        ? (health.rolling_hit_rate * 100).toFixed(0) + "%"
                        : "—"}
                    </td>
                    <td className="text-right font-mono text-(--color-text-dim)">
                      {health.total_signals}
                    </td>
                    <td className="px-2 py-1.5 text-[10px] text-(--color-text-secondary)">
                      <span
                        className="inline-block h-[8px] w-[8px] rounded-full"
                        style={{
                          background:
                            health.status === "healthy"
                              ? "#089981"
                              : health.status === "unhealthy"
                                ? "#F23645"
                                : "var(--color-text-dim)",
                        }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="m-0 text-[12px] text-(--color-text-secondary)">
            Model health tracking (rolling Sharpe, hit rate) will appear here once the live
            committee runner is deployed. Models flagged as unhealthy are automatically rotated.
          </p>
        )}
      </div>
    </div>
  );
}
