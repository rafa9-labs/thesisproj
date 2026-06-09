import { useCommitteeConfig, useCommitteeSnapshots, useCommitteeMetrics } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import apiClient from "@/api/client";
import { useState } from "react";

const healthCardStyle: React.CSSProperties = {
  background: "var(--color-elevated)",
  border: "1px solid var(--color-glass-border)",
  borderRadius: 4,
  padding: "10px 12px",
  textAlign: "center",
};

const thStyleH: React.CSSProperties = {
  padding: "4px 8px",
  textAlign: "left",
  color: "var(--color-text-muted)",
  fontWeight: 500,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  fontSize: 9,
  borderBottom: "1px solid var(--color-glass-border)",
};

const tdStyleH: React.CSSProperties = {
  padding: "5px 8px",
  color: "var(--color-text-secondary)",
  fontSize: 10,
};

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
      const { data } = await apiClient.post<{
        session_id: string; pair: string; timeframe: string; models: string[]; features: string[];
        snapshot_loaded: boolean; feature_count: number;
      }>("/live/deploy-committee", {
        pair: store.deployedPair,
        timeframe: store.deployedTimeframe,
        initial_equity: 10000.0,
        confidence_threshold: 0.55,
        execution_mode: store.executionMode,
      });
      store.setDeployedSession(data.session_id, data.pair, data.timeframe);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        ?? (err as { message?: string })?.message
        ?? "Deploy failed";
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
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Status */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 20,
        }}
      >
        <h2
          style={{
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: "0 0 16px",
          }}
        >
          Live Committee Status
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              marginLeft: 12,
              fontSize: 10,
              color: "var(--color-text-muted)",
              fontWeight: 400,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                display: "inline-block",
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

        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
          {!effectiveSessionId && (
            <>
              <button
                onClick={handleDeploy}
                disabled={deploying}
                style={{
                  background: deploying ? "var(--color-text-dim)" : "var(--color-accent-success)",
                  border: "none",
                  borderRadius: 4,
                  color: "var(--color-text-inverse)",
                  padding: "8px 20px",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: deploying ? "not-allowed" : "pointer",
                }}
              >
                {deploying ? "Deploying..." : "Deploy Live"}
              </button>
              <select
                value={store.executionMode}
                onChange={(e) => store.setExecutionMode(e.target.value)}
                style={{
                  background: "var(--color-input-bg)",
                  border: "1px solid var(--color-glass-border)",
                  borderRadius: 4,
                  color: "var(--color-text-primary)",
                  padding: "7px 10px",
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
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
              <span style={{
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                color: "var(--color-brand)",
                background: "rgba(0,229,255,0.08)",
                border: "1px solid rgba(0,229,255,0.2)",
                borderRadius: 3,
                padding: "4px 10px",
              }}>
                Session: {effectiveSessionId}
              </span>
              <button
                onClick={handleStop}
                style={{
                  background: "rgba(242,54,69,0.12)",
                  color: "var(--color-accent-danger)",
                  border: "1px solid rgba(242,54,69,0.25)",
                  borderRadius: 4,
                  padding: "6px 14px",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                Stop
              </button>
            </>
          )}
        </div>

        {deployError && (
          <div style={{
            padding: 10,
            background: "rgba(242,54,69,0.08)",
            border: "1px solid rgba(242,54,69,0.2)",
            borderRadius: 4,
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            color: "var(--color-accent-danger)",
            marginBottom: 12,
          }}>
            {deployError}
          </div>
        )}

        <p style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.6, margin: 0 }}>
          The live committee runner processes streaming OHLC bars, classifies the
          current market regime, routes predictions through the best models for
          that regime, and blends signals. Deploy a committee configuration to
          start live trading.
        </p>
      </div>

      {/* Active Config */}
      {config && (
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 20,
          }}
        >
          <h3
            style={{
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--color-text-primary)",
              margin: "0 0 12px",
            }}
          >
            Active Configuration
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(config.regimes).map(([regime, assignment]) => (
              <div
                key={regime}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  fontSize: 11,
                }}
              >
                <span
                  style={{
                    width: 120,
                    fontWeight: 500,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: "var(--color-brand)",
                    flexShrink: 0,
                  }}
                >
                  {regime.replace(/_/g, " ")}
                </span>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {assignment.models.map((model, idx) => (
                    <span
                      key={`${model}-${idx}`}
                      style={{
                        background: "var(--color-elevated)",
                        borderRadius: 3,
                        padding: "2px 8px",
                        fontSize: 10,
                        fontFamily: "var(--font-mono)",
                        color: "var(--color-text-secondary)",
                      }}
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
        <div
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-glass-border)",
            borderRadius: 4,
            padding: 20,
          }}
        >
          <h3
            style={{
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--color-text-primary)",
              margin: "0 0 12px",
            }}
          >
            Model Store Snapshots
          </h3>
          {snapshots.snapshots.map((snap) => (
            <div
              key={snap.version}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 0",
                borderBottom: "1px solid var(--color-glass-border)",
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 11,
                    fontFamily: "var(--font-mono)",
                    color: "var(--color-text-primary)",
                  }}
                >
                  {snap.version}
                </div>
                <div style={{ fontSize: 10, color: "var(--color-text-muted)", marginTop: 2 }}>
                  {snap.created_at}
                </div>
              </div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {snap.models.map((m) => (
                  <span
                    key={m}
                    style={{
                      background: "var(--color-elevated)",
                      borderRadius: 3,
                      padding: "1px 6px",
                      fontSize: 9,
                      fontFamily: "var(--font-mono)",
                      color: "var(--color-text-secondary)",
                    }}
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
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 20,
        }}
      >
        <h3
          style={{
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: "0 0 12px",
          }}
        >
          Health Monitoring
        </h3>

        {metrics ? (
          <>
            {/* Summary cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8, marginBottom: 16 }}>
              <div style={healthCardStyle}>
                <div style={{ fontSize: 18, fontFamily: "var(--font-mono)", fontWeight: 600, color: metrics.committee_healthy ? "#089981" : "#F23645" }}>
                  {metrics.committee_healthy ? "HEALTHY" : "UNHEALTHY"}
                </div>
                <div style={{ fontSize: 9, color: "var(--color-text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Status</div>
              </div>
              <div style={healthCardStyle}>
                <div style={{ fontSize: 18, fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--color-text-secondary)" }}>
                  {metrics.bar_count}
                </div>
                <div style={{ fontSize: 9, color: "var(--color-text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Bars</div>
              </div>
              <div style={healthCardStyle}>
                <div style={{ fontSize: 18, fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--color-text-secondary)" }}>
                  {metrics.non_zero_signals}
                </div>
                <div style={{ fontSize: 9, color: "var(--color-text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Signals</div>
              </div>
              <div style={healthCardStyle}>
                <div style={{ fontSize: 18, fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--color-text-secondary)" }}>
                  {metrics.current_regime.replace(/_/g, " ")}
                </div>
                <div style={{ fontSize: 9, color: "var(--color-text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Regime</div>
              </div>
            </div>

            {/* Per-model health table */}
            <div style={{ fontSize: 10, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: 8 }}>
              Model Health
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
              <thead>
                <tr>
                  <th style={thStyleH}>Model</th>
                  <th style={{ ...thStyleH, textAlign: "right" }}>Sharpe</th>
                  <th style={{ ...thStyleH, textAlign: "right" }}>Hit Rate</th>
                  <th style={{ ...thStyleH, textAlign: "right" }}>Signals</th>
                  <th style={thStyleH}>Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_model_health).map(([model, health]) => (
                  <tr key={model} style={{ borderBottom: "1px solid var(--color-glass-border)" }}>
                    <td style={{ ...tdStyleH, fontFamily: "var(--font-mono)" }}>{model}</td>
                    <td style={{ ...tdStyleH, textAlign: "right", fontFamily: "var(--font-mono)", color: (health.rolling_sharpe ?? 0) >= 0 ? "#089981" : "#F23645" }}>
                      {health.rolling_sharpe !== null ? health.rolling_sharpe.toFixed(2) : "—"}
                    </td>
                    <td style={{ ...tdStyleH, textAlign: "right", fontFamily: "var(--font-mono)", color: (health.rolling_hit_rate ?? 0) >= 0.5 ? "#089981" : (health.rolling_hit_rate ?? 0) >= 0.35 ? "#F2B436" : "#F23645" }}>
                      {health.rolling_hit_rate !== null ? (health.rolling_hit_rate * 100).toFixed(0) + "%" : "—"}
                    </td>
                    <td style={{ ...tdStyleH, textAlign: "right", fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>
                      {health.total_signals}
                    </td>
                    <td style={tdStyleH}>
                      <span style={{
                        display: "inline-block",
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: health.status === "healthy" ? "#089981"
                          : health.status === "unhealthy" ? "#F23645"
                          : "var(--color-text-dim)",
                      }} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: 0 }}>
            Model health tracking (rolling Sharpe, hit rate) will appear here
            once the live committee runner is deployed. Models flagged as
            unhealthy are automatically rotated.
          </p>
        )}
      </div>
    </div>
  );
}
