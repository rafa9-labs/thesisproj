import { useCommitteeConfig, useCommitteeSnapshots } from "@/api/queries";

export function LiveCommitteePanel() {
  const { data: config } = useCommitteeConfig();
  const { data: snapshots } = useCommitteeSnapshots();

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
                background: "var(--color-accent-success)",
                display: "inline-block",
              }}
            />
            Awaiting deployment
          </span>
        </h2>

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

      {/* Health Monitoring placeholder */}
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
        <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: 0 }}>
          Model health tracking (rolling Sharpe, hit rate) will appear here
          once the live committee runner is deployed. Models flagged as
          unhealthy are automatically rotated.
        </p>
      </div>
    </div>
  );
}
