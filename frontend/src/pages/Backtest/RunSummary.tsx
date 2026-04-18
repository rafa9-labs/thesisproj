import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { formatPercent } from "@/lib/formatters";

interface RunSummaryProps {
  open: boolean;
  onClose: () => void;
  onDeploy: () => void;
  warnings: number;
  errors: number;
}

export function RunSummary({ open, onClose, onDeploy, warnings, errors }: RunSummaryProps) {
  const store = useBacktestStore.getState();

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
      onClick={onClose}
    >
      <div
        className="flex w-[640px] max-h-[80vh] flex-col gap-4 overflow-y-auto rounded-lg border p-6"
        style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold" style={{ color: "var(--color-text-primary)" }}>
            Pre-Flight Summary
          </h2>
          <button
            onClick={onClose}
            className="text-xs"
            style={{ color: "var(--color-text-muted)" }}
          >
            ESC
          </button>
        </div>

        {/* Asset */}
        <Section title="Asset">
          <Row label="Pair" value={store.pair} />
          <Row label="Timeframe" value={store.timeframe} />
        </Section>

        {/* Models */}
        <Section title="Models">
          <div className="flex flex-wrap gap-2">
            {store.selectedModels.map((m) => (
              <span
                key={m}
                className="rounded-md border px-2 py-1 text-xs"
                style={{
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {modelDescriptions[m]?.name ?? m}
              </span>
            ))}
          </div>
        </Section>

        {/* Walk-forward */}
        <Section title="Walk-Forward">
          <Row label="Train" value={`${store.trainMonths} months`} />
          <Row label="Test" value={`${store.testMonths} months`} />
          <Row label="HPO Trials" value={String(store.nTrials)} />
          <Row label="Confidence" value={formatPercent(store.confidenceThreshold)} />
        </Section>

        {/* Execution */}
        <Section title="Execution">
          <Row label="Sizing" value={store.sizingMethod} />
          <Row label="Trailing" value={store.trailingMethod} />
          <Row label="Max DD" value={formatPercent(store.maxDrawdownPct)} />
          <Row label="Trading Costs" value={store.evalUseTradingCosts ? "Yes" : "No"} />
        </Section>

        {/* Validation summary */}
        {(warnings > 0 || errors > 0) && (
          <div
            className="rounded-md border p-3"
            style={{
              borderColor: errors > 0 ? "var(--color-accent-danger)" : "var(--color-accent-warning)",
              backgroundColor: errors > 0 ? "rgba(242,54,69,0.05)" : "rgba(255,152,0,0.05)",
            }}
          >
            <span className="text-xs font-semibold" style={{ color: errors > 0 ? "var(--color-accent-danger)" : "var(--color-accent-warning)" }}>
              {errors > 0 ? `${errors} error${errors > 1 ? "s" : ""}` : `${warnings} warning${warnings > 1 ? "s" : ""}`}
            </span>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 border-t pt-4" style={{ borderColor: "var(--color-border)" }}>
          <button
            onClick={onClose}
            className="rounded-md border px-4 py-2 text-xs font-semibold uppercase"
            style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          >
            Cancel
          </button>
          <button
            onClick={onDeploy}
            disabled={errors > 0}
            className="rounded-md px-6 py-2 text-xs font-bold uppercase transition-colors"
            style={{
              backgroundColor: errors > 0 ? "var(--color-border)" : "var(--color-accent)",
              color: errors > 0 ? "var(--color-text-muted)" : "var(--color-text-inverse)",
              letterSpacing: "0.05em",
              cursor: errors > 0 ? "not-allowed" : "pointer",
            }}
          >
            Deploy Backtest
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-secondary)" }}>
        {title}
      </h3>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}
