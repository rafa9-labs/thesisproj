import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { formatPercent } from "@/lib/formatters";

interface RunSummaryProps {
  open: boolean;
  onClose: () => void;
  onDeploy: () => void;
  warnings: number;
  errors: number;
  isSubmitting?: boolean;
}

export function RunSummary({
  open,
  onClose,
  onDeploy,
  warnings,
  errors,
  isSubmitting,
}: RunSummaryProps) {
  const store = useBacktestStore.getState();

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
      onClick={onClose}
    >
      <div
        className="flex max-h-[80vh] w-[640px] flex-col gap-4 overflow-y-auto rounded-sm border border-(--color-border) bg-(--color-surface) p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-(--color-text-primary)">
            Pre-Flight Summary
          </h2>
          <button onClick={onClose} className="text-xs text-(--color-text-muted)">
            ESC
          </button>
        </div>

        {/* Asset */}
        <Section title="Asset">
          <Row label="Pair" value={store.pair} />
          <Row label="Timeframe" value={store.timeframe} />
          {(store.startDate || store.endDate) && (
            <Row
              label="Date Range"
              value={`${store.startDate || "..."} → ${store.endDate || "..."}`}
            />
          )}
        </Section>

        {/* Models */}
        <Section title="Models">
          <div className="flex flex-wrap gap-2">
            {store.selectedModels.map((m) => (
              <span
                key={m}
                className="rounded-md border border-(--color-border) px-2 py-1 font-mono text-xs text-(--color-text-primary)"
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
          <Row label="Equity" value={`$${store.initialEquity.toLocaleString()}`} />
          <Row label="Leverage" value={`${store.maxLeverage}x`} />
          <Row label="Max DD" value={formatPercent(store.maxDrawdownPct)} />
          <Row label="Trading Costs" value={store.evalUseTradingCosts ? "Yes" : "No"} />
        </Section>

        {/* Features */}
        <Section title="Features">
          <Row label="Lags" value={`${store.lags} × ${store.lagDepth}`} />
          <Row label="FracDiff" value={store.useFracdiff ? `d=${store.fracdiffD}` : "Off"} />
          <Row
            label="News"
            value={
              store.useNews
                ? `${store.newsSentimentBackend}${store.newsEventFlags ? " + events" : ""}`
                : "Off"
            }
          />
        </Section>

        {/* Validation summary */}
        {(warnings > 0 || errors > 0) && (
          <div
            className="rounded-md border p-3"
            style={{
              borderColor:
                errors > 0 ? "var(--color-accent-danger)" : "var(--color-accent-warning)",
              backgroundColor: errors > 0 ? "rgba(239,68,68,0.05)" : "rgba(245,158,11,0.05)",
            }}
          >
            <span
              className="text-xs font-semibold"
              style={{
                color: errors > 0 ? "var(--color-accent-danger)" : "var(--color-accent-warning)",
              }}
            >
              {errors > 0
                ? `${errors} error${errors > 1 ? "s" : ""}`
                : `${warnings} warning${warnings > 1 ? "s" : ""}`}
            </span>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 border-t border-(--color-border) pt-4">
          <button
            onClick={onClose}
            className="rounded-md border border-(--color-border) px-4 py-2 text-xs font-semibold text-(--color-text-secondary) uppercase"
          >
            Cancel
          </button>
          <button
            onClick={onDeploy}
            disabled={errors > 0 || isSubmitting}
            className="rounded-md px-6 py-2 text-xs font-bold uppercase transition-colors"
            style={{
              backgroundColor:
                errors > 0 || isSubmitting ? "var(--color-border)" : "var(--color-brand)",
              color:
                errors > 0 || isSubmitting
                  ? "var(--color-text-muted)"
                  : "var(--color-text-inverse)",
              letterSpacing: "0.05em",
              cursor: errors > 0 || isSubmitting ? "not-allowed" : "pointer",
              opacity: isSubmitting ? 0.7 : 1,
            }}
          >
            {isSubmitting ? "Submitting..." : "Deploy Backtest"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        {title}
      </h3>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-(--color-text-muted)">{label}</span>
      <span className="font-mono text-(--color-text-primary)">{value}</span>
    </div>
  );
}
