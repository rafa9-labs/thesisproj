import { useEffect, useCallback } from "react";
import { X } from "lucide-react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { modelDescriptions, layout } from "@/lib/tokens";
import { SELECT_OPTIONS } from "@/lib/constants";
import { formatPercent } from "@/lib/formatters";

interface RunSummaryProps {
  open: boolean;
  mode: "summary" | "deploy";
  onClose: () => void;
  onDeploy: () => void;
  warnings: number;
  errors: number;
  isSubmitting?: boolean;
}

const FEATURE_LABELS: Record<string, string> = {
  useAdx: "ADX",
  useAtr: "ATR",
  useBbands: "Bollinger",
  useEma: "EMA",
  useSma: "SMA",
  useRsi: "RSI",
  useMacd: "MACD",
  useDonchian: "Donchian",
  useStoch: "Stochastic",
  useSar: "SAR",
  useMtfMa: "MTF MA",
  useMtfAlignment: "MTF Alignment",
  useMacdAtrRatio: "MACD/ATR Ratio",
  useTripleConfirm: "Triple Confirm",
  useTrendConfirm: "Trend Confirm",
  useVolManagedMom: "Vol-Managed Mom",
  useVmMom: "VM Mom",
  useMaSpread: "MA Spread",
  useSlopeDiff: "Slope Diff",
  useSqueezeBreakout: "Squeeze Breakout",
  useSqueezeExpansion: "Squeeze Expansion",
  useAtrChannelBreakout: "ATR Channel Breakout",
  useExtAtrLowAdx: "Ext ATR Low ADX",
  useReentryMom: "Re-entry Mom",
  useRvFeatures: "RV Features",
  useIndicatorStates: "Indicator States",
  usePriceMaZ: "Price-MA Z-Score",
  useCrossoverBins: "Crossover Bins",
  useNews: "News Sentiment",
  useFracdiff: "FracDiff",
};

function _lookupLabel(value: string, options: readonly { value: string; label: string }[]): string {
  return options.find((o) => o.value === value)?.label ?? value;
}

export function RunSummary({
  open,
  mode,
  onClose,
  onDeploy,
  warnings,
  errors,
  isSubmitting,
}: RunSummaryProps) {
  const s = useBacktestStore();
  const sidebarCollapsed = useSettingsStore((st) => st.sidebarCollapsed);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, handleKeyDown]);

  if (!open) return null;

  const models = (s.selectedModels as string[]) ?? [];
  const activeFeatureFlags = _FEATURE_FLAG_KEYS.filter((k) => s[k] === true);
  const hasMtf = s.useMtfMa === true || s.useMtfAlignment === true;
  const hyperparamOverrides = Object.keys(s).filter(
    (k) => k.includes("__") && s[k] !== undefined && s[k] !== null && s[k] !== "",
  );

  const sidebarWidth = sidebarCollapsed ? layout.sidebarCollapsed : layout.sidebarExpanded;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-[640px] max-w-[calc(100vw-2rem)] flex-col rounded-sm border border-(--color-border) bg-(--color-surface)"
        style={{
          position: "absolute",
          top: "50%",
          left: `calc((${sidebarWidth}px + 100vw) / 2)`,
          transform: "translate(-50%, -50%)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex shrink-0 items-center justify-between border-b border-(--color-border) px-6 py-4">
          <h2 className="text-sm font-semibold text-(--color-text-primary)">
            Configuration Summary
          </h2>
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-md text-(--color-text-muted) transition-colors hover:bg-(--color-glass-hover) hover:text-(--color-text-primary)"
          >
            <X size={15} />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="flex flex-col gap-6">
            {/* ── Asset & Timeframe ── */}
            <Section title="Asset & Timeframe">
              <Row label="Pair" value={String(s.pair)} />
              <Row label="Timeframe" value={String(s.timeframe)} />
              {(s.startDate || s.endDate) && (
                <Row
                  label="Date Range"
                  value={`${s.startDate || "..."} → ${s.endDate || "..."}`}
                />
              )}
            </Section>

            {/* ── Models ── */}
            <Section title="Models">
              {models.length === 0 ? (
                <span className="text-xs text-(--color-text-muted)">No models selected</span>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {models.map((m) => (
                    <span
                      key={m}
                      className="rounded-md border border-(--color-border) bg-(--color-elevated) px-2 py-0.5 font-mono text-[11px] text-(--color-text-primary)"
                    >
                      {modelDescriptions[m]?.name ?? m}
                    </span>
                  ))}
                </div>
              )}
            </Section>

            {/* ── Study & HPO ── */}
            <Section title="Study & HPO">
              <Row label="Train Window" value={`${s.trainMonths} months`} />
              <Row label="Test Window" value={`${s.testMonths} months`} />
              <Row
                label="Period Unit"
                value={_lookupLabel(String(s.periodUnit ?? "months"), SELECT_OPTIONS.periodUnit)}
              />
              <Row
                label="HPO Intensity"
                value={_lookupLabel(String(s.hpoIntensity), SELECT_OPTIONS.hpoIntensity)}
              />
              <Row
                label="Trials"
                value={`${s.nTrials}`}
              />
              <Row label="Repeats / Seeds" value={String(s.repeats)} />
              <Row label="Seed" value={String(s.seed)} />
              <Row
                label="Sampler"
                value={_lookupLabel(String(s.hpoSampler ?? "tpe"), SELECT_OPTIONS.hpoSampler)}
              />
              <Row
                label="HPO Mode"
                value={_lookupLabel(String(s.hpoMode ?? "static"), SELECT_OPTIONS.hpoMode)}
              />
              {s.hpoTwoPhase && (
                <>
                  <Row label="Phase 1 Sampler" value={String(s.phase1Sampler)} />
                  <Row label="Phase 1 Trials" value={String(s.phase1Trials)} />
                  <Row label="Phase 2 Trials" value={String(s.phase2Trials)} />
                  <Row label="Phase 2 Top-N" value={String(s.phase2TopN)} />
                </>
              )}
              <Row label="Confidence" value={formatPercent(s.confidenceThreshold)} />
              <Row label="Active Rate" value={formatPercent(s.targetActiveRate)} />
              <Row label="Coverage" value={formatPercent(s.targetCoverage)} />
              <Row
                label="Calibration"
                value={_lookupLabel(String(s.calibrateMethod), [
                  { value: "sigmoid", label: "Sigmoid (Platt)" },
                  { value: "isotonic", label: "Isotonic Regression" },
                ])}
              />
              <Row label="Trading Costs" value={s.evalUseTradingCosts ? "Yes" : "No"} />
              {s.evalUseTradingCosts && (
                <Row label="Slippage" value={`${s.slipNormBps} bps`} />
              )}
            </Section>

            {/* ── Features ── */}
            <Section title="Features">
              <Row
                label="Active Features"
                value={`${activeFeatureFlags.length}${hasMtf ? " (MTF)" : ""}`}
              />
              {activeFeatureFlags.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {activeFeatureFlags.map((k) => (
                    <span
                      key={k}
                      className="rounded bg-(--color-elevated) px-1.5 py-0.5 text-[10px] font-mono text-(--color-text-secondary)"
                    >
                      {FEATURE_LABELS[k] ?? k}
                    </span>
                  ))}
                </div>
              )}
              <Row label="Lags" value={`${s.lags} × ${s.lagDepth}`} />
              <Row
                label="FracDiff"
                value={s.useFracdiff ? `d=${s.fracdiffD}` : "Off"}
              />
              <Row label="Label Threshold" value={String(s.labelThreshold)} />
              <Row
                label="Triple Barrier"
                value={
                  s.useTripleBarrier
                    ? `PT=${s.tbPtMult} SL=${s.tbSlMult} Max=${s.tbMaxHolding} NZ=${s.tbNeutralZone}`
                    : "Off"
                }
              />
              <Row
                label="News"
                value={
                  s.useNews
                    ? `${s.newsSentimentBackend}${s.newsEventFlags ? " + events" : ""}`
                    : "Off"
                }
              />
              {s.llmSentimentEnabled && (
                <Row
                  label="LLM Sentiment"
                  value={`${s.llmBackend} / ${s.llmModel} (w=${s.llmWeight})`}
                />
              )}
            </Section>

            {/* ── Hyperparameters ── */}
            <Section title="Hyperparameters">
              {hyperparamOverrides.length === 0 ? (
                <span className="text-xs text-(--color-text-muted)">
                  Using model defaults (no per-model overrides)
                </span>
              ) : (
                <div className="flex flex-col gap-1.5">
                  {hyperparamOverrides.map((key) => (
                    <Row key={key} label={key} value={String(s[key])} />
                  ))}
                </div>
              )}
            </Section>

            {/* ── Execution ── */}
            <Section title="Execution">
              <Row
                label="Sizing"
                value={_lookupLabel(String(s.sizingMethod), SELECT_OPTIONS.sizingMethod)}
              />
              {s.sizingMethod === "kelly" && (
                <Row label="Kelly Fraction" value={String(s.kellyFraction)} />
              )}
              {s.sizingMethod === "fixed_fractional" && (
                <Row label="Risk Fraction" value={formatPercent(s.riskFraction)} />
              )}
              {s.sizingMethod === "atr" && (
                <>
                  <Row label="ATR Risk %" value={formatPercent(s.atrRiskPct)} />
                  <Row label="ATR SL Mult" value={String(s.atrSlMult)} />
                </>
              )}
              <Row
                label="Trailing"
                value={_lookupLabel(String(s.trailingMethod), SELECT_OPTIONS.trailingMethod)}
              />
              {s.trailingMethod !== "none" && (
                <Row label="Trailing Activation" value={formatPercent(s.trailingActivation)} />
              )}
              <Row label="Initial Equity" value={`$${Number(s.initialEquity).toLocaleString()}`} />
              <Row label="Max Leverage" value={`${s.maxLeverage}x`} />
              <Row label="Max Drawdown" value={formatPercent(s.maxDrawdownPct)} />
              <Row label="Max Consec. Losses" value={String(s.maxConsecutiveLosses)} />
              <Row label="Daily Loss Limit" value={formatPercent(s.dailyLossLimitPct)} />
            </Section>

            {/* Validation warnings/errors */}
            {(warnings > 0 || errors > 0) && (
              <div
                className="rounded-md border p-3"
                style={{
                  borderColor: errors > 0 ? "var(--color-accent-danger)" : "var(--color-accent-warning)",
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
                    ? `${errors} error${errors > 1 ? "s" : ""} — resolve before deploying`
                    : `${warnings} warning${warnings > 1 ? "s" : ""}`}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-end gap-3 border-t border-(--color-border) px-6 py-4">
          <button
            onClick={onClose}
            className="rounded-md border border-(--color-border) px-4 py-2 text-xs font-semibold text-(--color-text-secondary) uppercase transition-colors hover:text-(--color-text-primary)"
          >
            {mode === "deploy" ? "Cancel" : "Close"}
          </button>
          {mode === "deploy" && (
            <button
              onClick={onDeploy}
              disabled={errors > 0 || isSubmitting}
              className="flex items-center gap-2 rounded-md px-6 py-2 text-xs font-bold uppercase transition-all"
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
          )}
        </div>
      </div>
    </div>
  );
}

const _FEATURE_FLAG_KEYS = [
  "useAdx",
  "useAtr",
  "useBbands",
  "useEma",
  "useSma",
  "useRsi",
  "useMacd",
  "useStoch",
  "useSar",
  "useDonchian",
  "useFracdiff",
  "useCrossoverBins",
  "useMaSpread",
  "usePriceMaZ",
  "useIndicatorStates",
  "useMtfMa",
  "useMtfAlignment",
  "useMtfAlign",
  "useMacdAtrRatio",
  "useTripleConfirm",
  "useTrendConfirm",
  "useVolManagedMom",
  "useVmMom",
  "useSqueezeBreakout",
  "useSqueezeExpansion",
  "useAtrChannelBreakout",
  "useExtAtrLowAdx",
  "useReentryMom",
  "useSlopeDiff",
  "useRvFeatures",
  "useNews",
] as const;

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-[10px] font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
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
