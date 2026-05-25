import { useCallback } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { RANGES, SELECT_OPTIONS, STUDY_PRESETS } from "@/lib/constants";
import { ModelHyperparamsPanel } from "./ModelHyperparamsPanel";

const sectionClass = "rounded-xl border p-6";
const sectionStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};
const sectionTitleClass = "mb-1 text-[11px] font-medium uppercase tracking-[0.12em]";
const sectionTitleStyle: React.CSSProperties = { color: "var(--color-text-secondary)" };
const explainerClass = "mb-5 text-[11px] font-light leading-relaxed max-w-[720px]";
const explainerStyle: React.CSSProperties = { color: "var(--color-text-muted)" };

const PRESET_KEYS = ["diagnostic", "exploratory", "validation", "production"] as const;

const PRESET_COLORS: Record<string, string> = {
  green: "var(--color-accent-success)",
  yellow: "var(--color-accent-warning)",
  orange: "var(--color-accent)",
  red: "var(--color-accent-danger)",
};

interface HpoPanelProps {
  advancedMode: boolean;
  onToggleAdvanced: () => void;
}

export function HpoPanel({ advancedMode, onToggleAdvanced }: HpoPanelProps) {
  const setField = useBacktestStore((s) => s.setField);
  const applyStudyPreset = useBacktestStore((s) => s.applyStudyPreset);
  const s = useBacktestStore.getState();
  const activePreset = useBacktestStore((s) => s.activePreset);
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const nTrials = useBacktestStore((s) => s.nTrials);
  const hpoManualOverride = useBacktestStore((s) => s.hpoManualOverride);

  const handlePresetClick = useCallback((key: string) => {
    applyStudyPreset(key);
  }, [applyStudyPreset]);

  return (
    <div
      className="flex flex-col gap-6 rounded-xl border p-6"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      {/* Header with mode toggle */}
      <div className="flex items-center justify-between pb-2">
        <div className="flex flex-col gap-0.5">
          <h3
            className="text-[11px] font-medium uppercase tracking-[0.12em]"
            style={{ color: "var(--color-text-muted)" }}
          >
            Walk-Forward &amp; HPO
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onToggleAdvanced}
            className="rounded-md px-3 py-1 text-[10px] font-medium uppercase tracking-[0.1em] transition-all duration-200"
            style={{
              backgroundColor: !advancedMode ? "var(--color-brand)" : "var(--color-glass-hover)",
              color: !advancedMode ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              border: "1px solid",
              borderColor: !advancedMode ? "transparent" : "var(--color-glass-border)",
              boxShadow: !advancedMode ? "0 0 12px rgba(0,229,255,0.15)" : "none",
            }}
          >
            Preset
          </button>
          <button
            onClick={onToggleAdvanced}
            className="rounded-md px-3 py-1 text-[10px] font-medium uppercase tracking-[0.1em] transition-all duration-200"
            style={{
              backgroundColor: advancedMode ? "var(--color-brand)" : "var(--color-glass-hover)",
              color: advancedMode ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              border: "1px solid",
              borderColor: advancedMode ? "transparent" : "var(--color-glass-border)",
              boxShadow: advancedMode ? "0 0 12px rgba(0,229,255,0.15)" : "none",
            }}
          >
            Advanced
          </button>
        </div>
      </div>

      {!advancedMode ? (
        /* ── PRESET MODE ── */
        <section className={sectionClass} style={sectionStyle}>
          <div className="flex items-center gap-2 mb-1">
            <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
            <h4 className={sectionTitleClass} style={sectionTitleStyle}>Study Preset</h4>
          </div>
          <p className={explainerClass} style={explainerStyle}>
            Choose a study intensity. Higher presets run more HPO trials and seeds for greater statistical confidence.
          </p>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {PRESET_KEYS.map((key) => {
              const p = STUDY_PRESETS[key];
              const isSelected = activePreset === key;
              const color = PRESET_COLORS[p.badgeColor] ?? "var(--color-text-muted)";
              const tr = p.trialRange;
              const trialStr = tr.min === tr.max ? `${tr.min}` : `${tr.min}–${tr.max}`;
              const modelKey = selectedModels[0] ?? "logistic";
              const estMin = (p.estMinutes as Record<string, number>)[modelKey] ?? 30;

              return (
                <div
                  key={key}
                  onClick={() => handlePresetClick(key)}
                  className="rounded-lg border p-3 cursor-pointer transition-all duration-150"
                  style={{
                    borderColor: isSelected ? color : "var(--color-glass-border)",
                    backgroundColor: isSelected ? `${color}08` : "var(--color-elevated)",
                    boxShadow: isSelected ? `0 0 8px ${color}30` : "none",
                  }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] font-semibold uppercase tracking-[0.08em]" style={{ color }}>
                      {p.label}
                    </span>
                    <span
                      className="rounded px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider"
                      style={{
                        backgroundColor: `${color}18`,
                        color,
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {p.badge}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                    <span>{trialStr} tri</span>
                    <span style={{ color: "var(--color-glass-border)" }}>·</span>
                    <span>{p.repeats} run{p.repeats > 1 ? "s" : ""}</span>
                    {p.repeats > 1 && (
                      <>
                        <span style={{ color: "var(--color-glass-border)" }}>·</span>
                        <span>{(tr.min * p.repeats).toLocaleString()}–{(tr.max * p.repeats).toLocaleString()} total</span>
                      </>
                    )}
                  </div>

                  <div
                    className="mt-1.5 text-[9px]"
                    style={{ color: "var(--color-text-muted)" }}
                  >
                    ~{estMin > 120 ? `${(estMin / 60).toFixed(0)}h` : `${estMin}min`} / model
                  </div>
                </div>
              );
            })}
          </div>

          {activePreset && (
            <div
              className="mt-3 rounded-lg p-3 text-[11px] leading-relaxed"
              style={{
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-secondary)",
              }}
            >
              {STUDY_PRESETS[activePreset as keyof typeof STUDY_PRESETS]?.description}
            </div>
          )}
        </section>
      ) : (
        <>
          {/* ── ADVANCED MODE ── */}
          {/* ── Optimization Strategy ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Optimization Strategy</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Controls how aggressively the engine searches hyperparameters. Deeper searches find better configs but take longer.
            </p>

            <div className="flex flex-col gap-6">
              <div className="max-w-sm">
                <ParamSlider
                  label="HPO Trials"
                  value={hpoManualOverride ? nTrials : _effectiveMaxTrials(hpoIntensity)}
                  min={RANGES.nTrials.min}
                  max={RANGES.nTrials.max}
                  step={RANGES.nTrials.step}
                  description={`Set 0 for no HPO (default params only). Max ${RANGES.nTrials.max} for deep search. ${!hpoManualOverride ? "Click or drag to enable manual override." : "Override active."}`}
                  onChange={(v) => {
                    setField("nTrials", v);
                    setField("hpoManualOverride", true);
                  }}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-6">
                <div className="flex flex-col gap-1 rounded-lg p-3"
                  style={{ backgroundColor: "var(--color-elevated)" }}>
                  <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
                    HPO Trials
                  </span>
                  <div className="mt-2 text-[11px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                    <span style={{ color: "var(--color-brand)" }}>
                      {hpoManualOverride ? nTrials : _trialRangeByIntensity(hpoIntensity)}
                    </span>
                    <span style={{ color: "var(--color-text-muted)" }}> trials per model</span>
                    {hpoManualOverride && nTrials === 0 && (
                      <span style={{ color: "var(--color-accent-warning)", marginLeft: 8 }}>no HPO — defaults only</span>
                    )}
                  </div>
                  <div className="mt-0.5 text-[9px]" style={{ color: "var(--color-text-muted)" }}>
                    logistic={_trialForModel(hpoIntensity, "logistic")} · xgboost={_trialForModel(hpoIntensity, "xgboost")} ·
                    lstm={_trialForModel(hpoIntensity, "lstm")}
                  </div>
                  {hpoManualOverride && (
                    <div className="mt-1 text-[9px]" style={{ color: "var(--color-accent)" }}>
                      Manual override — {nTrials} trial{nTrials !== 1 ? "s" : ""} for all models
                    </div>
                  )}
                </div>
                <ParamSlider
                  label="Repeats / Seeds"
                  value={s.repeats}
                  min={RANGES.repeats.min}
                  max={RANGES.repeats.max}
                  step={RANGES.repeats.step}
                  description="How many independent runs with different seeds. More runs = higher confidence the result isn't seed-dependent."
                  onChange={(v) => setField("repeats", v)}
                />
                <ParamSelect
                  label="Direction"
                  value={s.optunaDirection}
                  options={[...SELECT_OPTIONS.optunaDirection]}
                  description="Whether to maximize Sharpe or minimize a metric like drawdown."
                  onChange={(v) => setField("optunaDirection", v as "maximize" | "minimize")}
                />
                <ParamSlider
                  label="Seed"
                  value={s.seed}
                  min={RANGES.seed.min}
                  max={RANGES.seed.max}
                  step={RANGES.seed.step}
                  description="Random seed for reproducible HPO runs. Same seed = same results."
                  onChange={(v) => setField("seed", v)}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                <ParamSlider
                  label="Max HPO Duration (min)"
                  value={s.maxHpoDurationMinutes ?? 0}
                  min={0}
                  max={120}
                  step={5}
                  description="Hard time limit. 0 = no limit. The optimizer returns the best config found so far if time runs out."
                  onChange={(v) => setField("maxHpoDurationMinutes", v)}
                />
                <div />
              </div>
            </div>
          </section>

          {/* ── Walk-Forward Windows ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-success)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Walk-Forward Windows</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Splits data into rolling train/test chunks. The model retrains on each window and tests on the next, preventing look-ahead bias.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-6">
              <ParamSlider
                label="Train Window"
                value={s.trainMonths}
                min={RANGES.trainMonths.min}
                max={RANGES.trainMonths.max}
                step={RANGES.trainMonths.step}
                description="Months of historical data used to train each fold."
                onChange={(v) => setField("trainMonths", v)}
              />
              <ParamSlider
                label="Test Window"
                value={s.testMonths}
                min={RANGES.testMonths.min}
                max={RANGES.testMonths.max}
                step={RANGES.testMonths.step}
                description="Months held out for validation after training."
                onChange={(v) => setField("testMonths", v)}
              />
              <ParamSelect
                label="Period Unit"
                value={s.periodUnit ?? "months"}
                options={["months", "weeks", "days"]}
                description="Granularity of each train/test window."
                onChange={(v) => setField("periodUnit", v as "months" | "weeks" | "days")}
              />
            </div>
          </section>

          {/* ── Quality Gates ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-warning)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Quality Gates</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Minimum thresholds a model must pass to be considered viable. Configs below any gate are discarded automatically.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-6">
              <ParamSlider
                label="Active Rate"
                value={s.targetActiveRate}
                min={RANGES.targetActiveRate.min}
                max={RANGES.targetActiveRate.max}
                step={RANGES.targetActiveRate.step}
                description="Minimum % of bars the model should signal trades on. Filters idle models."
                onChange={(v) => setField("targetActiveRate", v)}
              />
              <ParamSlider
                label="Coverage"
                value={s.targetCoverage}
                min={RANGES.targetCoverage.min}
                max={RANGES.targetCoverage.max}
                step={RANGES.targetCoverage.step}
                description="Minimum % of the test period the model must have exposure for."
                onChange={(v) => setField("targetCoverage", v)}
              />
              <ParamSlider
                label="Confidence"
                value={s.confidenceThreshold}
                min={RANGES.confidenceThreshold.min}
                max={RANGES.confidenceThreshold.max}
                step={RANGES.confidenceThreshold.step}
                description="Minimum prediction confidence. Higher = fewer but higher-conviction trades."
                onChange={(v) => setField("confidenceThreshold", v)}
              />
            </div>
          </section>

          {/* ── Execution Reality ── */}
          <section className={sectionClass} style={sectionStyle}>
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-danger)" }} />
              <h4 className={sectionTitleClass} style={sectionTitleStyle}>Execution Reality</h4>
            </div>
            <p className={explainerClass} style={explainerStyle}>
              Simulates real-world trading friction. Sharper backtests include slippage and spread to avoid overestimating returns.
            </p>

            <div className="flex flex-col gap-6">
              <div className="max-w-xs">
                <ParamToggle
                  label="Trading Costs"
                  checked={s.evalUseTradingCosts}
                  description="Enable spread + slippage simulation in the backtest loop."
                  onChange={(v) => setField("evalUseTradingCosts", v)}
                />
              </div>

              {s.evalUseTradingCosts && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                  <ParamSlider
                    label="Slippage (bps)"
                    value={s.slipNormBps}
                    min={RANGES.slipNormBps.min}
                    max={RANGES.slipNormBps.max}
                    step={RANGES.slipNormBps.step}
                    description="Average execution slippage in basis points. 1 bps = 0.01%."
                    onChange={(v) => setField("slipNormBps", v)}
                  />
                </div>
              )}
            </div>
          </section>

          {/* ── Per-Model Hyperparameters ── */}
          <ModelHyperparamsPanel />
        </>
      )}
    </div>
  );
}

// Per-model trial counts mirroring backend HPO_TRIAL_MAPS (api/schemas/backtest.py)
const _HPO_TRIALS: Record<string, Record<string, { r: number; b: number }>> = {
  light: { logistic: { r: 1, b: 1 }, svm: { r: 1, b: 1 }, decision_tree: { r: 1, b: 1 },
           random_forest: { r: 1, b: 1 }, xgboost: { r: 1, b: 1 },
           lstm: { r: 1, b: 1 }, cnn: { r: 1, b: 1 }, transformer: { r: 1, b: 1 },
           ensemble_adaptive_regime: { r: 1, b: 1 }, ensemble_cnn_lstm_xgboost: { r: 1, b: 1 }, dqn: { r: 1, b: 1 } },
  quick: { logistic: { r: 2, b: 2 }, svm: { r: 2, b: 2 }, decision_tree: { r: 2, b: 2 },
           random_forest: { r: 2, b: 2 }, xgboost: { r: 2, b: 2 },
           lstm: { r: 2, b: 2 }, cnn: { r: 2, b: 2 }, transformer: { r: 2, b: 2 },
           ensemble_adaptive_regime: { r: 2, b: 2 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 2 }, dqn: { r: 1, b: 1 } },
  standard: { logistic: { r: 5, b: 5 }, svm: { r: 5, b: 5 }, decision_tree: { r: 5, b: 5 },
             random_forest: { r: 5, b: 10 }, xgboost: { r: 5, b: 15 },
             lstm: { r: 3, b: 7 }, cnn: { r: 3, b: 7 }, transformer: { r: 3, b: 7 },
             ensemble_adaptive_regime: { r: 2, b: 3 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 3 }, dqn: { r: 2, b: 3 } },
  deep: { logistic: { r: 10, b: 10 }, svm: { r: 10, b: 10 }, decision_tree: { r: 5, b: 10 },
          random_forest: { r: 10, b: 20 }, xgboost: { r: 10, b: 30 },
          lstm: { r: 5, b: 15 }, cnn: { r: 5, b: 15 }, transformer: { r: 5, b: 15 },
          ensemble_adaptive_regime: { r: 3, b: 7 }, ensemble_cnn_lstm_xgboost: { r: 3, b: 7 }, dqn: { r: 3, b: 5 } },
};

function _trialForModel(intensity: string, model: string): number {
  const m = _HPO_TRIALS[intensity]?.[model];
  return m ? Math.max(m.r + m.b, 10) : 10;
}

function _trialRangeByIntensity(intensity: string): string {
  const models = _HPO_TRIALS[intensity];
  if (!models) return "10";
  const vals = Object.values(models).map((m) => Math.max(m.r + m.b, 10));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return min === max ? `${min}` : `${min}–${max}`;
}

function _effectiveMaxTrials(intensity: string): number {
  const models = _HPO_TRIALS[intensity];
  if (!models) return 10;
  const vals = Object.values(models).map((m) => Math.max(m.r + m.b, 10));
  return Math.max(...vals);
}
