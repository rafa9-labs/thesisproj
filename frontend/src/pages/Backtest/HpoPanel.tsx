import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";

const sectionClass = "rounded-xl border p-6";
const sectionStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};
const sectionTitleClass = "mb-1 text-[11px] font-medium uppercase tracking-[0.12em]";
const sectionTitleStyle: React.CSSProperties = { color: "var(--color-text-secondary)" };
const explainerClass = "mb-5 text-[11px] font-light leading-relaxed max-w-[720px]";
const explainerStyle: React.CSSProperties = { color: "var(--color-text-muted)" };

interface HpoPanelProps {
  advancedMode: boolean;
  onToggleAdvanced: () => void;
}

export function HpoPanel({ advancedMode, onToggleAdvanced }: HpoPanelProps) {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const showLogistic = selectedModels.includes("logistic");

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
              backgroundColor: advancedMode ? "var(--color-glass-hover)" : "var(--color-brand)",
              color: advancedMode ? "var(--color-text-muted)" : "var(--color-text-inverse)",
              border: "1px solid",
              borderColor: advancedMode ? "var(--color-glass-border)" : "transparent",
              boxShadow: advancedMode ? "none" : "0 0 12px rgba(0,229,255,0.15)",
            }}
          >
            Simple
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
            <ParamSelect
              label="HPO Intensity"
              value={hpoIntensity}
              options={[...SELECT_OPTIONS.hpoIntensity]}
              description="Preset search depth. Light = fast validation, Deep = production-grade tuning."
              onChange={(v) => setField("hpoIntensity", v as "light" | "quick" | "standard" | "deep")}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-6">
            <ParamSlider
              label="HPO Trials"
              value={s.nTrials}
              min={RANGES.nTrials.min}
              max={RANGES.nTrials.max}
              step={RANGES.nTrials.step}
              description="Number of hyperparameter combinations to test per model."
              onChange={(v) => setField("nTrials", v)}
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

      {/* ── Logistic Hyperparameters ── */}
      {showLogistic && (
        <div className="rounded-xl border p-6" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "rgba(255,255,255,0.02)" }}>
          <div className="flex items-center gap-2 mb-1">
            <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-accent-classical)" }} />
            <h4 className="text-[11px] font-medium uppercase tracking-[0.12em]" style={{ color: "var(--color-accent-classical)" }}>
              Logistic Regression — Hyperparameters
            </h4>
          </div>
          <p className={explainerClass} style={explainerStyle}>
            Fine-tune the classical baseline. These only appear when Logistic Regression is selected in your model pool.
          </p>

          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
              <ParamSelect
                label="Solver"
                value={s.logitSolver}
                options={[...SELECT_OPTIONS.logitSolver]}
                description="Optimization algorithm. lbfgs is stable; saga handles elastic-net."
                onChange={(v) => setField("logitSolver", v as typeof s.logitSolver)}
              />
              <ParamSelect
                label="Penalty"
                value={s.logitPenalty}
                options={[...SELECT_OPTIONS.logitPenalty]}
                description="Regularization type. L2 = standard ridge; elasticnet mixes L1 + L2."
                onChange={(v) => setField("logitPenalty", v as typeof s.logitPenalty)}
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-6">
              <ParamSlider
                label="C (Regularization)"
                value={s.logitC}
                min={0.001}
                max={10000}
                step={s.logitC < 1 ? 0.01 : s.logitC < 100 ? 1 : 100}
                description="Inverse regularization strength. Lower C = stronger penalty, simpler model."
                onChange={(v) => setField("logitC", v)}
              />
              <ParamSlider
                label="Max Iterations"
                value={s.logitMaxIter}
                min={100}
                max={5000}
                step={100}
                description="Solver convergence limit. Increase if convergence warnings appear."
                onChange={(v) => setField("logitMaxIter", v)}
              />
              <ParamSlider
                label="Tolerance"
                value={s.logitTol}
                min={0.00001}
                max={0.01}
                step={0.00001}
                description="Stopping criteria. Smaller = stricter convergence, longer training."
                onChange={(v) => setField("logitTol", v)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
