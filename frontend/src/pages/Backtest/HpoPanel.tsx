import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";

export function HpoPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const showLogistic = selectedModels.includes("logistic");

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <h3
        className="mb-3 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Walk-Forward &amp; HPO
      </h3>

      {/* HPO Intensity */}
      <div className="mb-4">
        <ParamSelect
          label="HPO Intensity"
          paramKey="hpo_intensity"
          value={hpoIntensity}
          options={[...SELECT_OPTIONS.hpoIntensity]}
          onChange={(v) => setField("hpoIntensity", v as "light" | "quick" | "standard" | "deep")}
        />
      </div>

      {/* HPO settings */}
      <div className="mb-4 grid grid-cols-3 gap-4">
        <ParamSlider
          label="HPO Trials"
          paramKey="n_trials"
          value={s.nTrials}
          min={RANGES.nTrials.min}
          max={RANGES.nTrials.max}
          step={RANGES.nTrials.step}
          onChange={(v) => setField("nTrials", v)}
        />
        <ParamSelect
          label="Direction"
          paramKey="optuna_direction"
          value={s.optunaDirection}
          options={[...SELECT_OPTIONS.optunaDirection]}
          onChange={(v) => setField("optunaDirection", v as "maximize" | "minimize")}
        />
        <ParamSlider
          label="Seed"
          paramKey="seed"
          value={s.seed}
          min={RANGES.seed.min}
          max={RANGES.seed.max}
          step={RANGES.seed.step}
          onChange={(v) => setField("seed", v)}
        />
      </div>

      {/* Walk-forward windows */}
      <div className="mb-4 grid grid-cols-2 gap-4">
        <ParamSlider
          label="Train Months"
          paramKey="train_months"
          value={s.trainMonths}
          min={RANGES.trainMonths.min}
          max={RANGES.trainMonths.max}
          step={RANGES.trainMonths.step}
          onChange={(v) => setField("trainMonths", v)}
        />
        <ParamSlider
          label="Test Months"
          paramKey="test_months"
          value={s.testMonths}
          min={RANGES.testMonths.min}
          max={RANGES.testMonths.max}
          step={RANGES.testMonths.step}
          onChange={(v) => setField("testMonths", v)}
        />
      </div>

      {/* Coverage & confidence */}
      <div className="mb-4 grid grid-cols-3 gap-4">
        <ParamSlider
          label="Active Rate"
          paramKey="target_active_rate"
          value={s.targetActiveRate}
          min={RANGES.targetActiveRate.min}
          max={RANGES.targetActiveRate.max}
          step={RANGES.targetActiveRate.step}
          onChange={(v) => setField("targetActiveRate", v)}
        />
        <ParamSlider
          label="Coverage"
          paramKey="target_coverage"
          value={s.targetCoverage}
          min={RANGES.targetCoverage.min}
          max={RANGES.targetCoverage.max}
          step={RANGES.targetCoverage.step}
          onChange={(v) => setField("targetCoverage", v)}
        />
        <ParamSlider
          label="Confidence"
          paramKey="confidence_threshold"
          value={s.confidenceThreshold}
          min={RANGES.confidenceThreshold.min}
          max={RANGES.confidenceThreshold.max}
          step={RANGES.confidenceThreshold.step}
          onChange={(v) => setField("confidenceThreshold", v)}
        />
      </div>

      {/* Trading costs */}
      <div className="flex gap-6">
        <ParamToggle
          label="Trading Costs"
          paramKey="eval_use_trading_costs"
          checked={s.evalUseTradingCosts}
          onChange={(v) => setField("evalUseTradingCosts", v)}
        />
        {s.evalUseTradingCosts && (
          <div className="flex-1">
            <ParamSlider
              label="Slippage (bps)"
              paramKey="slip_norm_bps"
              value={s.slipNormBps}
              min={RANGES.slipNormBps.min}
              max={RANGES.slipNormBps.max}
              step={RANGES.slipNormBps.step}
              onChange={(v) => setField("slipNormBps", v)}
            />
          </div>
        )}
      </div>

      {/* Logistic Hyperparameters */}
      {showLogistic && (
        <div className="mt-4 rounded-md border p-3" style={{ borderColor: "var(--color-border-subtle)" }}>
          <h4 className="mb-2 text-xs font-semibold uppercase" style={{ color: "var(--color-accent-classical)" }}>
            Logistic Regression — Hyperparameters
          </h4>
          <div className="mb-3 grid grid-cols-2 gap-4">
            <ParamSelect
              label="Solver"
              paramKey="logit_solver"
              value={s.logitSolver}
              options={[...SELECT_OPTIONS.logitSolver]}
              onChange={(v) => setField("logitSolver", v as typeof s.logitSolver)}
            />
            <ParamSelect
              label="Penalty"
              paramKey="logit_penalty"
              value={s.logitPenalty}
              options={[...SELECT_OPTIONS.logitPenalty]}
              onChange={(v) => setField("logitPenalty", v as typeof s.logitPenalty)}
            />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <ParamSlider
              label="C (Regularization)"
              paramKey="logit_C"
              value={s.logitC}
              min={0.001}
              max={10000}
              step={s.logitC < 1 ? 0.01 : s.logitC < 100 ? 1 : 100}
              description="Inverse regularization strength"
              onChange={(v) => setField("logitC", v)}
            />
            <ParamSlider
              label="Max Iterations"
              paramKey="logit_max_iter"
              value={s.logitMaxIter}
              min={100}
              max={5000}
              step={100}
              onChange={(v) => setField("logitMaxIter", v)}
            />
            <ParamSlider
              label="Tolerance"
              paramKey="logit_tol"
              value={s.logitTol}
              min={0.00001}
              max={0.01}
              step={0.00001}
              onChange={(v) => setField("logitTol", v)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
