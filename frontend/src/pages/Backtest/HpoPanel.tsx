import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";
import { ModelHyperparamsPanel } from "./ModelHyperparamsPanel";

export function HpoPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const nTrials = useBacktestStore((s) => s.nTrials);
  const hpoManualOverride = useBacktestStore((s) => s.hpoManualOverride);

  return (
    <Panel>
      <PanelHeader
        title="Walk-Forward & HPO"
        subtitle="Configure the hyperparameter search and rolling validation windows."
      />

      {/* ── Optimization Strategy ── */}
      <Section
        title="Optimization Strategy"
        description="Controls how aggressively the engine searches hyperparameters. Deeper searches find better configs but take longer."
      >
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
            <div className="flex flex-col gap-1 rounded-sm p-3"
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
            <ParamSelect
              label="HPO Sampler"
              value={s.hpoSampler ?? "tpe"}
              options={[...SELECT_OPTIONS.hpoSampler]}
              description="Optimization algorithm. TPE=fast Bayesian. Random=baseline. CMA-ES=broad exploration."
              onChange={(v) => setField("hpoSampler", v as string)}
            />
          </div>

          <div className="flex flex-col gap-4">
            <ParamToggle
              label="Two-Phase HPO"
              checked={s.hpoTwoPhase}
              description="Explore broadly first (Phase 1), then refine the best candidates (Phase 2)."
              onChange={(v) => setField("hpoTwoPhase", v)}
            />
            {s.hpoTwoPhase && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-6 ml-6 pl-4 border-l" style={{ borderColor: "var(--color-glass-border)" }}>
                <ParamSelect
                  label="Phase 1 Sampler"
                  value={s.phase1Sampler}
                  options={[...SELECT_OPTIONS.phase1Sampler]}
                  description="Broad exploration method."
                  onChange={(v) => setField("phase1Sampler", v as string)}
                />
                <ParamSlider
                  label="Phase 1 Trials"
                  value={s.phase1Trials}
                  min={RANGES.phase1Trials.min}
                  max={RANGES.phase1Trials.max}
                  step={RANGES.phase1Trials.step}
                  description="Budget for broad scan."
                  onChange={(v) => setField("phase1Trials", v)}
                />
                <ParamSlider
                  label="Phase 2 Trials"
                  value={s.phase2Trials}
                  min={RANGES.phase2Trials.min}
                  max={RANGES.phase2Trials.max}
                  step={RANGES.phase2Trials.step}
                  description="Budget for fine-tuning."
                  onChange={(v) => setField("phase2Trials", v)}
                />
                <ParamSlider
                  label="Top-N to Refine"
                  value={s.phase2TopN}
                  min={RANGES.phase2TopN.min}
                  max={RANGES.phase2TopN.max}
                  step={RANGES.phase2TopN.step}
                  description="Number of Phase 1 candidates to refine in Phase 2."
                  onChange={(v) => setField("phase2TopN", v)}
                />
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── Walk-Forward Windows ── */}
      <Section
        title="Walk-Forward Windows"
        description="Splits data into rolling train/test chunks. The model retrains on each window and tests on the next, preventing look-ahead bias."
      >

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
                options={[...SELECT_OPTIONS.periodUnit]}
                description="Granularity of each train/test window."
                onChange={(v) => setField("periodUnit", v as "months" | "weeks" | "days")}
              />
            </div>

            <div className="flex flex-col gap-4 mt-2">
              <ParamSelect
                label="HPO Mode"
                value={s.hpoMode ?? "static"}
                options={[...SELECT_OPTIONS.hpoMode]}
                description="Static runs HPO once on fold 1 and reuses params. Dynamic re-optimizes at each walk-forward step."
                onChange={(v) => setField("hpoMode", v as string)}
              />
              {s.hpoMode === "dynamic" && (
                <ParamSlider
                  label="Dynamic HPO Trials"
                  value={s.dynamicHpoTrials}
                  min={RANGES.dynamicHpoTrials.min}
                  max={RANGES.dynamicHpoTrials.max}
                  step={RANGES.dynamicHpoTrials.step}
                  description="Trials per walk-forward step in dynamic mode."
                  onChange={(v) => setField("dynamicHpoTrials", v)}
                />
              )}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-6">
                <ParamSlider
                  label="WFO Train Periods"
                  value={s.wfoTrainPeriods}
                  min={RANGES.wfoTrainPeriods.min}
                  max={RANGES.wfoTrainPeriods.max}
                  step={RANGES.wfoTrainPeriods.step}
                  description="Override train window in period units. 0 = use default from train months."
                  onChange={(v) => setField("wfoTrainPeriods", v)}
                />
                <ParamSlider
                  label="WFO Test Periods"
                  value={s.wfoTestPeriods}
                  min={RANGES.wfoTestPeriods.min}
                  max={RANGES.wfoTestPeriods.max}
                  step={RANGES.wfoTestPeriods.step}
                  description="Override test window in period units. 0 = use default from test months."
                  onChange={(v) => setField("wfoTestPeriods", v)}
                />
              </div>
            </div>
          </Section>

      {/* ── Quality Gates ── */}
      <Section
        title="Quality Gates"
        description="Minimum thresholds a model must pass to be considered viable. Configs below any gate are discarded automatically."
      >

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
      </Section>

      {/* ── Execution Reality ── */}
      <Section
        title="Execution Reality"
        description="Simulates real-world trading friction. Sharper backtests include slippage and spread to avoid overestimating returns."
      >

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
      </Section>

      {/* ── Per-Model Hyperparameters ── */}
      <ModelHyperparamsPanel />
    </Panel>
  );
  }

// Per-model trial counts mirroring backend HPO_TRIAL_MAPS (api/schemas/backtest.py)
const _HPO_TRIALS: Record<string, Record<string, { r: number; b: number }>> = {
  light: { logistic: { r: 1, b: 1 }, svm: { r: 1, b: 1 }, decision_tree: { r: 1, b: 1 },
           random_forest: { r: 1, b: 1 }, xgboost: { r: 1, b: 1 },
           lightgbm: { r: 1, b: 1 }, catboost: { r: 1, b: 1 },
           lstm: { r: 1, b: 1 }, cnn: { r: 1, b: 1 }, transformer: { r: 1, b: 1 },
           gru: { r: 1, b: 1 }, gru_lstm: { r: 1, b: 1 },
           ensemble_adaptive_regime: { r: 1, b: 1 }, ensemble_cnn_lstm_xgboost: { r: 1, b: 1 },
           meta_ensemble: { r: 1, b: 1 }, stacking_ensemble: { r: 1, b: 1 }, dqn: { r: 1, b: 1 } },
  quick: { logistic: { r: 2, b: 2 }, svm: { r: 2, b: 2 }, decision_tree: { r: 2, b: 2 },
           random_forest: { r: 2, b: 2 }, xgboost: { r: 2, b: 2 },
           lightgbm: { r: 2, b: 2 }, catboost: { r: 2, b: 2 },
           lstm: { r: 2, b: 2 }, cnn: { r: 2, b: 2 }, transformer: { r: 2, b: 2 },
           gru: { r: 2, b: 2 }, gru_lstm: { r: 2, b: 2 },
           ensemble_adaptive_regime: { r: 2, b: 2 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 2 },
           meta_ensemble: { r: 2, b: 2 }, stacking_ensemble: { r: 2, b: 2 }, dqn: { r: 1, b: 1 } },
  standard: { logistic: { r: 5, b: 5 }, svm: { r: 5, b: 5 }, decision_tree: { r: 5, b: 5 },
              random_forest: { r: 5, b: 10 }, xgboost: { r: 5, b: 15 },
              lightgbm: { r: 5, b: 15 }, catboost: { r: 5, b: 15 },
              lstm: { r: 3, b: 7 }, cnn: { r: 3, b: 7 }, transformer: { r: 3, b: 7 },
              gru: { r: 3, b: 7 }, gru_lstm: { r: 3, b: 7 },
              ensemble_adaptive_regime: { r: 2, b: 3 }, ensemble_cnn_lstm_xgboost: { r: 2, b: 3 },
              meta_ensemble: { r: 2, b: 3 }, stacking_ensemble: { r: 2, b: 3 }, dqn: { r: 2, b: 3 } },
  deep: { logistic: { r: 10, b: 10 }, svm: { r: 10, b: 10 }, decision_tree: { r: 5, b: 10 },
          random_forest: { r: 10, b: 20 }, xgboost: { r: 10, b: 30 },
          lightgbm: { r: 10, b: 30 }, catboost: { r: 10, b: 30 },
          lstm: { r: 5, b: 15 }, cnn: { r: 5, b: 15 }, transformer: { r: 5, b: 15 },
          gru: { r: 5, b: 15 }, gru_lstm: { r: 5, b: 15 },
          ensemble_adaptive_regime: { r: 3, b: 7 }, ensemble_cnn_lstm_xgboost: { r: 3, b: 7 },
          meta_ensemble: { r: 3, b: 7 }, stacking_ensemble: { r: 3, b: 7 }, dqn: { r: 3, b: 5 } },
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
