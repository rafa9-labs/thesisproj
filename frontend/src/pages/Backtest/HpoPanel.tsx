import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSelect } from "@/components/shared/ParamSelect";
import { Panel, PanelHeader, Section } from "@/components/shared/Panel";
import { RANGES, SELECT_OPTIONS } from "@/lib/constants";

export function HpoPanel() {
  const setField = useBacktestStore((s) => s.setField);
  const s = useBacktestStore.getState();
  const nTrials = useBacktestStore((s) => s.nTrials);

  return (
    <Panel>
      <PanelHeader
        title="Walk-Forward & HPO"
        subtitle="Configure the hyperparameter search and rolling validation windows."
      />

      <Section title="Optimization Strategy">
        <div className="flex flex-col gap-8">

          <div>
            <h5 className="text-xs font-bold tracking-widest uppercase text-(--color-brand)/80 mb-6 mt-8 border-b border-(--color-elevated) pb-2">
              SEARCH BUDGET
            </h5>
            <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSlider
                label="HPO TRIALS"
                value={nTrials}
                min={RANGES.nTrials.min}
                max={RANGES.nTrials.max}
                step={RANGES.nTrials.step}
                tooltip="Set 0 for no HPO (default params only). The optimizer will run this many trials per model to find the best hyperparameter configuration."
                onChange={(v) => setField("nTrials", v)}
              />
              <ParamSlider
                label="MAX DURATION (MIN)"
                value={s.maxHpoDurationMinutes ?? 0}
                min={RANGES.maxHpoDurationMinutes.min}
                max={RANGES.maxHpoDurationMinutes.max}
                step={RANGES.maxHpoDurationMinutes.step}
                tooltip="Hard time limit for the entire HPO run. 0 = no limit. The optimizer returns the best config found so far if time runs out."
                onChange={(v) => setField("maxHpoDurationMinutes", v)}
              />
            </div>
          </div>

          <div>
            <h5 className="text-xs font-bold tracking-widest uppercase text-(--color-brand)/80 mb-6 mt-8 border-b border-(--color-elevated) pb-2">
              ENGINE & REPRODUCIBILITY
            </h5>
            <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSelect
                label="HPO SAMPLER"
                value={s.hpoSampler ?? "tpe"}
                options={[...SELECT_OPTIONS.hpoSampler]}
                tooltip="Optimization algorithm. TPE = Bayesian Tree-structured Parzen Estimator. Random = baseline exploration. CMA-ES = broad evolutionary search."
                onChange={(v) => setField("hpoSampler", v as string)}
              />
              <ParamSelect
                label="DIRECTION"
                value={s.optunaDirection}
                options={[...SELECT_OPTIONS.optunaDirection]}
                tooltip="Whether to maximize Sharpe ratio or minimize a metric like max drawdown during HPO."
                onChange={(v) => setField("optunaDirection", v as "maximize" | "minimize")}
              />
              <ParamSlider
                label="REPEATS / SEEDS"
                value={s.repeats}
                min={RANGES.repeats.min}
                max={RANGES.repeats.max}
                step={RANGES.repeats.step}
                tooltip="How many independent runs with different random seeds. More runs = higher statistical confidence that the result is not seed-dependent."
                onChange={(v) => setField("repeats", v)}
              />
              <ParamSlider
                label="SEED"
                value={s.seed}
                min={RANGES.seed.min}
                max={RANGES.seed.max}
                step={RANGES.seed.step}
                tooltip="Random seed for reproducible HPO runs. Same seed + same config = identical results."
                onChange={(v) => setField("seed", v)}
              />
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <ParamToggle
              label="TWO-PHASE HPO"
              checked={s.hpoTwoPhase}
              tooltip="Explore broadly first (Phase 1), then refine the best candidates (Phase 2). Useful for large search spaces where a single-phase search may miss good regions."
              onChange={(v) => setField("hpoTwoPhase", v)}
            />
            {s.hpoTwoPhase && (
              <div className="ml-6 grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 border-l border-(--color-glass-border) pl-4">
                <ParamSelect
                  label="PHASE 1 SAMPLER"
                  value={s.phase1Sampler}
                  options={[...SELECT_OPTIONS.phase1Sampler]}
                  tooltip="Broad exploration method for the first phase. CMA-ES covers wide regions well before TPE refinement."
                  onChange={(v) => setField("phase1Sampler", v as string)}
                />
                <ParamSlider
                  label="PHASE 1 TRIALS"
                  value={s.phase1Trials}
                  min={RANGES.phase1Trials.min}
                  max={RANGES.phase1Trials.max}
                  step={RANGES.phase1Trials.step}
                  tooltip="Number of trials for broad exploration. More trials = wider coverage of the search space."
                  onChange={(v) => setField("phase1Trials", v)}
                />
                <ParamSlider
                  label="PHASE 2 TRIALS"
                  value={s.phase2Trials}
                  min={RANGES.phase2Trials.min}
                  max={RANGES.phase2Trials.max}
                  step={RANGES.phase2Trials.step}
                  tooltip="Number of trials for fine-tuning the best Phase 1 candidates."
                  onChange={(v) => setField("phase2Trials", v)}
                />
                <ParamSlider
                  label="TOP-N TO REFINE"
                  value={s.phase2TopN}
                  min={RANGES.phase2TopN.min}
                  max={RANGES.phase2TopN.max}
                  step={RANGES.phase2TopN.step}
                  tooltip="Number of best Phase 1 candidates to carry forward into Phase 2 refinement."
                  onChange={(v) => setField("phase2TopN", v)}
                />
              </div>
            )}
          </div>

        </div>
      </Section>

      <Section title="Walk-Forward Windows">
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            <ParamSlider
              label="TRAIN WINDOW"
              value={s.trainMonths}
              min={RANGES.trainMonths.min}
              max={RANGES.trainMonths.max}
              step={RANGES.trainMonths.step}
              tooltip="Months of historical data used to train the model before testing on the next window."
              onChange={(v) => setField("trainMonths", v)}
            />
            <ParamSlider
              label="TEST WINDOW"
              value={s.testMonths}
              min={RANGES.testMonths.min}
              max={RANGES.testMonths.max}
              step={RANGES.testMonths.step}
              tooltip="Months held out for validation after training. Shorter windows = more frequent retraining."
              onChange={(v) => setField("testMonths", v)}
            />
            <ParamSelect
              label="PERIOD UNIT"
              value={s.periodUnit ?? "months"}
              options={[...SELECT_OPTIONS.periodUnit]}
              tooltip="Granularity of each train/test window. Months is standard for FX; weeks/days for intraday strategies."
              onChange={(v) => setField("periodUnit", v as "months" | "weeks" | "days")}
            />
          </div>

          <div className="flex flex-col gap-4">
            <ParamSelect
              label="HPO MODE"
              value={s.hpoMode ?? "static"}
              options={[...SELECT_OPTIONS.hpoMode]}
              tooltip="Static: run HPO once on fold 1, reuse params for all subsequent folds. Dynamic: re-optimize hyperparameters at every walk-forward step."
              onChange={(v) => setField("hpoMode", v as string)}
            />
            {s.hpoMode === "dynamic" && (
              <ParamSlider
                label="DYNAMIC HPO TRIALS"
                value={s.dynamicHpoTrials}
                min={RANGES.dynamicHpoTrials.min}
                max={RANGES.dynamicHpoTrials.max}
                step={RANGES.dynamicHpoTrials.step}
                tooltip="Number of HPO trials per walk-forward step in dynamic mode."
                onChange={(v) => setField("dynamicHpoTrials", v)}
              />
            )}
            <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSlider
                label="WFO TRAIN PERIODS"
                value={s.wfoTrainPeriods}
                min={RANGES.wfoTrainPeriods.min}
                max={RANGES.wfoTrainPeriods.max}
                step={RANGES.wfoTrainPeriods.step}
                tooltip="Override train window in period units. 0 = use default from train months setting above."
                onChange={(v) => setField("wfoTrainPeriods", v)}
              />
              <ParamSlider
                label="WFO TEST PERIODS"
                value={s.wfoTestPeriods}
                min={RANGES.wfoTestPeriods.min}
                max={RANGES.wfoTestPeriods.max}
                step={RANGES.wfoTestPeriods.step}
                tooltip="Override test window in period units. 0 = use default from test months setting above."
                onChange={(v) => setField("wfoTestPeriods", v)}
              />
            </div>
          </div>
        </div>
      </Section>

      <Section title="Quality Gates">
        <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
          <ParamSlider
            label="ACTIVE RATE"
            value={s.targetActiveRate}
            min={RANGES.targetActiveRate.min}
            max={RANGES.targetActiveRate.max}
            step={RANGES.targetActiveRate.step}
            tooltip="Minimum % of bars the model should generate signals on. Filters out idle models that never trade. Set lower for sparse signal strategies."
            onChange={(v) => setField("targetActiveRate", v)}
          />
          <ParamSlider
            label="COVERAGE"
            value={s.targetCoverage}
            min={RANGES.targetCoverage.min}
            max={RANGES.targetCoverage.max}
            step={RANGES.targetCoverage.step}
            tooltip="Minimum % of the test period the model must maintain exposure. Filters models that are only active during tiny windows."
            onChange={(v) => setField("targetCoverage", v)}
          />
          <ParamSlider
            label="CONFIDENCE"
            value={s.confidenceThreshold}
            min={RANGES.confidenceThreshold.min}
            max={RANGES.confidenceThreshold.max}
            step={RANGES.confidenceThreshold.step}
            tooltip="Minimum prediction confidence to act on a signal. Higher = fewer but higher-conviction trades. 0.8 means only trades with >80% model confidence execute."
            onChange={(v) => setField("confidenceThreshold", v)}
          />
        </div>
      </Section>

      <Section title="Execution Reality">
        <div className="flex flex-col gap-6">
          <div className="max-w-xs">
            <ParamToggle
              label="TRADING COSTS"
              checked={s.evalUseTradingCosts}
              tooltip="Enable spread + slippage simulation in the backtest loop. Disable for raw signal quality assessment; enable for realistic P&L estimates."
              onChange={(v) => setField("evalUseTradingCosts", v)}
            />
          </div>

          {s.evalUseTradingCosts && (
            <div className="grid grid-cols-1 items-start gap-x-10 gap-y-8 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
              <ParamSlider
                label="SLIPPAGE (BPS)"
                value={s.slipNormBps}
                min={RANGES.slipNormBps.min}
                max={RANGES.slipNormBps.max}
                step={RANGES.slipNormBps.step}
                tooltip="Average execution slippage in basis points. 1 bps = 0.01% of notional. 0.25 bps is typical for EURUSD retail FX."
                onChange={(v) => setField("slipNormBps", v)}
              />
            </div>
          )}
        </div>
      </Section>

    </Panel>
  );
}
