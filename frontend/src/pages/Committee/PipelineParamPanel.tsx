import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { Section } from "@/components/shared/Panel";
import { ParamToggle } from "@/components/shared/ParamToggle";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";

const SELECT_OPTIONS = {
  sampler: [
    { value: "tpe", label: "TPE" },
    { value: "random", label: "Random Search" },
    { value: "grid", label: "Grid Search" },
  ],
  weighting: [
    { value: "sharpe_proportional", label: "Sharpe-Proportional" },
    { value: "equal", label: "Equal Weights" },
  ],
  proposer: [
    { value: "llm", label: "LLM Director" },
    { value: "hybrid_llm_ucb1", label: "LLM+UCB1 Hybrid" },
    { value: "ucb1", label: "UCB1 Bandit" },
    { value: "deterministic", label: "Greedy Deterministic" },
  ],
  llmBackend: [
    { value: "deepseek", label: "DeepSeek" },
    { value: "ollama", label: "Ollama (Local)" },
    { value: "openai", label: "OpenAI" },
    { value: "anthropic", label: "Anthropic (Claude)" },
  ],
};

const PHASE_DESCRIPTIONS: Record<number, string> = {
  1: "A shallow Random Forest is trained on the full candidate feature set (250+ indicators). Boruta-SHAP iteratively removes features that perform no better than randomly generated shadow features. The surviving feature list feeds all downstream phases, preventing overfitting from weak or noisy inputs.",
  2: "Hyperparameter optimization runs on every model that passed the feature sweep. CPU-based classical models (RF, XGB, LGBM, SVM, CatBoost, Logistic) are tuned in parallel via joblib. GPU deep-learning models (LSTM, CNN, Transformer, GRU) and ensembles are tuned sequentially to avoid VRAM exhaustion. ASHA pruning terminates underperforming trials early.",
  3: "The pipeline classifies each historical month into one of 7 market regimes (Trend Up, Trend Down, Sideways, Volatile, Quiet, Reversal, Breakout). For each regime, the top-K best-performing models are selected by regime-specific Sharpe ratio and assembled into a weighted committee. Weights are either Sharpe-proportional or equal.",
  4: "The assembled committee is validated through a 36-month walk-forward backtest. Three metrics are computed: Fold Consistency CV (stability across walk-forward folds), PBO (Probability of Backtest Overfitting), and a composite Trust Score. The validation is repeated across 3 different random seeds to ensure robustness. Optionally, a meta-learner gates the committee.",
  5: "An iterative optimizer proposes model swaps, additions, or removals per regime. Each proposal is stress-tested through a proxy walk-forward backtest. The proposer (LLM Director, UCB1 Bandit, Hybrid, or Greedy) decides whether to accept or reject based on Sharpe improvement. The loop stops when patience is exhausted, the hard gate is triggered, or proposals diverge.",
  6: "The final optimized committee is trained on the entire dataset and saved as a byte-for-byte reproducible artifact. This snapshot includes all model weights, feature lists, and regime assignments — ready for live trading deployment with no further training required.",
};

interface Props {
  selectedPhase: number;
}

export function PipelineParamPanel({ selectedPhase }: Props) {
  return (
    <div className="flex flex-col gap-3 pt-3">
      <p className="text-[10px] leading-relaxed text-(--color-text-dim)">
        Each phase in the Full Cycle pipeline consumes the output of the previous phase. Toggle phases on/off or adjust their parameters below. Click a phase node in the diagram above to reveal its controls.
      </p>
      {selectedPhase === 1 && <Phase1Controls />}
      {selectedPhase === 2 && <Phase2Controls />}
      {selectedPhase === 3 && <Phase3Controls />}
      {selectedPhase === 4 && <Phase4Controls />}
      {selectedPhase === 5 && <Phase5Controls />}
      {selectedPhase === 6 && <Phase6Info />}
    </div>
  );
}

function Phase1Controls() {
  const store = useFullCycleStore();

  return (
    <Section title="Phase 1: Feature Sweep">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[1]}
      </p>
      <ParamToggle
        label="Enable Feature Sweep"
        checked={!store.skipFeatureSweep}
        onChange={(v) => store.setSkipFeatureSweep(!v)}
        tooltip="When disabled, all features pass through unfiltered to downstream phases. Disable only when you trust the raw indicator set."
      />
      <div className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        <ParamSlider
          label="Estimators"
          value={store.sweepNEstimators}
          min={50} max={300} step={10}
          onChange={store.setSweepNEstimators}
          tooltip="Number of trees in the shallow Random Forest. Higher values increase feature importance stability at the cost of runtime."
        />
        <ParamSlider
          label="Max Depth"
          value={store.sweepMaxDepth}
          min={2} max={10} step={1}
          onChange={store.setSweepMaxDepth}
          tooltip="Maximum depth of each decision tree. Deeper trees capture more interactions but risk overfitting."
        />
      </div>
      <div className="mt-4">
        <ParamToggle
          label="Boruta-SHAP"
          checked={store.useBorutaShap}
          onChange={store.setUseBorutaShap}
          tooltip="Iteratively removes features that perform no better than randomly generated shadow features. Produces a robust minimal feature set."
        />
        {store.useBorutaShap && (
          <div className="mt-4 grid grid-cols-1 items-start gap-4 border-l-2 pl-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4" style={{ borderColor: "var(--color-glass-border)" }}>
            <ParamSlider
              label="Percentile"
              value={store.borutaPercentile}
              min={80} max={95} step={5}
              onChange={store.setBorutaPercentile}
              tooltip="Shadow feature comparison percentile. Higher values are more conservative — fewer features survive."
            />
            <ParamSlider
              label="Max Iterations"
              value={store.borutaMaxIter}
              min={10} max={50} step={5}
              onChange={store.setBorutaMaxIter}
              tooltip="Maximum Boruta-SHAP iterations. The algorithm stops early if all features are classified."
            />
          </div>
        )}
      </div>
      <div className="mt-4">
        <ParamToggle
          label="Debug Mode"
          checked={store.debugMode}
          onChange={store.setDebugMode}
          tooltip="Runs the sweep with reduced settings (fewer estimators, shallower trees) for faster iteration during development."
        />
      </div>
    </Section>
  );
}

function Phase2Controls() {
  const store = useFullCycleStore();

  return (
    <Section title="Phase 2: HPO Tuning">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[2]}
      </p>
      <ParamToggle
        label="Enable HPO Tuning"
        checked={store.enablePhase3}
        onChange={store.setEnablePhase3}
        tooltip="When disabled, models use default hyperparameters. Enable to let Optuna find optimal configurations per model."
      />
      <div className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        <ParamSelect
          label="Sampler"
          value={store.hpoSampler}
          options={SELECT_OPTIONS.sampler}
          onChange={store.setHpoSampler}
          tooltip="TPE (Tree-structured Parzen Estimator) is the recommended default. Random Search provides unbiased exploration. Grid Search exhausts the parameter space."
        />
        <ParamSlider
          label="CV Blocks"
          value={store.cvBlocks}
          min={2} max={10} step={1}
          onChange={store.setCvBlocks}
          tooltip="Number of cross-validation blocks per Optuna trial. Each block acts as a mini walk-forward fold for fitness evaluation."
        />
      </div>
    </Section>
  );
}

function Phase3Controls() {
  const store = useFullCycleStore();

  return (
    <Section title="Phase 3: Committee Build">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[3]}
      </p>
      <ParamToggle
        label="Enable Committee Assembly"
        checked={store.enablePhase4}
        onChange={store.setEnablePhase4}
        tooltip="When disabled, no per-regime committee is assembled. Downstream phases operate on individual model rankings."
      />
      <div className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        <ParamSlider
          label="Top-K Survivors"
          value={store.committeeTopK}
          min={1} max={5} step={1}
          onChange={store.setCommitteeTopK}
          tooltip="Number of top-performing models retained per market regime. Higher values increase diversity at the cost of signal dilution."
        />
        <ParamSlider
          label="Minimum Sharpe"
          value={store.committeeMinSharpe}
          min={-2} max={2} step={0.1}
          onChange={store.setCommitteeMinSharpe}
          tooltip="Minimum regime-specific Sharpe ratio required for a model to join the committee. Set negative values to allow inclusion of weak but diversifying models."
        />
        <ParamSelect
          label="Weighting"
          value={store.committeeWeightMethod}
          options={SELECT_OPTIONS.weighting}
          onChange={store.setCommitteeWeightMethod}
          tooltip="Sharpe-Proportional allocates more weight to models with higher regime-specific Sharpe. Equal Weights gives every model the same vote."
        />
      </div>
      <div className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        <ParamSlider
          label="Train Months"
          value={store.trainMonths}
          min={12} max={120} step={6}
          onChange={store.setTrainMonths}
          tooltip="Number of months used for training in each walk-forward fold. Longer windows capture more regimes but reduce reactivity."
        />
        <ParamSlider
          label="Test Months"
          value={store.testMonths}
          min={1} max={12} step={1}
          onChange={store.setTestMonths}
          tooltip="Number of months used for out-of-sample testing in each fold. 1 month provides monthly refit; longer windows reduce turnover."
        />
      </div>
    </Section>
  );
}

function Phase4Controls() {
  const store = useFullCycleStore();

  return (
    <Section title="Phase 4: Walk-Forward Validation">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[4]}
      </p>
      <ParamToggle
        label="Enable Validation"
        checked={store.enablePhase5}
        onChange={store.setEnablePhase5}
        tooltip="Runs 36-month walk-forward validation with fold consistency CV, PBO computation, trust score, and 3-seed robustness checks."
      />
    </Section>
  );
}

function Phase5Controls() {
  const store = useFullCycleStore();
  const showLlmBackend = store.proposer === "llm" || store.proposer === "hybrid_llm_ucb1";
  const showUcbC = store.proposer === "hybrid_llm_ucb1" || store.proposer === "ucb1";

  return (
    <Section title="Phase 5: Factory Optimization">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[5]}
      </p>
      <ParamToggle
        label="Enable Factory Optimization"
        checked={store.enablePhase6}
        onChange={store.setEnablePhase6}
        tooltip="When disabled, the committee is deployed as-is after walk-forward validation without further iterative optimization."
      />
      <div className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        <ParamSelect
          label="Proposer Strategy"
          value={store.proposer}
          options={SELECT_OPTIONS.proposer}
          onChange={store.setProposer}
          tooltip="LLM Director uses a language model to reason about model swaps. UCB1 Bandit balances exploration/exploitation. Hybrid combines both. Greedy always picks the best immediate swap."
        />
        {showLlmBackend && (
          <ParamSelect
            label="LLM Backend"
            value={store.llmBackend}
            options={SELECT_OPTIONS.llmBackend}
            onChange={store.setLlmBackend}
            tooltip="DeepSeek is the default (fast, cost-effective). Ollama runs locally for privacy. OpenAI and Anthropic provide alternative reasoning quality."
          />
        )}
        {showUcbC && (
          <ParamSlider
            label="UCB1 Exploration C"
            value={store.ucb1ExplorationC}
            min={0.5} max={5} step={0.1}
            onChange={store.setUcb1ExplorationC}
            tooltip="Exploration-exploitation trade-off coefficient. Higher values encourage trying untested model combinations."
          />
        )}
        <ParamSlider
          label="Max Iterations"
          value={store.maxIterations}
          min={3} max={50} step={1}
          onChange={store.setMaxIterations}
          tooltip="Maximum number of factory optimization iterations. Each iteration proposes, executes, and evaluates one committee modification."
        />
        <ParamSlider
          label="Factory Patience"
          value={store.factoryPatience}
          min={3} max={15} step={1}
          onChange={store.setFactoryPatience}
          tooltip="Number of consecutive iterations without Sharpe improvement before early stopping."
        />
        <ParamSlider
          label="Stopping Tolerance"
          value={store.stoppingTolerance}
          min={0.005} max={0.1} step={0.005}
          onChange={store.setStoppingTolerance}
          tooltip="Minimum Sharpe improvement required to consider an iteration as progress. Smaller values demand finer improvements."
        />
      </div>
    </Section>
  );
}

function Phase6Info() {

  return (
    <Section title="Phase 6: Snapshot &amp; Deployment">
      <p className="mb-4 text-[10px] leading-relaxed text-(--color-text-dim)">
        {PHASE_DESCRIPTIONS[6]}
      </p>
      <p className="text-[11px] text-(--color-text-muted)">
        Runs automatically after Phase 5 completes. No configuration required.
      </p>
    </Section>
  );
}
