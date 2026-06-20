import { TriangleAlert, Bookmark, Rocket, ArrowRight, FileText, CircleX, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { useRuntimeEstimate } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

interface Props {
  warnings: number;
  errors: number;
  errorMessages: string[];
  warningMessages: string[];
  canDeploy: boolean;
  isSubmitting: boolean;
  hasModels: boolean;
  hasPair: boolean;
  hasDates?: boolean;
  runningCount?: number;
  onDeploy: () => void;
  onViewSummary: () => void;
  onSavePreset?: () => void;
}

const MODEL_ABBREV: Record<string, string> = {
  logistic: "LogReg",
  svm: "SVM",
  random_forest: "RF",
  decision_tree: "DT",
  xgboost: "XGB",
  lightgbm: "LGBM",
  catboost: "CatB",
  cnn: "CNN",
  lstm: "LSTM",
  transformer: "Transf",
  gru: "GRU",
  gru_lstm: "GRU-LSTM",
  dqn: "DQN",
  ensemble_adaptive_regime: "Regime",
  ensemble_cnn_lstm_xgboost: "CNN+LSTM+XGB",
  meta_ensemble: "Committee",
  stacking_ensemble: "Stacking",
};

const _FEATURE_FLAGS: (keyof ReturnType<typeof useBacktestStore.getState>)[] = [
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
];

function RuntimeEstimate() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const testMonths = useBacktestStore((s) => s.testMonths);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);

  const { data, isLoading } = useRuntimeEstimate(
    selectedModels as string[],
    testMonths as number,
    hpoIntensity,
  );

  if (selectedModels.length === 0 || isLoading || !data) return null;

  const { estimated_minutes_low, estimated_minutes_high } = data;

  const fmt = (mins: number) => {
    if (mins < 1) return `${Math.round(mins * 60)}s`;
    if (mins < 60) return `${Math.round(mins)}min`;
    const h = Math.floor(mins / 60);
    const m = Math.round(mins % 60);
    return m > 0 ? `${h}h${m}m` : `${h}h`;
  };

  return (
    <>
      <span className="text-[10px] text-(--color-text-muted) opacity-30">|</span>
      <span className="font-mono text-[11px] text-(--color-text-muted) whitespace-nowrap">
        Est. {fmt(estimated_minutes_low)}–{fmt(estimated_minutes_high)}
      </span>
    </>
  );
}

function ConfigSummaryInline() {
  const s = useBacktestStore();

  const pair = (s.pair as string) || "—";
  const tf = (s.timeframe as string) || "—";
  const modelsRaw = (s.selectedModels as string[]) ?? [];
  const models: string[] = Array.isArray(modelsRaw) ? modelsRaw : [];
  const trainMonths = (s.trainMonths as number) ?? 36;
  const testMonths = (s.testMonths as number) ?? 1;
  const nTrials = (s.nTrials as number) ?? 10;

  const activeFeatures = _FEATURE_FLAGS.filter((k) => s[k] === true).length;
  const hasMtf = s.useMtfMa === true || s.useMtfAlignment === true;

  const modelText = (() => {
    if (models.length === 0) return null;
    if (models.length > 2) return `${models.length}M`;
    return models.map((m) => MODEL_ABBREV[m] ?? m).join(",");
  })();

  const pipe = (
    <span className="text-[10px] text-(--color-text-muted) opacity-30 mx-1.5">|</span>
  );

  return (
    <div className="flex min-w-0 items-center gap-0 truncate text-[11px] font-mono leading-none">
      {/* Pair · TF */}
      <span className="shrink-0 text-(--color-text-primary) whitespace-nowrap">
        {pair} · {tf}
      </span>

      {/* Models */}
      <>
        {pipe}
        {modelText ? (
          <span className="shrink-0 text-(--color-brand) whitespace-nowrap">{modelText}</span>
        ) : (
          <span className="shrink-0 text-[var(--color-accent-danger)]">—</span>
        )}
      </>

      {/* Walk-forward */}
      <>
        {pipe}
        <span className="flex shrink min-w-0 items-center gap-1 whitespace-nowrap">
          <span className="hidden lg:inline text-(--color-text-muted) uppercase tracking-wider" style={{ fontSize: 9 }}>
            W-Fwd
          </span>
          <span className="text-(--color-text-primary)">
            {trainMonths}mo/{testMonths}mo
          </span>
        </span>
      </>

      {/* HPO */}
      <>
        {pipe}
        <span className="flex shrink min-w-0 items-center gap-1 whitespace-nowrap">
          <span className="hidden xl:inline text-(--color-text-muted) uppercase tracking-wider" style={{ fontSize: 9 }}>
            HPO
          </span>
          <span className="text-(--color-text-primary)">{nTrials}Tri</span>
        </span>
      </>

      {/* Features */}
      <>
        {pipe}
        <span className="flex shrink min-w-0 items-center gap-1 whitespace-nowrap">
          <span className="text-(--color-text-primary)">{activeFeatures}</span>
          <span className="hidden 2xl:inline text-(--color-text-muted) uppercase tracking-wider" style={{ fontSize: 9 }}>
            Feat
          </span>
          {hasMtf && (
            <span className="text-(--color-text-muted)" style={{ fontSize: 9 }}>
              (MTF)
            </span>
          )}
        </span>
      </>
    </div>
  );
}

export function ValidationBar({
  warnings,
  errors,
  errorMessages,
  warningMessages,
  canDeploy,
  isSubmitting,
  hasModels,
  hasPair,
  hasDates,
  runningCount,
  onDeploy,
  onViewSummary,
  onSavePreset,
}: Props) {
  const hasDatesValue = hasDates ?? true;
  const missingItems: string[] = [];
  if (!hasPair) missingItems.push("a currency pair");
  if (!hasModels) missingItems.push("at least one model");
  if (!hasDatesValue) missingItems.push("date range");

  const ready = missingItems.length === 0 && errors === 0;
  const [showDetails, setShowDetails] = useState(false);
  const hasDetails = errorMessages.length > 0 || warningMessages.length > 0;

  return (
    <div className="sticky bottom-0 z-20 flex min-h-[56px] w-full flex-wrap items-center justify-between gap-x-3 gap-y-2 border-t border-(--color-glass-border) bg-(--color-surface) px-3 py-2 sm:px-6">
      {/* Left: status + summary tokens */}
      <div className="flex min-w-0 items-center gap-3">
        <div
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg sm:h-10 sm:w-10"
          style={{
            backgroundColor: ready ? "var(--color-brand-glow)" : "var(--color-glass)",
            border: `1px solid ${ready ? "var(--color-border-active)" : "var(--color-glass-border)"}`,
            color: ready ? "var(--color-brand)" : "var(--color-text-muted)",
          }}
        >
          <Rocket size={18} strokeWidth={1.75} />
        </div>

        <div className="flex min-w-0 flex-col gap-0.5">
          {missingItems.length > 0 ? (
            <>
              <span className="text-[12px] font-semibold text-(--color-text-secondary) sm:text-[13px]">
                Setup incomplete
              </span>
              <span className="font-mono text-[10px] text-(--color-text-muted) sm:text-[11px]">
                Select {missingItems.join(" and ")} to start
              </span>
            </>
          ) : errors > 0 ? (
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-1.5">
                <CircleX size={13} className="text-(--color-accent-danger)" />
                <span className="text-[12px] font-semibold text-(--color-accent-danger) sm:text-[13px]">
                  {errors} config error{errors > 1 ? "s" : ""} must be fixed
                </span>
                {hasDetails && (
                  <button
                    onClick={() => setShowDetails(!showDetails)}
                    className="ml-1 flex items-center gap-0.5 text-[9px] text-(--color-text-muted) transition-colors hover:text-(--color-text-secondary)"
                  >
                    {showDetails ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                    {showDetails ? "Hide" : "Details"}
                  </button>
                )}
              </div>
              {showDetails && (
                <div className="ml-5 flex flex-col gap-0.5">
                  {errorMessages.map((msg, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <CircleX size={9} className="mt-[3px] shrink-0 text-(--color-accent-danger)" />
                      <span className="font-mono text-[10px] leading-tight text-(--color-text-muted) sm:text-[11px]">
                        {msg}
                      </span>
                    </div>
                  ))}
                  {warningMessages.map((msg, i) => (
                    <div key={`w-${i}`} className="flex items-start gap-1.5">
                      <TriangleAlert size={9} className="mt-[3px] shrink-0 text-(--color-accent-warning)" />
                      <span className="font-mono text-[10px] leading-tight text-(--color-text-muted) sm:text-[11px]">
                        {msg}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="flex min-w-0 items-center gap-2 sm:gap-3">
              <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
                <span className="text-[13px] font-bold text-(--color-text-primary) sm:text-[14px]">
                  Ready
                </span>
                {warnings > 0 && (
                  <button
                    onClick={() => setShowDetails(!showDetails)}
                    className="flex items-center gap-1 transition-opacity hover:opacity-80"
                  >
                    <TriangleAlert size={11} className="text-(--color-accent-warning)" />
                    <span className="font-mono text-[9px] text-(--color-accent-warning) sm:text-[10px]">
                      {warnings}w
                    </span>
                    {showDetails ? <ChevronUp size={9} className="text-(--color-text-muted)" /> : <ChevronDown size={9} className="text-(--color-text-muted)" />}
                  </button>
                )}
              </div>

              <span className="hidden sm:block text-[10px] text-(--color-text-muted) opacity-30">|</span>

              <ConfigSummaryInline />

              <div className="hidden md:flex items-center">
                <RuntimeEstimate />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Expanded details panel (errors or warnings when Ready) */}
      {showDetails && !missingItems.length && hasDetails && (
        <div className="absolute bottom-full left-0 z-30 max-h-[240px] w-full overflow-y-auto border-t border-(--color-glass-border) bg-(--color-surface) px-4 py-3 sm:px-6">
          <div className="flex flex-col gap-1.5">
            {errorMessages.map((msg, i) => (
              <div key={i} className="flex items-start gap-2">
                <CircleX size={11} className="mt-[2px] shrink-0 text-(--color-accent-danger)" />
                <span className="font-mono text-[11px] leading-snug text-(--color-text-secondary)">
                  {msg}
                </span>
              </div>
            ))}
            {warningMessages.map((msg, i) => (
              <div key={`w-${i}`} className="flex items-start gap-2">
                <TriangleAlert size={11} className="mt-[2px] shrink-0 text-(--color-accent-warning)" />
                <span className="font-mono text-[11px] leading-snug text-(--color-text-secondary)">
                  {msg}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Right: actions */}
      <div className="flex shrink-0 flex-wrap items-center gap-1.5 sm:gap-3">
        {onSavePreset && canDeploy && (
          <button
            onClick={onSavePreset}
            className="hidden sm:flex h-10 items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-transparent px-3 text-[10px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase transition-colors duration-150 hover:border-(--color-border-active) hover:text-(--color-text-primary) sm:h-11 sm:px-4 sm:text-[11px]"
          >
            <Bookmark size={13} />
            Save Draft
          </button>
        )}

        {/* View Summary */}
        <button
          onClick={onViewSummary}
          className="flex h-10 items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-transparent px-3 text-[10px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase transition-colors duration-150 hover:border-(--color-border-active) hover:text-(--color-text-primary) sm:h-11 sm:px-4 sm:text-[11px]"
        >
          <FileText size={13} />
          <span className="hidden sm:inline">Summary</span>
        </button>

        {runningCount !== undefined && runningCount > 0 && (
          <span className="font-mono text-[10px] text-(--color-text-muted) tabular-nums">
            {runningCount}/4 slots used
          </span>
        )}
        {runningCount === 4 && (
          <span className="font-mono text-[10px] text-(--color-accent-warning)">
            All slots full
          </span>
        )}

        {/* Deploy */}
        <button
          onClick={onDeploy}
          disabled={!canDeploy || isSubmitting}
          className="flex h-10 items-center gap-1.5 rounded-md border-0 bg-(--color-brand) px-4 text-[11px] font-bold tracking-[0.08em] text-(--color-text-inverse) uppercase transition-all duration-150 hover:brightness-110 sm:h-11 sm:gap-2 sm:px-7 sm:text-[12px]"
          style={{
            cursor: canDeploy && !isSubmitting ? "pointer" : "not-allowed",
            opacity: canDeploy ? (isSubmitting ? 0.7 : 1) : 0.35,
            boxShadow: canDeploy ? "0 0 20px rgba(0,229,255,0.25)" : "none",
          }}
        >
          {isSubmitting ? "Submitting..." : "Deploy Backtest"}
          {!isSubmitting && <ArrowRight size={15} strokeWidth={2.25} />}
        </button>
      </div>
    </div>
  );
}
