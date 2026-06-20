import { Rocket, ArrowRight } from "lucide-react";
import { useFullCycleStore } from "@/stores/useFullCycleStore";

const PHASE_FLOW_NAMES = ["Sweep", "HPO", "Build", "W-Fwd", "Factory", "Snapshot"] as const;

interface Props {
  canDeploy: boolean;
  isSubmitting: boolean;
  onDeploy: () => void;
}

function PhaseFlowPills() {
  const s = useFullCycleStore();

  const phaseEnabled: boolean[] = [
    !s.skipFeatureSweep,
    s.enablePhase3,
    s.enablePhase4,
    s.enablePhase5,
    s.enablePhase6,
    s.enablePhase6,
  ];

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase shrink-0">
        Flow
      </span>
      <div className="flex items-center gap-0.5">
        {phaseEnabled.map((on, i) => (
          <div key={i} className="flex items-center gap-0.5">
            <span
              className="rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold !leading-none"
              style={{
                backgroundColor: on ? "var(--color-brand-glow)" : "transparent",
                color: on ? "var(--color-brand)" : "var(--color-text-dim)",
                textDecoration: on ? "none" : "line-through",
                opacity: on ? 1 : 0.4,
              }}
            >
              {i + 1}·{PHASE_FLOW_NAMES[i]}
            </span>
            {i < 5 && (
              <ArrowRight
                size={10}
                strokeWidth={1.5}
                className="shrink-0"
                style={{
                  color: on && phaseEnabled[i + 1] ? "var(--color-brand)" : "var(--color-text-dim)",
                  opacity: on && phaseEnabled[i + 1] ? 0.5 : 0.2,
                }}
              />
            )}
          </div>
        ))}
      </div>
      <span className="text-[9px] text-(--color-text-dim) shrink-0 font-mono ml-1">
        {phaseEnabled.filter(Boolean).length}/6
      </span>
    </div>
  );
}

function ConfigSummaryInline() {
  const s = useFullCycleStore();

  const pair = s.pair || "—";
  const tf = s.timeframe || "—";
  const modelCount = s.selectedModels.length;
  const topK = s.committeeTopK;

  const proposerLabel: Record<string, string> = {
    llm: "LLM",
    hybrid_llm_ucb1: "LLM+UCB1",
    ucb1: "UCB1",
    deterministic: "Greedy",
  };
  const proposerShort = proposerLabel[s.proposer] ?? s.proposer;
  const backend = s.llmBackend === "deepseek" ? "DeepSeek" : s.llmBackend;

  const pipe = (
    <span className="text-[10px] text-(--color-text-muted) opacity-30 mx-1.5">|</span>
  );

  return (
    <div className="flex min-w-0 items-center gap-0 truncate font-mono text-[11px] leading-none">
      <span className="shrink-0 text-(--color-text-primary) whitespace-nowrap">
        {pair} · {tf}
      </span>
      {pipe}
      <span className="shrink-0 text-(--color-brand) whitespace-nowrap">
        {modelCount > 0 ? `${modelCount} Model${modelCount !== 1 ? "s" : ""}` : "—"}
      </span>
      {pipe}
      <span className="flex shrink-0 items-center gap-1 whitespace-nowrap">
        <span className="text-(--color-text-muted) uppercase tracking-wider text-[9px]">
          Top-K
        </span>
        <span className="text-(--color-text-primary)">{topK}</span>
      </span>
      {pipe}
      <span className="flex shrink-0 items-center gap-1 whitespace-nowrap">
        <span className="text-(--color-text-muted) uppercase tracking-wider text-[9px]">
          {proposerShort}
        </span>
        {(s.proposer === "llm" || s.proposer === "hybrid_llm_ucb1") && (
          <span className="text-(--color-text-dim) text-[9px]">
            · {backend}
          </span>
        )}
      </span>
    </div>
  );
}

export function CommitteeValidationBar({ canDeploy, isSubmitting, onDeploy }: Props) {
  const hasModels = useFullCycleStore((s) => s.selectedModels.length > 0);
  const ready = hasModels;

  return (
    <div className="sticky bottom-0 z-20 flex min-h-[56px] items-center justify-between gap-3 border-t border-(--color-glass-border) bg-(--color-surface) px-4 py-2 sm:px-6">
      {/* Left: status + summary */}
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
          {!ready ? (
            <>
              <span className="text-[12px] font-semibold text-(--color-text-secondary) sm:text-[13px]">
                Setup incomplete
              </span>
              <span className="font-mono text-[10px] text-(--color-text-muted) sm:text-[11px]">
                Select at least one model to deploy
              </span>
            </>
          ) : (
            <div className="flex min-w-0 flex-col gap-1">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-bold text-(--color-text-primary) sm:text-[14px]">
                  Ready
                </span>
                <span className="hidden sm:block text-[10px] text-(--color-text-muted) opacity-30">|</span>
                <div className="hidden sm:flex">
                  <ConfigSummaryInline />
                </div>
              </div>
              <div className="md:flex">
                <PhaseFlowPills />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: action */}
      <div className="flex shrink-0 items-center gap-2 sm:gap-3">
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
          {isSubmitting ? "Deploying..." : "Deploy Committee Run"}
          {!isSubmitting && <ArrowRight size={15} strokeWidth={2.25} />}
        </button>
      </div>
    </div>
  );
}
