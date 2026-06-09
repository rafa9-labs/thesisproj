import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { modelDescriptions } from "@/lib/tokens";

export function FullCycleConfigSummaryBar() {
  const pair = useFullCycleStore((s) => s.pair);
  const timeframe = useFullCycleStore((s) => s.timeframe);
  const models = useFullCycleStore((s) => s.selectedModels);
  const trainMonths = useFullCycleStore((s) => s.trainMonths);
  const testMonths = useFullCycleStore((s) => s.testMonths);
  const proposer = useFullCycleStore((s) => s.proposer);
  const maxIterations = useFullCycleStore((s) => s.maxIterations);
  const sweepNEstimators = useFullCycleStore((s) => s.sweepNEstimators);
  const sweepMaxDepth = useFullCycleStore((s) => s.sweepMaxDepth);

  const modelNames = models.map(
    (m) => (modelDescriptions as Record<string, { name: string }>)[m]?.name ?? m,
  );
  const modelText =
    models.length === 0
      ? "\u2014"
      : models.length === 1
        ? modelNames[0]
        : `${modelNames[0]} +${models.length - 1}`;

  const proposerLabel = proposer === "deterministic" ? "Greedy" : proposer === "hybrid_llm_ucb1" ? "LLM+UCB1" : "LLM";

  const hasModels = models.length > 0;

  const readouts: { label: string; value: string; active?: boolean }[] = [
    { label: "ASSET", value: `${pair} \u00b7 ${timeframe}` },
    { label: "MODELS", value: modelText, active: hasModels },
    { label: "WFO", value: `${trainMonths}mo / ${testMonths}mo` },
    { label: "FACTORY", value: `${proposerLabel} \u00b7 ${maxIterations} iter`, active: true },
    { label: "SWEEP", value: `RF-${sweepNEstimators} \u00b7 d=${sweepMaxDepth}` },
  ];

  return (
    <div
      className="flex items-center px-6 pb-4"
      style={{
        borderBottom: "1px solid var(--color-glass-border)",
      }}
    >
      {readouts.map(({ label, value, active }, i) => (
        <div
          key={label}
          className="flex flex-col flex-1"
          style={{
            paddingLeft: i === 0 ? 0 : 24,
            paddingRight: 24,
          }}
        >
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--color-text-muted)",
              opacity: 0.7,
              marginBottom: 4,
            }}
          >
            {label}
          </span>
          <span
            style={{
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              fontWeight: 500,
              color: active
                ? "var(--color-brand)"
                : "var(--color-text-primary)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
            title={value}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}
