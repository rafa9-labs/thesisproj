import { useState } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { MODEL_CONSTRAINTS } from "@/lib/constants";
import { modelDescriptions } from "@/lib/tokens";
import { ChevronDown, ChevronRight, AlertTriangle, Lightbulb } from "lucide-react";

export function ParameterGuide() {
  const [open, setOpen] = useState(false);
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const lags = useBacktestStore((s) => s.lags);

  if (selectedModels.length === 0) return null;

  return (
    <div
      className="rounded-lg border overflow-hidden"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-3 py-2 text-left"
        style={{ cursor: "pointer", color: "var(--color-text-secondary)" }}
      >
        <div className="flex items-center gap-1.5">
          <Lightbulb size={12} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-[10px] font-semibold uppercase tracking-[0.06em]">
            Parameter Guide
          </span>
        </div>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>

      {open && (
        <div className="flex flex-col gap-2 px-3 pb-3 pt-1">
          {selectedModels.map((modelKey) => {
            const name = (modelDescriptions as Record<string, { name: string }>)[modelKey]?.name ?? modelKey;
            const constraints = MODEL_CONSTRAINTS[modelKey];
            return (
              <div
                key={modelKey}
                className="rounded p-2"
                style={{ backgroundColor: "var(--color-elevated)" }}
              >
                <span className="text-[10px] font-semibold uppercase tracking-[0.05em]" style={{ color: "var(--color-brand)" }}>
                  {name}
                </span>

                {constraints ? (
                  <div className="mt-1.5 flex flex-col gap-1.5">
                    {constraints.rules.map((rule, i) => (
                      <div key={`r-${i}`} className="flex items-start gap-1.5">
                        <AlertTriangle size={10} style={{ color: "var(--color-accent-warning)", marginTop: 1, flexShrink: 0 }} />
                        <span className="text-[9px] leading-relaxed" style={{ color: "var(--color-accent-warning)" }}>
                          {rule}
                        </span>
                      </div>
                    ))}
                    {constraints.tips.map((tip, i) => (
                      <div key={`t-${i}`} className="flex items-start gap-1.5">
                        <Lightbulb size={10} style={{ color: "var(--color-text-muted)", marginTop: 1, flexShrink: 0 }} />
                        <span className="text-[9px] leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
                          {tip}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>
                    No parameter guidance available.
                  </span>
                )}
              </div>
            );
          })}
          <span className="text-[8px]" style={{ color: "var(--color-text-muted)" }}>
            Based on known structural constraints and your own top-trial statistics.
          </span>
        </div>
      )}
    </div>
  );
}
