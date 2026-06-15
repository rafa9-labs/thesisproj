import { useModels } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelCategories, modelDescriptions } from "@/lib/tokens";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { AlertTriangle } from "lucide-react";
import type { ModelInfo } from "@/api/schemas";

const GPU_MODELS = new Set(["cnn", "lstm", "transformer", "dqn", "ensemble_adaptive_regime"]);

const CATEGORY_COLORS: Record<string, string> = {
  classical: "#22D3EE",
  deep:      "#A78BFA",
  rl:        "#F59E0B",
  ensemble:  "#EC4899",
};

export function ModelSelector() {
  const { data: models, isLoading } = useModels();
  const selected    = useBacktestStore((s) => s.selectedModels);
  const toggleModel = useBacktestStore((s) => s.toggleModel);
  const verbose     = useSettingsStore((s) => s.verboseMode);

  const modelsByCategory = categorizeModels(models ?? []);

  return (
    <div
      className="flex flex-col"
      style={{
        backgroundColor: "#1A1D27",
        border: "1px solid #2A2E39",
        borderRadius: 6,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{ borderBottom: "1px solid #2A2E39" }}
      >
        <div className="flex items-center gap-2">
          <div style={{ width: 2, height: 10, backgroundColor: "#A78BFA", borderRadius: 1 }} />
          <span
            className="text-[9px] font-semibold uppercase tracking-[0.14em]"
            style={{ color: "#787B86" }}
          >
            Model Selection
          </span>
        </div>
        <span
          className="text-[10px] tabular-nums"
          style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}
        >
          {selected.length}/5 selected
        </span>
      </div>

      {/* Table header */}
      <div
        className="grid px-3 py-1.5"
        style={{
          gridTemplateColumns: "20px 1fr 120px 140px",
          borderBottom: "1px solid #2A2E39",
          backgroundColor: "#131620",
        }}
      >
        <div />
        <span className="text-[9px] font-semibold uppercase tracking-[0.1em]" style={{ color: "#4A5568" }}>Model</span>
        <span className="text-[9px] font-semibold uppercase tracking-[0.1em]" style={{ color: "#4A5568" }}>Category</span>
        <span className="text-[9px] font-semibold uppercase tracking-[0.1em]" style={{ color: "#4A5568" }}>Notes</span>
      </div>

      {isLoading ? (
        <div className="flex flex-col gap-0">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={i}
              className="animate-pulse mx-3 my-1 rounded"
              style={{ height: 28, backgroundColor: "#2A2E39" }}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col">
          {Object.entries(modelCategories).map(([catKey, cat]) => {
            const catModels = modelsByCategory[catKey] ?? [];
            if (catModels.length === 0) return null;
            const catColor = CATEGORY_COLORS[catKey] ?? "#787B86";

            return (
              <div key={catKey}>
                {/* Category sub-header */}
                <div
                  className="grid px-3 py-1"
                  style={{
                    gridTemplateColumns: "20px 1fr 120px 140px",
                    borderBottom: "1px solid #1E222D",
                    backgroundColor: "rgba(255,255,255,0.012)",
                  }}
                >
                  <div />
                  <div className="flex items-center gap-1.5">
                    <div
                      style={{ width: 6, height: 6, borderRadius: "50%", backgroundColor: catColor, flexShrink: 0 }}
                    />
                    <span
                      className="text-[9px] font-semibold uppercase tracking-[0.12em]"
                      style={{ color: catColor }}
                    >
                      {cat.label}
                    </span>
                  </div>
                  <div />
                  <div />
                </div>

                {/* Model rows */}
                {catModels.map((m, rowIdx) => {
                  const isSelected = selected.includes(m.name);
                  const isFull     = selected.length >= 5 && !isSelected;
                  const needsGpu   = GPU_MODELS.has(m.name);
                  const desc       = modelDescriptions[m.name];
                  const isOdd      = rowIdx % 2 === 1;

                  return (
                    <button
                      key={m.name}
                      onClick={() => !isFull && toggleModel(m.name)}
                      disabled={isFull}
                      className="grid w-full px-3 text-left transition-colors duration-100"
                      style={{
                        gridTemplateColumns: "20px 1fr 120px 140px",
                        height: 34,
                        alignItems: "center",
                        borderBottom: "1px solid #1E222D",
                        backgroundColor: isSelected
                          ? "rgba(8,153,129,0.08)"
                          : isOdd
                          ? "rgba(255,255,255,0.012)"
                          : "transparent",
                        cursor: isFull ? "not-allowed" : "pointer",
                        opacity: isFull ? 0.35 : 1,
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected && !isFull)
                          (e.currentTarget as HTMLButtonElement).style.backgroundColor =
                            "rgba(255,255,255,0.03)";
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected)
                          (e.currentTarget as HTMLButtonElement).style.backgroundColor = isOdd
                            ? "rgba(255,255,255,0.012)"
                            : "transparent";
                      }}
                    >
                      {/* Checkbox */}
                      <div className="flex items-center justify-center">
                        <div
                          style={{
                            width: 12,
                            height: 12,
                            borderRadius: 3,
                            border: isSelected ? "1.5px solid #089981" : "1.5px solid #2A2E39",
                            backgroundColor: isSelected ? "#089981" : "transparent",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                            transition: "all 0.1s",
                          }}
                        >
                          {isSelected && (
                            <svg width="7" height="5" viewBox="0 0 7 5" fill="none">
                              <path d="M1 2.5L2.8 4L6 1" stroke="#0A0D12" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          )}
                        </div>
                      </div>

                      {/* Model name */}
                      <div className="flex flex-col justify-center min-w-0">
                        <span
                          className="text-[11px] font-medium truncate"
                          style={{ color: isSelected ? "#D1D4DC" : "#9CA3AF" }}
                        >
                          {desc?.name ?? m.display_name}
                        </span>
                        {verbose && desc?.short && (
                          <span
                            className="text-[9px] truncate"
                            style={{ color: "#4A5568" }}
                          >
                            {desc.short}
                          </span>
                        )}
                      </div>

                      {/* Category badge */}
                      <div>
                        <span
                          className="inline-flex items-center px-1.5 rounded text-[9px] font-medium uppercase tracking-wider"
                          style={{
                            backgroundColor: "rgba(255,255,255,0.04)",
                            border: `1px solid ${catColor}30`,
                            color: catColor,
                            lineHeight: "16px",
                          }}
                        >
                          {cat.label}
                        </span>
                      </div>

                      {/* GPU badge */}
                      <div>
                        {needsGpu && (
                          <span
                            className="inline-flex items-center gap-1 px-1.5 rounded text-[9px] font-medium"
                            style={{
                              backgroundColor: "rgba(245,158,11,0.06)",
                              border: "1px solid rgba(245,158,11,0.2)",
                              color: "#F59E0B",
                              lineHeight: "16px",
                            }}
                          >
                            <AlertTriangle size={9} strokeWidth={2} />
                            GPU rec.
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function categorizeModels(models: ModelInfo[]): Record<string, ModelInfo[]> {
  const result: Record<string, ModelInfo[]> = {};
  for (const model of models) {
    const cat = model.category;
    if (!result[cat]) result[cat] = [];
    result[cat].push(model);
  }
  return result;
}
