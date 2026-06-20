import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { FC_PRESETS } from "@/lib/constants";
import { Panel, PanelHeader } from "@/components/shared/Panel";
import { BaseModelsTab } from "./BaseModelsTab";
import { Bug, Cpu, Network, Layers, Bot } from "lucide-react";

const PRESET_ICONS: Record<string, React.ReactNode> = {
  debug: <Bug size={13} />,
  classical: <Cpu size={13} />,
  deep: <Network size={13} />,
  full: <Layers size={13} />,
  llm: <Bot size={13} />,
};
const PRESET_COLORS: Record<string, string> = {
  debug: "var(--color-text-muted)",
  classical: "var(--color-brand)",
  deep: "#a78bfa",
  full: "var(--color-accent-warning)",
  llm: "var(--color-accent-danger)",
};

export function ModelsTab() {
  const store = useFullCycleStore();

  return (
    <div className="flex flex-col gap-6">
      <Panel>
        <PanelHeader
          title="Quick Presets"
          subtitle="Start with a pre-configured model set or customize manually below."
        />
        <div className="flex flex-wrap gap-2">
          {Object.entries(FC_PRESETS).map(([key, preset]) => {
            const catColor = PRESET_COLORS[key] ?? "var(--color-text-muted)";
            const isActive = store.activePreset === key;
            return (
              <button
                key={key}
                onClick={() => store.applyPreset(key)}
                className="font-mono"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  background: isActive ? catColor : "var(--color-glass)",
                  border: `1px solid ${isActive ? "transparent" : "var(--color-glass-border)"}`,
                  borderRadius: 4,
                  padding: "6px 12px",
                  fontSize: 10,
                  color: isActive
                    ? key === "debug"
                      ? "var(--color-text-primary)"
                      : "var(--color-app)"
                    : "var(--color-text-secondary)",
                  cursor: "pointer",
                  fontWeight: isActive ? 600 : 400,
                }}
                title={preset.desc}
              >
                <span style={{ color: isActive ? "inherit" : catColor }}>
                  {PRESET_ICONS[key]}
                </span>
                {preset.label}
              </button>
            );
          })}
        </div>
      </Panel>

      <BaseModelsTab />
    </div>
  );
}
