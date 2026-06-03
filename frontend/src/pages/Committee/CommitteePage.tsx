import { useState } from "react";
import { TabBar } from "@/components/shared/TabBar";
import { FullCycleTab } from "./FullCycleTab";
import { FactoryTab } from "./FactoryTab";
import { RacecarTab } from "./RacecarTab";
import { RegimeHeatmap } from "./RegimeHeatmap";
import { CommitteeConfigPanel } from "./CommitteeConfigPanel";
import { CommitteeResultsTab } from "./CommitteeResultsTab";
import { LiveCommitteePanel } from "./LiveCommitteePanel";

const TABS = [
  { key: "fullcycle", label: "Full Cycle" },
  { key: "factory", label: "Factory" },
  { key: "racecar", label: "Racecar" },
  { key: "regime", label: "Regime Heatmap" },
  { key: "config", label: "Config" },
  { key: "results", label: "Results" },
  { key: "live", label: "Live" },
];

export function CommitteePage() {
  const [activeTab, setActiveTab] = useState("fullcycle");

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
      <div style={{ padding: "24px 24px 0" }}>
        <h1
          style={{
            fontSize: "18px",
            fontWeight: 600,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
            margin: 0,
          }}
        >
          Committee
        </h1>
        <div style={{ marginTop: 16 }}>
          <TabBar
            tabs={TABS}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        </div>
      </div>
      <div
        style={{
          flex: 1,
          overflow: "auto",
          padding: "0 24px 24px",
        }}
      >
        <div style={{ height: 24 }} />
        {activeTab === "fullcycle" && <FullCycleTab />}
        {activeTab === "factory" && <FactoryTab />}
        {activeTab === "racecar" && <RacecarTab />}
        {activeTab === "regime" && <RegimeHeatmap />}
        {activeTab === "config" && <CommitteeConfigPanel />}
        {activeTab === "results" && <CommitteeResultsTab />}
        {activeTab === "live" && <LiveCommitteePanel />}
      </div>
    </div>
  );
}
