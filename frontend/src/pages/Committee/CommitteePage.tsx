import { useState, useEffect } from "react";
import { TabBar } from "@/components/shared/TabBar";
import { FullCycleTab } from "./FullCycleTab";
import { RegimeHeatmap } from "./RegimeHeatmap";
import { CommitteeConfigPanel } from "./CommitteeConfigPanel";
import { LiveCommitteePanel } from "./LiveCommitteePanel";
import { RunHistoryTable } from "./RunHistoryTable";
import { useFullCycleStore } from "@/stores/useFullCycleStore";

const TABS = [
  { key: "pipeline", label: "Pipeline" },
  { key: "regime", label: "Regime" },
  { key: "history", label: "History" },
  { key: "live", label: "Live" },
  { key: "advanced", label: "Advanced" },
];

export function CommitteePage() {
  const [activeTab, setActiveTab] = useState("pipeline");
  const deployedSessionId = useFullCycleStore((s) => s.deployedSessionId);
  const setSelectedHistoryJobId = useFullCycleStore((s) => s.setSelectedHistoryJobId);

  useEffect(() => {
    if (deployedSessionId) setActiveTab("live");
  }, [deployedSessionId]);

  const handleHistorySelect = (jobId: string) => {
    setSelectedHistoryJobId(jobId);
    setActiveTab("pipeline");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
      <div style={{ padding: "24px 24px 0" }}>
        <h1 style={{ fontSize: "18px", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-text-primary)", margin: 0 }}>
          Committee
        </h1>
        <div style={{ marginTop: 16 }}><TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} /></div>
      </div>
      <div style={{ flex: 1, overflow: "auto", padding: "0 24px 24px" }}>
        <div style={{ height: 24 }} />
        {activeTab === "pipeline" && <FullCycleTab />}
        {activeTab === "regime" && <RegimeHeatmap />}
        {activeTab === "history" && <RunHistoryTable activeJobId={null} onSelect={handleHistorySelect} />}
        {activeTab === "live" && <LiveCommitteePanel sessionId={deployedSessionId} />}
        {activeTab === "advanced" && <CommitteeConfigPanel />}
      </div>
    </div>
  );
}
