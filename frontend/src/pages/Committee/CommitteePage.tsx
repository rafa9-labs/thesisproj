import { useState } from "react";
import { TabBar } from "@/components/shared/TabBar";
import { FullCycleTab } from "./FullCycleTab";
import { RegimeHeatmap } from "./RegimeHeatmap";
import { CommitteeConfigPanel } from "./CommitteeConfigPanel";
import { RunHistoryTable } from "./RunHistoryTable";
import { useFullCycleStore } from "@/stores/useFullCycleStore";

const TABS = [
  { key: "pipeline", label: "Pipeline" },
  { key: "regime", label: "Regime" },
  { key: "history", label: "History" },
  { key: "advanced", label: "Advanced" },
];

export function CommitteePage() {
  const [activeTab, setActiveTab] = useState("pipeline");
  const setSelectedHistoryJobId = useFullCycleStore((s) => s.setSelectedHistoryJobId);

  const handleHistorySelect = (jobId: string) => {
    setSelectedHistoryJobId(jobId);
    setActiveTab("pipeline");
  };

  return (
    <div className="flex flex-1 flex-col">
      <div className="p-[24px_24px_0]">
        <h1 className="m-0 text-[18px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
          Committee
        </h1>
        <div className="mt-[16px]">
          <TabBar tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
        </div>
      </div>
      <div className="flex-1 overflow-auto p-[0_24px_24px]">
        <div className="h-[24px]" />
        {activeTab === "pipeline" && <FullCycleTab />}
        {activeTab === "regime" && <RegimeHeatmap />}
        {activeTab === "history" && (
          <RunHistoryTable activeJobId={null} onSelect={handleHistorySelect} />
        )}
        {activeTab === "advanced" && <CommitteeConfigPanel />}
      </div>
    </div>
  );
}
