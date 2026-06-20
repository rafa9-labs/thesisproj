import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Trash2 } from "lucide-react";
import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { LogEntry } from "@/api/schemas";

const LEVEL_COLORS: Record<string, string> = {
  info: "var(--color-text-secondary)",
  warn: "var(--color-accent-warning)",
  error: "var(--color-accent-danger)",
  debug: "var(--color-text-dim)",
};

const LEVEL_BG: Record<string, string> = {
  info: "transparent",
  warn: "rgba(245,158,11,0.06)",
  error: "rgba(244,63,94,0.06)",
  debug: "transparent",
};

const PHASE_LABELS: Record<number, string> = {
  1: "P1",
  2: "P2",
  3: "P3",
  4: "P4",
  5: "P5",
};

function formatTime(ts: string): string {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "";
  }
}

export function CommitteeLogConsole() {
  const logs = useCommitteeMonitorStore((s) => s.logs);
  const logsAutoScroll = useCommitteeMonitorStore((s) => s.logsAutoScroll);
  const setLogsAutoScroll = useCommitteeMonitorStore((s) => s.setLogsAutoScroll);
  const selectedJobId = useCommitteeMonitorStore((s) => s.selectedJobId);

  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logsAutoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, logsAutoScroll]);

  if (!selectedJobId) return null;

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 40;
    if (isAtBottom !== logsAutoScroll) {
      setLogsAutoScroll(isAtBottom);
    }
  };

  return (
    <div className="shrink-0 border-t border-(--color-glass-border)">
      {/* Toggle bar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.06em] text-(--color-text-muted) transition-colors duration-150 sm:px-6"
        style={{
          backgroundColor: isOpen ? "var(--color-surface)" : "transparent",
        }}
      >
        <div className="flex items-center gap-2">
          <span className="font-mono text-(--color-text-dim)">{">_"}</span>
          <span>Live Log</span>
          {logs.length > 0 && (
            <span className="font-mono text-[9px] text-(--color-text-dim)">
              {logs.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isOpen && (
            <Trash2
              size={12}
              className="text-(--color-text-dim) hover:text-(--color-text-secondary)"
              onClick={(e) => {
                e.stopPropagation();
                useCommitteeMonitorStore.setState({ logs: [], logsNextIndex: 0 });
              }}
            />
          )}
          {isOpen ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </div>
      </button>

      {/* Log output */}
      {isOpen && (
        <div
          ref={containerRef}
          onScroll={handleScroll}
          className="overflow-y-auto px-3 py-1 sm:px-6"
          style={{ maxHeight: 180, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, backgroundColor: "#080c14" }}
        >
          {logs.length === 0 && (
            <div className="py-2 text-center text-[10px] text-(--color-text-dim)">
              Waiting for events...
            </div>
          )}
          {logs.map((entry) => (
            <LogLine key={entry.index} entry={entry} />
          ))}
          {!logsAutoScroll && logs.length > 0 && (
            <button
              onClick={() => {
                setLogsAutoScroll(true);
                if (containerRef.current) {
                  containerRef.current.scrollTop = containerRef.current.scrollHeight;
                }
              }}
              className="sticky bottom-0 left-0 right-0 mx-auto block rounded-[2px] bg-[rgba(0,229,255,0.08)] px-2 py-0.5 text-[9px] text-(--color-brand)"
            >
              Jump to latest
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  const color = LEVEL_COLORS[entry.level] || "var(--color-text-muted)";
  const bg = LEVEL_BG[entry.level] || "transparent";
  const phaseLabel = entry.phase_number ? PHASE_LABELS[entry.phase_number] : "   ";

  return (
    <div
      className="flex items-start gap-1.5 py-px leading-relaxed"
      style={{ backgroundColor: bg }}
    >
      <span className="shrink-0 text-(--color-text-dim)" style={{ width: 14 }}>
        {phaseLabel}
      </span>
      <span className="shrink-0 text-(--color-text-dim)" style={{ width: 68 }}>
        {formatTime(entry.timestamp)}
      </span>
      <span
        className="shrink-0 uppercase"
        style={{ color, width: 32, fontSize: 9 }}
      >
        {entry.level}
      </span>
      <span className="truncate" style={{ color }}>
        {entry.message}
      </span>
    </div>
  );
}

