import { useRef, useState, useEffect, useCallback } from "react";
import { Bug, X, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { useJobStore } from "@/stores/useJobStore";
import { wsManager } from "@/api/websocket";

interface LogEntry {
  id: number;
  ts: string;
  source: "ws" | "poll";
  raw: string;
}

const MAX_LOG = 100;

export function DebugOverlay({ jobId, pollCursor }: { jobId: string | null; pollCursor: number }) {
  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"log" | "counters">("log");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [counters, setCounters] = useState<Map<string, number>>(new Map());
  const [wsConnected, setWsConnected] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(0);

  const addLog = useCallback((source: "ws" | "poll", raw: unknown) => {
    const str = typeof raw === "string" ? raw : JSON.stringify(raw);
    let eventName = "unknown";
    try {
      const p = JSON.parse(str);
      eventName = p.event ?? "unknown";
    } catch { /* ignore */ }
    setLogs((prev) => {
      const entry: LogEntry = { id: nextId.current++, ts: new Date().toISOString().slice(11, 23), source, raw: str.slice(0, 300) };
      const next = [...prev, entry];
      return next.length > MAX_LOG ? next.slice(next.length - MAX_LOG) : next;
    });
    setCounters((prev) => {
      const next = new Map(prev);
      next.set(eventName, (next.get(eventName) ?? 0) + 1);
      return next;
    });
  }, []);

  const job = jobId ? useJobStore.getState().getJob(jobId) : undefined;

  useEffect(() => {
    if (!open) return;
    const check = setInterval(() => setWsConnected(wsManager.connected), 500);
    return () => clearInterval(check);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length, open]);

  useEffect(() => {
    if (!open || !jobId) return;
    const unsub = wsManager.subscribe((event) => addLog("ws", event));
    return unsub;
  }, [open, jobId, addLog]);

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-3 right-3 z-50 flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[10px] font-medium transition-all hover:brightness-110"
        style={{
          borderColor: "var(--color-border)",
          backgroundColor: "var(--color-surface)",
          color: "var(--color-text-muted)",
          opacity: 0.7,
        }}
        title="Open debug panel"
      >
        <Bug size={12} />
        Debug
      </button>
    );
  }

  const totalEvents = [...counters.entries()].reduce((s, [, c]) => s + c, 0);

  return (
    <div
      className="fixed bottom-3 right-3 z-50 flex flex-col rounded-lg border shadow-lg"
      style={{
        width: 480,
        maxHeight: "calc(100vh - 120px)",
        backgroundColor: "var(--color-surface)",
        borderColor: "var(--color-border)",
        fontFamily: "var(--font-mono)",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 rounded-t-lg border-b px-3 py-2"
        style={{ borderColor: "var(--color-border)", backgroundColor: "var(--color-elevated)" }}
      >
        <Bug size={12} style={{ color: "var(--color-accent-warning)" }} />
        <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-secondary)" }}>
          Debug
        </span>
        <div className="flex items-center gap-1.5 ml-2">
          {wsConnected ? (
            <Wifi size={10} style={{ color: "var(--color-accent-success)" }} />
          ) : (
            <WifiOff size={10} style={{ color: "var(--color-accent-danger)" }} />
          )}
          <span className="text-[9px]" style={{ color: wsConnected ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}>
            {wsConnected ? "WS" : "No WS"}
          </span>
        </div>
        {job && (
          <span className="text-[9px] ml-1" style={{ color: "var(--color-text-muted)" }}>
            | {job.status} | {job.progress}% | poll:{pollCursor}
          </span>
        )}
        <div className="flex-1" />
        <button
          onClick={() => setActiveTab(activeTab === "log" ? "counters" : "log")}
          className="rounded px-1.5 py-0.5 text-[9px] uppercase tracking-[0.06em] transition-colors"
          style={{
            color: "var(--color-text-muted)",
            backgroundColor: "var(--color-glass-hover)",
          }}
        >
          {activeTab === "log" ? "Counters" : "Log"}
        </button>
        <button
          onClick={() => setOpen(false)}
          className="rounded p-0.5 transition-colors"
          style={{ color: "var(--color-text-muted)" }}
        >
          <X size={12} />
        </button>
      </div>

      {/* Content */}
      <div className="overflow-y-auto" style={{ maxHeight: 400 }}>
        {activeTab === "counters" ? (
          <div className="flex flex-col gap-0.5 p-2">
            <div className="flex items-center gap-2 px-2 py-1 text-[9px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              <span className="flex-1">Event Type</span>
              <span>Count</span>
            </div>
            {[...counters.entries()]
              .sort((a, b) => b[1] - a[1])
              .map(([name, count]) => (
                <div key={name} className="flex items-center gap-2 rounded px-2 py-0.5" style={{ backgroundColor: "var(--color-glass-hover)" }}>
                  <span className="flex-1 text-[10px]" style={{ color: "var(--color-text-primary)" }}>
                    {name}
                  </span>
                  <span className="text-[10px] font-semibold" style={{ color: "var(--color-brand)" }}>
                    {count}
                  </span>
                </div>
              ))}
            {totalEvents > 0 && (
              <div className="flex items-center gap-2 rounded px-2 py-0.5 mt-1" style={{ backgroundColor: "var(--color-glass-hover)" }}>
                <span className="flex-1 text-[10px] font-semibold" style={{ color: "var(--color-text-secondary)" }}>
                  TOTAL
                </span>
                <span className="text-[10px] font-semibold" style={{ color: "var(--color-brand)" }}>
                  {totalEvents}
                </span>
              </div>
            )}
            {totalEvents === 0 && (
              <div className="flex items-center justify-center py-6">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                  No events received yet
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-0.5 p-2">
            {logs.length === 0 && (
              <div className="flex items-center justify-center py-6">
                <RefreshCw size={14} className="animate-spin" style={{ color: "var(--color-text-muted)" }} />
                <span className="ml-2 text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                  Waiting for events...
                </span>
              </div>
            )}
            {logs.map((entry) => (
              <div key={entry.id} className="flex gap-1.5 rounded px-1.5 py-0.5 text-[9px] leading-tight" style={{ backgroundColor: "var(--color-glass-hover)" }}>
                <span style={{ color: "var(--color-text-muted)", minWidth: 80 }}>{entry.ts}</span>
                <span
                  style={{
                    color: entry.source === "ws" ? "var(--color-accent-success)" : "var(--color-accent)",
                    minWidth: 28,
                  }}
                >
                  {entry.source === "ws" ? "WS" : "POLL"}
                </span>
                <span style={{ color: "var(--color-text-primary)", wordBreak: "break-all" }}>{entry.raw}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}
