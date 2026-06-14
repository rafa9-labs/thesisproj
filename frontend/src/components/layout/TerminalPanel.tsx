/* eslint-disable react-refresh/only-export-components */
import { useState, useRef, useEffect } from "react";
import { ChevronUp, ChevronDown, Trash2 } from "lucide-react";

interface LogEntry {
  ts: Date;
  level: "info" | "warn" | "error" | "debug";
  message: string;
}

export interface TerminalPanelStatusProps {
  apiOk: boolean;
  wsConnected: boolean;
}

const LEVEL_COLORS: Record<string, string> = {
  info: "var(--color-text-secondary)",
  warn: "var(--color-accent-warning)",
  error: "var(--color-accent-danger)",
  debug: "var(--color-text-muted)",
};

const MAX_LOGS = 500;

let logBuffer: LogEntry[] = [];
const listeners: Set<() => void> = new Set();

export function pushTerminalLog(level: LogEntry["level"], message: string) {
  logBuffer.push({ ts: new Date(), level, message });
  if (logBuffer.length > MAX_LOGS) logBuffer = logBuffer.slice(-MAX_LOGS);
  listeners.forEach((fn) => fn());
}

function StatusPip({ active, label }: { active: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <div className="relative">
        <div
          className="h-[5px] w-[5px] rounded-full"
          style={{
            backgroundColor: active ? "var(--color-accent-success)" : "var(--color-text-muted)",
          }}
        />
        {active && (
          <div className="absolute inset-0 animate-ping-brand rounded-full bg-(--color-accent-success) opacity-35" />
        )}
      </div>
      <span className="font-mono text-[9px] text-(--color-text-muted)">{label}</span>
    </div>
  );
}

export function TerminalPanel({
  apiOk = false,
  wsConnected = false,
}: Partial<TerminalPanelStatusProps>) {
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>(logBuffer);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const update = () => setLogs([...logBuffer]);
    listeners.add(update);
    return () => {
      listeners.delete(update);
    };
  }, []);

  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, open]);

  const clear = () => {
    logBuffer = [];
    setLogs([]);
  };

  return (
    <div
      className="flex shrink-0 flex-col overflow-hidden border-t border-(--color-border-subtle) bg-(--color-app) transition-[height] duration-200"
      style={{
        height: open ? 228 : 28,
      }}
    >
      {/* ── Status bar / toggle row — always 28px ───────────────── */}
      <div
        className="flex h-7 shrink-0 cursor-pointer items-center justify-between px-6 select-none"
        onClick={() => setOpen(!open)}
      >
        {/* Left: terminal toggle + status pips */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] tracking-[0.03em] text-(--color-text-muted)">
              {">_"}
            </span>
            <span
              className="font-mono text-[10px]"
              style={{
                color: open ? "var(--color-text-secondary)" : "var(--color-text-muted)",
              }}
            >
              Terminal
            </span>
            {logs.length > 0 && (
              <span className="font-mono text-[9px] text-(--color-text-muted)">
                [{logs.length}]
              </span>
            )}
          </div>
          <StatusPip active={apiOk} label="API" />
          <StatusPip active={wsConnected} label="WS" />
        </div>

        {/* Right: version + clear + chevron */}
        <div className="flex items-center gap-2">
          <span className="font-mono text-[9px] tracking-[0.04em] text-(--color-text-muted) opacity-60">
            v1.0.0 — KodaQuant
          </span>
          {open && (
            <span
              onClick={(e) => {
                e.stopPropagation();
                clear();
              }}
              className="rounded p-0.5 text-(--color-text-muted) hover:text-[var(--color-text-secondary)]"
              className="cursor-pointer"
              title="Clear logs"
            >
              <Trash2 size={10} />
            </span>
          )}
          {open ? (
            <ChevronDown size={10} className="text-(--color-text-muted)" />
          ) : (
            <ChevronUp size={10} className="text-(--color-text-muted)" />
          )}
        </div>
      </div>

      {/* ── Log output area ──────────────────────────────────────── */}
      {open && (
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-6 py-1 font-mono text-[11px]"
          style={{ lineHeight: 1.5 }}
        >
          {logs.length === 0 ? (
            <span className="text-(--color-text-muted)">
              No logs yet. Logs will appear here during backtest execution.
            </span>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="flex gap-2">
                <span className="whitespace-nowrap text-(--color-text-muted)">
                  {log.ts.toLocaleTimeString()}
                </span>
                <span
                  className="min-w-[36px] font-semibold"
                  style={{ color: LEVEL_COLORS[log.level] }}
                >
                  [{log.level.toUpperCase()}]
                </span>
                <span className="text-(--color-text-primary)">{log.message}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
