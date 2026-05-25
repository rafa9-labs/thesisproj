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
          style={{ backgroundColor: active ? "var(--color-accent-success)" : "var(--color-text-muted)" }}
        />
        {active && (
          <div
            className="absolute inset-0 animate-ping-brand rounded-full"
            style={{ backgroundColor: "var(--color-accent-success)", opacity: 0.35 }}
          />
        )}
      </div>
      <span style={{ color: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}>
        {label}
      </span>
    </div>
  );
}

export function TerminalPanel({ apiOk = false, wsConnected = false }: Partial<TerminalPanelStatusProps>) {
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>(logBuffer);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const update = () => setLogs([...logBuffer]);
    listeners.add(update);
    return () => { listeners.delete(update); };
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
      className="flex flex-col"
      style={{
        backgroundColor: "var(--color-app)",
        borderTop: "1px solid rgba(255,255,255,0.04)",
        height: open ? 228 : 28,
        overflow: "hidden",
        transition: "height 200ms ease",
        flexShrink: 0,
      }}
    >
      {/* ── Status bar / toggle row — always 28px ───────────────── */}
      <div
        className="flex items-center justify-between px-6 cursor-pointer select-none"
        style={{ height: 28, flexShrink: 0 }}
        onClick={() => setOpen(!open)}
      >
        {/* Left: terminal toggle + status pips */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span
              style={{
                color: "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.03em",
              }}
            >
              {">_"}
            </span>
            <span
              style={{
                color: open ? "var(--color-text-secondary)" : "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
                fontSize: 10,
              }}
            >
              Terminal
            </span>
            {logs.length > 0 && (
              <span
                style={{
                  color: "var(--color-text-muted)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 9,
                }}
              >
                [{logs.length}]
              </span>
            )}
          </div>
          <StatusPip active={apiOk} label="API" />
          <StatusPip active={wsConnected} label="WS" />
        </div>

        {/* Right: version + clear + chevron */}
        <div className="flex items-center gap-2">
          <span
            style={{
              color: "var(--color-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.04em",
              opacity: 0.6,
            }}
          >
            v1.0.0 — KodaQuant
          </span>
          {open && (
            <span
              onClick={(e) => { e.stopPropagation(); clear(); }}
              className="rounded p-0.5 hover:text-[var(--color-text-secondary)]"
              style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
              title="Clear logs"
            >
              <Trash2 size={10} />
            </span>
          )}
          {open ? (
            <ChevronDown size={10} style={{ color: "var(--color-text-muted)" }} />
          ) : (
            <ChevronUp size={10} style={{ color: "var(--color-text-muted)" }} />
          )}
        </div>
      </div>

      {/* ── Log output area ──────────────────────────────────────── */}
      {open && (
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-6 py-1"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            lineHeight: 1.5,
          }}
        >
          {logs.length === 0 ? (
            <span style={{ color: "var(--color-text-muted)" }}>
              No logs yet. Logs will appear here during backtest execution.
            </span>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="flex gap-2">
                <span style={{ color: "var(--color-text-muted)", whiteSpace: "nowrap" }}>
                  {log.ts.toLocaleTimeString()}
                </span>
                <span style={{ color: LEVEL_COLORS[log.level], fontWeight: 600, minWidth: 36 }}>
                  [{log.level.toUpperCase()}]
                </span>
                <span style={{ color: "var(--color-text-primary)" }}>{log.message}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
