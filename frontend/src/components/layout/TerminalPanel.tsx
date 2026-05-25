import { useState, useRef, useEffect } from "react";
import { Terminal, ChevronUp, ChevronDown, Trash2 } from "lucide-react";

interface LogEntry {
  ts: Date;
  level: "info" | "warn" | "error" | "debug";
  message: string;
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

export function TerminalPanel() {
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
      className="flex flex-col border-t transition-all duration-200"
      style={{
        borderColor: "var(--color-border)",
        backgroundColor: "var(--color-app)",
        height: open ? 200 : 32,
        overflow: "hidden",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between border-b px-3"
        style={{
          height: 32,
          borderColor: "var(--color-border)",
          backgroundColor: "var(--color-surface)",
          cursor: "pointer",
          flexShrink: 0,
        }}
      >
        <div className="flex items-center gap-2">
          <Terminal size={12} style={{ color: "var(--color-text-muted)" }} />
          <span
            className="text-[10px] font-semibold uppercase tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Terminal
          </span>
          {logs.length > 0 && (
            <span
              className="rounded-full px-1.5 text-[10px]"
              style={{
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {logs.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {open && (
            <span
              onClick={(e) => { e.stopPropagation(); clear(); }}
              className="rounded p-0.5"
              style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
              title="Clear logs"
            >
              <Trash2 size={11} />
            </span>
          )}
          {open ? (
            <ChevronDown size={12} style={{ color: "var(--color-text-muted)" }} />
          ) : (
            <ChevronUp size={12} style={{ color: "var(--color-text-muted)" }} />
          )}
        </div>
      </button>
      {open && (
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-3 py-1"
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
