import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

interface ConfigViewerProps {
  config: Record<string, unknown> | null;
}

function JsonValue({ value, indent }: { value: unknown; indent: number }) {
  if (value === null) {
    return <span style={{ color: "var(--color-text-muted)" }}>null</span>;
  }
  if (typeof value === "boolean") {
    return <span style={{ color: "var(--color-accent-info)" }}>{String(value)}</span>;
  }
  if (typeof value === "number") {
    return <span style={{ color: "var(--color-accent-warning)" }}>{value}</span>;
  }
  if (typeof value === "string") {
    return <span style={{ color: "var(--color-accent-success)" }}>"{value}"</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span style={{ color: "var(--color-text-muted)" }}>[]</span>;
    }
    return (
      <span className="flex flex-col">
        <span style={{ color: "var(--color-text-muted)" }}>[</span>
        {value.map((item, i) => (
          <span key={i} style={{ paddingLeft: indent + 12 }}>
            <JsonValue value={item} indent={indent + 12} />
            {i < value.length - 1 && <span style={{ color: "var(--color-text-muted)" }}>,</span>}
          </span>
        ))}
        <span style={{ paddingLeft: indent, color: "var(--color-text-muted)" }}>]</span>
      </span>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return <span style={{ color: "var(--color-text-muted)" }}>{"{}"}</span>;
    }
    return (
      <span className="flex flex-col">
        <span style={{ color: "var(--color-text-muted)" }}>{"{"}</span>
        {entries.map(([k, v], i) => (
          <span key={k} style={{ paddingLeft: indent + 12 }}>
            <span style={{ color: "var(--color-accent)" }}>"{k}"</span>
            <span style={{ color: "var(--color-text-muted)" }}>: </span>
            <JsonValue value={v} indent={indent + 12} />
            {i < entries.length - 1 && <span style={{ color: "var(--color-text-muted)" }}>,</span>}
          </span>
        ))}
        <span style={{ paddingLeft: indent, color: "var(--color-text-muted)" }}>{"}"}</span>
      </span>
    );
  }
  return <span>{String(value)}</span>;
}

export function ConfigViewer({ config }: ConfigViewerProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!config || Object.keys(config).length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] transition-colors"
        style={{ color: "var(--color-text-secondary)", cursor: "pointer" }}
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Best Configuration
      </button>
      {isOpen && (
        <div
          className="rounded-lg border p-4 overflow-auto"
          style={{
            backgroundColor: "var(--color-surface)",
            borderColor: "var(--color-border)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            lineHeight: 1.6,
            maxHeight: 400,
          }}
        >
          <JsonValue value={config} indent={0} />
        </div>
      )}
    </div>
  );
}
