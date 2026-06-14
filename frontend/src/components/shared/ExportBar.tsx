import { FileSpreadsheet, Image, FileJson } from "lucide-react";

interface ExportBarProps {
  onExportCsv?: () => void;
  onExportPng?: () => void;
  onExportJson?: () => void;
  disabled?: boolean;
}

export function ExportBar({ onExportCsv, onExportPng, onExportJson, disabled }: ExportBarProps) {
  const buttons = [
    { icon: <FileSpreadsheet size={14} />, label: "CSV", onClick: onExportCsv },
    { icon: <Image size={14} />, label: "PNG", onClick: onExportPng },
    { icon: <FileJson size={14} />, label: "JSON", onClick: onExportJson },
  ];

  return (
    <div className="flex gap-2">
      {buttons.map((btn) => {
        const isDisabled = disabled || !btn.onClick;
        return (
          <button
            key={btn.label}
            onClick={btn.onClick}
            disabled={isDisabled}
            className="flex items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-surface) px-3 py-1.5 text-xs font-semibold uppercase transition-all duration-150 hover:border-[var(--color-brand)]"
            style={{
              color: isDisabled ? "var(--color-text-muted)" : "var(--color-text-primary)",
              letterSpacing: "0.05em",
              cursor: isDisabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.5 : 1,
            }}
          >
            {btn.icon} {btn.label}
          </button>
        );
      })}
    </div>
  );
}
