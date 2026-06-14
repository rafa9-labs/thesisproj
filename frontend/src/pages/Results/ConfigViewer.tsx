import { Download, Upload } from "lucide-react";

interface ConfigViewerProps {
  config: Record<string, unknown> | null;
}

export function ConfigViewer({ config }: ConfigViewerProps) {
  if (!config || Object.keys(config).length === 0) {
    return null;
  }

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "kodaquant-config.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoad = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const loaded = JSON.parse(evt.target?.result as string);
          const configOverrides = loaded?.config ?? loaded;
          const backtestUrl = `/backtest?config=${encodeURIComponent(JSON.stringify(configOverrides))}`;
          window.open(backtestUrl, "_blank");
        } catch {
          alert("Invalid JSON config file.");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-[10px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase">
        Configuration
      </h3>
      <div className="flex items-center gap-2">
        <button
          onClick={handleExport}
          className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-white/[0.04] px-3 py-2 text-[10px] font-medium tracking-wider text-(--color-text-secondary) uppercase transition-all hover:brightness-110"
        >
          <Download size={11} />
          Export Config (.json)
        </button>
        <button
          onClick={handleLoad}
          className="flex cursor-pointer items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-white/[0.04] px-3 py-2 text-[10px] font-medium tracking-wider text-(--color-text-secondary) uppercase transition-all hover:brightness-110"
        >
          <Upload size={11} />
          Load Config Schema
        </button>
      </div>
    </div>
  );
}
