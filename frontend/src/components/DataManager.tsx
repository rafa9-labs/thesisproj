import { useRef } from "react";
import { Upload } from "lucide-react";
import { usePairs, useUploadCsv } from "@/api/queries";

export function DataManager() {
  const { data: pairs, isLoading } = usePairs();
  const uploadCsv = useUploadCsv();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadCsv.mutate(file);
    e.target.value = "";
  };

  return (
    <div className="flex flex-col gap-3 mt-3 border-t border-(--color-glass-border) pt-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-(--color-text-primary)">
          Uploaded Data
        </span>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadCsv.isPending}
          className="flex items-center gap-1.5 cursor-pointer rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-2.5 py-1 text-[10px] font-medium tracking-wider text-(--color-text-secondary) uppercase transition-all duration-200 hover:border-[var(--color-border-active)] disabled:opacity-50"
        >
          <Upload size={10} />
          {uploadCsv.isPending ? "Uploading…" : "Upload CSV"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleUpload}
          className="hidden"
        />
      </div>

      {uploadCsv.isError && (
        <p className="text-[10px] text-(--color-accent-danger)">
          Upload failed: {(uploadCsv.error as Error)?.message || "Unknown error"}
        </p>
      )}

      {isLoading ? (
        <p className="text-[10px] text-(--color-text-muted)">Loading pairs…</p>
      ) : !pairs || pairs.length === 0 ? (
        <p className="text-[10px] text-(--color-text-muted)">
          No data uploaded. Use Upload CSV or Demo Mode to add data.
        </p>
      ) : (
        <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto">
          {pairs.map((p) => (
            <div
              key={p.pair.symbol}
              className="flex items-center justify-between rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-2"
            >
              <div className="flex flex-col">
                <span className="font-mono text-[11px] font-semibold text-(--color-text-primary)">
                  {p.pair.symbol}
                </span>
                <span className="text-[9px] text-(--color-text-muted)">
                  {p.timeframes.length} timeframe{p.timeframes.length !== 1 ? "s" : ""}{" "}
                  ({p.timeframes.map((t) => t.timeframe).join(", ")})
                </span>
              </div>
              <div className="flex items-center gap-2">
                {p.timeframes.map((t) => (
                  <span
                    key={t.timeframe}
                    className="rounded bg-(--color-glass) px-1.5 py-0.5 font-mono text-[9px] text-(--color-text-secondary)"
                  >
                    {t.rows.toLocaleString()} bars
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
