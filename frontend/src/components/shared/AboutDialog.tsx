import { useState, useEffect } from "react";
import { X, ExternalLink } from "lucide-react";

interface AboutDialogProps {
  open: boolean;
  onClose: () => void;
}

const VERSION = "1.0.0-beta";

export function AboutDialog({ open, onClose }: AboutDialogProps) {
  const [electronVersion, setElectronVersion] = useState<string>("");

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (open && (window as any).__ELECTRON__?.getAppVersion) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__ELECTRON__.getAppVersion().then((v: string) => setElectronVersion(v));
    }
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6] backdrop-blur-[4px]"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="About KodaQuant"
    >
      <div
        className="w-[420px] animate-fade-in rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-8 shadow-2xl"
        style={{ boxShadow: "0 0 40px rgba(0,229,255,0.08)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="6" fill="var(--color-app, #050608)" />
              <polygon points="16,4 28,16 16,28 4,16" fill="var(--color-brand)" />
              <polygon points="16,8 24,16 16,24 8,16" fill="var(--color-app, #050608)" />
            </svg>
            <div className="flex flex-col">
              <span className="text-base font-semibold text-(--color-text-primary)">KodaQuant</span>
              <span className="font-mono text-[10px] text-(--color-text-muted)">
                v{electronVersion || VERSION}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-full bg-(--color-glass) p-1.5 transition hover:brightness-125"
          >
            <X size={14} className="text-(--color-text-muted)" />
          </button>
        </div>

        <div className="mb-6 flex flex-col gap-3">
          <p className="text-xs leading-relaxed text-(--color-text-secondary)">
            Professional walk-forward backtesting platform for forex traders. ML models, news
            sentiment, live monitoring, and institutional-grade execution simulation.
          </p>
          <p className="text-[10px] text-(--color-text-muted)">
            Copyright &copy; 2026 rafa9-labs. All rights reserved.
          </p>
        </div>

        <div className="flex flex-col gap-2 border-t border-(--color-glass-border) pt-4">
          <a
            href="https://github.com/rafa9-labs/kodaquant"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs text-(--color-brand) transition hover:opacity-80"
          >
            <ExternalLink size={12} />
            GitHub Repository
          </a>
        </div>
      </div>
    </div>
  );
}
