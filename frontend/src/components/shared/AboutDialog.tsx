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
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="About KodaQuant"
    >
      <div
        className="rounded-sm border p-8 w-[420px] shadow-2xl animate-fade-in"
        style={{
          backgroundColor: "var(--color-elevated)",
          borderColor: "var(--color-glass-border)",
          boxShadow: "0 0 40px rgba(0,229,255,0.08)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="6" fill="var(--color-app, #050608)"/>
              <polygon points="16,4 28,16 16,28 4,16" fill="var(--color-brand)"/>
              <polygon points="16,8 24,16 16,24 8,16" fill="var(--color-app, #050608)"/>
            </svg>
            <div className="flex flex-col">
              <span className="text-base font-semibold" style={{ color: "var(--color-text-primary)" }}>
                KodaQuant
              </span>
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                v{electronVersion || VERSION}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-full p-1.5 transition hover:brightness-125"
            style={{ backgroundColor: "var(--color-glass)" }}
          >
            <X size={14} style={{ color: "var(--color-text-muted)" }} />
          </button>
        </div>

        <div className="flex flex-col gap-3 mb-6">
          <p className="text-xs leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
            Professional walk-forward backtesting platform for forex traders. ML models, news sentiment, live monitoring, and institutional-grade execution simulation.
          </p>
          <p className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
            Copyright &copy; 2026 rafa9-labs. All rights reserved.
          </p>
        </div>

        <div className="flex flex-col gap-2 pt-4" style={{ borderTop: "1px solid var(--color-glass-border)" }}>
          <a
             href="https://github.com/rafa9-labs/kodaquant"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs transition hover:opacity-80"
            style={{ color: "var(--color-brand)" }}
          >
            <ExternalLink size={12} />
            GitHub Repository
          </a>
        </div>
      </div>
    </div>
  );
}