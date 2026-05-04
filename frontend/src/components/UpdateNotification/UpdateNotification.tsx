import { useState, useEffect } from "react";
import { Download, RefreshCw, X } from "lucide-react";

interface ElectronUpdateAPI {
  checkForUpdates?: () => Promise<unknown>;
  downloadUpdate?: () => Promise<unknown>;
  installUpdate?: () => Promise<unknown>;
  isUpdateDownloaded?: () => Promise<boolean>;
  onUpdateAvailable?: (cb: (info: UpdateInfo) => void) => void;
  onUpdateProgress?: (cb: (progress: DownloadProgress) => void) => void;
  onUpdateDownloaded?: (cb: (info: UpdateInfo) => void) => void;
  onUpdateNotAvailable?: (cb: () => void) => void;
  onTriggerUpdateCheck?: (cb: () => void) => void;
}

interface UpdateInfo {
  version: string;
  releaseDate?: string;
}

interface DownloadProgress {
  bytesPerSecond: number;
  percent: number;
  transferred: number;
  total: number;
}

function getElectronAPI(): ElectronUpdateAPI | null {
  return (window as Record<string, unknown>).electronAPI as ElectronUpdateAPI | null;
}

export function UpdateNotification() {
  const [updateAvailable, setUpdateAvailable] = useState<UpdateInfo | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [updateReady, setUpdateReady] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    const api = getElectronAPI();
    if (!api) return;

    api.onUpdateAvailable?.((info: UpdateInfo) => {
      setUpdateAvailable(info);
      setDownloadProgress(null);
      setDownloading(false);
    });

    api.onUpdateProgress?.((progress: DownloadProgress) => {
      setDownloadProgress(progress);
    });

    api.onUpdateDownloaded?.(() => {
      setUpdateReady(true);
      setDownloading(false);
      setDownloadProgress(null);
    });

    api.onUpdateNotAvailable?.(() => {
      setUpdateAvailable(null);
    });
  }, []);

  const handleCheckForUpdates = async () => {
    const api = getElectronAPI();
    if (!api?.checkForUpdates) return;
    setChecking(true);
    try {
      await api.checkForUpdates();
    } catch (_e) {
      void _e;
    }
    setChecking(false);
  };

  useEffect(() => {
    const api = getElectronAPI();
    if (!api) return;

    const handler = () => {
      handleCheckForUpdates();
    };
    api.onTriggerUpdateCheck?.(handler);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDownload = async () => {
    const api = getElectronAPI();
    if (!api?.downloadUpdate) return;
    setDownloading(true);
    try {
      await api.downloadUpdate();
    } catch (_e) {
      void _e;
    }
  };

  const handleInstall = async () => {
    const api = getElectronAPI();
    if (!api?.installUpdate) return;
    await api.installUpdate();
  };

  if (updateReady && !dismissed) {
    return (
      <div
        className="fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-lg border px-4 py-3 shadow-lg"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-accent)",
        }}
      >
        <RefreshCw size={18} style={{ color: "var(--color-accent)" }} />
        <div className="flex flex-col">
          <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
            Update ready — v{updateAvailable?.version}
          </span>
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Restart to install the latest version
          </span>
        </div>
        <button
          onClick={handleInstall}
          className="rounded-md px-3 py-1.5 text-xs font-semibold"
          style={{
            backgroundColor: "var(--color-accent)",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          Restart & Update
        </button>
        <button
          onClick={() => setDismissed(true)}
          style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  if (downloading && downloadProgress) {
    return (
      <div
        className="fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-lg border px-4 py-3 shadow-lg"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
        }}
      >
        <Download size={18} style={{ color: "var(--color-accent)" }} />
        <div className="flex flex-col">
          <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
            Downloading v{updateAvailable?.version}...
          </span>
          <div className="mt-1 h-1.5 w-48 rounded-full" style={{ backgroundColor: "var(--color-elevated)" }}>
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${downloadProgress.percent}%`,
                backgroundColor: "var(--color-accent)",
              }}
            />
          </div>
          <span className="mt-0.5 text-xs" style={{ color: "var(--color-text-muted)" }}>
            {(downloadProgress.bytesPerSecond / 1024 / 1024).toFixed(1)} MB/s
          </span>
        </div>
      </div>
    );
  }

  if (updateAvailable && !downloading && !dismissed) {
    return (
      <div
        className="fixed bottom-4 right-4 z-50 flex items-center gap-3 rounded-lg border px-4 py-3 shadow-lg"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
        }}
      >
        <RefreshCw size={18} style={{ color: "var(--color-accent)" }} />
        <div className="flex flex-col">
          <span className="text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
            Update available — v{updateAvailable.version}
          </span>
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Download and install the latest version
          </span>
        </div>
        <button
          onClick={handleDownload}
          className="rounded-md px-3 py-1.5 text-xs font-semibold"
          style={{
            backgroundColor: "var(--color-accent)",
            color: "#fff",
            cursor: "pointer",
          }}
        >
          Download
        </button>
        <button
          onClick={() => setDismissed(true)}
          style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  if (checking) {
    return (
      <div
        className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-lg border px-3 py-2"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
        }}
      >
        <RefreshCw size={14} className="animate-spin" style={{ color: "var(--color-text-muted)" }} />
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          Checking for updates...
        </span>
      </div>
    );
  }

  return null;
}