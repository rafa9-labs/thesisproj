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
  }, []);

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
      <div className="fixed right-4 bottom-4 z-50 flex items-center gap-3 rounded-sm border border-(--color-accent) bg-(--color-surface) px-4 py-3 shadow-lg">
        <RefreshCw size={18} className="text-(--color-accent)" />
        <div className="flex flex-col">
          <span className="text-sm font-medium text-(--color-text-primary)">
            Update ready — v{updateAvailable?.version}
          </span>
          <span className="text-xs text-(--color-text-muted)">
            Restart to install the latest version
          </span>
        </div>
        <button
          onClick={handleInstall}
          className="rounded-md bg-(--color-accent) px-3 py-1.5 text-xs font-semibold"
          style={{ color: "#fff" }}
          className="cursor-pointer"
        >
          Restart & Update
        </button>
        <button
          onClick={() => setDismissed(true)}
          className="text-(--color-text-muted)"
          className="cursor-pointer"
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  if (downloading && downloadProgress) {
    return (
      <div className="fixed right-4 bottom-4 z-50 flex items-center gap-3 rounded-sm border border-(--color-border) bg-(--color-surface) px-4 py-3 shadow-lg">
        <Download size={18} className="text-(--color-accent)" />
        <div className="flex flex-col">
          <span className="text-sm font-medium text-(--color-text-primary)">
            Downloading v{updateAvailable?.version}...
          </span>
          <div className="mt-1 h-1.5 w-48 rounded-full bg-(--color-elevated)">
            <div
              className="h-full rounded-full bg-(--color-accent) transition-all"
              style={{ width: `${downloadProgress.percent}%` }}
            />
          </div>
          <span className="mt-0.5 text-xs text-(--color-text-muted)">
            {(downloadProgress.bytesPerSecond / 1024 / 1024).toFixed(1)} MB/s
          </span>
        </div>
      </div>
    );
  }

  if (updateAvailable && !downloading && !dismissed) {
    return (
      <div className="fixed right-4 bottom-4 z-50 flex items-center gap-3 rounded-sm border border-(--color-border) bg-(--color-surface) px-4 py-3 shadow-lg">
        <RefreshCw size={18} className="text-(--color-accent)" />
        <div className="flex flex-col">
          <span className="text-sm font-medium text-(--color-text-primary)">
            Update available — v{updateAvailable.version}
          </span>
          <span className="text-xs text-(--color-text-muted)">
            Download and install the latest version
          </span>
        </div>
        <button
          onClick={handleDownload}
          className="rounded-md bg-(--color-accent) px-3 py-1.5 text-xs font-semibold"
          style={{ color: "#fff" }}
          className="cursor-pointer"
        >
          Download
        </button>
        <button
          onClick={() => setDismissed(true)}
          className="text-(--color-text-muted)"
          className="cursor-pointer"
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  if (checking) {
    return (
      <div className="fixed right-4 bottom-4 z-50 flex items-center gap-2 rounded-sm border border-(--color-border) bg-(--color-surface) px-3 py-2">
        <RefreshCw size={14} className="animate-spin text-(--color-text-muted)" />
        <span className="text-xs text-(--color-text-muted)">Checking for updates...</span>
      </div>
    );
  }

  return null;
}
