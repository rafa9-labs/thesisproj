/**
 * Auto-update module using electron-updater.
 *
 * On startup (production only), checks GitHub Releases for a new version.
 * If found, downloads the delta in the background and notifies the renderer.
 * The user can then restart to apply the update.
 *
 * Note: This only updates the Electron + React shell.
 * The Python backend (extraResources) requires a full reinstall.
 */

import { autoUpdater } from "electron-updater";
import { BrowserWindow, ipcMain } from "electron";

autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = true;

let _mainWindow: BrowserWindow | null = null;
let updateDownloaded = false;

export function setupAutoUpdater(mainWindow: BrowserWindow): void {
  _mainWindow = mainWindow;

  autoUpdater.on("update-available", (info) => {
    console.log(`[Updater] Update available: v${info.version}`);
    _mainWindow?.webContents.send("update-available", {
      version: info.version,
      releaseDate: info.releaseDate,
      releaseNotes: info.releaseNotes,
    });
    autoUpdater.downloadUpdate().catch((err) => {
      console.error("[Updater] Download failed:", err);
    });
  });

  autoUpdater.on("download-progress", (progress) => {
    _mainWindow?.webContents.send("update-progress", {
      bytesPerSecond: progress.bytesPerSecond,
      percent: progress.percent,
      transferred: progress.transferred,
      total: progress.total,
    });
  });

  autoUpdater.on("update-downloaded", (info) => {
    console.log(`[Updater] Update downloaded: v${info.version}`);
    updateDownloaded = true;
    _mainWindow?.webContents.send("update-downloaded", {
      version: info.version,
      releaseDate: info.releaseDate,
    });
  });

  autoUpdater.on("error", (err) => {
    console.error("[Updater] Error:", err?.message || err);
  });

  autoUpdater.on("update-not-available", () => {
    console.log("[Updater] App is up to date");
    _mainWindow?.webContents.send("update-not-available");
  });

  ipcMain.handle("check-for-updates", async () => {
    try {
      const result = await autoUpdater.checkForUpdates();
      return { available: !!result, version: result?.updateInfo?.version };
    } catch (err: any) {
      return { available: false, error: err?.message || "Check failed" };
    }
  });

  ipcMain.handle("download-update", async () => {
    try {
      await autoUpdater.downloadUpdate();
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err?.message || "Download failed" };
    }
  });

  ipcMain.handle("install-update", () => {
    if (updateDownloaded) {
      autoUpdater.quitAndInstall();
    }
  });

  ipcMain.handle("is-update-downloaded", () => updateDownloaded);

  autoUpdater.checkForUpdates().catch(() => {});
}

export function checkForUpdates(): void {
  autoUpdater.checkForUpdates().catch(() => {});
}