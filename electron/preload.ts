/**
 * Preload script — secure bridge between Electron and renderer.
 *
 * Exposes a minimal API via contextBridge so the React app
 * can communicate with the main process without full Node access.
 */

import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  onBackendStatus: (callback: (data: BackendStatus) => void) => {
    ipcRenderer.on("backend-status", (_event, data) => callback(data));
  },
  getAppVersion: () => ipcRenderer.invoke("get-app-version"),
  getPlatform: () => process.platform,
  openExternal: (url: string) => ipcRenderer.send("open-external", url),
  licenseActivate: (key: string) => ipcRenderer.invoke("license:activate", key),
  licenseStartTrial: () => ipcRenderer.invoke("license:start-trial"),
  licenseGetStatus: () => ipcRenderer.invoke("license:get-status"),
  checkForUpdates: () => ipcRenderer.invoke("check-for-updates"),
  downloadUpdate: () => ipcRenderer.invoke("download-update"),
  installUpdate: () => ipcRenderer.invoke("install-update"),
  isUpdateDownloaded: () => ipcRenderer.invoke("is-update-downloaded"),
  onUpdateAvailable: (callback: (info: UpdateInfo) => void) => {
    ipcRenderer.on("update-available", (_event, info) => callback(info));
  },
  onUpdateProgress: (callback: (progress: DownloadProgress) => void) => {
    ipcRenderer.on("update-progress", (_event, progress) => callback(progress));
  },
  onUpdateDownloaded: (callback: (info: UpdateInfo) => void) => {
    ipcRenderer.on("update-downloaded", (_event, info) => callback(info));
  },
  onUpdateNotAvailable: (callback: () => void) => {
    ipcRenderer.on("update-not-available", () => callback());
  },
  onTriggerUpdateCheck: (callback: () => void) => {
    ipcRenderer.on("trigger-update-check", () => callback());
  },
});

export interface BackendStatus {
  status: "starting" | "ready" | "error";
  port?: number;
  message?: string;
}

export interface UpdateInfo {
  version: string;
  releaseDate?: string;
  releaseNotes?: string;
}

export interface DownloadProgress {
  bytesPerSecond: number;
  percent: number;
  transferred: number;
  total: number;
}
