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
});

export interface BackendStatus {
  status: "starting" | "ready" | "error";
  port?: number;
  message?: string;
}
