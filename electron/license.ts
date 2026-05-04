/**
 * License check in Electron main process.
 *
 * On startup, checks the local license state via the backend API.
 * If no license and no active trial, blocks the UI until activation.
 */

import { app, BrowserWindow, dialog, ipcMain } from "electron";
import http from "http";

let licenseChecked = false;

export interface LicenseInfo {
  plan: "free" | "trial" | "pro" | "team";
  licensed: boolean;
  trial_active: boolean;
  trial_days_left: number;
  needs_activation: boolean;
  license_key: string;
  machine_id: string;
}

export async function checkLicense(
  port: number
): Promise<LicenseInfo> {
  try {
    const info = await fetchLicenseInfo(port);
    licenseChecked = true;
    return info;
  } catch {
    return {
      plan: "free",
      licensed: false,
      trial_active: false,
      trial_days_left: 0,
      needs_activation: true,
      license_key: "",
      machine_id: "",
    };
  }
}

function fetchLicenseInfo(port: number): Promise<LicenseInfo> {
  return new Promise((resolve, reject) => {
    const req = http.get(
      `http://127.0.0.1:${port}/api/v1/license/status`,
      { timeout: 5000 },
      (res) => {
        let body = "";
        res.on("data", (chunk: Buffer) => (body += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(body) as LicenseInfo);
          } catch {
            reject(new Error("Invalid license response"));
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("License check timeout"));
    });
  });
}

export async function activateLicense(
  port: number,
  licenseKey: string
): Promise<{ success: boolean; plan?: string; error?: string }> {
  return new Promise((resolve) => {
    const data = JSON.stringify({ license_key: licenseKey });
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path: "/api/v1/license/activate",
        method: "POST",
        headers: { "Content-Type": "application/json", "Content-Length": data.length },
        timeout: 15000,
      },
      (res) => {
        let body = "";
        res.on("data", (chunk: Buffer) => (body += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(body));
          } catch {
            resolve({ success: false, error: "Invalid response" });
          }
        });
      }
    );
    req.on("error", () => resolve({ success: false, error: "Connection failed" }));
    req.write(data);
    req.end();
  });
}

export async function startTrial(
  port: number
): Promise<{ success: boolean; days_left?: number; error?: string }> {
  return new Promise((resolve) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port,
        path: "/api/v1/license/trial",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        timeout: 10000,
      },
      (res) => {
        let body = "";
        res.on("data", (chunk: Buffer) => (body += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(body));
          } catch {
            resolve({ success: false, error: "Invalid response" });
          }
        });
      }
    );
    req.on("error", () => resolve({ success: false, error: "Connection failed" }));
    req.end();
  });
}

export function registerLicenseIPC(port: number): void {
  ipcMain.handle("license:activate", async (_event, key: string) => {
    return activateLicense(port, key);
  });

  ipcMain.handle("license:start-trial", async () => {
    return startTrial(port);
  });

  ipcMain.handle("license:get-status", async () => {
    try {
      return await fetchLicenseInfo(port);
    } catch {
      return null;
    }
  });
}