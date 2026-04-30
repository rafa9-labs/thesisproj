/**
 * Splash screen — shown while the Python backend starts up.
 *
 * Displays a simple loading window with the app name and a CSS spinner.
 * Automatically closed when the backend is ready.
 */

import { BrowserWindow } from "electron";
import path from "path";

const SPLASH_HTML = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 480px;
      height: 280px;
      background: #131722;
      color: #E0E3EB;
      font-family: 'Segoe UI', -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      -webkit-app-region: drag;
    }
    .logo {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 24px;
      color: #E0E3EB;
    }
    .logo span {
      color: #089981;
    }
    .spinner {
      width: 32px;
      height: 32px;
      border: 3px solid #2A2E39;
      border-top: 3px solid #089981;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-bottom: 16px;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .status {
      font-size: 12px;
      color: #787B86;
      font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
    }
  </style>
</head>
<body>
  <div class="logo"><span>FX</span> ML Backtester</div>
  <div class="spinner"></div>
  <div class="status" id="status">Starting backend...</div>
</body>
</html>`;

export function createSplashWindow(): BrowserWindow {
  const splash = new BrowserWindow({
    width: 480,
    height: 280,
    frame: false,
    transparent: false,
    resizable: false,
    center: true,
    show: false,
    backgroundColor: "#131722",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splash.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(SPLASH_HTML)}`
  );

  splash.once("ready-to-show", () => {
    splash.show();
  });

  return splash;
}

export function updateSplashStatus(splash: BrowserWindow, message: string): void {
  if (splash.isDestroyed()) return;
  splash.webContents.executeJavaScript(
    `document.getElementById('status').textContent = ${JSON.stringify(message)};`
  ).catch(() => {});
}

export function destroySplash(splash: BrowserWindow): void {
  if (!splash.isDestroyed()) {
    splash.close();
  }
}