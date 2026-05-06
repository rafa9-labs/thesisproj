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
      height: 300px;
      background: #050608;
      color: #E0E3EB;
      font-family: 'Segoe UI', -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      -webkit-app-region: drag;
    }
    .logo-svg {
      margin-bottom: 16px;
    }
    .title {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.06em;
      margin-bottom: 24px;
      color: #E8ECF1;
    }
    .title span {
      color: #00E5FF;
    }
    .progress-track {
      width: 220px;
      height: 3px;
      background: #1A1F2A;
      border-radius: 2px;
      margin-bottom: 14px;
      overflow: hidden;
    }
    .progress-fill {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #00E5FF, #0099AA);
      border-radius: 2px;
      transition: width 0.4s ease;
    }
    .status {
      font-size: 11px;
      color: #4A5568;
      font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
    }
  </style>
</head>
<body>
  <svg class="logo-svg" width="48" height="48" viewBox="0 0 32 32" fill="none">
    <rect width="32" height="32" rx="6" fill="#050608"/>
    <polygon points="16,4 28,16 16,28 4,16" fill="#00E5FF"/>
    <polygon points="16,8 24,16 16,24 8,16" fill="#050608"/>
  </svg>
  <div class="title"><span>Koda</span>Quant</div>
  <div class="progress-track"><div class="progress-fill" id="progress"></div></div>
  <div class="status" id="status">Starting backend...</div>
</body>
</html>`;

export function createSplashWindow(): BrowserWindow {
  const splash = new BrowserWindow({
    width: 480,
    height: 300,
    frame: false,
    transparent: false,
    resizable: false,
    center: true,
    show: false,
    backgroundColor: "#050608",
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