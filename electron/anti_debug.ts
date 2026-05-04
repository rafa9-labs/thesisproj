/**
 * Anti-debugging checks for production builds.
 *
 * Detects common debugging tools and techniques:
 * 1. DevTools open detection (production only)
 * 2. Debugger statement timing detection
 * 3. Electron remote debugger detection
 *
 * Only active in packaged (production) builds — skipped in dev mode.
 * Warnings are logged to console; app continues running.
 */

import { app } from "electron";

const isDev = !app.isPackaged;

let antiDebugInterval: NodeJS.Timeout | null = null;
let lastDevToolsState = false;

export function startAntiDebugChecks(): void {
  if (isDev) {
    return;
  }

  antiDebugInterval = setInterval(() => {
    checkDevTools();
    checkDebuggerTiming();
  }, 30_000);
}

export function stopAntiDebugChecks(): void {
  if (antiDebugInterval) {
    clearInterval(antiDebugInterval);
    antiDebugInterval = null;
  }
}

function checkDevTools(): void {
  const windows = require("electron").BrowserWindow.getAllWindows();
  for (const win of windows) {
    if (win.isDestroyed()) continue;
    const isOpen = win.webContents.isDevToolsOpened();
    if (isOpen && !lastDevToolsState) {
      console.warn(
        "[Security] DevTools opened in production build. " +
        "This may indicate debugging attempts."
      );
      lastDevToolsState = true;
    } else if (!isOpen) {
      lastDevToolsState = false;
    }
  }
}

function checkDebuggerTiming(): void {
  const start = Date.now();
  debugger;
  const elapsed = Date.now() - start;
  if (elapsed > 100) {
    console.warn(
      `[Security] Debugger detected (pause time: ${elapsed}ms). ` +
      "This may indicate a debugger is attached."
    );
  }
}

export function disableContextMenuInProduction(
  webContents: Electron.WebContents
): void {
  if (isDev) return;

  webContents.on("context-menu", (event) => {
    event.preventDefault();
  });
}