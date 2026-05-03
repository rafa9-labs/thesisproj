/**
 * Electron main process — FX ML Backtester desktop shell.
 *
 * Responsibilities:
 * 1. Show splash screen while backend starts
 * 2. Spawn FastAPI Python backend (with auto-restart)
 * 3. Wait for backend health check to pass
 * 4. Open BrowserWindow loading the React frontend
 * 5. Graceful shutdown with SIGTERM → wait → SIGKILL
 * 6. System tray + native menus
 * 7. Dynamic port discovery (9000-9999)
 */

import { app, BrowserWindow, Menu, Tray, shell } from "electron";
import path from "path";
import { PythonManager } from "./python";
import { waitForBackend } from "./health";
import { buildMenu } from "./menu";
import { createTray } from "./tray";
import { createSplashWindow, updateSplashStatus, destroySplash } from "./splash";
import { getProjectRoot, getUserDataDir } from "./utils";

const isDev = !app.isPackaged;
const isWin = process.platform === "win32";

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let pythonManager: PythonManager | null = null;
let splashWindow: BrowserWindow | null = null;
let backendPort = 8001;
let PROJECT_ROOT = "";

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 800,
    title: "FX ML Backtester",
    backgroundColor: "#131722",
    show: false,
    webPreferences: {
      preload: path.resolve(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
    if (isDev) {
      mainWindow?.webContents.openDevTools({ mode: "detach" });
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

async function loadFrontend() {
  if (isDev) {
    const devUrl = process.env.VITE_DEV_SERVER_URL ?? "http://localhost:5173";
    await mainWindow?.loadURL(devUrl);
  } else {
    // In production, the backend serves the frontend on the same port.
    // This ensures API calls (/api/v1/*) resolve to the same origin.
    const backendUrl = `http://127.0.0.1:${backendPort}`;
    console.log(`[Electron] Loading frontend from ${backendUrl}`);
    await mainWindow?.loadURL(backendUrl);
  }
}

async function startBackend() {
  pythonManager = new PythonManager(PROJECT_ROOT, isDev, (status, message) => {
    console.log(`[Electron] Backend status: ${status} - ${message}`);
    if (splashWindow && !splashWindow.isDestroyed()) {
      updateSplashStatus(splashWindow, message);
    }
    mainWindow?.webContents.send("backend-status", { status, message });
  });

  updateSplashStatus(splashWindow!, "Finding available port...");
  backendPort = await pythonManager.start();
  console.log(`[Electron] Backend starting on port ${backendPort}`);

  updateSplashStatus(splashWindow!, "Waiting for backend...");
  const ok = await waitForBackend(backendPort, 30_000);
  if (!ok) {
    console.error("[Electron] Backend failed to start within 30s");
    updateSplashStatus(splashWindow!, "Backend failed to start. Check logs.");
    mainWindow?.webContents.send("backend-status", {
      status: "error",
      message: "Backend failed to start",
    });
  } else {
    console.log("[Electron] Backend ready on port", backendPort);
    mainWindow?.webContents.send("backend-status", {
      status: "ready",
      port: backendPort,
    });
  }
}

async function cleanup() {
  if (pythonManager) {
    await pythonManager.stop();
    pythonManager = null;
  }
  if (tray) {
    tray.destroy();
    tray = null;
  }
}

app.whenReady().then(async () => {
  PROJECT_ROOT = getProjectRoot(isDev);
  console.log(`[Electron] Project root: ${PROJECT_ROOT}`);

  Menu.setApplicationMenu(buildMenu(isDev));

  splashWindow = createSplashWindow();

  createWindow();
  tray = createTray(mainWindow!);

  await startBackend();

  destroySplash(splashWindow!);
  splashWindow = null;

  await loadFrontend();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", async () => {
  await cleanup();
  if (!isWin) {
    app.quit();
  }
});

app.on("before-quit", async () => {
  await cleanup();
});

app.on("will-quit", () => {
  cleanup();
});