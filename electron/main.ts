/**
 * Electron main process — KodaQuant desktop shell.
 *
 * Responsibilities:
 * 1. Show splash screen while backend starts
 * 2. Spawn FastAPI Python backend (with auto-restart)
 * 3. Wait for backend health check to pass
 * 4. Open BrowserWindow loading the React frontend
 * 5. Graceful shutdown with SIGTERM → wait → SIGKILL
 * 6. System tray + native menus
 * 7. Dynamic port discovery (9000-9999)
 * 8. Auto-update via electron-updater (production only)
 * 9. License verification on startup
 * 10. Sentry crash reporting (opt-in)
 */

import { app, BrowserWindow, Menu, Tray, shell, ipcMain } from "electron";
import path from "path";
import { PythonManager } from "./python";
import { waitForBackend } from "./health";
import { buildMenu } from "./menu";
import { createTray } from "./tray";
import { createSplashWindow, updateSplashStatus, destroySplash } from "./splash";
import { getProjectRoot, getUserDataDir } from "./utils";
import { checkLicense, registerLicenseIPC, LicenseInfo } from "./license";
import { startAntiDebugChecks, stopAntiDebugChecks, disableContextMenuInProduction } from "./anti_debug";
import { setupAutoUpdater } from "./updater";
import { initSentry } from "./sentry";

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
    title: "KodaQuant",
    backgroundColor: "#050608",
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

  disableContextMenuInProduction(mainWindow.webContents);
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
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(`document.getElementById('progress').style.width='20%';`).catch(() => {});
  }
  backendPort = await pythonManager.start();
  console.log(`[Electron] Backend starting on port ${backendPort}`);

  updateSplashStatus(splashWindow!, "Waiting for backend...");
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(`document.getElementById('progress').style.width='60%';`).catch(() => {});
  }
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
    updateSplashStatus(splashWindow!, "Ready!");
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.webContents.executeJavaScript(`document.getElementById('progress').style.width='100%';`).catch(() => {});
    }
    mainWindow?.webContents.send("backend-status", {
      status: "ready",
      port: backendPort,
    });
  }

  registerLicenseIPC(backendPort);

  const licenseInfo = await checkLicense(backendPort);
  console.log(`[Electron] License: plan=${licenseInfo.plan}, needs_activation=${licenseInfo.needs_activation}`);
}

async function cleanup() {
  stopAntiDebugChecks();
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

  startAntiDebugChecks();

  await loadFrontend();

  ipcMain.handle("get-app-version", () => app.getVersion());

  if (!isDev) {
    initSentry(process.env.SENTRY_DSN);
    setupAutoUpdater(mainWindow!);
  }

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

app.on("will-quit", async () => {
  await cleanup();
});

process.on("SIGTERM", async () => {
  await cleanup();
  app.quit();
});

process.on("SIGINT", async () => {
  await cleanup();
  app.quit();
});