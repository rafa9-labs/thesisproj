/**
 * Electron main process — FX ML Backtester desktop shell.
 *
 * Responsibilities:
 * 1. Spawn FastAPI Python backend as child process
 * 2. Wait for backend health check to pass
 * 3. Open BrowserWindow loading the React frontend
 * 4. Graceful shutdown (SIGTERM → wait → SIGKILL)
 * 5. System tray + native menus
 */

import { app, BrowserWindow, Menu, Tray, shell } from "electron";
import path from "path";
import { spawnPythonBackend } from "./python";
import { waitForBackend } from "./health";
import { buildMenu } from "./menu";
import { createTray } from "./tray";

const isDev = !app.isPackaged;
const isWin = process.platform === "win32";

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let pythonProc: ReturnType<typeof spawnPythonBackend> | null = null;
let backendPort = 8000;

const PROJECT_ROOT = isDev
  ? path.resolve(__dirname, "..")
  : path.resolve(process.resourcesPath, "app");

const FRONTEND_DIST = isDev
  ? path.resolve(PROJECT_ROOT, "frontend", "dist")
  : path.resolve(PROJECT_ROOT, "frontend", "dist");

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 800,
    title: "FX ML Backtester",
    backgroundColor: "#0a0e17",
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
    const indexPath = path.join(FRONTEND_DIST, "index.html");
    await mainWindow?.loadFile(indexPath);
  }
}

async function startBackend() {
  pythonProc = spawnPythonBackend(PROJECT_ROOT, backendPort, isDev);
  const ok = await waitForBackend(backendPort, 30_000);
  if (!ok) {
    console.error("[Electron] Backend failed to start within 30s");
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

function cleanup() {
  if (pythonProc) {
    console.log("[Electron] Shutting down Python backend...");
    pythonProc.kill();
    pythonProc = null;
  }
  if (tray) {
    tray.destroy();
    tray = null;
  }
}

app.whenReady().then(async () => {
  Menu.setApplicationMenu(buildMenu(isDev));

  createWindow();
  tray = createTray(mainWindow!);

  await startBackend();
  await loadFrontend();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  cleanup();
  if (!isWin) {
    app.quit();
  }
});

app.on("before-quit", () => {
  cleanup();
});

app.on("will-quit", () => {
  cleanup();
});
