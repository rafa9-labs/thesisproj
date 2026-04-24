# Sprint 9: Electron Desktop Shell — Complete Implementation Plan

> **Estimate**: 11 hours across 4 phases
> **Stack**: Electron 28 + Node.js + PyInstaller + electron-builder
> **Target**: Native-feeling Windows desktop application wrapping React + FastAPI
> **Product**: FX ML Backtester Pro — sold via Paddle as one-time + subscription

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Use Cases](#2-use-cases)
3. [Python Subsystem](#3-python-subsystem)
4. [Electron Main Process](#4-electron-main-process)
5. [Build & Packaging](#5-build--packaging)
6. [Payment & Licensing Infrastructure](#6-payment--licensing-infrastructure)
7. [Account & Database Architecture](#7-account--database-architecture)
8. [Implementation Phases](#8-implementation-phases)
9. [Professional Safeguards](#9-professional-safeguards)
10. [Testing Strategy](#10-testing-strategy)
11. [File Structure](#11-file-structure)
12. [Dependencies](#12-dependencies)
13. [Completion Criteria](#13-completion-criteria)

---

## 1. Architecture

### 1.1 System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ELECTRON DESKTOP APP                               │
│                                                                       │
│  ┌────────────────────────┐     ┌──────────────────────────────────┐ │
│  │  Main Process (Node)   │     │  Renderer Process (Chromium)     │ │
│  │                        │     │                                   │ │
│  │  electron/main.ts      │     │  React App (Sprint 8)            │ │
│  │  ├─ App lifecycle      │────▶│  BrowserWindow                   │ │
│  │  ├─ Window management  │     │  loads: http://localhost:{port}  │ │
│  │  ├─ IPC handlers       │◀───▶│                                   │ │
│  │  ├─ System tray        │     │  Pages:                          │ │
│  │  ├─ Native menus       │     │  ├─ Dashboard                    │ │
│  │  ├─ Auto-update check  │     │  ├─ Backtest Config              │ │
│  │  ├─ License validation │     │  ├─ Results & Charts             │ │
│  │  └─ Crash reporting    │     │  ├─ Model Comparison             │ │
│  │                        │     │  ├─ News & Sentiment             │ │
│  │                        │     │  └─ Settings                     │ │
│  └────────────────────────┘     └──────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Python Subprocess (FastAPI + Pipeline Engine)                  │  │
│  │                                                                  │  │
│  │  ├─ uvicorn serving REST + WebSocket on localhost:{dynamicPort} │  │
│  │  ├─ Pipeline engine (backtester, models, features)              │  │
│  │  ├─ SQLite data layer (pairs, jobs, results)                    │  │
│  │  ├─ Celery worker (optional, for long background jobs)          │  │
│  │  └─ All Python deps: TF, PyTorch, scikit-learn, XGBoost, etc.  │  │
│  │                                                                  │  │
│  │  Bundled via PyInstaller into build/python/                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Local Data (User's Machine)                                    │  │
│  │                                                                  │  │
│  │  %APPDATA%/fx-ml-backtester/                                    │  │
│  │  ├─ data/forex.db          — SQLite database (pairs, jobs)      │  │
│  │  ├─ data/csv/              — Downloaded CSV files               │  │
│  │  ├─ results/               — Backtest results (Parquet/JSON)    │  │
│  │  ├─ cache/features/        — Feature disk cache                 │  │
│  │  ├─ cache/news/            — News cache (Parquet)               │  │
│  │  ├─ config/settings.json   — User settings (encrypted S10)      │  │
│  │  ├─ license/license.key    — Paddle license (encrypted S10)     │  │
│  │  └─ logs/                  — Application logs                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Process Lifecycle

```
User double-clicks FX Backtester.exe
        │
        ▼
┌─ Electron main.ts ──────────────────────────────────────────────────┐
│  1. Check single-instance lock (prevent multiple windows)           │
│  2. Create %APPDATA%/fx-ml-backtester/ directory structure          │
│  3. Find available port (9000-9999 range)                           │
│  4. Spawn Python subprocess:                                        │
│     build/python/forex_pipeline.exe                                 │
│       --host 127.0.0.1 --port {dynamicPort}                         │
│       --data-dir "%APPDATA%/fx-ml-backtester/data"                  │
│  5. Show splash screen ("Starting FX Backtester...")                │
│  6. Health check: GET http://localhost:{port}/api/v1/health         │
│     Poll every 500ms, max 60 retries (30 seconds)                   │
│  7. Health OK → Create BrowserWindow, load http://localhost:{port}  │
│  8. Close splash screen                                             │
│  9. Set up system tray, menus, IPC                                  │
│  10. Main loop — app runs until user quits                          │
│                                                                      │
│  On quit:                                                            │
│  11. SIGTERM to Python process                                       │
│  12. Wait 5 seconds for graceful shutdown                           │
│  13. If still running: SIGKILL                                       │
│  14. Release single-instance lock                                    │
│  15. Exit                                                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 IPC Communication (Main ↔ Renderer)

```
Renderer (React)                    Main Process (Electron)
      │                                      │
      │  ipcRenderer.invoke('get-python-status')  │
      │─────────────────────────────────────────▶│
      │                                      │  Returns: 'starting' | 'ready' | 'error'
      │  ◀──────────────────────────────────────│
      │                                      │
      │  ipcRenderer.invoke('get-app-version')    │
      │─────────────────────────────────────────▶│
      │                                      │  Returns: '1.0.0'
      │  ◀──────────────────────────────────────│
      │                                      │
      │  ipcRenderer.invoke('open-data-folder')   │
      │─────────────────────────────────────────▶│
      │                                      │  Opens Explorer at %APPDATA%/...
      │  ◀──────────────────────────────────────│
      │                                      │
      │  ipcRenderer.invoke('check-for-updates')  │
      │─────────────────────────────────────────▶│
      │                                      │  Checks GitHub Releases
      │  ◀── { hasUpdate, version, downloadUrl } │
      │                                      │
      │  ipcRenderer.invoke('validate-license', key)│
      │─────────────────────────────────────────▶│
      │                                      │  Paddle API validation
      │  ◀── { valid, expiresAt, features }  │
      │                                      │
      │  ipcRenderer.invoke('get-machine-id')     │
      │─────────────────────────────────────────▶│
      │                                      │  Hardware fingerprint
      │  ◀── 'A4F2-8C1D-E7B3-9K5M'           │
      │                                      │
```

**IPC API Surface** (preload.ts exposes these to renderer):

| Channel | Direction | Parameters | Returns | Sprint |
|---------|-----------|------------|---------|--------|
| `get-python-status` | R→M | — | `PythonStatus` | S9 |
| `get-app-version` | R→M | — | `string` | S9 |
| `get-machine-id` | R→M | — | `string` | S9 |
| `open-data-folder` | R→M | — | `void` (opens Explorer) | S9 |
| `open-external-link` | R→M | `url: string` | `void` | S9 |
| `check-for-updates` | R→M | — | `UpdateCheckResult` | S11 |
| `download-update` | R→M | — | `void` (triggers install) | S11 |
| `validate-license` | R→M | `key: string` | `LicenseResult` | S10 |
| `activate-license` | R→M | `key: string` | `LicenseResult` | S10 |
| `deactivate-license` | R→M | — | `void` | S10 |
| `get-license-status` | R→M | — | `LicenseStatus` | S10 |
| `on-python-crash` | M→R | — | `CrashInfo` | S9 |
| `on-update-available` | M→R | — | `UpdateInfo` | S11 |

---

## 2. Use Cases

### 2.1 Desktop App Use Cases

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-D01 | **Launch Application** | User | Double-click desktop shortcut → Splash screen → Python starts → Health check passes → Main window appears (target: < 8 seconds cold start) |
| UC-D02 | **Minimize to Tray** | User | Click close (X) → App minimizes to system tray → Tray icon shows status → Double-click tray icon to restore window |
| UC-D03 | **Quick Backtest** | User | Launch app → Dashboard loads → Click NEW BACKTEST → Select EURUSD H1 → Select XGBoost → DEPLOY BACKTEST → Watch progress → View results |
| UC-D04 | **Navigate with Keyboard** | User | Ctrl+1-6 switches pages → Ctrl+N new backtest → Ctrl+E export results → Escape closes modals |
| UC-D05 | **Background Backtest** | User | Start backtest → Minimize to tray → Tray icon turns yellow (running) → Backtest completes → Tray notification "Backtest Complete" → Restore window to view results |
| UC-D06 | **Update Application** | User | App checks for updates on launch → "Update available: v1.1.0" notification → Click "Download & Install" → App downloads update → Prompts restart → Applies update |
| UC-D07 | **Clean Uninstall** | User | Uninstall via Windows Settings → Prompt "Keep your data?" → Yes: data preserved in %APPDATA% → No: data deleted → Clean removal |
| UC-D08 | **First Run Setup** | New user | Launch app → Welcome wizard → "Choose data directory" → "Enter OANDA API key (optional)" → "Select default currency pair" → "Run sample backtest?" → Yes → Quick EURUSD smoke test → Results shown |
| UC-D09 | **Backend Crash Recovery** | User | Running backtest → Python crashes → Modal: "Engine stopped unexpectedly. Restarting..." → Python restarts → "Resume last backtest?" → Yes → Job resumes or offers restart |
| UC-D10 | **Offline Grace** | Licensed user | No internet → License check uses cached validation (7-day grace) → App works normally → Day 8: "Please connect to verify license" → 3-day hard warning → Day 11: features restricted |

### 2.2 Licensing & Payment Use Cases (Sprint 10)

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-L01 | **Free Trial Start** | New user | First launch → 14-day trial activates automatically → Full access to all features → Header shows "Trial: 12 days remaining" → Countdown continues |
| UC-L02 | **Trial Expiry** | Trial user | Day 15 → Modal: "Trial expired. Activate to continue." → Restricted mode: 3 models only (logistic, xgboost, random_forest), no advanced execution, no news features → "Activate Now" button |
| UC-L03 | **License Purchase** | Prospect | Click "Upgrade" in app → Opens Paddle checkout in browser → User pays £149 → Paddle sends license key via email → User enters key in Settings → License activated |
| UC-L04 | **License Activation** | Licensed user | Settings → License → Enter key → Click "Activate" → Online validation via Paddle API → Machine bound → Full access unlocked |
| UC-L05 | **Machine Transfer** | Licensed user | Settings → License → "Deactivate This Machine" → Confirmation → Machine deactivated → License freed → Activate on new machine |
| UC-L06 | **Annual Renewal** | Licensed user | Email reminder 30 days before expiry → Paddle subscription auto-renews → If cancelled: app continues with last version, no updates → "Updates expired" badge |
| UC-L07 | **Refund Handling** | Support | Paddle issues refund → Webhook to Paddle API → License deactivated → User's app shows "License revoked" |
| UC-L08 | **Volume Licensing** | Enterprise | Contact sales → Custom Paddle plan → Multiple license keys → Each key activates on one machine |

### 2.3 Data Management Use Cases

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-DM01 | **Data Directory Setup** | User | First launch or Settings → Choose data directory → Default: %APPDATA%/fx-ml-backtester → Custom: browse to any folder → Verify write permissions → Migrate existing data |
| UC-DM02 | **Download New Pair** | User | Settings → Data Sources → Enter OANDA key → Click "Download" next to unavailable pair → Progress bar → Data appears in CSV directory |
| UC-DM03 | **Update Historical Data** | User | Settings → Data Sources → "Update All" → API fetches latest candles → Appends to existing CSV → Verifies no gaps |
| UC-DM04 | **Backup Data** | User | Settings → "Export Data Archive" → Creates ZIP of csv_data/ + results/ → User chooses save location |
| UC-DM05 | **Clear Cache** | User | Settings → "Clear Feature Cache" → Deletes .feature_cache/ and news_cache/ → Next backtest recomputes features |

---

## 3. Python Subsystem

### 3.1 FastAPI Static File Serving (New)

For production mode, FastAPI serves the built React app as static files. This eliminates the need for a separate web server.

```
# api/main.py — add at the end of the file

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# In production (Electron), serve React build from static/
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    # Mount static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    # Catch-all: serve index.html for React Router
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        file_path = os.path.join(static_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(static_dir, "index.html"))
```

### 3.2 CLI Entry Point for PyInstaller

Create a standalone entry point that starts uvicorn with the FastAPI app:

```
# run_server.py — PyInstaller entry point

import uvicorn
import sys
import os

def main():
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    data_dir = os.environ.get("FX_DATA_DIR", "")

    if data_dir:
        os.environ["API_DB_PATH"] = os.path.join(data_dir, "forex.db")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=5,
    )

if __name__ == "__main__":
    main()
```

### 3.3 PyInstaller Spec

```
# forex_pipeline.spec

# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect hidden imports for ML libraries
hidden_imports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "api",
    "api.main",
    "api.routers",
    "pipeline",
    "pipeline.backtester",
    "models",
    "news",
    "schemas",
    "rl",
]

# Collect all submodules for key packages
for pkg in ["pipeline", "models", "api", "news", "schemas", "rl"]:
    hidden_imports.extend(collect_submodules(pkg))

# TensorFlow and PyTorch have many hidden deps
try:
    tf_datas, tf_binaries, tf_hidden = collect_all("tensorflow")
    hidden_imports.extend(tf_hidden)
except ImportError:
    pass

try:
    torch_datas, torch_binaries, torch_hidden = collect_all("torch")
    hidden_imports.extend(torch_hidden)
except ImportError:
    pass

a = Analysis(
    ["run_server.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("csv_data", "csv_data"),
        ("hpo", "hpo"),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter", "IPython", "notebook"],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="forex_pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can break TensorFlow
    console=True,  # Keep console for log output
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="forex_pipeline",
)
```

### 3.4 Environment Variables for Electron → Python

| Variable | Set By | Default | Purpose |
|----------|--------|---------|---------|
| `API_HOST` | Electron main.ts | `127.0.0.1` | FastAPI bind address |
| `API_PORT` | Electron main.ts | Dynamic (9000-9999) | FastAPI bind port |
| `FX_DATA_DIR` | Electron main.ts | `%APPDATA%/fx-ml-backtester/data` | Data directory |
| `API_DB_PATH` | run_server.py | `{FX_DATA_DIR}/forex.db` | SQLite database path |
| `API_REDIS_URL` | Electron main.ts | — (optional) | Redis for Celery |
| `API_DEBUG` | Electron main.ts | `false` | Debug mode |
| `FX_LICENSE_KEY` | Electron main.ts | — | Validated license key |
| `FX_MACHINE_ID` | Electron main.ts | — | Hardware fingerprint |
| `FX_APP_MODE` | Electron main.ts | `desktop` | Tells API it's running inside Electron |

---

## 4. Electron Main Process

### 4.1 main.ts — Application Entry

Responsibilities:
1. Single-instance lock
2. Create user data directories
3. Find available port
4. Spawn Python subprocess
5. Health check polling
6. Create BrowserWindow
7. Set up tray + menus
8. IPC handler registration
9. Graceful shutdown

```
Lifecycle:

app.on('ready')
  ├── acquireSingleInstanceLock()
  ├── ensureUserDataDirs()
  ├── findAvailablePort()
  ├── spawnPythonProcess(port)
  ├── createSplashWindow()
  ├── pollHealthCheck(port, maxRetries=60, interval=500ms)
  │   ├── OK: createMainWindow(port), closeSplash()
  │   └── TIMEOUT: showError("Failed to start engine"), quit()
  ├── setupSystemTray()
  ├── setupApplicationMenu()
  └── registerIpcHandlers()

app.on('window-all-closed')
  └── If not darwin: do nothing (tray keeps app alive)

app.on('before-quit')
  ├── killPythonProcess()
  └── releaseSingleInstanceLock()

app.on('activate')  [macOS only]
  └── createMainWindow if null
```

### 4.2 window.ts — BrowserWindow Configuration

```typescript
const WINDOW_CONFIG = {
  minWidth: 1280,
  minHeight: 800,
  width: 1440,
  height: 900,
  backgroundColor: "#131722",          // Matches TradingView Mirage
  titleBarStyle: "hidden",              // Custom title bar via React
  titleBarOverlay: {
    color: "#1E222D",
    symbolColor: "#EDEFF5",
    height: 48,
  },
  webPreferences: {
    preload: path.join(__dirname, "preload.js"),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    webSecurity: true,
    allowRunningInsecureContent: false,
  },
  show: false,                          // Show after ready-to-show
  icon: path.join(__dirname, "assets/icon.ico"),
};
```

### 4.3 python.ts — Python Process Manager

```typescript
class PythonManager {
  private process: ChildProcess | null = null;
  private port: number;
  private dataDir: string;
  private status: "stopped" | "starting" | "ready" | "error" = "stopped";
  private restartCount: number = 0;
  private maxRestarts: number = 3;

  async start(): Promise<void>
  async stop(): Promise<void>
  async restart(): Promise<void>

  private async findAvailablePort(): Promise<number>
  private spawnProcess(): ChildProcess
  private async healthCheck(maxRetries: number, interval: number): Promise<void>
  private handleCrash(code: number, signal: string): void
  private forwardLogs(data: string): void

  getStatus(): PythonStatus
  getPort(): number
}

interface PythonStatus {
  state: "stopped" | "starting" | "ready" | "error";
  port: number | null;
  pid: number | null;
  uptime: number;          // seconds since ready
  restartCount: number;
  lastError: string | null;
}
```

**Port Discovery**:
- Scan ports 9000-9999
- Try creating a temporary TCP server on each port
- First available port wins
- Pass to Python via `API_PORT` env var
- After Python starts, health check on that port

**Health Check**:
```
GET http://localhost:{port}/api/v1/health
Expected: { "status": "ok", "version": "1.0.0" }
Poll every 500ms
Max 60 retries (30 seconds)
Timeout → show error dialog
```

**Crash Recovery**:
```
Python process exits unexpectedly (code != 0):
  1. Log crash details to file
  2. If restartCount < maxRestarts (3):
     a. Send IPC 'on-python-crash' to renderer
     b. Wait 2 seconds
     c. Restart Python process
     d. Increment restartCount
     e. Re-run health check
  3. If restartCount >= maxRestarts:
     a. Show error dialog: "Engine failed to start after 3 attempts."
     b. Offer: [Retry] [Open Logs] [Quit]
```

**Graceful Shutdown**:
```
User quits app:
  1. Send SIGTERM to Python process
  2. Wait up to 5 seconds (Python finishes current request)
  3. If process still running: SIGKILL (force terminate)
  4. Clean up temp files, release port
```

### 4.4 tray.ts — System Tray

```typescript
interface TrayConfig {
  icon: string;                          // Path to tray icon
  tooltip: string;                       // "FX ML Backtester — Idle"
}

// Tray context menu:
const trayMenu = [
  { label: "FX ML Backtester", enabled: false },  // Header
  { type: "separator" },
  { label: "Show Window", click: showMainWindow },
  { type: "separator" },
  { label: "Status: Idle", enabled: false },       // Dynamic
  { type: "separator" },
  { label: "New Backtest", click: () => navigateTo("/backtest") },
  { label: "Dashboard", click: () => navigateTo("/") },
  { type: "separator" },
  { label: "Quit", click: quitApp },
];
```

**Tray Status Updates** (via IPC from renderer):
| Status | Icon | Tooltip |
|--------|------|---------|
| Idle | Default | "FX ML Backtester — Ready" |
| Backtesting | Yellow overlay | "FX ML Backtester — Running backtest (67%)" |
| Error | Red overlay | "FX ML Backtester — Engine error" |
| Updating | Blue overlay | "FX ML Backtester — Installing update..." |

### 4.5 menu.ts — Application Menu

```typescript
const applicationMenu = [
  {
    label: "File",
    submenu: [
      { label: "New Backtest", accelerator: "CmdOrCtrl+N", click: newBacktest },
      { type: "separator" },
      { label: "Settings", accelerator: "CmdOrCtrl+,", click: openSettings },
      { label: "Open Data Folder", click: openDataFolder },
      { type: "separator" },
      { label: "Exit", accelerator: "Alt+F4", click: quitApp },
    ],
  },
  {
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      { role: "selectAll" },
    ],
  },
  {
    label: "View",
    submenu: [
      { label: "Dashboard", accelerator: "CmdOrCtrl+1", click: () => navigateTo("/") },
      { label: "Backtest", accelerator: "CmdOrCtrl+2", click: () => navigateTo("/backtest") },
      { label: "Results", accelerator: "CmdOrCtrl+3", click: () => navigateTo("/results") },
      { label: "Compare", accelerator: "CmdOrCtrl+4", click: () => navigateTo("/compare") },
      { label: "News", accelerator: "CmdOrCtrl+5", click: () => navigateTo("/news") },
      { label: "Settings", accelerator: "CmdOrCtrl+6", click: () => navigateTo("/settings") },
      { type: "separator" },
      { label: "Toggle Terminal", accelerator: "CmdOrCtrl+`", click: toggleTerminal },
      { label: "Toggle Verbose Mode", accelerator: "CmdOrCtrl+Shift+V", click: toggleVerbose },
      { type: "separator" },
      { label: "Reload", accelerator: "CmdOrCtrl+R", role: "reload" },
      { label: "Developer Tools", accelerator: "F12", role: "toggleDevTools", visible: false },
    ],
  },
  {
    label: "Backtest",
    submenu: [
      { label: "Deploy", accelerator: "CmdOrCtrl+Enter", click: deployBacktest },
      { label: "Terminate Run", accelerator: "CmdOrCtrl+Shift+Escape", click: terminateRun },
      { type: "separator" },
      { label: "Export Results", accelerator: "CmdOrCtrl+E", click: exportResults },
      { label: "Compare Models", accelerator: "CmdOrCtrl+Shift+C", click: openCompare },
    ],
  },
  {
    label: "Help",
    submenu: [
      { label: "Documentation", click: () => openExternal("https://docs.fxbacktester.com") },
      { label: "Check for Updates", click: checkForUpdates },
      { type: "separator" },
      { label: "View Logs", click: openLogsFolder },
      { label: "Open Developer Tools", accelerator: "Alt+Shift+F12", role: "toggleDevTools" },
      { type: "separator" },
      { label: "About FX ML Backtester", click: showAboutDialog },
    ],
  },
];
```

### 4.6 preload.ts — IPC Bridge

```typescript
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  // Python engine
  getPythonStatus: () => ipcRenderer.invoke("get-python-status"),
  restartEngine: () => ipcRenderer.invoke("restart-engine"),

  // App info
  getAppVersion: () => ipcRenderer.invoke("get-app-version"),
  getMachineId: () => ipcRenderer.invoke("get-machine-id"),
  getPlatform: () => process.platform,

  // File system
  openDataFolder: () => ipcRenderer.invoke("open-data-folder"),
  openExternalLink: (url: string) => ipcRenderer.invoke("open-external-link", url),
  selectDirectory: () => ipcRenderer.invoke("select-directory"),

  // Updates (S11)
  checkForUpdates: () => ipcRenderer.invoke("check-for-updates"),
  onUpdateAvailable: (callback: (info: any) => void) =>
    ipcRenderer.on("on-update-available", (_, info) => callback(info)),

  // Licensing (S10)
  validateLicense: (key: string) => ipcRenderer.invoke("validate-license", key),
  activateLicense: (key: string) => ipcRenderer.invoke("activate-license", key),
  deactivateLicense: () => ipcRenderer.invoke("deactivate-license"),
  getLicenseStatus: () => ipcRenderer.invoke("get-license-status"),

  // Events (main → renderer)
  onPythonCrash: (callback: (info: any) => void) =>
    ipcRenderer.on("on-python-crash", (_, info) => callback(info)),
});
```

---

## 5. Build & Packaging

### 5.1 Build Pipeline Overview

```
┌─ Step 1: Build React ──────────────────────────────────────────┐
│  cd frontend && npm run build                                  │
│  Output: frontend/dist/ (HTML, JS, CSS, assets)                │
└────────────────────────┬───────────────────────────────────────┘
                         │ Copy dist/ → static/
                         ▼
┌─ Step 2: Build Python ─────────────────────────────────────────┐
│  PyInstaller forex_pipeline.spec                               │
│  Output: build/python/forex_pipeline/ (exe + all deps)         │
│  Includes: static/ (React build), csv_data/, hpo/              │
│  Size estimate: 2-4GB (TensorFlow + PyTorch are large)         │
└────────────────────────┬───────────────────────────────────────┘
                         │ Copy build/python/ → extraResources
                         ▼
┌─ Step 3: Build Electron ───────────────────────────────────────┐
│  electron-builder --win                                        │
│  Output: dist/FXBacktester-Setup-1.0.0.exe (NSIS installer)    │
│  Size estimate: 2.5-4.5GB (includes Python bundle)             │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 electron-builder.yml

```yaml
appId: com.fxbacktester.desktop
productName: "FX ML Backtester"
copyright: "Copyright © 2026 FX Backtester"

directories:
  output: dist
  buildResources: build

files:
  - electron/dist/**/*         # Compiled Electron main process
  - frontend/dist/**/*         # React build (also in Python bundle)
  - package.json

extraResources:
  - from: build/python/forex_pipeline
    to: python
    filter:
      - "**/*"

win:
  target:
    - target: nsis
      arch: [x64]
  icon: build/icon.ico
  artifactName: "FXBacktester-Setup-${version}.${ext}"

nsis:
  oneClick: false
  perMachine: false
  allowToChangeInstallationDirectory: true
  installerIcon: build/icon.ico
  uninstallerIcon: build/icon.ico
  installerHeaderIcon: build/icon.ico
  createDesktopShortcut: true
  createStartMenuShortcut: true
  shortcutName: "FX ML Backtester"
  include: build/nsis-installer.nsh
  license: LICENSE

mac:
  target:
    - target: dmg
      arch: [x64, arm64]
  icon: build/icon.icns
  category: public.app-category.finance

linux:
  target:
    - target: AppImage
      arch: [x64]
  icon: build/icon.png
  category: Office

publish:
  provider: github
  owner: rafa9-labs
  repo: thesisproj
```

### 5.3 Build Scripts

```bat
REM scripts/build_python.bat — Build Python bundle
@echo off
echo Building Python bundle with PyInstaller...
rmdir /s /q build\python 2>nul

REM Copy React build into static/ for FastAPI to serve
rmdir /s /q static 2>nul
xcopy /s /e /i frontend\dist static

REM Run PyInstaller
pyinstaller forex_pipeline.spec --clean --noconfirm

echo Python bundle ready: build\python\forex_pipeline\
```

```bat
REM scripts/build_electron.bat — Build Electron app
@echo off
echo Building Electron app...

REM Step 1: Build React
cd frontend
call npm run build
cd ..

REM Step 2: Build Python (if not already built)
if not exist build\python\forex_pipeline\forex_pipeline.exe (
    call scripts\build_python.bat
)

REM Step 3: Compile Electron main process
call npx tsc -p tsconfig.electron.json

REM Step 4: Build Electron installer
call npx electron-builder --win

echo Done! Installer: dist\FXBacktester-Setup-1.0.0.exe
pause
```

```bat
REM scripts/dev.bat — Development mode
@echo off
echo Starting development environment...

REM Start FastAPI (in a new terminal)
start "FastAPI" cmd /c "python run_server.py"

REM Start Vite dev server + Electron
cd frontend
call npx concurrently ^
    "npm run dev" ^
    "npx wait-on http://localhost:5173 && npx electron . --dev"
```

---

## 6. Payment & Licensing Infrastructure

### 6.1 Paddle Integration Architecture (Sprint 10)

```
┌──────────────────────────────────────────────────────────────────┐
│                    PADDLE PAYMENT FLOW                             │
│                                                                   │
│  User clicks "Upgrade" in app                                     │
│        │                                                          │
│        ▼                                                          │
│  Electron opens Paddle Checkout in default browser               │
│  https://buy.paddle.com/product/{product_id}                     │
│        │                                                          │
│        ▼                                                          │
│  User completes payment (card/PayPal)                             │
│        │                                                          │
│        ▼                                                          │
│  Paddle sends license key to user via email                       │
│  Paddle fires Webhook: "license_created"                          │
│        │                                                          │
│        ▼                                                          │
│  User enters license key in Settings → License                    │
│        │                                                          │
│        ▼                                                          │
│  Electron → POST Paddle API: /api/2.0/license/verify             │
│  { license_key, machine_id }                                      │
│        │                                                          │
│        ▼                                                          │
│  Paddle responds: { valid, activation_id, expires_at }           │
│        │                                                          │
│        ▼                                                          │
│  Electron saves:                                                  │
│  - Encrypted license key to %APPDATA%/license/license.key         │
│  - Activation ID + expiry to encrypted SQLite                    │
│  - Sets isLicensed = true in store                                │
│        │                                                          │
│        ▼                                                          │
│  React app updates: full access unlocked, trial badge removed    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Paddle Product Configuration

| Config | Value |
|--------|-------|
| **Product Name** | FX Backtester Pro |
| **Product Type** | Software (desktop) |
| **Pricing Model** | Hybrid: One-time + Annual updates subscription |
| **One-time Price** | £149 |
| **Annual Updates** | £49/year (optional, auto-renew) |
| **Trial Period** | 14 days, full access |
| **Trial → Paid** | Automatic restriction on day 15 |
| **Money-back** | 30-day guarantee |
| **Currency** | GBP, USD, EUR supported |

**Paddle Product IDs** (to be configured):

| ID | Name | Price | Type |
|----|------|-------|------|
| `prod_pro` | FX Backtester Pro | £149 one-time | Perpetual license |
| `prod_updates` | Annual Updates | £49/year | Subscription |
| `prod_team` | FX Backtester Team | £299 one-time | Multi-seat license |
| `prod_team_updates` | Team Annual Updates | £99/year | Multi-seat subscription |

### 6.3 Feature Gating (Trial vs Licensed)

| Feature | Free Trial (14 days) | Trial Expired | Licensed |
|---------|---------------------|---------------|----------|
| Dashboard | Full | View only | Full |
| Backtest — logistic, xgboost, random_forest | Full | Full | Full |
| Backtest — CNN, LSTM, Transformer, DQN | Full | Locked | Full |
| Backtest — Ensemble models | Full | Locked | Full |
| Execution — Fixed sizing | Full | Full | Full |
| Execution — Kelly, ATR, Trailing | Full | Locked | Full |
| Risk Management | Full | Locked | Full |
| News & Sentiment | Full | Locked | Full |
| Model Comparison | Full | View only | Full |
| Results Export | Full | CSV only | Full (CSV + PNG + JSON) |
| Settings — GPU config | Full | Full | Full |
| Settings — Data sources | Full | Full | Full |
| Settings — License | Full | Full | Full |
| Auto-updates | Yes | No | Yes |

### 6.4 License Validation Flow

```typescript
// Electron main process — license.ts (Sprint 10)

interface LicenseResult {
  valid: boolean;
  activated: boolean;
  expiresAt: Date | null;
  productId: string | null;
  error: string | null;
}

async function validateLicense(key: string, machineId: string): Promise<LicenseResult> {
  // Step 1: Online validation via Paddle API
  try {
    const response = await fetch("https://vendor-api.paddle.com/api/2.0/license/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        vendor_id: process.env.PADDLE_VENDOR_ID,
        vendor_auth_code: process.env.PADDLE_AUTH_CODE,
        license_key: key,
        product_id: process.env.PADDLE_PRODUCT_ID,
      }),
    });

    const data = await response.json();

    if (data.success) {
      // Step 2: Activate on this machine
      await activateOnMachine(key, machineId, data.activation_id);
      return { valid: true, activated: true, expiresAt: data.expires_at, productId: data.product_id, error: null };
    }

    return { valid: false, activated: false, expiresAt: null, productId: null, error: data.error.message };
  } catch (networkError) {
    // Step 3: Offline grace period — check cached validation
    return checkCachedValidation(machineId);
  }
}

async function checkCachedValidation(machineId: string): Promise<LicenseResult> {
  // Read last successful validation from encrypted SQLite
  const cached = await readCachedLicense(machineId);

  if (!cached) {
    return { valid: false, activated: false, expiresAt: null, productId: null, error: "No cached license" };
  }

  const daysSinceLastCheck = (Date.now() - cached.lastCheckedAt.getTime()) / (1000 * 60 * 60 * 24);

  if (daysSinceLastCheck <= 7) {
    // Grace period: valid for 7 days offline
    return { valid: true, activated: true, expiresAt: cached.expiresAt, productId: cached.productId, error: null };
  }

  if (daysSinceLastCheck <= 10) {
    // Warning period: valid but show warning
    return { valid: true, activated: true, expiresAt: cached.expiresAt, productId: cached.productId,
             error: "Please connect to the internet to verify your license" };
  }

  // Hard expired
  return { valid: false, activated: false, expiresAt: null, productId: null, error: "License verification expired. Please connect to the internet." };
}
```

### 6.5 Machine Fingerprinting (Sprint 10)

```typescript
// electron/fingerprint.ts

import * as os from "os";
import * as crypto from "crypto";

function getMachineFingerprint(): string {
  const components = [
    getCPUSerial(),       // CPU identifier
    getMACAddress(),      // Primary network adapter MAC
    getDiskSerial(),      // System drive serial number
    getMotherboardID(),   // Baseboard serial (Windows: wmic baseboard get serialnumber)
    getHostname(),        // Computer name (fallback)
  ];

  const raw = components.filter(Boolean).join("|");
  const hash = crypto.createHash("sha256").update(raw).digest("hex");

  // Format as human-readable: A4F2-8C1D-E7B3-9K5M
  return hash.substring(0, 19).match(/.{4}/g)!.join("-").toUpperCase();
}

// Returns stable identifier across reboots
// On Windows: uses wmic commands (no admin required)
// Graceful degradation: if any component fails, uses what's available
```

---

## 7. Account & Database Architecture

### 7.1 Local SQLite Database

The app uses a local SQLite database (WAL mode for concurrent reads) stored in the user's data directory.

```
%APPDATA%/fx-ml-backtester/data/forex.db
```

**Schema**:

```sql
-- Job tracking (managed by api/services/)
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'backtest',  -- 'backtest' | 'download'
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'running' | 'completed' | 'failed'
    pair TEXT,                               -- 'EURUSD'
    models TEXT,                             -- JSON array: '["logistic","xgboost"]'
    config TEXT,                             -- JSON: full config_overrides
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

-- Job results (metrics per model)
CREATE TABLE job_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    model TEXT NOT NULL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    total_return REAL,
    win_rate REAL,
    total_trades INTEGER,
    profit_factor REAL,
    avg_trade REAL,
    equity_curve BLOB,                       -- Compressed binary (zstd)
    monthly_results BLOB,                    -- Compressed JSON
    best_config TEXT,                        -- JSON
    param_importances TEXT,                  -- JSON
    trial_values TEXT,                       -- JSON
    created_at TEXT NOT NULL
);

-- User settings (encrypted in Sprint 10)
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- License state (Sprint 10)
CREATE TABLE license (
    id INTEGER PRIMARY KEY CHECK (id = 1),   -- Single row
    license_key TEXT,                         -- Encrypted
    activation_id TEXT,                       -- Encrypted
    machine_id TEXT,                          -- Hardware fingerprint
    product_id TEXT,
    expires_at TEXT,
    last_checked_at TEXT,
    is_valid INTEGER DEFAULT 0
);

-- Data source tracking
CREATE TABLE data_sources (
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    file_path TEXT NOT NULL,
    row_count INTEGER,
    start_date TEXT,
    end_date TEXT,
    last_updated TEXT,
    PRIMARY KEY (pair, timeframe)
);

-- Feature cache index
CREATE TABLE feature_cache_index (
    cache_key TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    size_bytes INTEGER
);

-- Indexes
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX idx_job_results_job ON job_results(job_id);
```

### 7.2 Data Directory Layout

```
%APPDATA%/fx-ml-backtester/
├── data/
│   ├── forex.db                    — SQLite database
│   ├── csv/
│   │   ├── EURUSD_10_years_H1_OANDA.csv
│   │   ├── GBPUSD_10_years_H1_OANDA.csv
│   │   └── ... (18 files)
│   └── results/
│       ├── 2026-04-18_abc123/
│       │   ├── metrics.json
│       │   ├── equity_curve.parquet
│       │   ├── trades.parquet
│       │   └── config.json
│       └── ... (one dir per job)
│
├── cache/
│   ├── features/                   — Parquet feature cache
│   │   ├── abc123.parquet
│   │   └── ...
│   └── news/                       — News article cache
│       ├── articles.parquet
│       └── scores.parquet
│
├── config/
│   └── settings.json               — User preferences (encrypted S10)
│
├── license/
│   └── license.key                 — License state (encrypted S10)
│
├── logs/
│   ├── app.log                     — Electron main process log
│   ├── python.log                  — Python/FastAPI log
│   └── crash-YYYYMMDD-HHMMSS.log  — Crash reports
│
└── updates/
    └── download/                   — Temporary update downloads
```

### 7.3 Settings Persistence

User settings stored in `settings.json` (plain JSON for v1, encrypted SQLite in S10):

```json
{
  "version": 1,
  "general": {
    "verboseMode": false,
    "sidebarCollapsed": false,
    "terminalCollapsed": true,
    "dataDir": "%APPDATA%/fx-ml-backtester/data"
  },
  "api": {
    "port": null,
    "timeout": 30000
  },
  "gpu": {
    "threadBudget": 4,
    "mixedPrecision": true
  },
  "dataSource": {
    "oandaApiKey": null,
    "defaultPair": "EURUSD",
    "defaultTimeframe": "H1"
  },
  "pipeline": {
    "configOverrides": {}
  }
}
```

### 7.4 Log Rotation

```
logs/app.log — Electron main process
logs/python.log — FastAPI/uvicorn output

Rotation:
- Max file size: 10MB
- Max files: 5
- Naming: app.log, app.log.1, app.log.2, ..., app.log.5
- Oldest file deleted when rotating
- Crash logs: separate files with timestamp (never rotated)
```

---

## 8. Implementation Phases

### Phase 9.1: Electron Scaffold (Est: 3h)

**Goal**: Bootable Electron app that loads the React frontend from a running FastAPI instance.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 1.1 | Add Electron deps to root `package.json` | `package.json` | `npm install` succeeds |
| 1.2 | Create `electron/main.ts` — app lifecycle, window creation, dev/prod mode | `electron/main.ts` | Compiles without errors |
| 1.3 | Create `electron/window.ts` — BrowserWindow config (1280×800, dark bg, custom title bar) | `electron/window.ts` | Window opens with correct size and color |
| 1.4 | Create `electron/preload.ts` — expose IPC APIs (v1: get-python-status, get-app-version, open-data-folder) | `electron/preload.ts` | `window.electronAPI` accessible in renderer |
| 1.5 | Configure `tsconfig.electron.json` for main process compilation | `tsconfig.electron.json` | `npx tsc -p tsconfig.electron.json` succeeds |
| 1.6 | Add scripts to `package.json`: `dev`, `build`, `build:electron` | `package.json` | Scripts run without errors |
| 1.7 | Dev mode: concurrently start Vite + wait-on + Electron | `scripts/dev.bat` | `npm run dev` launches app |
| 1.8 | Prod mode: detect bundled Python, construct URL from dynamic port | `electron/main.ts` | Prod build loads React from FastAPI |

**Exit Criteria**: `npm run dev` launches Electron window with React app loaded. Window has correct dimensions and dark theme.

### Phase 9.2: Python Backend Lifecycle (Est: 3h)

**Goal**: Electron spawns and manages the FastAPI Python process automatically.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 2.1 | Create `electron/python.ts` — PythonManager class with spawn, stop, restart, health check | `electron/python.ts` | Python process spawns and stops |
| 2.2 | Port discovery — find available port in 9000-9999 range | `electron/python.ts` | Finds available port reliably |
| 2.3 | Health check polling — GET /api/v1/health every 500ms, max 30s | `electron/python.ts` | Health check passes when Python ready |
| 2.4 | Splash screen — show loading window with progress while Python starts | `electron/main.ts` update | Splash shows during startup |
| 2.5 | Graceful shutdown — SIGTERM → wait 5s → SIGKILL | `electron/python.ts` | Python stops cleanly on app quit |
| 2.6 | Log forwarding — pipe Python stdout/stderr to Electron console + file | `electron/python.ts` | Logs appear in console and log file |
| 2.7 | Auto-restart on crash — max 3 restart attempts, then error dialog | `electron/python.ts` | Crashes trigger restart, dialog after 3 failures |
| 2.8 | IPC handlers — `get-python-status`, `restart-engine` | `electron/ipc.ts` | Renderer can query Python status |
| 2.9 | Create `run_server.py` — PyInstaller-compatible entry point | `run_server.py` | `python run_server.py` starts FastAPI |
| 2.10 | Environment variable passing — host, port, data dir, license key | `electron/python.ts` | Python receives correct env vars |

**Exit Criteria**: App launches, shows splash, starts Python, health check passes, React app loads. App quit kills Python cleanly.

### Phase 9.3: Native Menus + System Tray (Est: 2h)

**Goal**: Desktop-native menus and system tray integration.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 3.1 | Create `electron/menu.ts` — full application menu (File, Edit, View, Backtest, Help) | `electron/menu.ts` | Menu appears in app |
| 3.2 | Keyboard shortcuts — Ctrl+1-6 for pages, Ctrl+N for new backtest, Ctrl+Enter for deploy | `electron/menu.ts` | Shortcuts navigate correctly |
| 3.3 | Create `electron/tray.ts` — system tray icon with context menu | `electron/tray.ts` | Tray icon appears |
| 3.4 | Tray status updates — icon changes based on active job status (via IPC from renderer) | `electron/tray.ts` | Icon turns yellow during backtest |
| 3.5 | Minimize-to-tray — close button minimizes to tray instead of quitting | `electron/main.ts` update | Close minimizes, tray icon persists |
| 3.6 | Single instance lock — prevent multiple app instances | `electron/main.ts` update | Second launch focuses existing window |
| 3.7 | Create app icons — icon.ico (Windows), icon.icns (macOS), icon.png (Linux) | `build/icon.ico`, etc. | Icons appear in taskbar and title bar |

**Exit Criteria**: App menu works with shortcuts, tray icon shows status, close minimizes to tray, single instance enforced.

### Phase 9.4: Build Pipeline (Est: 3h)

**Goal**: Production Windows build (.exe installer) that bundles everything.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 4.1 | Create `forex_pipeline.spec` — PyInstaller config for Python + all ML deps | `forex_pipeline.spec` | PyInstaller build succeeds |
| 4.2 | Add static file serving to FastAPI — mount `frontend/dist/` in production | `api/main.py` update | FastAPI serves React app |
| 4.3 | Create `electron-builder.yml` — NSIS installer config, Python as extraResource | `electron-builder.yml` | electron-builder reads config |
| 4.4 | Create `scripts/build_python.bat` — automate PyInstaller build | `scripts/build_python.bat` | Script produces Python bundle |
| 4.5 | Create `scripts/build_electron.bat` — full build pipeline (React + Python + Electron) | `scripts/build_electron.bat` | Script produces .exe installer |
| 4.6 | Code signing — configure self-signed cert for dev builds | `electron-builder.yml` | Build signs the .exe |
| 4.7 | Bundle size optimization — tree-shake unused TensorFlow/PyTorch ops, exclude dev deps | `forex_pipeline.spec` | Bundle < 4GB |
| 4.8 | Create `scripts/dev.bat` — dev mode launcher (FastAPI + Vite + Electron) | `scripts/dev.bat` | Dev mode works with one command |
| 4.9 | Test installer on clean Windows 10/11 VM | Manual | App installs, launches, runs backtest |
| 4.10 | Test uninstall — verify clean removal (with "keep data" option) | Manual | Uninstall works correctly |

**Exit Criteria**: `scripts\build_electron.bat` produces a working `.exe` installer. App installs and runs on a clean Windows machine.

---

## 9. Professional Safeguards

### 9.1 Process Management Safeguards

| Guard | Implementation | Where |
|-------|---------------|-------|
| **Single instance** | `app.requestSingleInstanceLock()` — second launch focuses existing window | `electron/main.ts` |
| **Zombie process cleanup** | On unexpected exit, kill Python subprocess via `tree-kill` (kills entire process tree) | `electron/python.ts` |
| **Orphan port cleanup** | Before starting Python, check if port is already in use. If yes, try to kill the process using it | `electron/python.ts` |
| **Memory leak detection** | Monitor Electron renderer memory via `process.memoryUsage()`. Warn if > 2GB | `electron/main.ts` |
| **Graceful degradation** | If Python fails to start 3 times, show error with "Open Logs" and "Report Issue" buttons | `electron/python.ts` |
| **Uncaught exception handler** | `process.on('uncaughtException')` → log to crash file, show dialog, offer restart | `electron/main.ts` |
| **Unhandled rejection handler** | `process.on('unhandledRejection')` → log, continue (don't crash) | `electron/main.ts` |

### 9.2 Security Safeguards

| Guard | Implementation | Where |
|-------|---------------|-------|
| **Context isolation** | `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` | `electron/window.ts` |
| **Preload whitelist** | Only explicitly exposed APIs available in renderer | `electron/preload.ts` |
| **No remote module** | `@electron/remote` not used — all IPC via `ipcRenderer.invoke` | Throughout |
| **Navigation restriction** | `webContents.on('will-navigate')` blocks navigation away from localhost | `electron/window.ts` |
| **No shell.openExternal abuse** | All external links opened via explicit IPC `open-external-link` with URL validation | `electron/preload.ts` |
| **Content Security Policy** | CSP header set by FastAPI: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'` | `api/main.py` |
| **HTTPS enforcement** | Paddle API calls use HTTPS only | `electron/license.ts` |
| **Secret storage** | OANDA API key stored in encrypted SQLite, never in plaintext files | `api/storage.py` (S10) |
| **DevTools disabled** | In production build, DevTools accessible only via hidden shortcut (Alt+Shift+F12) | `electron/main.ts` |
| **Code signing** | Production builds signed with EV code signing certificate | `electron-builder.yml` |

### 9.3 Data Integrity Safeguards

| Guard | Implementation | Where |
|-------|---------------|-------|
| **SQLite WAL mode** | Write-Ahead Logging for concurrent read/write without corruption | `api/main.py` |
| **Database backup** | Auto-backup `forex.db` to `forex.db.backup` on every app start | `electron/main.ts` |
| **Schema migration** | Version-tracked schema migrations on app update | `api/main.py` |
| **File lock on CSV** | Prevent concurrent writes to CSV data files | `pipeline/data_downloader.py` |
| **Data validation on load** | Verify CSV has expected columns, no NaN in price columns, correct row count | `pipeline/backtester/composed.py` |
| **Corrupt cache detection** | Parquet cache files validated by checksum before use | `pipeline/feature_cache.py` |

### 9.4 Update Safeguards

| Guard | Implementation | Where |
|-------|---------------|-------|
| **Signature verification** | electron-builder verifies update signature before installing | `electron-builder.yml` |
| **Rollback** | Keep previous version directory. If new version crashes on startup, auto-rollback | `electron/updater.ts` (S11) |
| **Differential updates** | Only download changed files (blockmap-based) | `electron-builder` |
| **Background download** | Download update in background, prompt to install when ready | `electron/updater.ts` (S11) |
| **Skip version** | User can skip a specific update version | Update notification UI |

### 9.5 Crash Reporting Safeguards

| Guard | Implementation | Where |
|-------|---------------|-------|
| **Local crash logs** | Every crash writes to `logs/crash-YYYYMMDD-HHMMSS.log` with full stack trace | `electron/main.ts` |
| **Sentry integration** | (S11) Opt-in crash reporting to Sentry for both Electron and Python errors | `electron/sentry.ts` |
| **Breadcrumbs** | Last 20 user actions logged in crash report for debugging | Sentry integration |
| **Crash count tracking** | If app crashes 3 times in a row on startup, offer "Safe Mode" (no plugins, default config) | `electron/main.ts` |

---

## 10. Testing Strategy

### 10.1 Unit Tests (Vitest)

| Category | What | Tool | Target |
|----------|------|------|--------|
| **Port discovery** | Finds available port, skips occupied ports | Vitest | 100% branch coverage |
| **Machine fingerprint** | Returns stable ID across calls, handles missing components | Vitest | All fallback paths |
| **IPC handlers** | All IPC channels return expected shapes | Vitest | All registered channels |
| **Config defaults** | Default settings match expected values | Vitest | All settings keys |

### 10.2 Integration Tests (Playwright + Electron)

| Category | What | Tool | Target |
|----------|------|------|--------|
| **App lifecycle** | Launch → Python starts → Health check → Window shows → Quit kills Python | Playwright + Electron | Full lifecycle |
| **Crash recovery** | Kill Python process → Auto-restart → App recovers | Playwright + Electron | Recovery flow |
| **Tray behavior** | Close window → Tray icon present → Double-click → Window restores | Playwright + Electron | All tray actions |
| **Keyboard shortcuts** | Ctrl+1-6 navigate, Ctrl+N new backtest, etc. | Playwright + Electron | All shortcuts |

### 10.3 Build Verification Tests

| Test | What | Pass Criteria |
|------|------|---------------|
| **Python bundle size** | PyInstaller output is reasonable | < 5GB |
| **Electron build succeeds** | `build_electron.bat` completes without errors | Exit code 0 |
| **Installer size** | Final .exe is reasonable | < 5GB |
| **Clean install** | Install on clean Windows 10 VM | App launches, no errors |
| **Clean uninstall** | Uninstall with "delete data" | No files remain in %APPDATA% |
| **Upgrade install** | Install v1.1 over v1.0 | Settings and data preserved |
| **First run** | Launch on clean system | Welcome wizard appears |
| **Cold start time** | Time from double-click to UI ready | < 8 seconds (SSD) |
| **Memory at idle** | Memory usage after launch, no backtest running | < 500MB |
| **GPU detection** | CUDA available on machine with NVIDIA GPU | GPU status shows correctly |

---

## 11. File Structure

```
project root/
├── electron/
│   ├── main.ts                        # Electron main process entry
│   ├── preload.ts                     # IPC bridge (contextBridge)
│   ├── python.ts                      # PythonManager class
│   ├── window.ts                      # BrowserWindow config + creation
│   ├── tray.ts                        # System tray icon + menu
│   ├── menu.ts                        # Application menu bar
│   ├── ipc.ts                         # IPC handler registration
│   ├── fingerprint.ts                 # Machine ID generation (S10 stub)
│   ├── license.ts                     # License validation stub (S10)
│   ├── updater.ts                     # Auto-update stub (S11)
│   └── utils.ts                       # Port discovery, path helpers
│
├── scripts/
│   ├── build_python.bat               # PyInstaller build
│   ├── build_electron.bat             # Full build pipeline
│   └── dev.bat                        # Dev mode launcher
│
├── build/                             # Build artifacts (gitignored)
│   ├── icon.ico                       # Windows app icon
│   ├── icon.icns                      # macOS app icon
│   ├── icon.png                       # Linux app icon
│   └── nsis-installer.nsh             # Custom NSIS installer steps
│
├── static/                            # React build copy for FastAPI (gitignored)
│
├── run_server.py                      # PyInstaller entry point
├── forex_pipeline.spec                # PyInstaller configuration
├── electron-builder.yml               # electron-builder configuration
├── tsconfig.electron.json             # TypeScript config for Electron
│
├── frontend/                          # React app (Sprint 8)
│   ├── dist/                          # Build output (gitignored)
│   └── ...
│
├── api/                               # FastAPI backend (Sprint 7 — COMPLETE)
├── pipeline/                          # Backtester engine
├── models/                            # ML models
├── news/                              # News & sentiment
├── tests/                             # 436 tests
│
├── package.json                       # Root package.json (Electron)
├── SPRINT_8_PLAN.md                   # React frontend plan
├── SPRINT_9_PLAN.md                   # This file
├── ROADMAP.md                         # Full product roadmap
├── README.md                          # Project documentation
└── CLAUDE.md                          # AI assistant context
```

---

## 12. Dependencies

### 12.1 Root package.json (Electron)

```json
{
  "name": "fx-ml-backtester",
  "version": "1.0.0",
  "private": true,
  "main": "electron/dist/main.js",
  "scripts": {
    "dev": "scripts\\dev.bat",
    "build": "scripts\\build_electron.bat",
    "build:python": "scripts\\build_python.bat",
    "build:react": "cd frontend && npm run build",
    "build:electron-ts": "npx tsc -p tsconfig.electron.json",
    "electron:dev": "npx concurrently \"cd frontend && npm run dev\" \"npx wait-on http://localhost:5173 && npx electron . --dev\"",
    "electron:preview": "npx electron ."
  },
  "dependencies": {},
  "devDependencies": {
    "electron": "^28.1.0",
    "electron-builder": "^24.9.1",
    "typescript": "^5.3.3",
    "@types/node": "^20.11.0",
    "concurrently": "^8.2.2",
    "wait-on": "^7.2.0",
    "tree-kill": "^1.2.2",
    "portfinder": "^1.0.32",
    "electron-devtools-installer": "^3.2.0"
  }
}
```

### 12.2 Python Build Dependencies

```
# Added to requirements.txt for PyInstaller builds
pyinstaller>=6.3.0
pyinstaller-hooks-contrib>=2024.1
zstandard>=0.22.0           # For compressed equity curve storage
```

---

## 13. Completion Criteria

### Phase Gate Checklist

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 1 | Electron app launches and loads React UI | Double-click .exe or `npm run dev` |
| 2 | Python process starts automatically on launch | Check Task Manager for python/forex_pipeline process |
| 3 | Health check passes within 30 seconds | Splash screen → main window transition |
| 4 | Cold start time < 8 seconds (SSD) | Stopwatch from double-click to UI visible |
| 5 | Graceful shutdown kills Python process | Close app, verify no orphan python processes |
| 6 | System tray shows status correctly | Idle, running, error states |
| 7 | Minimize-to-tray works | Close button → tray, double-click → restore |
| 8 | Single instance lock works | Second launch focuses existing window |
| 9 | Application menu with keyboard shortcuts | Ctrl+1-6, Ctrl+N, Ctrl+Enter all work |
| 10 | Full backtest runs end-to-end in Electron | Deploy backtest, see progress, view results |
| 11 | PyInstaller build succeeds | `scripts\build_python.bat` completes |
| 12 | electron-builder produces .exe installer | `scripts\build_electron.bat` produces installer |
| 13 | Installer works on clean Windows 10 VM | Install → Launch → Run backtest → Results |
| 14 | Uninstall cleans up properly | Uninstall → No orphan files (unless "keep data") |
| 15 | Memory at idle < 500MB | Task Manager memory check |
| 16 | Crash recovery works | Kill Python process → App restarts it |
| 17 | Log files created and rotated | Check `logs/` directory |
| 18 | IPC communication works | Settings page can query machine ID, Python status |

### Not in Scope for Sprint 9

| Feature | Sprint | Reason |
|---------|--------|--------|
| License enforcement | S10 | Needs Paddle SDK integration |
| Encrypted storage | S10 | Needs cryptography setup |
| Auto-update system | S11 | Needs GitHub Releases + electron-updater |
| Sentry crash reporting | S11 | Needs Sentry project setup |
| macOS DMG build | Post-launch | Windows-only for v1 |
| Linux AppImage build | Post-launch | Windows-only for v1 |
| Auto-launch on startup | Post-launch | Nice-to-have |
| File association (.fxbacktest) | Post-launch | Nice-to-have |
| Multi-window support | Post-launch | Single window for v1 |
| Deep linking (fx://) | Post-launch | Nice-to-have |

---

## Appendix A: Sprint 8 ↔ 9 Integration Points

These are the specific integration points between the React frontend (Sprint 8) and Electron shell (Sprint 9):

| Integration Point | Sprint 8 Responsibility | Sprint 9 Responsibility |
|-------------------|------------------------|------------------------|
| **API URL** | Configurable in settings store, default `http://localhost:8000` | Electron injects dynamic port via IPC or env var |
| **Python status** | Header shows Python status (green/red dot) | Electron exposes `get-python-status` IPC |
| **Machine ID** | Settings page displays machine ID | Electron exposes `get-machine-id` IPC |
| **Data directory** | Settings page shows/browses data directory | Electron exposes `open-data-folder` and `select-directory` IPC |
| **External links** | All links open via `window.electronAPI.openExternalLink()` | Electron validates URL and opens in default browser |
| **Window controls** | React renders custom title bar (minimize, maximize, close) | Electron provides `titleBarOverlay` config |
| **Tray status** | React sends job status updates to Electron via IPC | Electron updates tray icon based on status |
| **Keyboard shortcuts** | React handles Ctrl+1-6 within app | Electron registers global shortcuts in menu |
| **License UI** | Settings → License panel (stubs, reads from IPC) | Electron handles actual Paddle validation |
| **Updates** | Update notification component (stub) | Electron checks GitHub Releases |

## Appendix B: Post-Sprint 9 Sprint Sequence

| Sprint | Topic | Depends On | Est |
|--------|-------|-----------|-----|
| **S10** | Security & Licensing (Paddle) | S9 (Electron shell) | 12-15h |
| **S11** | Installer & Auto-Update | S10 (code signing) | 6-8h |
| **S12** | Commercial Infrastructure | S11 (auto-update) | 8-10h |
| **S13** | Beta & Launch | S12 (marketing site) | 6-8h + 2 week beta |

## Appendix C: Cold Start Budget Breakdown

```
Double-click FXBacktester.exe
        │
        ├─ 0-200ms     Electron binary loads, main.ts executes
        ├─ 200-500ms   Single instance check, directory creation
        ├─ 500-800ms   Port discovery, Python process spawn
        ├─ 800ms-4s    Python starts: imports TensorFlow, PyTorch, FastAPI
        │               (This is the bottleneck — TF import alone is ~2-3s)
        ├─ 4-5s        FastAPI initializes, mounts routes, connects SQLite
        ├─ 5-5.5s      Health check passes
        ├─ 5.5-6s      BrowserWindow created, React bundle starts loading
        ├─ 6-7s        React hydrates, fonts load, API calls fire
        ├─ 7-8s        Dashboard renders with data
        │
        └─ 8s          App ready to use

Optimization opportunities (post-launch):
- Lazy TensorFlow import (only import when deep model is selected)
- Python warm start (pre-fork on system boot)
- React code splitting (load pages on demand)
- Service worker caching (cache API responses)
```
