/**
 * Utility functions for the FX ML Backtester Electron shell.
 *
 * Port discovery, path resolution, and data directory management.
 */

import net from "net";
import path from "path";
import fs from "fs";
import { app } from "electron";

const PORT_RANGE_START = 9000;
const PORT_RANGE_END = 9999;

/**
 * Find the next available port in range [9000, 9999].
 */
export async function findAvailablePort(startFrom?: number): Promise<number> {
  const start = startFrom ?? PORT_RANGE_START;
  for (let port = start; port <= PORT_RANGE_END; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found in range ${PORT_RANGE_START}-${PORT_RANGE_END}`);
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => {
      server.close();
      resolve(false);
    });
    server.once("listening", () => {
      server.close();
      resolve(true);
    });
    server.listen(port, "127.0.0.1");
  });
}

/**
 * Resolve the project root path.
 *
 * In dev:  __dirname = frontend/electron-dist → go up 2 levels = project root
 * In prod: Use the exe directory (e.g. release/win-unpacked/) as project root.
 *          This is where the app binary lives alongside the resources directory.
 */
export function getProjectRoot(isDev: boolean): string {
  if (isDev) {
    return path.resolve(__dirname, "..", "..");
  }
  return path.dirname(app.getPath("exe"));
}

/**
 * Get the path to the Python backend executable.
 */
export function getBackendExePath(isDev: boolean, projectRoot: string): string {
  if (isDev) {
    return "python";
  }
  const exeName = process.platform === "win32" ? "fx_backend.exe" : "fx_backend";
  return path.resolve(process.resourcesPath, "backend", exeName);
}

/**
 * Get the user data directory for persistent storage.
 *
 * In dev: project root (same as source data)
 * In prod: %APPDATA%/FX ML Backtester/ (Windows) or equivalent
 *
 * This is where the database, results, and cache are stored.
 */
export function getUserDataDir(isDev: boolean): string {
  if (isDev) {
    return process.cwd();
  }
  const dataDir = path.join(app.getPath("userData"), "data");
  ensureDir(dataDir);
  return dataDir;
}

/**
 * Get the reference data directory (CSV files, HPO configs).
 *
 * In dev: project root (csv_data/, hpo/)
 * In prod: extraResources directory (csv_data/ bundled alongside app)
 *
 * electron-builder.yml maps csv_data/ → process.resourcesPath/csv_data/
 */
export function getReferenceDataDir(isDev: boolean, projectRoot: string): string {
  if (isDev) {
    return projectRoot;
  }
  return process.resourcesPath;
}

/**
 * Ensure a directory exists, creating it recursively if needed.
 */
function ensureDir(dirPath: string): void {
  try {
    fs.mkdirSync(dirPath, { recursive: true });
  } catch {
    // Directory may already exist or may be created by another process
  }
}

/**
 * Set up the data directory structure in production mode.
 * Creates subdirectories for database, results, and cache.
 */
export function ensureDataDirs(dataDir: string): void {
  const subdirs = ["", "results", "cache", "logs"];
  for (const subdir of subdirs) {
    ensureDir(path.join(dataDir, subdir));
  }
}