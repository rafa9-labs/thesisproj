/**
 * Utility functions for the FX ML Backtester Electron shell.
 *
 * Port discovery, path resolution, and data directory management.
 */

import net from "net";
import path from "path";
import fs from "fs";

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
 * In dev: the project repository root.
 * In prod: directory containing the app executable.
 */
export function getProjectRoot(isDev: boolean): string {
  if (isDev) {
    const { app } = require("electron");
    return app.getAppPath();
  }
  // In production, the app is inside the asar or resources directory
  const { app } = require("electron");
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
  const { app } = require("electron");
  const dataDir = path.join(app.getPath("userData"), "data");
  ensureDir(dataDir);
  return dataDir;
}

/**
 * Get the reference data directory (CSV files, HPO configs).
 *
 * In dev: project root (csv_data/, hpo/)
 * In prod: resources directory (bundled with app)
 */
export function getReferenceDataDir(isDev: boolean, projectRoot: string): string {
  if (isDev) {
    return projectRoot;
  }
  return path.resolve(process.resourcesPath, "app.asar.unpacked") || projectRoot;
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