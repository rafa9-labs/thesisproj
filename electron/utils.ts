/**
 * Port discovery utilities for FX ML Backtester Electron shell.
 */

import net from "net";

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
 * In dev: parent of the electron-dist directory.
 * In prod: process.resourcesPath/app.
 */
export function getProjectRoot(isDev: boolean): string {
  if (isDev) {
    // In dev, __dirname points to electron-dist/ (compiled output)
    // Project root is the parent directory
    const { app } = require("electron");
    return app.getAppPath();
  }
  return require("path").resolve(process.resourcesPath, "app");
}

/**
 * Get the path to the Python backend executable.
 */
export function getBackendExePath(isDev: boolean, projectRoot: string): string {
  const path = require("path");
  if (isDev) {
    return "python"; // Use system Python in dev
  }
  const exeName = process.platform === "win32" ? "fx_backend.exe" : "fx_backend";
  return path.resolve(process.resourcesPath, "backend", exeName);
}

/**
 * Get the data directory path.
 * In dev: project root.
 * In prod: next to the app executable.
 */
export function getDataDir(isDev: boolean, projectRoot: string): string {
  if (isDev) {
    return projectRoot;
  }
  // In production, data lives next to the app
  const { app } = require("electron");
  return path.join(path.dirname(app.getPath("exe")), "data");
}

const path = require("path");