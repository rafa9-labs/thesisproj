/**
 * Python backend lifecycle — spawn FastAPI as a child process.
 *
 * In dev mode: runs `uvicorn` directly.
 * In production: runs the PyInstaller-bundled `fx_backend` executable.
 *
 * PythonManager handles:
 * - Dynamic port discovery (9000-9999)
 * - Auto-restart on crash (max 3 retries)
 * - Graceful shutdown (SIGTERM → wait → SIGKILL)
 * - Environment variable setup (FX_APP_MODE, FX_DATA_DIR, API_DB_PATH)
 */

import { spawn, exec, ChildProcess } from "child_process";
import path from "path";
import { findAvailablePort, getBackendExePath, getUserDataDir, ensureDataDirs } from "./utils";

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;
const SHUTDOWN_TIMEOUT_MS = 8000;
const isWin = process.platform === "win32";

export class PythonManager {
  private proc: ChildProcess | null = null;
  private port: number = 8001;
  private retries: number = 0;
  private isDev: boolean;
  private projectRoot: string;
  private onStatus: (status: string, message: string) => void;
  private shuttingDown: boolean = false;

  constructor(
    projectRoot: string,
    isDev: boolean,
    onStatus: (status: string, message: string) => void = () => {},
  ) {
    this.projectRoot = projectRoot;
    this.isDev = isDev;
    this.onStatus = onStatus;
  }

  getPort(): number {
    return this.port;
  }

  getProcess(): ChildProcess | null {
    return this.proc;
  }

  /** Fixed port for dev mode — must match vite.config.ts proxy target. */
  static readonly DEV_PORT = 8002;

  async start(): Promise<number> {
    if (this.isDev) {
      this.port = PythonManager.DEV_PORT;
    } else {
      this.port = await findAvailablePort();
    }
    this.shuttingDown = false;
    this._spawn();
    return this.port;
  }

  private _spawn(): void {
    let cmd: string;
    let args: string[];

    if (this.isDev) {
      cmd = "python";
      args = ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", String(this.port)];
    } else {
      cmd = getBackendExePath(false, this.projectRoot);
      args = ["--host", "127.0.0.1", "--port", String(this.port)];
    }

    console.log(`[Python] Spawning: ${cmd} ${args.join(" ")}`);

    const dataDir = getUserDataDir(this.isDev);
    ensureDataDirs(dataDir);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
      TF_CPP_MIN_LOG_LEVEL: "3",
      FX_APP_MODE: this.isDev ? "dev" : "desktop",
      API_PORT: String(this.port),
    };

    if (!this.isDev) {
      env.API_DB_PATH = path.join(dataDir, "forex.db");
      env.FX_DATA_DIR = dataDir;
      env.CSV_DATA_DIR = path.join(process.resourcesPath, "csv_data");
    }

    this.proc = spawn(cmd, args, {
      cwd: this.isDev ? this.projectRoot : path.dirname(cmd),
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    this.proc.stdout?.on("data", (data: Buffer) => {
      const msg = data.toString().trim();
      if (msg) console.log(`[Python:out] ${msg}`);
    });

    this.proc.stderr?.on("data", (data: Buffer) => {
      const msg = data.toString().trim();
      if (msg) console.log(`[Python:err] ${msg}`);
    });

    this.proc.on("exit", (code, signal) => {
      console.log(`[Python] Exited with code=${code} signal=${signal}`);
      if (!this.shuttingDown && this.retries < MAX_RETRIES) {
        this.retries++;
        console.log(`[Python] Restarting (attempt ${this.retries}/${MAX_RETRIES})...`);
        this.onStatus("restarting", `Backend crashed, restarting (attempt ${this.retries})`);
        setTimeout(() => this._spawn(), RETRY_DELAY_MS);
      } else if (!this.shuttingDown) {
        this.onStatus("error", "Backend crashed and max retries exceeded");
      }
    });

    this.proc.on("error", (err) => {
      console.error(`[Python] Failed to start: ${err.message}`);
      this.onStatus("error", `Backend failed to start: ${err.message}`);
    });
  }

  async stop(): Promise<void> {
    this.shuttingDown = true;
    if (!this.proc || this.proc.exitCode !== null) return;

    const pid = this.proc.pid;
    console.log(`[Python] Shutting down (pid=${pid})...`);

    if (isWin && pid) {
      await this._killWindows(pid);
    } else {
      this.proc.kill("SIGTERM");
    }

    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        console.log("[Python] Force killing...");
        if (isWin && pid) {
          exec(`taskkill /F /T /PID ${pid}`, () => {});
        } else {
          this.proc?.kill("SIGKILL");
        }
        resolve();
      }, SHUTDOWN_TIMEOUT_MS);

      this.proc?.on("exit", () => {
        clearTimeout(timeout);
        resolve();
      });
    });

    this.proc = null;
  }

  private _killWindows(pid: number): Promise<void> {
    return new Promise((resolve) => {
      exec(`taskkill /PID ${pid} /T`, (err) => {
        if (err) {
          console.log(`[Python] taskkill /T failed, trying /F...`);
          exec(`taskkill /F /T /PID ${pid}`, () => resolve());
        } else {
          resolve();
        }
      });
    });
  }
}

/**
 * Legacy function for backward compatibility.
 */
export function spawnPythonBackend(
  projectRoot: string,
  port: number,
  isDev: boolean,
): ChildProcess {
  const manager = new PythonManager(projectRoot, isDev);
  manager.start();
  return manager.getProcess()!;
}