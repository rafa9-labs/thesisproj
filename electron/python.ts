/**
 * Python backend lifecycle — spawn FastAPI as a child process.
 *
 * In dev mode: runs `uvicorn` directly.
 * In production: runs the PyInstaller-bundled executable.
 */

import { spawn, ChildProcess } from "child_process";
import path from "path";

export function spawnPythonBackend(
  projectRoot: string,
  port: number,
  isDev: boolean,
): ChildProcess {

  let cmd: string;
  let args: string[];

  if (isDev) {
    cmd = "python";
    args = ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", String(port)];
  } else {
    const exeName = process.platform === "win32" ? "fx_backend.exe" : "fx_backend";
    const exePath = path.resolve(process.resourcesPath, "backend", exeName);
    cmd = exePath;
    args = ["--host", "127.0.0.1", "--port", String(port)];
  }

  console.log(`[Python] Spawning: ${cmd} ${args.join(" ")}`);

  const proc = spawn(cmd, args, {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
      TF_CPP_MIN_LOG_LEVEL: "3",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  proc.stdout?.on("data", (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[Python:out] ${msg}`);
  });

  proc.stderr?.on("data", (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) console.log(`[Python:err] ${msg}`);
  });

  proc.on("exit", (code, signal) => {
    console.log(`[Python] Exited with code=${code} signal=${signal}`);
  });

  proc.on("error", (err) => {
    console.error(`[Python] Failed to start: ${err.message}`);
  });

  return proc;
}
