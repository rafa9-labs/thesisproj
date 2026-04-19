/**
 * Backend health check — polls /api/v1/health until the FastAPI
 * server is ready or the timeout expires.
 */

import http from "http";

export async function waitForBackend(
  port: number,
  timeoutMs: number,
): Promise<boolean> {
  const start = Date.now();
  const interval = 500;

  while (Date.now() - start < timeoutMs) {
    try {
      const ok = await checkHealth(port);
      if (ok) return true;
    } catch {
      // not ready yet
    }
    await sleep(interval);
  }
  return false;
}

function checkHealth(port: number): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const req = http.get(
      `http://127.0.0.1:${port}/api/v1/health`,
      { timeout: 2000 },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            const data = JSON.parse(body);
            resolve(data.status === "ok" || data.status === "healthy");
          } catch {
            resolve(false);
          }
        });
      },
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("timeout"));
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
