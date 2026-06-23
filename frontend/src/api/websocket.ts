type WsListener = (event: unknown) => void;

interface ConnectionState {
  ws: WebSocket | null;
  listeners: Set<WsListener>;
  reconnectAttempts: number;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  watchDogTimer: ReturnType<typeof setInterval> | null;
  lastMsgTime: number;
  isTerminal: boolean;
  watchdogRetries: number;
}

const MAX_RECONNECT_ATTEMPTS = 10;
const WATCHDOG_INTERVAL = 15_000;
const WATCHDOG_TIMEOUT = 60_000;

export class WebSocketManager {
  private connections: Map<string, ConnectionState> = new Map();
  private baseUrl: string;

  constructor() {
    const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
    this.baseUrl = apiBase.replace(/^http/, "ws").replace(/\/api\/v1\/?$/, "");
    if (!this.baseUrl.startsWith("ws")) {
      this.baseUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
    }
  }

  connect(jobId: string) {
    const existing = this.connections.get(jobId);
    if (existing?.ws?.readyState === WebSocket.OPEN) return;

    this.disconnect(jobId);

    const state: ConnectionState = {
      ws: null,
      listeners: new Set(),
      reconnectAttempts: 0,
      reconnectTimer: null,
      watchDogTimer: null,
      lastMsgTime: Date.now(),
      isTerminal: false,
      watchdogRetries: 0,
    };
    this.connections.set(jobId, state);
    this.doConnect(jobId, state);
  }

  private doConnect(jobId: string, state: ConnectionState) {
    const url = `${this.baseUrl}/api/v1/backtest/${jobId}/ws`;
    if (import.meta.env.DEV) console.log("[WS] connecting to:", url);
    state.ws = new WebSocket(url);

    state.ws.onopen = () => {
      state.reconnectAttempts = 0;
      state.lastMsgTime = Date.now();
      this.startWatchDog(jobId, state);
      if (import.meta.env.DEV) console.log("[WS] connected:", url);
    };

    state.ws.onmessage = (msg) => {
      state.lastMsgTime = Date.now();
      try {
        const parsed = JSON.parse(msg.data);
        if (import.meta.env.DEV && parsed.event !== "heartbeat")
          console.log("[WS] received:", parsed.event, parsed.job_id?.slice(0, 8));
        state.listeners.forEach((fn) => fn(parsed));
      } catch {
        if (import.meta.env.DEV) console.warn("[WS] failed to parse:", msg.data.slice(0, 100));
        state.listeners.forEach((fn) => fn(msg.data));
      }
    };

    state.ws.onclose = (ev) => {
      this.stopWatchDog(state);
      if (import.meta.env.DEV)
        console.warn("[WS] closed, code:", ev.code, "wasClean:", ev.wasClean, "reason:", ev.reason);
      if (state.isTerminal) {
        state.ws = null;
        return;
      }
      if (!ev.wasClean && state.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 30_000);
        state.reconnectAttempts++;
        state.reconnectTimer = setTimeout(() => this.doConnect(jobId, state), delay);
      }
    };

    state.ws.onerror = () => {
      if (import.meta.env.DEV) console.error("[WS] error event, readyState:", state.ws?.readyState);
      state.ws?.close();
    };
  }

  private startWatchDog(jobId: string, state: ConnectionState) {
    this.stopWatchDog(state);
    state.watchDogTimer = setInterval(() => {
      if (state.isTerminal) {
        this.stopWatchDog(state);
        state.ws?.close();
        state.ws = null;
        return;
      }
      const elapsed = Date.now() - state.lastMsgTime;
      if (elapsed > WATCHDOG_TIMEOUT && state.ws) {
        state.watchdogRetries++;
        if (state.watchdogRetries > 5) {
          if (import.meta.env.DEV) console.warn("[WS] watchdog: max retries for job, giving up:", jobId.slice(0, 8));
          this.stopWatchDog(state);
          state.ws.onclose = null;
          state.ws.close();
          state.ws = null;
          state.isTerminal = true;
          return;
        }
        if (import.meta.env.DEV) console.warn("[WS] watchdog: no message for 60s, reconnecting job:", jobId.slice(0, 8));
        state.ws.onclose = null;
        state.ws.close();
        state.ws = null;
        this.doConnect(jobId, state);
      }
    }, WATCHDOG_INTERVAL);
  }

  private stopWatchDog(state: ConnectionState) {
    if (state.watchDogTimer) {
      clearInterval(state.watchDogTimer);
      state.watchDogTimer = null;
    }
  }

  disconnect(jobId: string) {
    const state = this.connections.get(jobId);
    if (!state) return;

    this.stopWatchDog(state);
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    if (state.ws) {
      state.ws.onclose = null;
      state.ws.close();
      state.ws = null;
    }
    this.connections.delete(jobId);
  }

  markTerminal(jobId: string) {
    const state = this.connections.get(jobId);
    if (!state) return;
    state.isTerminal = true;
    this.stopWatchDog(state);
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    if (state.ws) {
      state.ws.onclose = null;
      state.ws.close();
      state.ws = null;
    }
  }

  subscribe(jobId: string, fn: WsListener): () => void {
    const state = this.connections.get(jobId);
    if (!state) {
      if (import.meta.env.DEV) console.warn("[WS] subscribe called for unknown job:", jobId.slice(0, 8));
      return () => {};
    }
    state.listeners.add(fn);
    return () => state.listeners.delete(fn);
  }

  get connected(): boolean {
    for (const state of this.connections.values()) {
      if (state.ws?.readyState === WebSocket.OPEN) return true;
    }
    return false;
  }

  isConnected(jobId: string): boolean {
    return this.connections.get(jobId)?.ws?.readyState === WebSocket.OPEN;
  }
}

export const wsManager = new WebSocketManager();
