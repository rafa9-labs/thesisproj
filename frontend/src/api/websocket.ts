type WsListener = (event: unknown) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private jobId: string | null = null;
  private baseUrl: string;
  private listeners: Set<WsListener> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
    this.baseUrl = apiBase.replace(/^http/, "ws").replace(/\/api\/v1\/?$/, "");
    if (!this.baseUrl.startsWith("ws")) {
      this.baseUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
    }
  }

  connect(jobId: string) {
    if (this.jobId === jobId && this.ws?.readyState === WebSocket.OPEN) return;
    this.reconnectAttempts = 0;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.jobId = jobId;
    this.doConnect();
  }

  private doConnect() {
    if (!this.jobId) return;
    const url = `${this.baseUrl}/api/v1/backtest/${this.jobId}/ws`;
    if (import.meta.env.DEV) console.log("[WS] connecting to:", url);
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      if (import.meta.env.DEV) console.log("[WS] connected:", url);
    };

    this.ws.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (import.meta.env.DEV) console.log("[WS] received:", parsed.event, parsed.job_id?.slice(0, 8));
        this.listeners.forEach((fn) => fn(parsed));
      } catch {
        if (import.meta.env.DEV) console.warn("[WS] failed to parse:", msg.data.slice(0, 100));
        this.listeners.forEach((fn) => fn(msg.data));
      }
    };

    this.ws.onclose = (ev) => {
      if (import.meta.env.DEV) console.warn("[WS] closed, code:", ev.code, "wasClean:", ev.wasClean, "reason:", ev.reason);
      if (!ev.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30_000);
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => this.doConnect(), delay);
      }
    };

    this.ws.onerror = () => {
      if (import.meta.env.DEV) console.error("[WS] error event, readyState:", this.ws?.readyState);
      this.ws?.close();
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.jobId = null;
  }

  subscribe(fn: WsListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const wsManager = new WebSocketManager();
