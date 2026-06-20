type WsListener = (event: unknown) => void;

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_RECONNECT_DELAY = 1_000;
const MAX_RECONNECT_DELAY = 30_000;

function jitter(delay: number): number {
  return delay + Math.random() * delay * 0.5;
}

let _beforeunloadBound: (() => void) | null = null;

function installBeforeunload(closeFn: () => void) {
  if (_beforeunloadBound) return;
  _beforeunloadBound = () => closeFn();
  window.addEventListener("beforeunload", _beforeunloadBound);
}

function uninstallBeforeunload() {
  if (!_beforeunloadBound) return;
  window.removeEventListener("beforeunload", _beforeunloadBound);
  _beforeunloadBound = null;
}

function generateClientId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function buildWsUrl(clientId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/v1/news/ws?client_id=${encodeURIComponent(clientId)}`;
}

class NewsWebSocketManager {
  readonly clientId: string;
  private ws: WebSocket | null = null;
  private url: string;
  private listeners: Set<WsListener> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _intentionalClose = false;

  constructor() {
    this.clientId = generateClientId();
    this.url = buildWsUrl(this.clientId);
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      return;
    }
    this._intentionalClose = false;
    this.reconnectAttempts = 0;
    this.clearReconnectTimer();
    this.doConnect();
  }

  private doConnect() {
    if (import.meta.env.DEV) console.log("[WS-NEWS] connecting to:", this.url);
    this.ws = new WebSocket(this.url);

    installBeforeunload(() => {
      if (import.meta.env.DEV) console.log("[WS-NEWS] beforeunload — closing WS");
      this._intentionalClose = true;
      this.clearReconnectTimer();
      if (this.ws) {
        this.ws.onclose = null;
        this.ws.close();
        this.ws = null;
      }
    });

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      if (import.meta.env.DEV) console.log("[WS-NEWS] connected, client_id:", this.clientId);
    };

    this.ws.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (import.meta.env.DEV) {
          console.log("[WS-NEWS] received:", parsed.event, "listeners:", this.listeners.size);
        }
        this.listeners.forEach((fn) => fn(parsed));
      } catch {
        if (import.meta.env.DEV) console.warn("[WS-NEWS] failed to parse:", msg.data.slice(0, 100));
      }
    };

    this.ws.onclose = (ev) => {
      if (import.meta.env.DEV) {
        console.warn("[WS-NEWS] closed, code:", ev.code, "intentional:", this._intentionalClose);
      }
      this.ws = null;
      if (!this._intentionalClose && this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          jitter(BASE_RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts)),
          MAX_RECONNECT_DELAY,
        );
        this.reconnectAttempts++;
        if (import.meta.env.DEV) console.log("[WS-NEWS] reconnecting in", delay, "ms");
        this.reconnectTimer = setTimeout(() => this.doConnect(), delay);
      }
    };

    this.ws.onerror = () => {
      if (import.meta.env.DEV) console.error("[WS-NEWS] error, readyState:", this.ws?.readyState);
      this.ws?.close();
    };
  }

  disconnect() {
    this._intentionalClose = true;
    this.clearReconnectTimer();
    uninstallBeforeunload();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }

  subscribe(fn: WsListener): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

export const newsWsManager = new NewsWebSocketManager();
