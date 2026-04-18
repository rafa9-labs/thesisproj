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
    const apiBase = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
    this.baseUrl = apiBase.replace(/^http/, "ws").replace(/\/api\/v1\/?$/, "");
  }

  connect(jobId: string) {
    this.disconnect();
    this.jobId = jobId;
    this.reconnectAttempts = 0;
    this.doConnect();
  }

  private doConnect() {
    if (!this.jobId) return;
    const url = `${this.baseUrl}/api/v1/backtest/${this.jobId}/ws`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        this.listeners.forEach((fn) => fn(parsed));
      } catch {
        this.listeners.forEach((fn) => fn(msg.data));
      }
    };

    this.ws.onclose = (ev) => {
      if (!ev.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30_000);
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => this.doConnect(), delay);
      }
    };

    this.ws.onerror = () => {
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
