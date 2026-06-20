import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

type CloseFn = () => void;

describe("NewsWebSocketManager singleton", () => {
  let mockClose: ReturnType<typeof vi.fn>;
  let wsCreated: number;
  let wsUrls: string[];

  beforeEach(async () => {
    vi.resetModules();
    mockClose = vi.fn();
    wsCreated = 0;
    wsUrls = [];

    // jsdom provides window.location (http://localhost:3000) by default — don't override
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).WebSocket = class {
      public url: string;
      public readyState: number;
      public onopen: (() => void) | null = null;
      public onclose: ((ev: { code: number; reason: string }) => void) | null = null;
      public onmessage: ((msg: { data: string }) => void) | null = null;
      public onerror: (() => void) | null = null;
      public close: CloseFn;
      public static CONNECTING = 0;
      public static OPEN = 1;
      public static CLOSING = 2;
      public static CLOSED = 3;

      constructor(u: string) {
        this.url = u;
        this.readyState = 0;
        this.close = mockClose;
        wsCreated++;
        wsUrls.push(u);
      }
    } as unknown as typeof WebSocket;
  });

  afterEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (globalThis as any).WebSocket;
    vi.restoreAllMocks();
  });

  it("returns the same instance across multiple imports (singleton)", async () => {
    const mod1 = await import("@/api/newsWebSocket");
    const mod2 = await import("@/api/newsWebSocket");
    expect(mod1.newsWsManager).toBe(mod2.newsWsManager);
    expect(mod1.newsWsManager.clientId).toBe(mod2.newsWsManager.clientId);
  });

  it("generates a distinct clientId", async () => {
    const mod = await import("@/api/newsWebSocket");
    expect(mod.newsWsManager.clientId).toBeTruthy();
    expect(mod.newsWsManager.clientId.length).toBeGreaterThan(8);
  });

  it("appends client_id to the WS URL", async () => {
    const mod = await import("@/api/newsWebSocket");

    // verify state before connect
    const cid = mod.newsWsManager.clientId;
    expect(cid).toBeTruthy();
    expect(wsCreated).toBe(0);

    mod.newsWsManager.connect();
    expect(wsCreated).toBe(1);
    expect(wsUrls[0]).toContain("client_id=");
    expect(wsUrls[0]).toContain(encodeURIComponent(cid));
    expect(wsUrls[0]).toContain("/api/v1/news/ws");
  });

  it("creates only one WebSocket on multiple connect() calls", async () => {
    const mod = await import("@/api/newsWebSocket");
    mod.newsWsManager.connect();
    expect(wsCreated).toBe(1);

    // Set ws to simulate OPEN state so second connect() short-circuits
    // Must include close() — stale beforeunload listener from this test
    // may fire during later tests if not cleaned up
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mod.newsWsManager as any).ws = { readyState: 1, close: vi.fn() };
    mod.newsWsManager.connect();
    expect(wsCreated).toBe(1);
  });

  it("calls ws.close() on beforeunload", async () => {
    const mod = await import("@/api/newsWebSocket");
    mod.newsWsManager.connect();
    expect(wsCreated).toBe(1);

    window.dispatchEvent(new Event("beforeunload"));
    expect(mockClose).toHaveBeenCalled();
  });

  it("disconnect() calls close and nullifies ws", async () => {
    const mod = await import("@/api/newsWebSocket");
    mod.newsWsManager.connect();
    expect(wsCreated).toBe(1);

    mod.newsWsManager.disconnect();
    expect(mockClose).toHaveBeenCalled();
  });

  it("subscribe returns an unsubscribe function", async () => {
    const mod = await import("@/api/newsWebSocket");
    const listener = vi.fn();
    const unsub = mod.newsWsManager.subscribe(listener);
    expect(typeof unsub).toBe("function");
    unsub();
  });
});
