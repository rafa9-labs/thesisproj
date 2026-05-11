import { describe, it, expect } from "vitest";

function parseMonthToTs(monthStr: string): number | null {
  const parts = monthStr.split("-");
  if (parts.length < 2) return null;
  const year = parseInt(parts[0], 10);
  const month = parseInt(parts[1], 10);
  if (isNaN(year) || isNaN(month)) return null;
  return Math.floor(Date.UTC(year, month - 1, 1) / 1000);
}

describe("Sprint 2: parseMonthToTs", () => {
  it("parses 2024-01 to unix timestamp", () => {
    const ts = parseMonthToTs("2024-01");
    expect(ts).toBeGreaterThan(0);
    const d = new Date(ts * 1000);
    expect(d.getUTCFullYear()).toBe(2024);
    expect(d.getUTCMonth()).toBe(0);
  });

  it("parses 2023-12 to unix timestamp", () => {
    const ts = parseMonthToTs("2023-12");
    const d = new Date(ts * 1000);
    expect(d.getUTCFullYear()).toBe(2023);
    expect(d.getUTCMonth()).toBe(11);
  });

  it("returns null for empty string", () => {
    expect(parseMonthToTs("")).toBeNull();
  });

  it("returns null for invalid format", () => {
    expect(parseMonthToTs("invalid")).toBeNull();
  });

  it("returns null for partial format", () => {
    expect(parseMonthToTs("2024")).toBeNull();
  });

  it("different months produce different timestamps", () => {
    const jan = parseMonthToTs("2024-01");
    const feb = parseMonthToTs("2024-02");
    expect(feb).toBeGreaterThan(jan!);
  });
});

describe("Sprint 2: walk-forward month logic", () => {
  const monthlyResults = [
    { month: "2024-01", return_pct: 2.5, win_rate: 0.6, trades: 12, sharpe: 1.2 },
    { month: "2024-02", return_pct: -1.3, win_rate: 0.45, trades: 8, sharpe: -0.5 },
    { month: "2024-03", return_pct: 0.8, win_rate: 0.55, trades: 10, sharpe: 0.7 },
  ];

  it("computes correct count of positive months", () => {
    const positive = monthlyResults.filter((m) => (m.return_pct ?? 0) >= 0);
    expect(positive).toHaveLength(2);
  });

  it("computes correct count of negative months", () => {
    const negative = monthlyResults.filter((m) => (m.return_pct ?? 0) < 0);
    expect(negative).toHaveLength(1);
  });

  it("parses all months to valid timestamps", () => {
    const timestamps = monthlyResults.map((m) => parseMonthToTs(m.month));
    expect(timestamps.every((t) => t !== null)).toBe(true);
  });

  it("timestamps are in chronological order", () => {
    const timestamps = monthlyResults.map((m) => parseMonthToTs(m.month)!);
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
    }
  });
});

describe("Sprint 2: equity PnL calculation", () => {
  it("computes PnL from equity start and current", () => {
    const start = 10000;
    const current = 10250;
    const pnl = current - start;
    const pnlPct = (pnl / start) * 100;
    expect(pnl).toBe(250);
    expect(pnlPct).toBeCloseTo(2.5);
  });

  it("handles negative PnL", () => {
    const start = 10000;
    const current = 9750;
    const pnl = current - start;
    const pnlPct = (pnl / start) * 100;
    expect(pnl).toBe(-250);
    expect(pnlPct).toBeCloseTo(-2.5);
  });

  it("handles zero start equity", () => {
    const start = 0;
    const current = 100;
    const pnlPct = start !== 0 ? ((current - start) / start) * 100 : null;
    expect(pnlPct).toBeNull();
  });
});

describe("Sprint 2: playback speed intervals", () => {
  it("interval at 1x speed is 200ms", () => {
    const speed = 1;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    expect(intervalMs).toBe(200);
  });

  it("interval at 2x speed is 100ms", () => {
    const speed = 2;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    expect(intervalMs).toBe(100);
  });

  it("interval at 5x speed is 40ms", () => {
    const speed = 5;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    expect(intervalMs).toBe(40);
  });

  it("interval at 10x speed is 20ms", () => {
    const speed = 10;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    expect(intervalMs).toBe(20);
  });

  it("interval never falls below 10ms", () => {
    const speed = 100;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    expect(intervalMs).toBe(10);
  });
});

describe("Sprint 2: trade filtering by timestamp", () => {
  const trades = [
    { trade_id: 1, entry_time: 100, exit_time: 150, direction: "BUY" as const, entry_price: 1.08, exit_price: 1.085, pnl_pct: 0.46 },
    { trade_id: 2, entry_time: 200, exit_time: 250, direction: "SELL" as const, entry_price: 1.083, exit_price: 1.078, pnl_pct: -0.46 },
    { trade_id: 3, entry_time: 300, exit_time: 350, direction: "BUY" as const, entry_price: 1.08, exit_price: 1.09, pnl_pct: 0.93 },
  ];

  it("filters trades with entry_time before current timestamp", () => {
    const currentTs = 220;
    const visible = trades.filter((t) => t.entry_time <= currentTs);
    expect(visible).toHaveLength(2);
  });

  it("returns all trades when current timestamp is after last entry", () => {
    const currentTs = 400;
    const visible = trades.filter((t) => t.entry_time <= currentTs);
    expect(visible).toHaveLength(3);
  });

  it("returns no trades when current timestamp is before first entry", () => {
    const currentTs = 50;
    const visible = trades.filter((t) => t.entry_time <= currentTs);
    expect(visible).toHaveLength(0);
  });

  it("identifies open trade (entered but not exited)", () => {
    const currentTs = 220;
    const openTrade = trades.find((t) => t.entry_time <= currentTs && t.exit_time > currentTs);
    expect(openTrade).toBeDefined();
    expect(openTrade!.trade_id).toBe(2);
  });
});