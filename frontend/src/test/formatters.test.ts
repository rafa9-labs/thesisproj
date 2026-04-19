import { describe, it, expect } from "vitest";
import {
  formatMetric,
  formatPercent,
  formatPrice,
  formatPips,
  formatInt,
  formatRelativeTime,
  colorForReturn,
} from "@/lib/formatters";

describe("formatMetric", () => {
  it("formats a number to 2 decimal places", () => {
    expect(formatMetric(1.473)).toBe("1.47");
  });
  it("formats with custom decimals", () => {
    expect(formatMetric(1.5, 3)).toBe("1.500");
  });
  it("returns em-dash for null", () => {
    expect(formatMetric(null)).toBe("—");
  });
  it("returns em-dash for undefined", () => {
    expect(formatMetric(undefined)).toBe("—");
  });
  it("formats zero", () => {
    expect(formatMetric(0)).toBe("0.00");
  });
  it("formats negative numbers", () => {
    expect(formatMetric(-3.1)).toBe("-3.10");
  });
});

describe("formatPercent", () => {
  it("formats positive percentage with plus sign", () => {
    expect(formatPercent(0.124)).toBe("+12.4%");
  });
  it("formats negative percentage without plus sign", () => {
    expect(formatPercent(-0.085)).toBe("-8.5%");
  });
  it("formats zero", () => {
    expect(formatPercent(0)).toBe("+0.0%");
  });
  it("returns em-dash for null", () => {
    expect(formatPercent(null)).toBe("—");
  });
  it("uses custom decimals", () => {
    expect(formatPercent(0.5, 2)).toBe("+50.00%");
  });
});

describe("formatPrice", () => {
  it("formats to 5 decimals by default", () => {
    expect(formatPrice(1.08423)).toBe("1.08423");
  });
  it("returns em-dash for null", () => {
    expect(formatPrice(null)).toBe("—");
  });
});

describe("formatPips", () => {
  it("formats positive pips with plus sign", () => {
    expect(formatPips(12)).toBe("+12.0 pips");
  });
  it("formats negative pips", () => {
    expect(formatPips(-8.5)).toBe("-8.5 pips");
  });
  it("returns em-dash for null", () => {
    expect(formatPips(null)).toBe("—");
  });
});

describe("formatInt", () => {
  it("formats integers with locale", () => {
    expect(formatInt(120000)).toBe("120,000");
  });
  it("returns em-dash for null", () => {
    expect(formatInt(null)).toBe("—");
  });
});

describe("formatRelativeTime", () => {
  it("returns em-dash for null", () => {
    expect(formatRelativeTime(null)).toBe("—");
  });
  it("returns em-dash for undefined", () => {
    expect(formatRelativeTime(undefined)).toBe("—");
  });
  it("returns 'just now' for very recent", () => {
    expect(formatRelativeTime(new Date().toISOString())).toBe("just now");
  });
  it("returns minutes ago", () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    expect(formatRelativeTime(fiveMinAgo)).toBe("5m ago");
  });
  it("returns hours ago", () => {
    const threeHoursAgo = new Date(Date.now() - 3 * 3600_000).toISOString();
    expect(formatRelativeTime(threeHoursAgo)).toBe("3h ago");
  });
  it("returns days ago", () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 86400_000).toISOString();
    expect(formatRelativeTime(twoDaysAgo)).toBe("2d ago");
  });
});

describe("colorForReturn", () => {
  it("returns success color for positive", () => {
    expect(colorForReturn(0.05)).toBe("var(--color-accent-success)");
  });
  it("returns danger color for negative", () => {
    expect(colorForReturn(-0.05)).toBe("var(--color-accent-danger)");
  });
  it("returns success color for zero", () => {
    expect(colorForReturn(0)).toBe("var(--color-accent-success)");
  });
  it("returns muted color for null", () => {
    expect(colorForReturn(null)).toBe("var(--color-text-muted)");
  });
});
