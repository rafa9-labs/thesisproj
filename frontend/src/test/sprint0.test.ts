import { describe, it, expect } from "vitest";
import { TIMEFRAMES } from "@/lib/constants";

describe("Sprint 0: TIMEFRAMES constant", () => {
  it("exports 3 V1 timeframes", () => {
    expect(TIMEFRAMES).toHaveLength(3);
  });

  it("includes M30, H1, H4", () => {
    const keys = TIMEFRAMES.map((tf) => tf.key);
    expect(keys).toEqual(["M30", "H1", "H4"]);
  });

  it("each timeframe has key and label", () => {
    for (const tf of TIMEFRAMES) {
      expect(tf.key).toBe(tf.label);
      expect(tf.key).toMatch(/^(M30|H1|H4)$/);
    }
  });
});
