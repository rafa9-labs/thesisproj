import { describe, it, expect, beforeEach } from "vitest";
import { useSettingsStore } from "@/stores/useSettingsStore";

describe("useSettingsStore", () => {
  beforeEach(() => {
    localStorage.clear();
    useSettingsStore.setState({
      verboseMode: false,
      theme: "dark",
      apiUrl: "http://localhost:8000",
      dataDir: "",
      oandaApiKey: null,
      threadBudget: 4,
      mixedPrecision: true,
      sidebarCollapsed: false,
      terminalCollapsed: true,
    });
  });

  it("has correct defaults", () => {
    const state = useSettingsStore.getState();
    expect(state.verboseMode).toBe(false);
    expect(state.theme).toBe("dark");
    expect(state.apiUrl).toBe("http://localhost:8000");
    expect(state.threadBudget).toBe(4);
    expect(state.mixedPrecision).toBe(true);
    expect(state.oandaApiKey).toBeNull();
  });

  it("updates a field", () => {
    useSettingsStore.getState().setField("verboseMode", true);
    expect(useSettingsStore.getState().verboseMode).toBe(true);
  });

  it("updates threadBudget", () => {
    useSettingsStore.getState().setField("threadBudget", 8);
    expect(useSettingsStore.getState().threadBudget).toBe(8);
  });

  it("persists to localStorage", () => {
    useSettingsStore.getState().setField("apiUrl", "http://192.168.1.100:8000");
    const saved = JSON.parse(localStorage.getItem("fx-backtester-settings")!);
    expect(saved.apiUrl).toBe("http://192.168.1.100:8000");
  });

  it("loads from localStorage on next instance", () => {
    localStorage.setItem(
      "fx-backtester-settings",
      JSON.stringify({ verboseMode: true, threadBudget: 12 }),
    );
    useSettingsStore.getState().loadFromStorage();
    expect(useSettingsStore.getState().verboseMode).toBe(true);
    expect(useSettingsStore.getState().threadBudget).toBe(12);
  });

  it("handles corrupted localStorage gracefully", () => {
    localStorage.setItem("fx-backtester-settings", "{invalid json!!!");
    useSettingsStore.getState().loadFromStorage();
    expect(useSettingsStore.getState().verboseMode).toBe(false);
  });

  it("sets oandaApiKey to string", () => {
    useSettingsStore.getState().setField("oandaApiKey", "test-key-123");
    expect(useSettingsStore.getState().oandaApiKey).toBe("test-key-123");
  });

  it("sets oandaApiKey back to null", () => {
    useSettingsStore.getState().setField("oandaApiKey", "test-key");
    useSettingsStore.getState().setField("oandaApiKey", null);
    expect(useSettingsStore.getState().oandaApiKey).toBeNull();
  });
});
