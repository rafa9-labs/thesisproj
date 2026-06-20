import { create } from "zustand";

interface DashboardState {
  activePair: string;
  activeTimeframe: string;
  previousPair: string | null;
  setActivePair: (pair: string) => void;
  setActiveTimeframe: (tf: string) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  activePair: "EURUSD",
  activeTimeframe: "M30",
  previousPair: null,
  setActivePair: (pair) =>
    set((s) => ({
      previousPair: s.activePair !== pair ? s.activePair : s.previousPair,
      activePair: pair,
    })),
  setActiveTimeframe: (tf) => set({ activeTimeframe: tf }),
}));
