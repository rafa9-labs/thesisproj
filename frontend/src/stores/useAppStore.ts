import { create } from "zustand";

interface AppState {
  demoMode: boolean;
  demoSeeded: boolean;
  setDemoMode: (v: boolean) => void;
  setDemoSeeded: (v: boolean) => void;
}

export const useAppStore = create<AppState>()((set) => ({
  demoMode: false,
  demoSeeded: false,
  setDemoMode: (v) => set({ demoMode: v }),
  setDemoSeeded: (v) => set({ demoSeeded: v }),
}));
