import { create } from "zustand";

interface SettingsState {
  verboseMode: boolean;
  theme: "dark";
  apiUrl: string;
  dataDir: string;
  oandaApiKey: string | null;
  oandaAccountId: string | null;
  threadBudget: number;
  mixedPrecision: boolean;
  ramLimit: number;
  sidebarCollapsed: boolean;
  terminalCollapsed: boolean;
  liveNewsBlendEnabled: boolean;
  liveNewsBlendWeight: number;
  maxConcurrentBacktests: number;
  gpuEnabled: boolean;
  maxConcurrentGpu: number;
  notificationsEnabled: boolean;
  notificationSound: boolean;
}

interface SettingsActions {
  setField: <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => void;
  loadFromStorage: () => void;
  saveToStorage: () => void;
}

const STORAGE_KEY = "fx-backtester-settings";

const DEFAULTS: SettingsState = {
  verboseMode: false,
  theme: "dark",
  apiUrl: "http://localhost:8000",
  dataDir: "",
  oandaApiKey: null,
  oandaAccountId: null,
  threadBudget: 4,
  mixedPrecision: true,
  ramLimit: 32,
  sidebarCollapsed: false,
  terminalCollapsed: true,
  liveNewsBlendEnabled: false,
  liveNewsBlendWeight: 0.1,
  maxConcurrentBacktests: 4,
  gpuEnabled: true,
  maxConcurrentGpu: 1,
  notificationsEnabled: true,
  notificationSound: true,
};

function loadSaved(): Partial<SettingsState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return {};
}

export const useSettingsStore = create<SettingsState & SettingsActions>()((set, get) => ({
  ...DEFAULTS,
  ...loadSaved(),

  setField: (key, value) => {
    set({ [key]: value } as Partial<SettingsState>);
    get().saveToStorage();
  },

  loadFromStorage: () => {
    const saved = loadSaved();
    set(saved);
  },

  saveToStorage: () => {
    try {
      const {
        verboseMode,
        theme,
        apiUrl,
        threadBudget,
        mixedPrecision,
        ramLimit,
        sidebarCollapsed,
        terminalCollapsed,
        oandaApiKey,
        oandaAccountId,
        dataDir,
        liveNewsBlendEnabled,
        liveNewsBlendWeight,
        notificationsEnabled,
        notificationSound,
      } = get();
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          verboseMode,
          theme,
          apiUrl,
          threadBudget,
          mixedPrecision,
          ramLimit,
          sidebarCollapsed,
          terminalCollapsed,
          oandaApiKey,
          oandaAccountId,
          dataDir,
          liveNewsBlendEnabled,
          liveNewsBlendWeight,
          notificationsEnabled,
          notificationSound,
        }),
      );
    } catch {
      /* ignore */
    }
  },
}));
