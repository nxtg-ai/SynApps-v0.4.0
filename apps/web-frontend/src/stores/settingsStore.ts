import { create } from "zustand";

interface SettingsValues {
  openaiKey: string;
  stabilityKey: string;
  darkMode: boolean;
  animationsEnabled: boolean;
  browserNotifications: boolean;
  emailNotifications: boolean;
  emailAddress: string;
  autoSaveEnabled: boolean;
  compactMode: boolean;
  logLevel: string;
}

interface SettingsStoreState extends SettingsValues {
  setSettings: (patch: Partial<SettingsValues>) => void;
  loadSettings: () => void;
  saveSettings: () => void;
  resetSettings: () => void;
}

const defaultSettings: SettingsValues = {
  openaiKey: "",
  stabilityKey: "",
  darkMode: false,
  animationsEnabled: true,
  browserNotifications: true,
  emailNotifications: false,
  emailAddress: "",
  autoSaveEnabled: true,
  compactMode: false,
  logLevel: "info",
};

const safeGetItem = (key: string): string | null => {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(key);
};

const safeSetItem = (key: string, value: string): void => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, value);
};

const safeRemoveItem = (key: string): void => {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(key);
};

const readBoolean = (key: string, fallback: boolean): boolean => {
  const value = safeGetItem(key);
  return value === null ? fallback : value === "true";
};

export const useSettingsStore = create<SettingsStoreState>((set, get) => ({
  ...defaultSettings,
  setSettings: (patch) => set(patch),
  loadSettings: () => {
    set({
      openaiKey: safeGetItem("openai_key") ?? defaultSettings.openaiKey,
      stabilityKey: safeGetItem("stability_key") ?? defaultSettings.stabilityKey,
      darkMode: readBoolean("dark_mode", defaultSettings.darkMode),
      animationsEnabled: readBoolean("animations_enabled", defaultSettings.animationsEnabled),
      browserNotifications: readBoolean("browser_notifications", defaultSettings.browserNotifications),
      emailNotifications: readBoolean("email_notifications", defaultSettings.emailNotifications),
      emailAddress: safeGetItem("email_address") ?? defaultSettings.emailAddress,
      autoSaveEnabled: readBoolean("auto_save", defaultSettings.autoSaveEnabled),
      compactMode: readBoolean("compact_mode", defaultSettings.compactMode),
      logLevel: safeGetItem("log_level") ?? defaultSettings.logLevel,
    });
  },
  saveSettings: () => {
    const state = get();
    safeSetItem("openai_key", state.openaiKey);
    safeSetItem("stability_key", state.stabilityKey);
    safeSetItem("dark_mode", String(state.darkMode));
    safeSetItem("animations_enabled", String(state.animationsEnabled));
    safeSetItem("browser_notifications", String(state.browserNotifications));
    safeSetItem("email_notifications", String(state.emailNotifications));
    safeSetItem("email_address", state.emailAddress);
    safeSetItem("auto_save", String(state.autoSaveEnabled));
    safeSetItem("compact_mode", String(state.compactMode));
    safeSetItem("log_level", state.logLevel);
  },
  resetSettings: () => {
    set({ ...defaultSettings });
    safeRemoveItem("openai_key");
    safeRemoveItem("stability_key");
    safeRemoveItem("dark_mode");
    safeRemoveItem("animations_enabled");
    safeRemoveItem("browser_notifications");
    safeRemoveItem("email_notifications");
    safeRemoveItem("email_address");
    safeRemoveItem("auto_save");
    safeRemoveItem("compact_mode");
    safeRemoveItem("log_level");
  },
}));
