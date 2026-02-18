import { act } from '@testing-library/react';
import { useSettingsStore } from './settingsStore';
import { vi } from 'vitest';

const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

describe('useSettingsStore', () => {
  beforeEach(() => {
    // Clear mock localStorage before each test
    window.localStorage.clear();
    // Reset the store to its initial state using the store's own reset action
    act(() => {
      useSettingsStore.getState().resetSettings();
    });
  });

  it('should have default initial settings after reset', () => {
    const state = useSettingsStore.getState();
    expect(state.openaiKey).toBe("");
    expect(state.stabilityKey).toBe("");
    expect(state.darkMode).toBe(false);
    expect(state.animationsEnabled).toBe(true);
    expect(state.browserNotifications).toBe(true);
    expect(state.emailNotifications).toBe(false);
    expect(state.emailAddress).toBe("");
    expect(state.autoSaveEnabled).toBe(true);
    expect(state.compactMode).toBe(false);
    expect(state.logLevel).toBe("info");
  });

  it('should set an individual setting correctly using setSettings', () => {
    const { setSettings } = useSettingsStore.getState();
    act(() => {
      setSettings({ darkMode: true });
    });
    expect(useSettingsStore.getState().darkMode).toBe(true);
  });

  it('should load settings from localStorage', () => {
    window.localStorage.setItem("dark_mode", "true");
    window.localStorage.setItem("openai_key", "test-openai-key");
    window.localStorage.setItem("auto_save", "false");

    act(() => {
      useSettingsStore.getState().loadSettings();
    });

    const state = useSettingsStore.getState();
    expect(state.darkMode).toBe(true);
    expect(state.openaiKey).toBe("test-openai-key");
    expect(state.autoSaveEnabled).toBe(false);
    // Other settings should remain default if not in localStorage
    expect(state.stabilityKey).toBe("");
  });

  it('should save settings to localStorage', () => {
    act(() => {
      useSettingsStore.getState().setSettings({
        openaiKey: "saved-openai-key",
        darkMode: true,
        autoSaveEnabled: false,
      });
      useSettingsStore.getState().saveSettings();
    });

    expect(window.localStorage.getItem("openai_key")).toBe("saved-openai-key");
    expect(window.localStorage.getItem("dark_mode")).toBe("true");
    expect(window.localStorage.getItem("auto_save")).toBe("false");
    expect(window.localStorage.getItem("log_level")).toBe("info"); // Default value should be saved
  });

  it('should reset settings to default and clear localStorage', () => {
    // Set some non-default values and save them
    act(() => {
      useSettingsStore.getState().setSettings({
        openaiKey: "some-key",
        darkMode: true,
      });
      useSettingsStore.getState().saveSettings();
    });

    // Verify they are saved
    expect(window.localStorage.getItem("openai_key")).toBe("some-key");
    expect(useSettingsStore.getState().darkMode).toBe(true);

    // Reset settings
    act(() => {
      useSettingsStore.getState().resetSettings();
    });

    // Verify store is reset to defaults
    const state = useSettingsStore.getState();
    expect(state.openaiKey).toBe("");
    expect(state.darkMode).toBe(false);

    // Verify localStorage is cleared for these keys
    expect(window.localStorage.getItem("openai_key")).toBeNull();
    expect(window.localStorage.getItem("dark_mode")).toBeNull();
  });
});
