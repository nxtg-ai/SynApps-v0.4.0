/**
 * SettingsPage component
 * Allows user to configure SynApps settings
 */
import React, { useEffect } from 'react';
import MainLayout from '../../components/Layout/MainLayout';
import { useSettingsStore } from '../../stores/settingsStore';
import './SettingsPage.css';

const SettingsPage: React.FC = () => {
  const openaiKey = useSettingsStore((state) => state.openaiKey);
  const stabilityKey = useSettingsStore((state) => state.stabilityKey);
  const darkMode = useSettingsStore((state) => state.darkMode);
  const animationsEnabled = useSettingsStore((state) => state.animationsEnabled);
  const browserNotifications = useSettingsStore((state) => state.browserNotifications);
  const emailNotifications = useSettingsStore((state) => state.emailNotifications);
  const emailAddress = useSettingsStore((state) => state.emailAddress);
  const autoSaveEnabled = useSettingsStore((state) => state.autoSaveEnabled);
  const compactMode = useSettingsStore((state) => state.compactMode);
  const logLevel = useSettingsStore((state) => state.logLevel);
  const setSettings = useSettingsStore((state) => state.setSettings);
  const loadSettings = useSettingsStore((state) => state.loadSettings);
  const saveSettingsStore = useSettingsStore((state) => state.saveSettings);
  const resetSettingsStore = useSettingsStore((state) => state.resetSettings);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);
  
  // Save settings to localStorage
  const saveSettings = () => {
    saveSettingsStore();
    alert('Settings saved successfully');
  };
  
  // Reset settings to defaults
  const resetSettings = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      resetSettingsStore();
      alert('Settings reset to defaults');
    }
  };
  
  return (
    <MainLayout title="Settings">
      <div className="settings-page">
        <div className="settings-container">
          <div className="settings-section">
            <h2>API Keys</h2>
            <div className="settings-form">
              <div className="form-group">
                <label htmlFor="openai_key">OpenAI API Key</label>
                <input
                  type="password"
                  id="openai_key"
                  value={openaiKey}
                  onChange={(e) => setSettings({ openaiKey: e.target.value })}
                  placeholder="sk-..."
                />
                <div className="input-description">
                  Required for Writer applet using gpt-4o and AI code suggestions.
                </div>
              </div>
              
              <div className="form-group">
                <label htmlFor="stability_key">Stability AI API Key</label>
                <input
                  type="password"
                  id="stability_key"
                  value={stabilityKey}
                  onChange={(e) => setSettings({ stabilityKey: e.target.value })}
                  placeholder="sk_..."
                />
                <div className="input-description">
                  Required for Artist applet using Stable Diffusion.
                </div>
              </div>
            </div>
          </div>
          
          <div className="settings-section">
            <h2>Appearance</h2>
            <div className="settings-form">
              <div className="form-group checkbox">
                <label htmlFor="dark_mode">
                  <input
                    type="checkbox"
                    id="dark_mode"
                    checked={darkMode}
                    onChange={(e) => setSettings({ darkMode: e.target.checked })}
                  />
                  <span>Dark Mode (Coming Soon)</span>
                </label>
              </div>
              
              <div className="form-group checkbox">
                <label htmlFor="animations_enabled">
                  <input
                    type="checkbox"
                    id="animations_enabled"
                    checked={animationsEnabled}
                    onChange={(e) => setSettings({ animationsEnabled: e.target.checked })}
                  />
                  <span>Enable Animations</span>
                </label>
              </div>
            </div>
          </div>
          
          <div className="settings-section">
            <h2>Notifications</h2>
            <div className="settings-form">
              <div className="form-group checkbox">
                <label htmlFor="browser_notifications">
                  <input
                    type="checkbox"
                    id="browser_notifications"
                    checked={browserNotifications}
                    onChange={(e) => setSettings({ browserNotifications: e.target.checked })}
                  />
                  <span>Browser Notifications</span>
                </label>
              </div>
              
              <div className="form-group checkbox">
                <label htmlFor="email_notifications">
                  <input
                    type="checkbox"
                    id="email_notifications"
                    checked={emailNotifications}
                    onChange={(e) => setSettings({ emailNotifications: e.target.checked })}
                  />
                  <span>Email Notifications (Coming Soon)</span>
                </label>
              </div>
              
              {emailNotifications && (
                <div className="form-group">
                  <label htmlFor="email_address">Email Address</label>
                  <input
                    type="email"
                    id="email_address"
                    value={emailAddress}
                    onChange={(e) => setSettings({ emailAddress: e.target.value })}
                    placeholder="you@example.com"
                  />
                </div>
              )}
            </div>
          </div>
          
          <div className="settings-section">
            <h2>User Interface</h2>
            <div className="settings-form">
              <div className="form-group checkbox">
                <label htmlFor="auto_save">
                  <input
                    type="checkbox"
                    id="auto_save"
                    checked={autoSaveEnabled}
                    onChange={(e) => setSettings({ autoSaveEnabled: e.target.checked })}
                  />
                  <span>Auto-save Workflows</span>
                </label>
              </div>
              
              <div className="form-group checkbox">
                <label htmlFor="compact_mode">
                  <input
                    type="checkbox"
                    id="compact_mode"
                    checked={compactMode}
                    onChange={(e) => setSettings({ compactMode: e.target.checked })}
                  />
                  <span>Compact Mode (Coming Soon)</span>
                </label>
              </div>
            </div>
          </div>
          
          <div className="settings-section">
            <h2>Advanced</h2>
            <div className="settings-form">
              <div className="form-group">
                <label htmlFor="log_level">Log Level</label>
                <select
                  id="log_level"
                  value={logLevel}
                  onChange={(e) => setSettings({ logLevel: e.target.value })}
                >
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warn">Warning</option>
                  <option value="error">Error</option>
                </select>
              </div>
            </div>
          </div>
          
          <div className="settings-actions">
            <button className="save-button" onClick={saveSettings}>
              Save Settings
            </button>
            <button className="reset-button" onClick={resetSettings}>
              Reset to Defaults
            </button>
          </div>
        </div>
        
        <div className="settings-info">
          <div className="info-section">
            <h3>About SynApps</h3>
            <p>
              SynApps MVP v0.4.0 - A web-based visual platform for modular AI agents.
            </p>
            <p>
              SynApps allows indie creators to build autonomous AI applets like LEGO blocks, 
              with each applet being a small agent with a specialized skill.
            </p>
          </div>
          
          <div className="info-section">
            <h3>Quick Links</h3>
            <ul>
              <li><a href="https://github.com/nxtg-ai/SynApps-v0.4.0" target="_blank" rel="noopener noreferrer">GitHub Repository</a></li>
              <li><a href="https://github.com/nxtg-ai/SynApps-v0.4.0/tree/master/docs" target="_blank" rel="noopener noreferrer">Documentation</a></li>
              <li><a href="https://discord.gg/UUt6Yfk7NX" target="_blank" rel="noopener noreferrer">Discord Community</a></li>
            </ul>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default SettingsPage;
