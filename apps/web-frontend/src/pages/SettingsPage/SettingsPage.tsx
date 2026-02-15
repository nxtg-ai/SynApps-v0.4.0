/**
 * SettingsPage component
 * Allows user to configure SynApps settings
 */
import React, { useState, useEffect } from 'react';
import MainLayout from '../../components/Layout/MainLayout';
import './SettingsPage.css';

const SettingsPage: React.FC = () => {
  // API Key settings
  const [openaiKey, setOpenaiKey] = useState<string>('');
  const [stabilityKey, setStabilityKey] = useState<string>('');
  
  // App appearance settings
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [animationsEnabled, setAnimationsEnabled] = useState<boolean>(true);
  
  // Notification settings
  const [browserNotifications, setBrowserNotifications] = useState<boolean>(true);
  const [emailNotifications, setEmailNotifications] = useState<boolean>(false);
  const [emailAddress, setEmailAddress] = useState<string>('');
  
  // UI settings
  const [autoSaveEnabled, setAutoSaveEnabled] = useState<boolean>(true);
  const [compactMode, setCompactMode] = useState<boolean>(false);
  
  // Other settings
  const [logLevel, setLogLevel] = useState<string>('info');
  
  // Load settings from localStorage
  useEffect(() => {
    // API Keys
    const savedOpenaiKey = localStorage.getItem('openai_key');
    if (savedOpenaiKey) setOpenaiKey(savedOpenaiKey);
    
    const savedStabilityKey = localStorage.getItem('stability_key');
    if (savedStabilityKey) setStabilityKey(savedStabilityKey);
    
    // App appearance
    const savedDarkMode = localStorage.getItem('dark_mode');
    if (savedDarkMode) setDarkMode(savedDarkMode === 'true');
    
    const savedAnimations = localStorage.getItem('animations_enabled');
    if (savedAnimations) setAnimationsEnabled(savedAnimations === 'true');
    
    // Notifications
    const savedBrowserNotifications = localStorage.getItem('browser_notifications');
    if (savedBrowserNotifications) setBrowserNotifications(savedBrowserNotifications === 'true');
    
    const savedEmailNotifications = localStorage.getItem('email_notifications');
    if (savedEmailNotifications) setEmailNotifications(savedEmailNotifications === 'true');
    
    const savedEmailAddress = localStorage.getItem('email_address');
    if (savedEmailAddress) setEmailAddress(savedEmailAddress);
    
    // UI settings
    const savedAutoSave = localStorage.getItem('auto_save');
    if (savedAutoSave) setAutoSaveEnabled(savedAutoSave === 'true');
    
    const savedCompactMode = localStorage.getItem('compact_mode');
    if (savedCompactMode) setCompactMode(savedCompactMode === 'true');
    
    // Other settings
    const savedLogLevel = localStorage.getItem('log_level');
    if (savedLogLevel) setLogLevel(savedLogLevel);
  }, []);
  
  // Save settings to localStorage
  const saveSettings = () => {
    // API Keys
    localStorage.setItem('openai_key', openaiKey);
    localStorage.setItem('stability_key', stabilityKey);
    
    // App appearance
    localStorage.setItem('dark_mode', darkMode.toString());
    localStorage.setItem('animations_enabled', animationsEnabled.toString());
    
    // Notifications
    localStorage.setItem('browser_notifications', browserNotifications.toString());
    localStorage.setItem('email_notifications', emailNotifications.toString());
    localStorage.setItem('email_address', emailAddress);
    
    // UI settings
    localStorage.setItem('auto_save', autoSaveEnabled.toString());
    localStorage.setItem('compact_mode', compactMode.toString());
    
    // Other settings
    localStorage.setItem('log_level', logLevel);
    
    // Show success message
    alert('Settings saved successfully');
  };
  
  // Reset settings to defaults
  const resetSettings = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      // API Keys
      setOpenaiKey('');
      setStabilityKey('');
      
      // App appearance
      setDarkMode(false);
      setAnimationsEnabled(true);
      
      // Notifications
      setBrowserNotifications(true);
      setEmailNotifications(false);
      setEmailAddress('');
      
      // UI settings
      setAutoSaveEnabled(true);
      setCompactMode(false);
      
      // Other settings
      setLogLevel('info');
      
      // Clear localStorage
      localStorage.removeItem('openai_key');
      localStorage.removeItem('stability_key');
      localStorage.removeItem('dark_mode');
      localStorage.removeItem('animations_enabled');
      localStorage.removeItem('browser_notifications');
      localStorage.removeItem('email_notifications');
      localStorage.removeItem('email_address');
      localStorage.removeItem('auto_save');
      localStorage.removeItem('compact_mode');
      localStorage.removeItem('log_level');
      
      // Show success message
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
                  onChange={(e) => setOpenaiKey(e.target.value)}
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
                  onChange={(e) => setStabilityKey(e.target.value)}
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
                    onChange={(e) => setDarkMode(e.target.checked)}
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
                    onChange={(e) => setAnimationsEnabled(e.target.checked)}
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
                    onChange={(e) => setBrowserNotifications(e.target.checked)}
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
                    onChange={(e) => setEmailNotifications(e.target.checked)}
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
                    onChange={(e) => setEmailAddress(e.target.value)}
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
                    onChange={(e) => setAutoSaveEnabled(e.target.checked)}
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
                    onChange={(e) => setCompactMode(e.target.checked)}
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
                  onChange={(e) => setLogLevel(e.target.value)}
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
