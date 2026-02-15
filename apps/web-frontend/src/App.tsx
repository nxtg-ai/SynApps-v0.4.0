/**
 * Main App component
 */
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage/DashboardPage';
import EditorPage from './pages/EditorPage/EditorPage';
import HistoryPage from './pages/HistoryPage/HistoryPage';
import AppletLibraryPage from './pages/AppletLibraryPage/AppletLibraryPage';
import SettingsPage from './pages/SettingsPage/SettingsPage';
import NotFoundPage from './pages/NotFoundPage/NotFoundPage';
import webSocketService from './services/WebSocketService';
import './App.css';

const App: React.FC = () => {
  // Connect to WebSocket on app start
  useEffect(() => {
    // Connect to the WebSocket server
    webSocketService.connect();
    
    // Clean up on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, []);
  
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/editor/:flowId?" element={<EditorPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/applets" element={<AppletLibraryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
