/**
 * Main App component
 */
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage/DashboardPage';
import EditorPage from './pages/EditorPage/EditorPage';
import HistoryPage from './pages/HistoryPage/HistoryPage';
import AppletLibraryPage from './pages/AppletLibraryPage/AppletLibraryPage';
import SettingsPage from './pages/SettingsPage/SettingsPage';
import NotFoundPage from './pages/NotFoundPage/NotFoundPage';
import webSocketService from './services/WebSocketService';
import { Button } from './components/ui/button';

const AppRoutes: React.FC = () => {
  const location = useLocation();
  const showShortcut = location.pathname !== '/dashboard';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {showShortcut ? (
        <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-sm">
          <div className="mx-auto flex h-12 max-w-7xl items-center justify-end px-4">
            <Button asChild size="sm" variant="outline">
              <Link to="/dashboard">Dashboard</Link>
            </Button>
          </div>
        </header>
      ) : null}

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
  );
};

const App: React.FC = () => {
  useEffect(() => {
    webSocketService.connect();

    return () => {
      webSocketService.disconnect();
    };
  }, []);

  return (
    <Router>
      <AppRoutes />
    </Router>
  );
};

export default App;
