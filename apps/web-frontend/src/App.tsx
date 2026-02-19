/**
 * Main App component
 */
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import webSocketService from './services/WebSocketService';
import { Button } from './components/ui/button';
import { useSettingsStore } from './stores/settingsStore';

const DashboardPage = React.lazy(() => import('./pages/DashboardPage/DashboardPage'));
const EditorPage = React.lazy(() => import('./pages/EditorPage/EditorPage'));
const HistoryPage = React.lazy(() => import('./pages/HistoryPage/HistoryPage'));
const AppletLibraryPage = React.lazy(() => import('./pages/AppletLibraryPage/AppletLibraryPage'));
const SettingsPage = React.lazy(() => import('./pages/SettingsPage/SettingsPage'));
const NotFoundPage = React.lazy(() => import('./pages/NotFoundPage/NotFoundPage'));

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

      <React.Suspense
        fallback={
          <div className="mx-auto flex min-h-[40vh] max-w-7xl items-center justify-center px-4 text-sm text-slate-300">
            Loading page...
          </div>
        }
      >
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/editor/:flowId?" element={<EditorPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/applets" element={<AppletLibraryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </React.Suspense>
    </div>
  );
};

const App: React.FC = () => {
  const loadSettings = useSettingsStore((state) => state.loadSettings);
  const darkMode = useSettingsStore((state) => state.darkMode);

  useEffect(() => {
    webSocketService.connect();

    return () => {
      webSocketService.disconnect();
    };
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  return (
    <Router>
      <AppRoutes />
    </Router>
  );
};

export default App;
