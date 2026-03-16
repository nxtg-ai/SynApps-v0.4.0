/**
 * MainLayout component
 * Provides consistent layout with sidebar navigation and header
 */
import React, { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import NotificationCenter from '../Notifications/NotificationCenter';
import { useAuthStore } from '../../stores/authStore';
import { authService } from '../../services/AuthService';
import webSocketService from '../../services/WebSocketService';
import './MainLayout.css';

interface MainLayoutProps {
  children: ReactNode;
  title: string;
  actions?: ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children, title, actions }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const clearAuth = useAuthStore((s) => s.clearAuth);

  const handleLogout = async () => {
    const refreshToken =
      typeof window !== 'undefined' ? window.localStorage.getItem('refresh_token') : null;

    // Clear client-side auth immediately (resilient: even if server call fails)
    webSocketService.disconnect();
    clearAuth();
    navigate('/login', { replace: true });

    if (refreshToken) {
      try {
        await authService.logout(refreshToken);
      } catch {
        // Server-side revocation failed – tokens already cleared locally
      }
    }
  };
  
  // Navigation items
  const navItems = [
    { path: '/dashboard', icon: '🏠', label: 'Dashboard' },
    { path: '/editor', icon: '🔄', label: 'Workflow Editor' },
    { path: '/history', icon: '📋', label: 'Run History' },
    { path: '/applets', icon: '🧩', label: 'Applet Library' },
    { path: '/settings', icon: '⚙️', label: 'Settings' }
  ];
  
  return (
    <div className="main-layout">
      <aside className="sidebar">
        <div className="logo">
          <img src="logo50.png" alt="Logo" className="logo-icon" />
          <span className="logo-text">SynApps</span>
        </div>
        
        <nav className="nav-menu">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-item ${location.pathname.startsWith(item.path) ? 'active' : ''}`}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </Link>
          ))}
        </nav>
        
        {user && (
          <div className="sidebar-user">
            <span className="sidebar-user-email" title={user.email}>{user.email}</span>
            <button className="sidebar-logout-btn" onClick={handleLogout}>
              Sign out
            </button>
          </div>
        )}

        <div className="version-info">
          <span><a href="https://github.com/nxtg-ai/SynApps-v0.4.0" target="_blank" rel="noopener noreferrer">SynApps v1.0</a></span>
        </div>
      </aside>
      
      <main className="main-content">
        <header className="header">
          <h1 className="page-title">{title}</h1>
          
          <div className="header-actions">
            {actions}
            <NotificationCenter />
          </div>
        </header>
        
        <div className="content">
          {children}
        </div>
      </main>
    </div>
  );
};

export default MainLayout;
