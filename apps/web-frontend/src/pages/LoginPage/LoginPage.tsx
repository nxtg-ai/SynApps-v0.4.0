import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authService } from '../../services/AuthService';
import { useAuthStore } from '../../stores/authStore';
import './LoginPage.css';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const tokens = await authService.login({ email, password });
      const user = await authService.getMe(tokens.access_token);
      setAuth(tokens, user);
      navigate('/dashboard', { replace: true });
    } catch (err: any) {
      const msg = err.response?.data?.detail;
      setError(typeof msg === 'string' ? msg : 'Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <img src="/logo50.png" alt="SynApps" className="auth-logo-img" />
          <span className="auth-logo-text">SynApps</span>
        </div>

        <h1 className="auth-title">Sign in</h1>

        {error && <div className="auth-error">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <label className="auth-label">
            Email
            <input
              type="email"
              className="auth-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
              autoComplete="email"
            />
          </label>

          <label className="auth-label">
            Password
            <input
              type="password"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </label>

          <button type="submit" className="auth-button" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <p className="auth-footer">
          Don&apos;t have an account?{' '}
          <Link to="/register" className="auth-link">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
