import { create } from 'zustand';
import { AuthTokenResponse, UserProfile } from '../types';

interface AuthStoreState {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  /** Hydrate state from localStorage (synchronous, no network). */
  loadAuth: () => void;

  /** Persist tokens + user after a successful login/register. */
  setAuth: (tokens: AuthTokenResponse, user: UserProfile) => void;

  /** Clear all auth state and localStorage keys. */
  clearAuth: () => void;
}

const safeGetItem = (key: string): string | null => {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(key);
};

const safeSetItem = (key: string, value: string): void => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(key, value);
};

const safeRemoveItem = (key: string): void => {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(key);
};

export const useAuthStore = create<AuthStoreState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,

  loadAuth: () => {
    const accessToken = safeGetItem('access_token');
    const userJson = safeGetItem('auth_user');

    if (accessToken && userJson) {
      try {
        const user = JSON.parse(userJson) as UserProfile;
        set({ user, isAuthenticated: true, isLoading: false });
        return;
      } catch {
        // corrupt data – fall through to unauthenticated
      }
    }
    set({ user: null, isAuthenticated: false, isLoading: false });
  },

  setAuth: (tokens: AuthTokenResponse, user: UserProfile) => {
    safeSetItem('access_token', tokens.access_token);
    safeSetItem('refresh_token', tokens.refresh_token);
    safeSetItem('auth_user', JSON.stringify(user));
    set({ user, isAuthenticated: true, isLoading: false });
  },

  clearAuth: () => {
    safeRemoveItem('access_token');
    safeRemoveItem('refresh_token');
    safeRemoveItem('auth_user');
    set({ user: null, isAuthenticated: false, isLoading: false });
  },
}));
