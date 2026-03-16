/**
 * AuthService - Handles authentication API calls.
 * Uses its own bare axios instance (no auth interceptors) to avoid circular deps.
 */
import axios, { AxiosInstance } from 'axios';
import { AuthCredentials, AuthTokenResponse, UserProfile } from '../types';

class AuthService {
  private api: AxiosInstance;

  constructor() {
    const baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    this.api = axios.create({
      baseURL,
      timeout: 15000,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async register(credentials: AuthCredentials): Promise<AuthTokenResponse> {
    const response = await this.api.post<AuthTokenResponse>('/auth/register', credentials);
    return response.data;
  }

  async login(credentials: AuthCredentials): Promise<AuthTokenResponse> {
    const response = await this.api.post<AuthTokenResponse>('/auth/login', credentials);
    return response.data;
  }

  async refresh(refreshToken: string): Promise<AuthTokenResponse> {
    const response = await this.api.post<AuthTokenResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    });
    return response.data;
  }

  async logout(refreshToken: string): Promise<void> {
    await this.api.post('/auth/logout', { refresh_token: refreshToken });
  }

  async getMe(accessToken: string): Promise<UserProfile> {
    const response = await this.api.get<UserProfile>('/auth/me', {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return response.data;
  }
}

export const authService = new AuthService();
export default authService;
