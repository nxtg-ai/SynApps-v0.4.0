/**
 * ApiService - Handles HTTP communication with the backend
 */
import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import {
  Flow,
  AppletMetadata,
  WorkflowRunStatus,
  CodeSuggestionRequest,
  CodeSuggestionResponse
} from '../types';

// ── Token refresh queue ────────────────────────────────────────────────
// Prevents multiple concurrent refresh calls when several 401s fire at once.
let isRefreshing = false;
let refreshSubscribers: Array<(token: string) => void> = [];

function onRefreshed(token: string) {
  refreshSubscribers.forEach((cb) => cb(token));
  refreshSubscribers = [];
}

function addRefreshSubscriber(cb: (token: string) => void) {
  refreshSubscribers.push(cb);
}

class ApiService {
  private api: AxiosInstance;

  constructor() {
    const baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    this.api = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // ── Request interceptor: attach Bearer token ──────────────────────
    this.api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
      const token =
        typeof window !== 'undefined' ? window.localStorage.getItem('access_token') : null;
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // ── Response interceptor: auto-refresh on 401 ─────────────────────
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          if (isRefreshing) {
            // Another refresh is in-flight – queue this request
            return new Promise((resolve) => {
              addRefreshSubscriber((newToken: string) => {
                if (originalRequest.headers) {
                  originalRequest.headers.Authorization = `Bearer ${newToken}`;
                }
                resolve(this.api(originalRequest));
              });
            });
          }

          isRefreshing = true;

          try {
            // Dynamic import to avoid circular dependency with AuthService
            const { authService } = await import('./AuthService');
            const refreshToken =
              typeof window !== 'undefined'
                ? window.localStorage.getItem('refresh_token')
                : null;

            if (!refreshToken) {
              throw new Error('No refresh token');
            }

            const tokens = await authService.refresh(refreshToken);

            // Persist new tokens
            window.localStorage.setItem('access_token', tokens.access_token);
            window.localStorage.setItem('refresh_token', tokens.refresh_token);

            isRefreshing = false;
            onRefreshed(tokens.access_token);

            // Retry the original request with the new token
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`;
            }
            return this.api(originalRequest);
          } catch (refreshError) {
            isRefreshing = false;
            refreshSubscribers = [];

            // Clear auth state – user must log in again
            window.localStorage.removeItem('access_token');
            window.localStorage.removeItem('refresh_token');
            window.localStorage.removeItem('auth_user');

            // Redirect to login (only in browser)
            if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
              window.location.href = '/login';
            }

            return Promise.reject(refreshError);
          }
        }

        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      },
    );
  }

  /**
   * Unwrap a paginated response: { items: T[], total, ... } → T[]
   * Falls back to returning data as-is if it's already an array.
   */
  private unwrapPaginated<T>(data: T[] | { items: T[] }): T[] {
    if (Array.isArray(data)) return data;
    if (data && typeof data === 'object' && 'items' in data) return data.items;
    return [];
  }

  /**
   * Get all available applets
   */
  public async getApplets(): Promise<AppletMetadata[]> {
    const response = await this.api.get('/applets');
    return this.unwrapPaginated(response.data);
  }

  /**
   * Get all flows
   */
  public async getFlows(): Promise<Flow[]> {
    const response = await this.api.get('/flows');
    return this.unwrapPaginated(response.data);
  }

  /**
   * Get a specific flow
   */
  public async getFlow(flowId: string): Promise<Flow> {
    const response = await this.api.get(`/flows/${flowId}`);
    return response.data;
  }

  /**
   * Create or update a flow
   */
  public async saveFlow(flow: Flow): Promise<{ id: string }> {
    const response = await this.api.post('/flows', flow);
    return response.data;
  }

  /**
   * Delete a flow
   */
  public async deleteFlow(flowId: string): Promise<void> {
    await this.api.delete(`/flows/${flowId}`);
  }

  /**
   * Run a flow with the given input data
   */
  public async runFlow(flowId: string, inputData: Record<string, any>): Promise<{ run_id: string }> {
    const response = await this.api.post(`/flows/${flowId}/run`, { input: inputData });
    return response.data;
  }

  /**
   * Get all workflow runs
   */
  public async getRuns(): Promise<WorkflowRunStatus[]> {
    const response = await this.api.get('/runs');
    return this.unwrapPaginated(response.data);
  }

  /**
   * Get a specific workflow run
   */
  public async getRun(runId: string): Promise<WorkflowRunStatus> {
    const response = await this.api.get(`/runs/${runId}`);
    return response.data;
  }

  /**
   * Get AI code suggestions
   */
  public async getCodeSuggestion(request: CodeSuggestionRequest): Promise<CodeSuggestionResponse> {
    const response = await this.api.post('/ai/suggest', request);
    return response.data;
  }

  /**
   * Export a flow as JSON
   */
  public async exportFlow(flowId: string): Promise<any> {
    const response = await this.api.get(`/flows/${flowId}/export`);
    return response.data;
  }

  /**
   * Import a flow from JSON
   */
  public async importFlow(flowData: any): Promise<{ id: string }> {
    const response = await this.api.post('/flows/import', flowData);
    return response.data;
  }
}

// Create a singleton instance
export const apiService = new ApiService();
export default apiService;
