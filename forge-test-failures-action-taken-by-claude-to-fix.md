   11 -  CodeSuggestionResponse                                
       12 -} from '../types';                                      
       13 -                                                        
       14 -class ApiService {                                      
       15 -  private api: AxiosInstance;                           
       16 -                                                        
       17 -  constructor() {                                       
       18 -    const baseURL = process.env.REACT_APP_API_URL || 'htt
          -p://localhost:8000';                                    
       19 -                                                        
       20 -    this.api = axios.create({                           
       21 -      baseURL,                                          
       22 -      timeout: 30000,                                   
       23 -      headers: {                                        
       24 -        'Content-Type': 'application/json',             
       25 -      },                                                
       26 -    });                                                 
       27 -                                                        
       28 -    // Add response interceptor for error handling      
       29 -    this.api.interceptors.response.use(                 
       30 -      response => response,                             
       31 -      error => {                                        
       32 -        console.error('API Error:', error.response?.data 
          -|| error.message);                                      
       33 -        return Promise.reject(error);                   
       34 -      }                                                 
       35 -    );                                                  
       36 -  }                                                     
       37 -                                                        
       38 -  /**                                                   
       39 -   * Get all available applets                          
       40 -   */                                                   
       41 -  public async getApplets(): Promise<AppletMetadata[]> {
       42 -    const response = await this.api.get('/applets');    
       43 -    return response.data;                               
       44 -  }                                                     
       45 -                                                        
       46 -  /**                                                   
       47 -   * Get all flows                                      
       48 -   */                                                   
       49 -  public async getFlows(): Promise<Flow[]> {            
       50 -    const response = await this.api.get('/flows');      
       51 -    return response.data;                               
       52 -  }                                                     
       53 -                                                        
       54 -  /**                                                   
       55 -   * Get a specific flow                                
       56 -   */                                                   
       57 -  public async getFlow(flowId: string): Promise<Flow> { 
       58 -    const response = await this.api.get(`/flows/${flowId}
          -`);                                                     
       59 -    return response.data;                               
       60 -  }                                                     
       61 -                                                        
       62 -  /**                                                   
       63 -   * Create or update a flow                            
       64 -   */                                                   
       65 -  public async saveFlow(flow: Flow): Promise<{ id: string
          - }> {                                                   
       66 -    const response = await this.api.post('/flows', flow);
          -                                                         
       67 -    return response.data;                               
       68 -  }                                                     
       69 -                                                        
       70 -  /**                                                   
       71 -   * Delete a flow                                      
       72 -   */                                                   
       73 -  public async deleteFlow(flowId: string): Promise<void> 
          -{                                                       
       74 -    await this.api.delete(`/flows/${flowId}`);          
       75 -  }                                                     
       76 -                                                        
       77 -  /**                                                   
       78 -   * Run a flow with the given input data               
       79 -   */                                                   
       80 -  public async runFlow(flowId: string, inputData: Record<
          -string, any>): Promise<{ run_id: string }> {            
       81 -    const response = await this.api.post(`/flows/${flowId
          -}/run`, inputData);                                     
       82 -    return response.data;                               
       83 -  }                                                     
       84 -                                                        
       85 -  /**                                                   
       86 -   * Get all workflow runs                              
       87 -   */                                                   
       88 -  public async getRuns(): Promise<WorkflowRunStatus[]> {
       89 -    const response = await this.api.get('/runs');       
       90 -    return response.data;                               
       91 -  }                                                     
       92 -                                                        
       93 -  /**                                                   
       94 -   * Get a specific workflow run                        
       95 -   */                                                   
       96 -  public async getRun(runId: string): Promise<WorkflowRun
          -Status> {                                               
       97 -    const response = await this.api.get(`/runs/${runId}`)
          -;                                                       
       98 -    return response.data;                               
       99 -  }                                                     
      100 -                                                        
      101 -  /**                                                   
      102 -   * Get AI code suggestions                            
      103 -   */                                                   
      104 -  public async getCodeSuggestion(request: CodeSuggestionR
          -equest): Promise<CodeSuggestionResponse> {              
      105 -    const response = await this.api.post('/ai/suggest', r
          -equest);                                                
      106 -    return response.data;                               
      107 -  }                                                     
      108 -}                                                       
      109 -                                                        
      110 -// Create a singleton instance                          
      111 -export const apiService = new ApiService();             
      112 -export default apiService;                              
        1 +/**                                                      
        2 + * ApiService - Handles HTTP communication with the backe
          +nd                                                       
        3 + */                                                      
        4 +import axios, { AxiosInstance, AxiosRequestConfig, Intern
          +alAxiosRequestConfig } from 'axios';                     
        5 +import {                                                 
        6 +  Flow,                                                  
        7 +  FlowTemplate,                                          
        8 +  AppletMetadata,                                        
        9 +  WorkflowRunStatus,                                     
       10 +  CodeSuggestionRequest,                                 
       11 +  CodeSuggestionResponse                                 
       12 +} from '../types';                                       
       13 +                                                         
       14 +// ── Token refresh queue ───────────────────────────────
          +─────────────────                                        
       15 +// Prevents multiple concurrent refresh calls when severa
          +l 401s fire at once.                                     
       16 +let isRefreshing = false;                                
       17 +let refreshSubscribers: Array<(token: string) => void> = 
          +[];                                                      
       18 +                                                         
       19 +function onRefreshed(token: string) {                    
       20 +  refreshSubscribers.forEach((cb) => cb(token));         
       21 +  refreshSubscribers = [];                               
       22 +}                                                        
       23 +                                                         
       24 +function addRefreshSubscriber(cb: (token: string) => void
          +) {                                                      
       25 +  refreshSubscribers.push(cb);                           
       26 +}                                                        
       27 +                                                         
       28 +class ApiService {                                       
       29 +  private api: AxiosInstance;                            
       30 +                                                         
       31 +  constructor() {                                        
       32 +    const baseURL = process.env.REACT_APP_API_URL || 'htt
          +p://localhost:8000';                                     
       33 +                                                         
       34 +    this.api = axios.create({                            
       35 +      baseURL,                                           
       36 +      timeout: 30000,                                    
       37 +      headers: {                                         
       38 +        'Content-Type': 'application/json',              
       39 +      },                                                 
       40 +    });                                                  
       41 +                                                         
       42 +    // ── Request interceptor: attach Bearer token ──────
          +────────────────                                         
       43 +    this.api.interceptors.request.use((config: InternalAx
          +iosRequestConfig) => {                                   
       44 +      const token =                                      
       45 +        typeof window !== 'undefined' ? window.localStora
          +ge.getItem('access_token') : null;                       
       46 +      if (token && config.headers) {                     
       47 +        config.headers.Authorization = `Bearer ${token}`;
       48 +      }                                                  
       49 +      return config;                                     
       50 +    });                                                  
       51 +                                                         
       52 +    // ── Response interceptor: auto-refresh on 401 ─────
          +────────────────                                         
       53 +    this.api.interceptors.response.use(                  
       54 +      (response) => response,                            
       55 +      async (error) => {                                 
       56 +        const originalRequest = error.config as InternalA
          +xiosRequestConfig & { _retry?: boolean };                
       57 +                                                         
       58 +        if (error.response?.status === 401 && !originalRe
          +quest._retry) {                                          
       59 +          originalRequest._retry = true;                 
       60 +                                                         
       61 +          if (isRefreshing) {                            
       62 +            // Another refresh is in-flight – queue this 
          +request                                                  
       63 +            return new Promise((resolve) => {            
       64 +              addRefreshSubscriber((newToken: string) => 
          +{                                                        
       65 +                if (originalRequest.headers) {           
       66 +                  originalRequest.headers.Authorization =
          + `Bearer ${newToken}`;                                   
       67 +                }                                        
       68 +                resolve(this.api(originalRequest));      
       69 +              });                                        
       70 +            });                                          
       71 +          }                                              
       72 +                                                         
       73 +          isRefreshing = true;                           
       74 +                                                         
       75 +          try {                                          
       76 +            // Dynamic import to avoid circular dependenc
          +y with AuthService                                       
       77 +            const { authService } = await import('./AuthS
          +ervice');                                                
       78 +            const refreshToken =                         
       79 +              typeof window !== 'undefined'              
       80 +                ? window.localStorage.getItem('refresh_to
          +ken')                                                    
       81 +                : null;                                  
       82 +                                                         
       83 +            if (!refreshToken) {                         
       84 +              throw new Error('No refresh token');       
       85 +            }                                            
       86 +                                                         
       87 +            const tokens = await authService.refresh(refr
          +eshToken);                                               
       88 +                                                         
       89 +            // Persist new tokens                        
       90 +            window.localStorage.setItem('access_token', t
          +okens.access_token);                                     
       91 +            window.localStorage.setItem('refresh_token', 
          +tokens.refresh_token);                                   
       92 +                                                         
       93 +            isRefreshing = false;                        
       94 +            onRefreshed(tokens.access_token);            
       95 +                                                         
       96 +            // Retry the original request with the new to
          +ken                                                      
       97 +            if (originalRequest.headers) {               
       98 +              originalRequest.headers.Authorization = `Be
          +arer ${tokens.access_token}`;                            
       99 +            }                                            
      100 +            return this.api(originalRequest);            
      101 +          } catch (refreshError) {                       
      102 +            isRefreshing = false;                        
      103 +            refreshSubscribers = [];                     
      104 +                                                         
      105 +            // Clear auth state – user must log in again 
      106 +            window.localStorage.removeItem('access_token'
          +);                                                       
      107 +            window.localStorage.removeItem('refresh_token
          +');                                                      
      108 +            window.localStorage.removeItem('auth_user'); 
      109 +                                                         
      110 +            // Redirect to login (only in browser)       
      111 +            if (typeof window !== 'undefined' && window.l
          +ocation.pathname !== '/login') {                         
      112 +              window.location.href = '/login';           
      113 +            }                                            
      114 +                                                         
      115 +            return Promise.reject(refreshError);         
      116 +          }                                              
      117 +        }                                                
      118 +                                                         
      119 +        console.error('API Error:', error.response?.data 
          +|| error.message);                                       
      120 +        return Promise.reject(error);                    
      121 +      },                                                 
      122 +    );                                                   
      123 +  }                                                      
      124 +                                                         
      125 +  /**                                                    
      126 +   * Get all available applets                           
      127 +   */                                                    
      128 +  public async getApplets(): Promise<AppletMetadata[]> { 
      129 +    const response = await this.api.get('/applets');     
      130 +    return response.data;                                
      131 +  }                                                      
      132 +                                                         
      133 +  /**                                                    
      134 +   * Get all flows                                       
      135 +   */                                                    
      136 +  public async getFlows(): Promise<Flow[]> {             
      137 +    const response = await this.api.get('/flows');       
      138 +    return response.data;                                
      139 +  }                                                      
      140 +                                                         
      141 +  /**                                                    
      142 +   * Get a specific flow                                 
      143 +   */                                                    
      144 +  public async getFlow(flowId: string): Promise<Flow> {  
      145 +    const response = await this.api.get(`/flows/${flowId}
          +`);                                                      
      146 +    return response.data;                                
      147 +  }                                                      
      148 +                                                         
      149 +  /**                                                    
      150 +   * Create or update a flow                             
      151 +   */                                                    
      152 +  public async saveFlow(flow: Flow): Promise<{ id: string
          + }> {                                                    
      153 +    const response = await this.api.post('/flows', flow);
      154 +    return response.data;                                
      155 +  }                                                      
      156 +                                                         
      157 +  /**                                                    
      158 +   * Delete a flow                                       
      159 +   */                                                    
      160 +  public async deleteFlow(flowId: string): Promise<void> 
          +{                                                        
      161 +    await this.api.delete(`/flows/${flowId}`);           
      162 +  }                                                      
      163 +                                                         
      164 +  /**                                                    
      165 +   * Run a flow with the given input data                
      166 +   */                                                    
      167 +  public async runFlow(flowId: string, inputData: Record<
          +string, any>): Promise<{ run_id: string }> {             
      168 +    const response = await this.api.post(`/flows/${flowId
          +}/run`, inputData);                                      
      169 +    return response.data;                                
      170 +  }                                                      
      171 +                                                         
      172 +  /**                                                    
      173 +   * Get all workflow runs                               
      174 +   */                                                    
      175 +  public async getRuns(): Promise<WorkflowRunStatus[]> { 
      176 +    const response = await this.api.get('/runs');        
      177 +    return response.data;                                
      178 +  }                                                      
      179 +                                                         
      180 +  /**                                                    
      181 +   * Get a specific workflow run                         
      182 +   */                                                    
      183 +  public async getRun(runId: string): Promise<WorkflowRun
          +Status> {                                                
      184 +    const response = await this.api.get(`/runs/${runId}`)
          +;                                                        
      185 +    return response.data;                                
      186 +  }                                                      
      187 +                                                         
      188 +  /**                                                    
      189 +   * Get AI code suggestions                             
      190 +   */                                                    
      191 +  public async getCodeSuggestion(request: CodeSuggestionR
          +equest): Promise<CodeSuggestionResponse> {               
      192 +    const response = await this.api.post('/ai/suggest', r
          +equest);                                                 
      193 +    return response.data;                                
      194 +  }                                                      
      195 +}                                                        
      196 +                                                         
      197 +// Create a singleton instance                           
      198 +export const apiService = new ApiService();              
      199 +export default apiService;                               

● Step 5: WebSocket auth

● Update(apps/web-frontend/src/services/WebSocketService.ts)
  ⎿  Added 7 lines, removed 1 line
      33      this.socket = new WebSocket(wsUrl);
      34  
      35      this.socket.onopen = () => {
      36 -                                                          
      36        this.isConnected = true;
      37        if (this.reconnectTimer) {
      38          clearTimeout(this.reconnectTimer);
      39          this.reconnectTimer = null;
      40        }
      41 +                                                          
      42 +      // Send auth token if available                     
      43 +      const token =                                       
      44 +        typeof window !== 'undefined' ? window.localStorag
         +e.getItem('access_token') : null;                         
      45 +      if (token && this.socket) {                         
      46 +        this.socket.send(JSON.stringify({ type: 'auth', to
         +ken }));                                                  
      47 +      }                                                   
      48      };
      49  
      50      this.socket.onmessage = (event) => {

● Step 6 & 7: Login and Register pages

  Let me check the existing page structure for conventions.

● Read 1 file (ctrl+o to expand)

● Bash(mkdir -p
      /home/axw/projects/synapps/apps/web-frontend/src/pages/LoginPage
      /home/axw/projects/synapps/apps/web-frontend/src/pages/RegisterPage)
  ⎿  Done

● Write(apps/web-frontend/src/pages/LoginPage/LoginPage.tsx)
  ⎿  Wrote 88 lines to
     apps/web-frontend/src/pages/LoginPage/LoginPage.tsx
      1 import React, { useState } from 'react';
      2 import { Link, useNavigate } from 'react-router-dom';
      3 import { authService } from '../../services/AuthService';
      4 import { useAuthStore } from '../../stores/authStore';
      5 import './LoginPage.css';
      6
      7 const LoginPage: React.FC = () => {
      8   const navigate = useNavigate();
      9   const setAuth = useAuthStore((s) => s.setAuth);
     … +78 lines (ctrl+o to expand)

● Write(apps/web-frontend/src/pages/LoginPage/LoginPage.css)
  ⎿  Wrote 127 lines to
     apps/web-frontend/src/pages/LoginPage/LoginPage.css
       1 /* Shared auth page styles (LoginPage + RegisterPage) */
       2
       3 .auth-page {
       4   min-height: 100vh;
       5   display: flex;
       6   align-items: center;
       7   justify-content: center;
       8   background-color: #0b1120;
       9   padding: 24px;
      10 }
     … +117 lines (ctrl+o to expand)

● Write(apps/web-frontend/src/pages/RegisterPage/RegisterPage.tsx)
  ⎿  Wrote 116 lines to
     apps/web-frontend/src/pages/RegisterPage/RegisterPage.tsx
      1 import React, { useState } from 'react';
      2 import { Link, useNavigate } from 'react-router-dom';
      3 import { authService } from '../../services/AuthService';
      4 import { useAuthStore } from '../../stores/authStore';
      5 import '../LoginPage/LoginPage.css'; // reuse shared auth st
        yles
      6
      7 const RegisterPage: React.FC = () => {
      8   const navigate = useNavigate();
      9   const setAuth = useAuthStore((s) => s.setAuth);
     … +106 lines (ctrl+o to expand)

● Step 8: ProtectedRoute and GuestRoute

● Write(apps/web-frontend/src/components/ProtectedRoute.tsx)
  ⎿  Wrote 19 lines to
     apps/web-frontend/src/components/ProtectedRoute.tsx
       1 import React from 'react';
       2 import { Navigate, useLocation } from 'react-router-dom';
       3 import { useAuthStore } from '../stores/authStore';
       4
       5 const ProtectedRoute: React.FC<{ children: React.ReactNode 
         }> = ({ children }) => {
       6   const isAuthenticated = useAuthStore((s) => s.isAuthentic
         ated);
       7   const isLoading = useAuthStore((s) => s.isLoading);
       8   const location = useLocation();
       9
      10   if (isLoading) return null; // wait for loadAuth to hydra
         te
     … +9 lines (ctrl+o to expand)

● Write(apps/web-frontend/src/components/GuestRoute.tsx)
  ⎿  Wrote 18 lines to apps/web-frontend/src/components/GuestRoute.tsx
      1 import React from 'react';
      2 import { Navigate } from 'react-router-dom';
      3 import { useAuthStore } from '../stores/authStore';
      4
      5 const GuestRoute: React.FC<{ children: React.ReactNode }> = 
        ({ children }) => {
      6   const isAuthenticated = useAuthStore((s) => s.isAuthentica
        ted);
      7   const isLoading = useAuthStore((s) => s.isLoading);
      8
      9   if (isLoading) return null;
     … +8 lines (ctrl+o to expand)

● Step 9: Update App.tsx

● Write(apps/web-frontend/src/App.tsx)
  ⎿  Added 31 lines, removed 11 lines
       6  import webSocketService from './services/WebSocketService'
          ;
       7  import { Button } from './components/ui/button';
       8  import { useSettingsStore } from './stores/settingsStore';
       9 +import { useAuthStore } from './stores/authStore';        
      10 +import ProtectedRoute from './components/ProtectedRoute'; 
      11 +import GuestRoute from './components/GuestRoute';         
      12  
      13  const DashboardPage = React.lazy(() => import('./pages/Das
          hboardPage/DashboardPage'));
      14  const EditorPage = React.lazy(() => import('./pages/Editor
          Page/EditorPage'));
     ...
      16  const AppletLibraryPage = React.lazy(() => import('./pages
          /AppletLibraryPage/AppletLibraryPage'));
      17  const SettingsPage = React.lazy(() => import('./pages/Sett
          ingsPage/SettingsPage'));
      18  const NotFoundPage = React.lazy(() => import('./pages/NotF
          oundPage/NotFoundPage'));
      19 +const LoginPage = React.lazy(() => import('./pages/LoginPa
         +ge/LoginPage'));                                          
      20 +const RegisterPage = React.lazy(() => import('./pages/Regi
         +sterPage/RegisterPage'));                                 
      21  
      22  const AppRoutes: React.FC = () => {
      23    const location = useLocation();
      19 -  const showShortcut = location.pathname !== '/dashboard';
      24 +  const isAuthenticated = useAuthStore((s) => s.isAuthenti
         +cated);                                                   
      25 +  const showShortcut = isAuthenticated && location.pathnam
         +e !== '/dashboard';                                       
      26  
      27    return (
      28      <div className="min-h-screen bg-slate-950 text-slate-1
          00">
     ...
      44          }
      45        >
      46          <Routes>
      47 +          {/* Guest-only routes */}                       
      48 +          <Route path="/login" element={<GuestRoute><Login
         +Page /></GuestRoute>} />                                  
      49 +          <Route path="/register" element={<GuestRoute><Re
         +gisterPage /></GuestRoute>} />                            
      50 +                                                          
      51 +          {/* Protected routes */}                        
      52            <Route path="/" element={<Navigate to="/dashboar
          d" replace />} />
      42 -          <Route path="/dashboard" element={<DashboardPage
         - />} />                                                   
      43 -          <Route path="/editor/:flowId?" element={<EditorP
         -age />} />                                                
      44 -          <Route path="/history" element={<HistoryPage />}
         - />                                                       
      45 -          <Route path="/applets" element={<AppletLibraryPa
         -ge />} />                                                 
      46 -          <Route path="/settings" element={<SettingsPage /
         ->} />                                                     
      53 +          <Route path="/dashboard" element={<ProtectedRout
         +e><DashboardPage /></ProtectedRoute>} />                  
      54 +          <Route path="/editor/:flowId?" element={<Protect
         +edRoute><EditorPage /></ProtectedRoute>} />               
      55 +          <Route path="/history" element={<ProtectedRoute>
         +<HistoryPage /></ProtectedRoute>} />                      
      56 +          <Route path="/applets" element={<ProtectedRoute>
         +<AppletLibraryPage /></ProtectedRoute>} />                
      57 +          <Route path="/settings" element={<ProtectedRoute
         +><SettingsPage /></ProtectedRoute>} />                    
      58            <Route path="*" element={<NotFoundPage />} />
      59          </Routes>
      60        </React.Suspense>
     ...
      65  const App: React.FC = () => {
      66    const loadSettings = useSettingsStore((state) => state.l
          oadSettings);
      67    const darkMode = useSettingsStore((state) => state.darkM
          ode);
      68 +  const loadAuth = useAuthStore((s) => s.loadAuth);       
      69 +  const isAuthenticated = useAuthStore((s) => s.isAuthenti
         +cated);                                                   
      70  
      71 +  // Hydrate auth state from localStorage (synchronous – n
         +o flash)                                                  
      72    useEffect(() => {
      59 -    webSocketService.connect();                           
      73 +    loadAuth();                                           
      74 +  }, [loadAuth]);                                         
      75  
      61 -    return () => {                                        
      62 -      webSocketService.disconnect();                      
      63 -    };                                                    
      64 -  }, []);                                                 
      76 +  // Only connect WebSocket when authenticated            
      77 +  useEffect(() => {                                       
      78 +    if (isAuthenticated) {                                
      79 +      webSocketService.connect();                         
      80 +      return () => {                                      
      81 +        webSocketService.disconnect();                    
      82 +      };                                                  
      83 +    }                                                     
      84 +  }, [isAuthenticated]);                                  
      85  
      86    useEffect(() => {
      87      loadSettings();

● Step 10: Update MainLayout with user info and logout

● Update(apps/web-frontend/src/components/Layout/MainLayout.tsx)
  ⎿  Added 4 lines, removed 1 line
       3   * Provides consistent layout with sidebar navigation and
          header
       4   */
       5  import React, { ReactNode } from 'react';
       6 -import { Link, useLocation } from 'react-router-dom';     
       6 +import { Link, useLocation, useNavigate } from 'react-rout
         +er-dom';                                                  
       7  import NotificationCenter from '../Notifications/Notificat
          ionCenter';
       8 +import { useAuthStore } from '../../stores/authStore';    
       9 +import { authService } from '../../services/AuthService'; 
      10 +import webSocketService from '../../services/WebSocketServ
         +ice';                                                     
      11  import './MainLayout.css';
      12  
      13  interface MainLayoutProps {

● Update(apps/web-frontend/src/components/Layout/MainLayout.tsx)
  ⎿  Added 21 lines
      18  
      19  const MainLayout: React.FC<MainLayoutProps> = ({ children,
           title, actions }) => {
      20    const location = useLocation();
      21 +  const navigate = useNavigate();                         
      22 +  const user = useAuthStore((s) => s.user);               
      23 +  const clearAuth = useAuthStore((s) => s.clearAuth);     
      24 +                                                          
      25 +  const handleLogout = async () => {                      
      26 +    const refreshToken =                                  
      27 +      typeof window !== 'undefined' ? window.localStorage.
         +getItem('refresh_token') : null;                          
      28 +                                                          
      29 +    // Clear client-side auth immediately (resilient: even
         + if server call fails)                                    
      30 +    webSocketService.disconnect();                        
      31 +    clearAuth();                                          
      32 +    navigate('/login', { replace: true });                
      33 +                                                          
      34 +    if (refreshToken) {                                   
      35 +      try {                                               
      36 +        await authService.logout(refreshToken);           
      37 +      } catch {                                           
      38 +        // Server-side revocation failed – tokens already 
         +cleared locally                                           
      39 +      }                                                   
      40 +    }                                                     
      41 +  };                                                      
      42  
      43    // Navigation items
      44    const navItems = [

● Now add the user info + logout button in the sidebar, above version-info.

● Update(apps/web-frontend/src/components/Layout/MainLayout.tsx)
  ⎿  Added 9 lines
      70            ))}
      71          </nav>
      72  
      73 +        {user && (                                        
      74 +          <div className="sidebar-user">                  
      75 +            <span className="sidebar-user-email" title={us
         +er.email}>{user.email}</span>                             
      76 +            <button className="sidebar-logout-btn" onClick
         +={handleLogout}>                                          
      77 +              Sign out                                    
      78 +            </button>                                     
      79 +          </div>                                          
      80 +        )}                                                
      81 +                                                          
      82          <div className="version-info">
      83            <span><a href="https://github.com/nxtg-ai/SynApp
          s-v0.4.0" target="_blank" rel="noopener noreferrer">SynApp
          s-v0.4.0 GitHub</a></span>
      84          </div>

● Now add the CSS for the user section.

● Update(apps/web-frontend/src/components/Layout/MainLayout.css)
  ⎿  Added 33 lines
       64    font-size: 1.1rem;
       65  }
       66  
       67 +.sidebar-user {                                          
       68 +  padding: 12px 24px;                                    
       69 +  border-top: 1px solid rgba(255, 255, 255, 0.08);       
       70 +  display: flex;                                         
       71 +  flex-direction: column;                                
       72 +  gap: 8px;                                              
       73 +}                                                        
       74 +                                                         
       75 +.sidebar-user-email {                                    
       76 +  font-size: 0.8rem;                                     
       77 +  color: rgba(255, 255, 255, 0.65);                      
       78 +  overflow: hidden;                                      
       79 +  text-overflow: ellipsis;                               
       80 +  white-space: nowrap;                                   
       81 +}                                                        
       82 +                                                         
       83 +.sidebar-logout-btn {                                    
       84 +  background: none;                                      
       85 +  border: 1px solid rgba(255, 255, 255, 0.15);           
       86 +  color: rgba(255, 255, 255, 0.65);                      
       87 +  font-size: 0.8rem;                                     
       88 +  padding: 6px 12px;                                     
       89 +  border-radius: 6px;                                    
       90 +  cursor: pointer;                                       
       91 +  transition: all 0.2s;                                  
       92 +}                                                        
       93 +                                                         
       94 +.sidebar-logout-btn:hover {                              
       95 +  color: white;                                          
       96 +  border-color: rgba(255, 255, 255, 0.35);               
       97 +  background-color: rgba(255, 255, 255, 0.08);           
       98 +}                                                        
       99 +                                                         
      100  .version-info {
      101    padding: 16px 24px;
      102    font-size: 0.8rem;

● Step 11: Verify - typecheck and tests

● Bash(cd /home/axw/projects/synapps/apps/web-frontend && npx tsc --noEmit
      2>&1 | head -80)
  ⎿  src/services/WebSocketService.test.ts(81,64): error TS7031: Bindi
     ng element 'eventName' implicitly has an 'any' type.
     src/stores/executionStore.test.ts(40,11): error TS2322: Type 'str
     … +29 lines (ctrl+o to expand)

● All these errors are pre-existing (in test files, not related to my
  changes). Let me confirm by checking if they reference any auth-related
  files.

● Bash(npx tsc --noEmit 2>&1 | grep -i
      "auth\|login\|register\|Protected\|Guest\|authStore" || echo "No
      auth-related type errors")
  ⎿  No auth-related type errors

● No type errors from the auth changes. Now run the tests.

● Bash(npx vitest run 2>&1 | tail -40)
  ⎿   ❯ src/services/ApiService.test.ts:156:24

     ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯[8/10]⎯
     … +35 lines (ctrl+o to expand)

● The existing ApiService tests mock axios.create but the mock doesn't
  include interceptors.request. Let me read the test file to understand the
   mock setup.

● Read 1 file (ctrl+o to expand)

● The mock instance is missing interceptors.request. I need to add it to
  the mock.

● Update(src/services/ApiService.test.ts)
  ⎿  Added 5 lines
      21    const del = vi.fn();
      22    const use = vi.fn();
      23  
      24 +  const requestUse = vi.fn();                             
      25 +                                                          
      26    const instance = {
      27      get,
      28      post,
      29      delete: del,
      30      interceptors: {
      31 +      request: {                                          
      32 +        use: requestUse,                                  
      33 +      },                                                  
      34        response: {
      35          use,
      36        },

● Now I also need to update the assertion that checks interceptor call
  count since we now have 2 interceptor registrations (1 request + 1
  response).

● Update(src/services/ApiService.test.ts)
  ⎿  Added 1 line
      78        },
      79      });
      80      expect(mockInterceptorUse).toHaveBeenCalledTimes(1);
      81 +    expect(requestUse).toHaveBeenCalledTimes(1);          
      82    });
      83  
      84    it('uses REACT_APP_API_URL when provided', async () => {

● Wait, I also need to add requestUse to the return object so it's
  available in the test scope. Let me check...

● Read 1 file (ctrl+o to expand)

● I need to export requestUse from the hoisted block too.

● Update(src/services/ApiService.test.ts)
  ⎿  Added 1 line
      13    mockPost,
      14    mockDelete,
      15    mockInterceptorUse,
      16 +  mockRequestInterceptorUse,                              
      17    mockAxiosInstance,
      18  } = vi.hoisted(() => {
      19    const create = vi.fn();

● Update(src/services/ApiService.test.ts)
  ⎿  Added 1 line
      44      mockPost: post,
      45      mockDelete: del,
      46      mockInterceptorUse: use,
      47 +    mockRequestInterceptorUse: requestUse,                
      48      mockAxiosInstance: instance,
      49    };
      50  });

● Now update the assertion to use the correct variable name.

● Update(src/services/ApiService.test.ts)
  ⎿  Added 1 line, removed 1 line
      80        },
      81      });
      82      expect(mockInterceptorUse).toHaveBeenCalledTimes(1);
      83 -    expect(requestUse).toHaveBeenCalledTimes(1);          
      83 +    expect(mockRequestInterceptorUse).toHaveBeenCalledTime
         +s(1);                                                     
      84    });
      85  
      86    it('uses REACT_APP_API_URL when provided', async () => {

● Now let me also update the test for the error interceptor since the
  response interceptor call index may have changed. Let me check the test
  for that.

● Read 1 file (ctrl+o to expand)

● The test at line 102 gets the first call to mockInterceptorUse (the
  response interceptor). Since we only have one response.use call, calls[0]
   is still correct. The response interceptor is still called exactly once
  on interceptors.response.use. The request interceptor is on a separate
  mock (mockRequestInterceptorUse). This should be fine.

  However, the response interceptor's error handler now has more complex
  logic (401 refresh). The test expects a simple rethrow, but now the 401
  path does different things. Let me check what error the test passes — it
  doesn't have status: 401 or a config object, so it should fall through to
   the generic error handler path.

● Read 1 file (ctrl+o to expand)

● The error mock has no status on response, so error.response?.status ===
  401 is undefined === 401 → false. It will go straight to the
  console.error + reject path. This should still pass.

  Let me run the tests again.

● Bash(npx vitest run 2>&1 | tail -30)
  ⎿  When testing, code that causes React state updates should be wrap
     ped into act(...):
     … +21 lines (ctrl+o to expand)

● All 101 tests pass, 13 test files all green.

● All done. Here's a summary of what was implemented:

  New files (8):
  - src/types/index.ts — Added AuthTokenResponse, UserProfile,
  AuthCredentials interfaces
  - src/services/AuthService.ts — Singleton with register, login, refresh,
  logout, getMe (own axios instance, no auth interceptors)
  - src/stores/authStore.ts — Zustand store with loadAuth, setAuth,
  clearAuth + localStorage persistence
  - src/pages/LoginPage/LoginPage.tsx + .css — Login form with dark theme
  - src/pages/RegisterPage/RegisterPage.tsx — Register form with confirm
  password validation (reuses LoginPage.css)
  - src/components/ProtectedRoute.tsx — Redirects to /login when
  unauthenticated
  - src/components/GuestRoute.tsx — Redirects to /dashboard when already
  authenticated

  Modified files (5):
  - src/services/ApiService.ts — Request interceptor (Bearer token from
  localStorage) + response interceptor (401 auto-refresh with retry queue,
  redirect to /login on failure)
  - src/services/ApiService.test.ts — Added request.use mock to match new
  interceptor setup
  - src/services/WebSocketService.ts — Sends {"type":"auth","token":"..."}
  on socket open
  - src/App.tsx — Auth hydration on mount, login/register routes
  (GuestRoute), protected existing routes (ProtectedRoute), conditional WS
  connect
  - src/components/Layout/MainLayout.tsx + .css — User email display +
  sign-out button in sidebar

  Verification: 101 tests pass, no auth-related type errors.