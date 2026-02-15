/**
 * ApiService - Handles HTTP communication with the backend
 */
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { 
  Flow, 
  FlowTemplate, 
  AppletMetadata, 
  WorkflowRunStatus,
  CodeSuggestionRequest,
  CodeSuggestionResponse
} from '../types';

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
    
    // Add response interceptor for error handling
    this.api.interceptors.response.use(
      response => response,
      error => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }
  
  /**
   * Get all available applets
   */
  public async getApplets(): Promise<AppletMetadata[]> {
    const response = await this.api.get('/applets');
    return response.data;
  }
  
  /**
   * Get all flows
   */
  public async getFlows(): Promise<Flow[]> {
    const response = await this.api.get('/flows');
    return response.data;
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
    const response = await this.api.post(`/flows/${flowId}/run`, inputData);
    return response.data;
  }
  
  /**
   * Get all workflow runs
   */
  public async getRuns(): Promise<WorkflowRunStatus[]> {
    const response = await this.api.get('/runs');
    return response.data;
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
}

// Create a singleton instance
export const apiService = new ApiService();
export default apiService;
