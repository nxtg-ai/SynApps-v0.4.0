import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import apiService from './ApiService';
import { Flow, AppletMetadata, WorkflowRunStatus, CodeSuggestionRequest, CodeSuggestionResponse } from '../types';

// Mock the entire axios module
vi.mock('axios');

// Cast axios to a mocked AxiosInstance
const mockedAxios = axios as unknown as { create: ReturnType<typeof vi.fn> };

describe('ApiService', () => {
  let apiInstance: ReturnType<typeof apiService.constructor>;

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset the mock for axios.create for each test
    mockedAxios.create = vi.fn(() => ({
      get: vi.fn(),
      post: vi.fn(),
      delete: vi.fn(),
      interceptors: {
        response: {
          use: vi.fn(),
        },
      },
    }));
    // Re-instantiate apiService to ensure it uses the fresh mock
    apiInstance = new (apiService as any).constructor();
  });

  it('should initialize with the correct base URL and default headers', () => {
    // Check that axios.create was called with the correct config
    expect(mockedAxios.create).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: 'http://localhost:8000', // Default from ApiService
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
        },
      })
    );
  });

  describe('getApplets', () => {
    it('should fetch all applets successfully', async () => {
      const mockApplets: AppletMetadata[] = [{ id: 'applet1', name: 'Test Applet', description: 'desc', inputs: [], outputs: [] }];
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockApplets });

      const applets = await apiInstance.getApplets();
      expect(applets).toEqual(mockApplets);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/applets');
    });

    it('should handle errors when fetching applets', async () => {
      const errorMessage = 'Network Error';
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockRejectedValue(new Error(errorMessage));

      await expect(apiInstance.getApplets()).rejects.toThrow(errorMessage);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/applets');
    });
  });

  describe('getFlows', () => {
    it('should fetch all flows successfully', async () => {
      const mockFlows: Flow[] = [{ id: 'flow1', name: 'Flow 1', nodes: [], edges: [] }];
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockFlows });

      const flows = await apiInstance.getFlows();
      expect(flows).toEqual(mockFlows);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/flows');
    });
  });

  describe('getFlow', () => {
    it('should fetch a specific flow successfully', async () => {
      const mockFlow: Flow = { id: 'flow1', name: 'Flow 1', nodes: [], edges: [] };
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockFlow });

      const flow = await apiInstance.getFlow('flow1');
      expect(flow).toEqual(mockFlow);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/flows/flow1');
    });
  });

  describe('saveFlow', () => {
    it('should save a flow successfully', async () => {
      const mockFlow: Flow = { id: 'flow1', name: 'Flow 1', nodes: [], edges: [] };
      const mockResponse = { id: 'flow1' };
      (apiInstance['api'].post as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockResponse });

      const response = await apiInstance.saveFlow(mockFlow);
      expect(response).toEqual(mockResponse);
      expect(apiInstance['api'].post).toHaveBeenCalledWith('/flows', mockFlow);
    });
  });

  describe('deleteFlow', () => {
    it('should delete a flow successfully', async () => {
      (apiInstance['api'].delete as ReturnType<typeof vi.fn>).mockResolvedValue({});

      await apiInstance.deleteFlow('flow1');
      expect(apiInstance['api'].delete).toHaveBeenCalledWith('/flows/flow1');
    });
  });

  describe('runFlow', () => {
    it('should run a flow successfully', async () => {
      const mockInputData = { param1: 'value1' };
      const mockResponse = { run_id: 'run1' };
      (apiInstance['api'].post as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockResponse });

      const response = await apiInstance.runFlow('flow1', mockInputData);
      expect(response).toEqual(mockResponse);
      expect(apiInstance['api'].post).toHaveBeenCalledWith('/flows/flow1/run', mockInputData);
    });
  });

  describe('getRuns', () => {
    it('should fetch all workflow runs successfully', async () => {
      const mockRuns: WorkflowRunStatus[] = [{ flow_id: 'flow1', status: 'running', progress: 0, total_steps: 1 }];
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockRuns });

      const runs = await apiInstance.getRuns();
      expect(runs).toEqual(mockRuns);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/runs');
    });
  });

  describe('getRun', () => {
    it('should fetch a specific workflow run successfully', async () => {
      const mockRun: WorkflowRunStatus = { flow_id: 'flow1', status: 'completed', progress: 1, total_steps: 1 };
      (apiInstance['api'].get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockRun });

      const run = await apiInstance.getRun('run1');
      expect(run).toEqual(mockRun);
      expect(apiInstance['api'].get).toHaveBeenCalledWith('/runs/run1');
    });
  });

  describe('getCodeSuggestion', () => {
    it('should get AI code suggestions successfully', async () => {
      const mockRequest: CodeSuggestionRequest = { code: 'print("hello")', hint: 'add comment' };
      const mockResponse: CodeSuggestionResponse = { suggestion: 'print("hello") # add comment' };
      (apiInstance['api'].post as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockResponse });

      const response = await apiInstance.getCodeSuggestion(mockRequest);
      expect(response).toEqual(mockResponse);
      expect(apiInstance['api'].post).toHaveBeenCalledWith('/ai/suggest', mockRequest);
    });
  });
});