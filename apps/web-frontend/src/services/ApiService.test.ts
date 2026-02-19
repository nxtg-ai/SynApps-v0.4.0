import { beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  AppletMetadata,
  CodeSuggestionRequest,
  CodeSuggestionResponse,
  Flow,
  WorkflowRunStatus,
} from '../types';

const {
  mockCreate,
  mockGet,
  mockPost,
  mockDelete,
  mockInterceptorUse,
  mockAxiosInstance,
} = vi.hoisted(() => {
  const create = vi.fn();
  const get = vi.fn();
  const post = vi.fn();
  const del = vi.fn();
  const use = vi.fn();

  const instance = {
    get,
    post,
    delete: del,
    interceptors: {
      response: {
        use,
      },
    },
  };

  return {
    mockCreate: create,
    mockGet: get,
    mockPost: post,
    mockDelete: del,
    mockInterceptorUse: use,
    mockAxiosInstance: instance,
  };
});

vi.mock('axios', () => ({
  default: {
    create: mockCreate,
  },
  create: mockCreate,
}));

const loadApiService = async () => {
  vi.resetModules();
  const module = await import('./ApiService');
  return module.default;
};

describe('ApiService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllEnvs();
    mockCreate.mockReturnValue(mockAxiosInstance as never);
  });

  it('creates axios client with default settings and interceptor', async () => {
    await loadApiService();

    expect(mockCreate).toHaveBeenCalledWith({
      baseURL: 'http://localhost:8000',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    expect(mockInterceptorUse).toHaveBeenCalledTimes(1);
  });

  it('uses REACT_APP_API_URL when provided', async () => {
    vi.stubEnv('REACT_APP_API_URL', 'https://api.example.test');

    await loadApiService();

    expect(mockCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        baseURL: 'https://api.example.test',
      }),
    );
  });

  it('logs and rethrows interceptor errors', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await loadApiService();
    const [, onError] = mockInterceptorUse.mock.calls[0];
    const error = { response: { data: { message: 'boom' } } };

    await expect(onError(error)).rejects.toBe(error);
    expect(errorSpy).toHaveBeenCalledWith('API Error:', { message: 'boom' });
  });

  it('getApplets calls /applets and returns response data', async () => {
    const apiService = await loadApiService();
    const applets: AppletMetadata[] = [
      {
        type: 'writer',
        name: 'Writer',
        description: 'Writes text',
        version: '1.0.0',
        capabilities: ['draft'],
      },
    ];
    mockGet.mockResolvedValue({ data: applets });

    const result = await apiService.getApplets();

    expect(mockGet).toHaveBeenCalledWith('/applets');
    expect(result).toEqual(applets);
  });

  it('getFlows/getFlow call expected endpoints', async () => {
    const apiService = await loadApiService();
    const flow: Flow = { id: 'flow-1', name: 'Flow 1', nodes: [], edges: [] };
    mockGet.mockResolvedValueOnce({ data: [flow] });
    mockGet.mockResolvedValueOnce({ data: flow });

    const flows = await apiService.getFlows();
    const oneFlow = await apiService.getFlow('flow-1');

    expect(mockGet).toHaveBeenNthCalledWith(1, '/flows');
    expect(mockGet).toHaveBeenNthCalledWith(2, '/flows/flow-1');
    expect(flows).toEqual([flow]);
    expect(oneFlow).toEqual(flow);
  });

  it('saveFlow posts flow and returns created id', async () => {
    const apiService = await loadApiService();
    const flow: Flow = { id: '', name: 'New Flow', nodes: [], edges: [] };
    mockPost.mockResolvedValue({ data: { id: 'flow-new' } });

    const result = await apiService.saveFlow(flow);

    expect(mockPost).toHaveBeenCalledWith('/flows', flow);
    expect(result).toEqual({ id: 'flow-new' });
  });

  it('deleteFlow deletes by flow id', async () => {
    const apiService = await loadApiService();
    mockDelete.mockResolvedValue({});

    await apiService.deleteFlow('flow-delete');

    expect(mockDelete).toHaveBeenCalledWith('/flows/flow-delete');
  });

  it('runFlow posts run payload and returns run id', async () => {
    const apiService = await loadApiService();
    const payload = { prompt: 'hello' };
    mockPost.mockResolvedValue({ data: { run_id: 'run-1' } });

    const result = await apiService.runFlow('flow-1', payload);

    expect(mockPost).toHaveBeenCalledWith('/flows/flow-1/run', payload);
    expect(result).toEqual({ run_id: 'run-1' });
  });

  it('getRuns/getRun call expected endpoints', async () => {
    const apiService = await loadApiService();
    const run: WorkflowRunStatus = {
      run_id: 'run-1',
      flow_id: 'flow-1',
      status: 'running',
      progress: 0,
      total_steps: 1,
      start_time: 1,
      results: {},
    };
    mockGet.mockResolvedValueOnce({ data: [run] });
    mockGet.mockResolvedValueOnce({ data: run });

    const runs = await apiService.getRuns();
    const oneRun = await apiService.getRun('run-1');

    expect(mockGet).toHaveBeenNthCalledWith(1, '/runs');
    expect(mockGet).toHaveBeenNthCalledWith(2, '/runs/run-1');
    expect(runs).toEqual([run]);
    expect(oneRun).toEqual(run);
  });

  it('getCodeSuggestion posts request and returns response', async () => {
    const apiService = await loadApiService();
    const request: CodeSuggestionRequest = { code: 'print("x")', hint: 'comment' };
    const response: CodeSuggestionResponse = {
      original: 'print("x")',
      suggestion: 'print("x")  # comment',
      diff: '+ # comment',
    };
    mockPost.mockResolvedValue({ data: response });

    const result = await apiService.getCodeSuggestion(request);

    expect(mockPost).toHaveBeenCalledWith('/ai/suggest', request);
    expect(result).toEqual(response);
  });
});
