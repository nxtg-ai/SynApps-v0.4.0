import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import HistoryPage from './HistoryPage';
import apiService from '../../services/ApiService';
import { vi, type Mock } from 'vitest';

const { mockApiService } = vi.hoisted(() => ({
  mockApiService: {
    getRuns: vi.fn(),
    getFlow: vi.fn(),
    getRun: vi.fn(),
  },
}));

// Mock the API service
vi.mock('../../services/ApiService', () => ({
  __esModule: true,
  apiService: mockApiService,
  default: mockApiService,
}));

// Mock MainLayout component to simplify testing
vi.mock('../../components/Layout/MainLayout', () => {
  return {
    __esModule: true,
    default: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
  };
});

describe('HistoryPage Component', () => {
  const mockRuns = [
    {
      run_id: 'run-1',
      flow_id: 'flow-1',
      status: 'completed',
      start_time: 1622548800, // June 1, 2021
      end_time: 1622548860, // 1 minute later
      results: {
        'node-1': { type: 'writer', output: 'Test output', input: 'Test input' },
      },
      input_data: { prompt: 'Test prompt' },
    },
    {
      run_id: 'run-2',
      flow_id: 'flow-2',
      status: 'running',
      start_time: 1622635200, // June 2, 2021
      results: {},
      input_data: { prompt: 'Another test' },
    },
  ];

  // Define a type for the mock flow data
  interface MockFlow {
    id: string;
    name: string;
    nodes: Array<{
      id: string;
      type: string;
      position: { x: number; y: number };
      data?: { label: string; [key: string]: any };
    }>;
    edges: any[];
  }

  // Define the mockFlows with proper typing
  const mockFlows: Record<string, MockFlow> = {
    'flow-1': {
      id: 'flow-1',
      name: 'Test Flow 1',
      nodes: [
        {
          id: 'start',
          type: 'start',
          position: { x: 0, y: 0 },
          data: { label: 'Start' },
        },
        {
          id: 'node-1',
          type: 'writer',
          position: { x: 100, y: 100 },
          data: { label: 'Writer Node' },
        },
        {
          id: 'end',
          type: 'end',
          position: { x: 200, y: 200 },
          data: { label: 'End' },
        },
      ],
      edges: [],
    },
    'flow-2': {
      id: 'flow-2',
      name: 'Test Flow 2',
      nodes: [],
      edges: [],
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    window.history.pushState({}, '', '/history');
    (apiService.getRuns as Mock).mockResolvedValue(mockRuns);
    (apiService.getFlow as Mock).mockImplementation((flowId: string) => {
      return Promise.resolve(mockFlows[flowId as keyof typeof mockFlows] || {});
    });
  });

  test('renders loading state initially', () => {
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  test('renders workflow runs after loading', async () => {
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });
    
    expect(screen.getByText(/Test Flow 1/i)).toBeInTheDocument();
    expect(screen.getByText(/completed/i)).toBeInTheDocument();
  });

  test('displays run details when a run is selected', async () => {
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });
    
    expect(screen.getByText(/Run Detail/i)).toBeInTheDocument();
    expect(screen.getByText(/Flow Name/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Test Flow 2/i).length).toBeGreaterThan(0);
  });

  test('handles URL parameters for selecting a specific run', async () => {
    window.history.pushState({}, '', '/history?run=run-2');
    
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });
    
    expect(screen.getByText(/run-2/i)).toBeInTheDocument();
    expect(screen.getAllByText(/running/i).length).toBeGreaterThan(0);
  });

  test('formats duration correctly', async () => {
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Test Flow 1/i));
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });
    expect(screen.getByText(/1m 0s/i)).toBeInTheDocument(); // 60 seconds = 1m 0s
  });

  test('handles API errors gracefully', async () => {
    (apiService.getRuns as Mock).mockRejectedValue(new Error('API error'));
    
    render(
      <BrowserRouter>
        <HistoryPage />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });
    
    expect(screen.getByText(/No workflow runs found/i)).toBeInTheDocument();
  });
});
