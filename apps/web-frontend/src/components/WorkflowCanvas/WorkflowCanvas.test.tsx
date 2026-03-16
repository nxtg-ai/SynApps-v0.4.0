import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Connection, Edge, Node } from '@xyflow/react';
import WorkflowCanvas from './WorkflowCanvas';
import { useExecutionStore } from '@/stores/executionStore';
import { useWorkflowStore } from '@/stores/workflowStore';
import type { Flow, WorkflowRunStatus } from '@/types';

const {
  mockReactFlow,
  mockMiniMap,
  mockControls,
  mockBackground,
  mockAddEdge,
  mockApplyNodeChanges,
  mockApplyEdgeChanges,
} = vi.hoisted(() => {
  const reactFlow = vi.fn();
  const miniMap = vi.fn();
  const controls = vi.fn();
  const background = vi.fn();
  const addEdge = vi.fn((edge: Edge, edges: Edge[]) => [...edges, edge]);
  const applyNodeChanges = vi.fn((changes: any[], nodes: Node[]) => {
    let updated = [...nodes];
    for (const change of changes) {
      if (change.type === 'remove') {
        updated = updated.filter((node) => node.id !== change.id);
      } else if (change.type === 'position') {
        updated = updated.map((node) =>
          node.id === change.id ? { ...node, position: change.position ?? node.position } : node,
        );
      } else if (change.type === 'select') {
        updated = updated.map((node) =>
          node.id === change.id ? { ...node, selected: Boolean(change.selected) } : node,
        );
      }
    }
    return updated;
  });
  const applyEdgeChanges = vi.fn((changes: any[], edges: Edge[]) => {
    let updated = [...edges];
    for (const change of changes) {
      if (change.type === 'remove') {
        updated = updated.filter((edge) => edge.id !== change.id);
      }
    }
    return updated;
  });

  return {
    mockReactFlow: reactFlow,
    mockMiniMap: miniMap,
    mockControls: controls,
    mockBackground: background,
    mockAddEdge: addEdge,
    mockApplyNodeChanges: applyNodeChanges,
    mockApplyEdgeChanges: applyEdgeChanges,
  };
});

const { mockGenerateId } = vi.hoisted(() => ({
  mockGenerateId: vi.fn(() => 'generated-id'),
}));

const { mockSubscribe } = vi.hoisted(() => ({
  mockSubscribe: vi.fn(),
}));


vi.mock('@xyflow/react', async () => {
  const actual = await vi.importActual<typeof import('@xyflow/react')>('@xyflow/react');
  return {
    ...actual,
    ReactFlowProvider: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="mock-react-flow-provider">{children}</div>
    ),
    ReactFlow: (props: any) => {
      mockReactFlow(props);
      return (
        <div data-testid="mock-react-flow">
          {(props.nodes ?? []).map((node: any) => (
            <div key={node.id} data-testid={`rf-node-${node.id}`} data-id={node.id} data-status={node.data?.status} />
          ))}
          {props.children}
        </div>
      );
    },
    Background: (props: any) => {
      mockBackground(props);
      return <div data-testid="mock-background" />;
    },
    Controls: (props: any) => {
      mockControls(props);
      return <div data-testid="mock-controls" />;
    },
    MiniMap: (props: any) => {
      mockMiniMap(props);
      return <div data-testid="mock-minimap" />;
    },
    addEdge: mockAddEdge,
    applyNodeChanges: mockApplyNodeChanges,
    applyEdgeChanges: mockApplyEdgeChanges,
  };
});

vi.mock('@/utils/flowUtils', () => ({
  generateId: mockGenerateId,
}));

vi.mock('../../services/WebSocketService', () => ({
  __esModule: true,
  default: {
    subscribe: mockSubscribe,
  },
}));


vi.mock('./NodeContextMenu', () => ({
  __esModule: true,
  default: ({ nodeId, onDelete, onOpenConfig, onClose }: any) => (
    <div data-testid="mock-node-context-menu">
      <span>{nodeId}</span>
      <button type="button" onClick={() => onOpenConfig(nodeId)}>
        Open Config
      </button>
      <button type="button" onClick={() => onDelete(nodeId)}>
        Delete Node
      </button>
      <button type="button" onClick={onClose}>
        Close Menu
      </button>
    </div>
  ),
}));

vi.mock('./NodeConfigModal', () => ({
  __esModule: true,
  default: ({ isOpen, nodeId, onSave, onClose }: any) =>
    isOpen ? (
      <div data-testid="mock-node-config-modal">
        <span>{nodeId}</span>
        <button type="button" onClick={() => onSave(nodeId, { label: 'Updated label', systemPrompt: 'Prompt' })}>
          Save Config
        </button>
        <button type="button" onClick={onClose}>
          Close Config
        </button>
      </div>
    ) : null,
}));

const baseFlow: Flow = {
  id: 'flow-1',
  name: 'Test flow',
  nodes: [
    { id: 'start', type: 'start', position: { x: 0, y: 0 }, data: { label: 'Start' } },
    { id: 'writer-1', type: 'writer', position: { x: 100, y: 0 }, data: { label: 'Writer' } },
    { id: 'end', type: 'end', position: { x: 200, y: 0 }, data: { label: 'End' } },
  ],
  edges: [
    { id: 'edge-1', source: 'start', target: 'writer-1' },
    { id: 'edge-2', source: 'writer-1', target: 'end' },
  ],
};

const getLatestReactFlowProps = () => {
  const calls = mockReactFlow.mock.calls;
  if (calls.length === 0) {
    throw new Error('ReactFlow was not called');
  }
  return calls[calls.length - 1][0];
};

describe('WorkflowCanvas', () => {
  let statusSubscriber: ((status: WorkflowRunStatus) => void) | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    statusSubscriber = undefined;
    useWorkflowStore.getState().resetWorkflowState();
    useExecutionStore.getState().resetExecutionState();

    mockSubscribe.mockImplementation((type: string, callback: (status: WorkflowRunStatus) => void) => {
      if (type === 'workflow.status') {
        statusSubscriber = callback;
      }
      return vi.fn();
    });
  });

  it('renders react-flow shell and toggles virtualization by workflow size', () => {
    render(<WorkflowCanvas flow={baseFlow} />);
    expect(screen.getByTestId('mock-react-flow')).toBeInTheDocument();
    expect(screen.getByTestId('mock-minimap')).toBeInTheDocument();
    expect(screen.getByTestId('mock-controls')).toBeInTheDocument();
    expect(getLatestReactFlowProps().onlyRenderVisibleElements).toBe(false);

    const largeFlow: Flow = {
      ...baseFlow,
      nodes: Array.from({ length: 120 }).map((_, index) => ({
        id: `node-${index}`,
        type: 'writer',
        position: { x: index * 10, y: index * 5 },
        data: { label: `Writer ${index}` },
      })),
      edges: [],
    };
    render(<WorkflowCanvas flow={largeFlow} />);
    expect(getLatestReactFlowProps().onlyRenderVisibleElements).toBe(true);
  });

  it('handles connect events and queues flow updates', async () => {
    const onFlowChange = vi.fn();
    mockGenerateId.mockReturnValueOnce('edge-ui').mockReturnValueOnce('edge-flow');

    render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} />);

    const connection: Connection = { source: 'start', target: 'end', sourceHandle: null, targetHandle: null };
    act(() => {
      getLatestReactFlowProps().onConnect(connection);
    });

    expect(mockAddEdge).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'edge-ui',
        source: 'start',
        target: 'end',
      }),
      expect.any(Array),
    );

    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          edges: expect.arrayContaining([
            expect.objectContaining({
              id: 'edge-flow',
              source: 'start',
              target: 'end',
            }),
          ]),
        }),
      );
    });
  });

  it('handles node and edge change callbacks', async () => {
    const onFlowChange = vi.fn();
    render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} />);

    act(() => {
      getLatestReactFlowProps().onNodesChange([{ type: 'remove', id: 'writer-1' }]);
    });
    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.not.arrayContaining([expect.objectContaining({ id: 'writer-1' })]),
          edges: [],
        }),
      );
    });

    act(() => {
      getLatestReactFlowProps().onEdgesChange([{ type: 'remove', id: 'edge-1' }]);
    });
    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          edges: [expect.objectContaining({ id: 'edge-2' })],
        }),
      );
    });
  });

  it('handles drag-over and drop to add a new node', async () => {
    const onFlowChange = vi.fn();
    mockGenerateId.mockReturnValue('new-node-id');
    render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} />);

    const reactFlowInstance = {
      screenToFlowPosition: vi.fn(() => ({ x: 40, y: 60 })),
    };
    act(() => {
      getLatestReactFlowProps().onInit(reactFlowInstance);
    });

    const dragOverEvent = {
      preventDefault: vi.fn(),
      dataTransfer: { dropEffect: '' },
    } as unknown as React.DragEvent<HTMLDivElement>;
    act(() => {
      getLatestReactFlowProps().onDragOver(dragOverEvent);
    });
    expect(dragOverEvent.preventDefault).toHaveBeenCalledTimes(1);
    expect(dragOverEvent.dataTransfer.dropEffect).toBe('move');

    const dropEvent = {
      preventDefault: vi.fn(),
      clientX: 400,
      clientY: 200,
      dataTransfer: {
        getData: vi.fn(() => JSON.stringify({ type: 'artist', data: { label: 'Artist' } })),
      },
    } as unknown as React.DragEvent<HTMLDivElement>;

    act(() => {
      getLatestReactFlowProps().onDrop(dropEvent);
    });

    expect(dropEvent.preventDefault).toHaveBeenCalledTimes(1);
    expect(reactFlowInstance.screenToFlowPosition).toHaveBeenCalledWith({ x: 400, y: 200 });

    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.arrayContaining([
            expect.objectContaining({
              id: 'new-node-id',
              type: 'artist',
              position: { x: 40, y: 60 },
            }),
          ]),
        }),
      );
    });
  });

  it('opens node context menu, config modal, and saves updated node data', async () => {
    const onFlowChange = vi.fn();
    const user = userEvent.setup();
    render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} />);

    const nodeElement = document.createElement('div');
    vi.spyOn(nodeElement, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 20, 100, 80));

    act(() => {
      getLatestReactFlowProps().onNodeContextMenu(
        {
          preventDefault: vi.fn(),
          currentTarget: nodeElement,
        },
        { id: 'writer-1', type: 'writer' },
      );
    });

    expect(screen.getByTestId('mock-node-context-menu')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Open Config' }));
    expect(screen.getByTestId('mock-node-config-modal')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Save Config' }));

    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.arrayContaining([
            expect.objectContaining({
              id: 'writer-1',
              data: expect.objectContaining({
                label: 'Updated label',
                systemPrompt: 'Prompt',
              }),
            }),
          ]),
        }),
      );
    });
  });

  it('deletes a specific node from context menu and supports keyboard delete/escape', async () => {
    const onFlowChange = vi.fn();
    const user = userEvent.setup();
    const { container } = render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} />);

    const nodeElement = document.createElement('div');
    vi.spyOn(nodeElement, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 20, 100, 80));

    act(() => {
      getLatestReactFlowProps().onNodeContextMenu(
        {
          preventDefault: vi.fn(),
          currentTarget: nodeElement,
        },
        { id: 'writer-1', type: 'writer' },
      );
    });

    await user.click(screen.getByRole('button', { name: 'Delete Node' }));
    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.not.arrayContaining([expect.objectContaining({ id: 'writer-1' })]),
        }),
      );
    });

    act(() => {
      getLatestReactFlowProps().onNodesChange([{ type: 'select', id: 'start', selected: true }]);
    });
    fireEvent.keyDown(container.querySelector('.workflow-canvas-container') as HTMLElement, { key: 'Delete' });
    await waitFor(() => {
      expect(onFlowChange).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.not.arrayContaining([expect.objectContaining({ id: 'start' })]),
        }),
      );
    });

    act(() => {
      getLatestReactFlowProps().onNodeContextMenu(
        {
          preventDefault: vi.fn(),
          currentTarget: nodeElement,
        },
        { id: 'end', type: 'end' },
      );
    });
    expect(screen.getByTestId('mock-node-context-menu')).toBeInTheDocument();
    fireEvent.keyDown(container.querySelector('.workflow-canvas-container') as HTMLElement, { key: 'Escape' });
    expect(screen.queryByTestId('mock-node-context-menu')).not.toBeInTheDocument();
  });

  it('updates execution status UI and triggers workflow animations from websocket updates', async () => {
    render(<WorkflowCanvas flow={baseFlow} />);

    expect(mockSubscribe).toHaveBeenCalledWith('workflow.status', expect.any(Function));
    expect(statusSubscriber).toBeTypeOf('function');

    act(() => {
      statusSubscriber?.({
        run_id: 'run-1',
        flow_id: 'flow-1',
        status: 'running',
        current_applet: 'writer-1',
        completed_applets: ['start'],
        progress: 1,
        total_steps: 3,
        start_time: 1,
        results: {},
      } as WorkflowRunStatus);
    });

    expect(screen.getByText('Status: running')).toBeInTheDocument();
    expect(screen.getByText('1 / 3 steps')).toBeInTheDocument();

    // Verify node status is updated via CSS classes (anime.js removed — CSS-driven animations)
    const writerNode = screen.getByTestId('rf-node-writer-1');
    expect(writerNode).toHaveAttribute('data-status', 'running');

    act(() => {
      statusSubscriber?.({
        run_id: 'run-2',
        flow_id: 'flow-1',
        status: 'error',
        current_applet: 'writer-1',
        completed_applets: ['start'],
        progress: 2,
        total_steps: 3,
        start_time: 1,
        end_time: 2,
        error: 'node failed',
        results: {},
      } as WorkflowRunStatus);
    });

    expect(screen.getByText('Error: node failed')).toBeInTheDocument();
  });

  it('prevents mutating actions in readonly mode', async () => {
    const onFlowChange = vi.fn();
    render(<WorkflowCanvas flow={baseFlow} onFlowChange={onFlowChange} readonly />);

    act(() => {
      getLatestReactFlowProps().onConnect({ source: 'start', target: 'end' });
      getLatestReactFlowProps().onNodesChange([{ type: 'remove', id: 'writer-1' }]);
      getLatestReactFlowProps().onEdgesChange([{ type: 'remove', id: 'edge-1' }]);
    });

    expect(onFlowChange).not.toHaveBeenCalled();
    expect(screen.queryByTestId('mock-node-context-menu')).not.toBeInTheDocument();
  });
});
