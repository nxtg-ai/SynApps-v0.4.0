import { vi } from 'vitest';
import { create } from 'zustand';
import { useWorkflowStore, WorkflowState } from './workflowStore';

// Mock initial state for testing
const initialState: WorkflowState = {
  nodes: [],
  edges: [],
  setNodes: vi.fn(),
  setEdges: vi.fn(),
  addNode: vi.fn(),
  updateNode: vi.fn(),
  updateEdge: vi.fn(),
  onNodesChange: vi.fn(),
  onEdgesChange: vi.fn(),
};

describe('useWorkflowStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useWorkflowStore.setState(initialState);
  });

  it('should have a default initial state', () => {
    // We need to create a fresh store to test the actual default state
    const freshStore = create(useWorkflowStore);
    const state = freshStore.getState();
    expect(state.nodes).toEqual([]);
    expect(state.edges).toEqual([]);
  });

  it('should update nodes when setNodes is called', () => {
    const { setNodes } = useWorkflowStore.getState();
    const newNodes = [{ id: '1', type: 'input', data: { label: 'Test Node' }, position: { x: 0, y: 0 } }];
    setNodes(newNodes);
    expect(useWorkflowStore.getState().nodes).toEqual(newNodes);
  });

  it('should update edges when setEdges is called', () => {
    const { setEdges } = useWorkflowStore.getState();
    const newEdges = [{ id: 'e1-2', source: '1', target: '2' }];
    setEdges(newEdges);
    expect(useWorkflowStore.getState().edges).toEqual(newEdges);
  });

  it('should add a node when addNode is called', () => {
    const { addNode } = useWorkflowStore.getState();
    const newNode = { id: '2', type: 'output', data: { label: 'Another Node' }, position: { x: 100, y: 100 } };
    addNode(newNode);
    const { nodes } = useWorkflowStore.getState();
    expect(nodes).toHaveLength(1);
    expect(nodes[0]).toEqual(newNode);
  });
});
