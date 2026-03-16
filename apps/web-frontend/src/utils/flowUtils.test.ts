import { describe, it, expect, vi, beforeEach } from 'vitest';
import { generateId, cloneFlow, createFlowFromTemplate, validateFlow } from './flowUtils';
import { Flow, FlowTemplate } from '@/types';
import * as uuid from 'uuid'; // Import uuid to mock its functions


// Globally mock uuid.v4. By default, it will call the original uuid.v4,
// but we can override it in specific tests.
vi.mock('uuid', async (importOriginal) => {
  const actual = await importOriginal<typeof uuid>();
  return {
    ...actual,
    v4: vi.fn(() => actual.v4()), // Default mock calls the actual v4
  };
});

describe('flowUtils', () => {
  describe('generateId', () => {
    it('should generate a unique ID string', () => {
      // In this test, generateId should use the actual uuidv4 because
      // the default mock implementation calls actual.v4()
      const id1 = generateId();
      const id2 = generateId();
      expect(typeof id1).toBe('string');
      expect(id1.length).toBeGreaterThan(0);
      expect(id1).not.toEqual(id2); // Very high probability of uniqueness
    });
  });

  describe('cloneFlow', () => {
    it('should create a deep clone of a flow', () => {
      const originalFlow: Flow = {
        id: 'flow-1',
        name: 'Original Flow',
        nodes: [{ id: 'node-1', type: 'start', position: { x: 0, y: 0 }, data: {} }],
        edges: [{ id: 'edge-1', source: 'node-1', target: 'node-2', sourceHandle: null, targetHandle: null }],
      };

      const clonedFlow = cloneFlow(originalFlow);

      // Should be deeply equal in value
      expect(clonedFlow).toEqual(originalFlow);
      // But not the same object reference
      expect(clonedFlow).not.toBe(originalFlow);
      expect(clonedFlow.nodes[0]).not.toBe(originalFlow.nodes[0]);
      expect(clonedFlow.edges[0]).not.toBe(originalFlow.edges[0]);

      // Modifying the clone should not affect the original
      clonedFlow.name = 'Cloned Flow';
      clonedFlow.nodes[0].id = 'node-1-cloned';
      expect(originalFlow.name).toBe('Original Flow');
      expect(originalFlow.nodes[0].id).toBe('node-1');
    });
  });

  describe('createFlowFromTemplate', () => {
    const mockTemplate: FlowTemplate = {
      id: 'template-1',
      name: 'Test Template',
      description: 'A test template',
      flow: {
        id: 'flow-template-id',
        name: 'Template Flow',
        nodes: [
          { id: 'node-template-1', type: 'start', position: { x: 0, y: 0 }, data: {} },
          { id: 'node-template-2', type: 'default', position: { x: 100, y: 100 }, data: {} },
          { id: 'node-template-3', type: 'end', position: { x: 200, y: 200 }, data: {} },
        ],
        edges: [
          { id: 'edge-template-1', source: 'node-template-1', target: 'node-template-2', sourceHandle: null, targetHandle: null },
          { id: 'edge-template-2', source: 'node-template-2', target: 'node-template-3', sourceHandle: null, targetHandle: null },
        ],
      },
    };

    beforeEach(() => {
      // Clear mocks and then set specific return values for uuid.v4 for this block
      vi.clearAllMocks();
      vi.mocked(uuid.v4)
        .mockReturnValueOnce('new-flow-id')
        .mockReturnValueOnce('new-node-1')
        .mockReturnValueOnce('new-node-2')
        .mockReturnValueOnce('new-node-3')
        .mockReturnValueOnce('new-edge-1')
        .mockReturnValueOnce('new-edge-2');
    });

    it('should create a new flow with new IDs for flow, nodes, and edges', () => {
      const newFlow = createFlowFromTemplate(mockTemplate);

      // Verify flow ID
      expect(newFlow.id).not.toBe(mockTemplate.flow.id);
      expect(newFlow.id).toBe('new-flow-id');

      // Verify node IDs and data
      expect(newFlow.nodes).toHaveLength(mockTemplate.flow.nodes.length);
      newFlow.nodes.forEach((node, index) => {
        expect(node.id).not.toBe(mockTemplate.flow.nodes[index].id);
        expect(node.data).toEqual(mockTemplate.flow.nodes[index].data);
        // Check specific new IDs
        if (index === 0) expect(node.id).toBe('new-node-1');
        if (index === 1) expect(node.id).toBe('new-node-2');
        if (index === 2) expect(node.id).toBe('new-node-3');
      });

      // Verify edge IDs and source/target mapping
      expect(newFlow.edges).toHaveLength(mockTemplate.flow.edges.length);
      newFlow.edges.forEach((edge, index) => {
        expect(edge.id).not.toBe(mockTemplate.flow.edges[index].id);
        // Check specific new IDs
        if (index === 0) {
          expect(edge.id).toBe('new-edge-1');
          expect(edge.source).toBe('new-node-1');
          expect(edge.target).toBe('new-node-2');
        }
        if (index === 1) {
          expect(edge.id).toBe('new-edge-2');
          expect(edge.source).toBe('new-node-2');
          expect(edge.target).toBe('new-node-3');
        }
      });
    });

    it('should maintain original node/edge properties other than IDs', () => {
      const newFlow = createFlowFromTemplate(mockTemplate);

      // Check node types and positions
      expect(newFlow.nodes[0].type).toBe('start');
      expect(newFlow.nodes[0].position).toEqual({ x: 0, y: 0 });
      expect(newFlow.nodes[1].type).toBe('default');
      expect(newFlow.nodes[2].type).toBe('end');

      // Check edge handles
      expect(newFlow.edges[0].sourceHandle).toBeNull();
      expect(newFlow.edges[0].targetHandle).toBeNull();
    });
  });

  describe('validateFlow', () => {
    const validFlow: Flow = {
      id: 'valid-flow',
      name: 'Valid Flow',
      nodes: [
        { id: 'start-node', type: 'start', position: { x: 0, y: 0 }, data: {} },
        { id: 'middle-node', type: 'default', position: { x: 50, y: 50 }, data: {} },
        { id: 'end-node', type: 'end', position: { x: 100, y: 100 }, data: {} },
      ],
      edges: [
        { id: 'e1', source: 'start-node', target: 'middle-node', sourceHandle: null, targetHandle: null },
        { id: 'e2', source: 'middle-node', target: 'end-node', sourceHandle: null, targetHandle: null },
      ],
    };

    it('should return valid true for a valid flow', () => {
      const result = validateFlow(validFlow);
      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('should return valid false and error for no nodes', () => {
      const flowNoNodes: Flow = { ...validFlow, nodes: [] };
      const result = validateFlow(flowNoNodes);
      expect(result.valid).toBe(false);
      expect(result.errors).toEqual(['Flow has no nodes']);
    });

    it('should return valid false and error for no start node', () => {
      const flowNoStart: Flow = {
        ...validFlow,
        nodes: validFlow.nodes.filter(node => node.type !== 'start'),
      };
      const result = validateFlow(flowNoStart);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Flow has no start node');
    });

    it('should return valid false and error for no end node', () => {
      const flowNoEnd: Flow = {
        ...validFlow,
        nodes: validFlow.nodes.filter(node => node.type !== 'end'),
      };
      const result = validateFlow(flowNoEnd);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Flow has no end node');
    });

    it('should return valid false and error for isolated nodes', () => {
      const flowIsolatedNode: Flow = {
        ...validFlow,
        nodes: [
          ...validFlow.nodes,
          { id: 'isolated-node', type: 'default', position: { x: 200, y: 200 }, data: {} },
        ],
      };
      const result = validateFlow(flowIsolatedNode);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Flow has 1 isolated nodes');
    });

    it('should return valid false and multiple errors for multiple issues', () => {
      const flowMultipleIssues: Flow = {
        id: 'invalid-flow',
        name: 'Invalid Flow',
        nodes: [
          { id: 'isolated-node-1', type: 'default', position: { x: 200, y: 200 }, data: {} }, // Isolated
          { id: 'isolated-node-2', type: 'default', position: { x: 250, y: 250 }, data: {} }, // Isolated
          { id: 'start-node', type: 'start', position: { x: 0, y: 0 }, data: {} }, // No end node
        ],
        edges: [],
      };
      const result = validateFlow(flowMultipleIssues);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Flow has no end node');
      expect(result.errors).toContain('Flow has 3 isolated nodes'); // Now expecting 3 isolated nodes
      expect(result.errors).toHaveLength(2);
    });
  });
});
