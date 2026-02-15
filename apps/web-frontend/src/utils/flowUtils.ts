/**
 * Flow utilities
 */
import { v4 as uuidv4 } from 'uuid';
import { Flow, FlowTemplate } from '../types';

/**
 * Generate a unique ID for flows, nodes, etc.
 */
export const generateId = (): string => {
  return uuidv4();
};

/**
 * Create a deep clone of a flow
 */
export const cloneFlow = (flow: Flow): Flow => {
  return JSON.parse(JSON.stringify(flow));
};

/**
 * Create a new flow from a template
 * Generates new IDs for the flow and all nodes and edges
 */
export const createFlowFromTemplate = (template: FlowTemplate): Flow => {
  // Clone the template flow
  const flowClone = cloneFlow(template.flow);
  
  // Generate a new ID for the flow
  const newFlow: Flow = {
    ...flowClone,
    id: generateId()
  };
  
  // Create a mapping of old node IDs to new node IDs
  const nodeIdMap: Record<string, string> = {};
  
  // Generate new IDs for all nodes
  newFlow.nodes = newFlow.nodes.map(node => {
    const oldId = node.id;
    const newId = generateId();
    
    // Store the mapping
    nodeIdMap[oldId] = newId;
    
    return {
      ...node,
      id: newId
    };
  });
  
  // Update all edges with the new node IDs
  newFlow.edges = newFlow.edges.map(edge => {
    return {
      ...edge,
      id: generateId(),
      source: nodeIdMap[edge.source],
      target: nodeIdMap[edge.target]
    };
  });
  
  return newFlow;
};

/**
 * Validate a flow to ensure it has nodes and valid connections
 */
export const validateFlow = (flow: Flow): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  
  // Check if the flow has nodes
  if (!flow.nodes || flow.nodes.length === 0) {
    errors.push('Flow has no nodes');
    return { valid: false, errors };
  }
  
  // Check for start and end nodes
  const hasStart = flow.nodes.some(node => node.type === 'start');
  const hasEnd = flow.nodes.some(node => node.type === 'end');
  
  if (!hasStart) {
    errors.push('Flow has no start node');
  }
  
  if (!hasEnd) {
    errors.push('Flow has no end node');
  }
  
  // Check if all nodes are connected
  const connectedNodes = new Set<string>();
  
  // Add source and target nodes from all edges
  flow.edges.forEach(edge => {
    connectedNodes.add(edge.source);
    connectedNodes.add(edge.target);
  });
  
  // Check for any isolated nodes
  const isolatedNodes = flow.nodes.filter(node => !connectedNodes.has(node.id));
  
  if (isolatedNodes.length > 0) {
    errors.push(`Flow has ${isolatedNodes.length} isolated nodes`);
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
};
