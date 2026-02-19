/**
 * WorkflowCanvas component
 * Visual editor for creating and visualizing AI workflows
 */
import React, { useState, useEffect, useCallback, useRef, KeyboardEvent } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  Connection,
  Edge,
  Node,
  applyNodeChanges,
  applyEdgeChanges,
  OnConnect,
  OnEdgesChange,
  OnNodesChange,
  ConnectionLineType,
  NodeRemoveChange,
  ReactFlowInstance
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import anime from 'animejs';
import { Flow, WorkflowRunStatus } from '../../types';
import { generateId } from '../../utils/flowUtils';
import './WorkflowCanvas.css';
import { useExecutionStore } from '../../stores/executionStore';
import { useWorkflowStore } from '../../stores/workflowStore';

// Custom node types
import AppletNode from './nodes/AppletNode';
import StartNode from './nodes/StartNode';
import EndNode from './nodes/EndNode';

// Handlers for workflow execution
import webSocketService from '../../services/WebSocketService';
import NodeContextMenu from './NodeContextMenu';
import './NodeContextMenu.css';
import NodeConfigModal from './NodeConfigModal';

interface WorkflowCanvasProps {
  flow?: Flow | null;
  onFlowChange?: (flow: Flow) => void;
  readonly?: boolean;
}

const nodeTypes = {
  applet: AppletNode,
  start: StartNode,
  end: EndNode,
  writer: AppletNode,
  memory: AppletNode,
  artist: AppletNode,
};

const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({ flow: propFlow, onFlowChange: propOnFlowChange, readonly = false }) => {
  const storeFlow = useWorkflowStore((state) => state.flow);
  const setStoreFlow = useWorkflowStore((state) => state.setFlow);
  const flow =
    propFlow ??
    storeFlow ??
    ({
      id: "",
      name: "",
      nodes: [],
      edges: [],
    } as Flow);
  const applyFlowChange = propOnFlowChange ?? ((updatedFlow: Flow) => setStoreFlow(updatedFlow));

  // Convert Flow to ReactFlow nodes and edges
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const enableVirtualization = nodes.length >= 100;
  const runStatus = useExecutionStore((state) => state.runStatus);
  const setRunStatus = useExecutionStore((state) => state.setRunStatus);
  const completedNodes = useExecutionStore((state) => state.completedNodes);
  const setCompletedNodes = useExecutionStore((state) => state.setCompletedNodes);
  
  // State for context menu
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    nodeId: string | null;
    nodeType: string | null;
    nodeRect: DOMRect | null;
  }>({
    visible: false,
    nodeId: null,
    nodeType: null,
    nodeRect: null
  });
  
  // State for node configuration modal
  const [configModal, setConfigModal] = useState<{
    isOpen: boolean;
    nodeId: string | null;
    nodeType: string | null;
    nodeData: any | null;
  }>({
    isOpen: false,
    nodeId: null,
    nodeType: null,
    nodeData: null
  });
  
  // Reference to the anime.js timeline
  const animationRef = useRef<anime.AnimeTimelineInstance | null>(null);
  
  // Reference to the ReactFlow wrapper div
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance<Node, Edge> | null>(null);
  
  // Reference to track pending flow updates that should happen after render
  const pendingFlowUpdateRef = useRef<{
    type: 'nodeChange' | 'nodeDelete' | 'edgeChange' | 'connect';
    nodes?: Node[];
    edges?: Edge[];
    nodeId?: string;
    connection?: Connection;
  } | null>(null);
  
  // Use ref to track previous flow value to prevent unnecessary updates
  const prevFlowRef = useRef<string>('');

  // Process flow updates after render
  useEffect(() => {
    // Skip if no pending updates
    if (!pendingFlowUpdateRef.current) return;
    
    const pendingUpdate = pendingFlowUpdateRef.current;
    pendingFlowUpdateRef.current = null; // Clear the pending update
    
    // Handle different types of updates
    if (pendingUpdate.type === 'nodeDelete' && pendingUpdate.nodeId) {
      // Update the flow by removing the node and its connected edges
      const updatedFlow = {
        ...flow,
        nodes: flow.nodes.filter(node => node.id !== pendingUpdate.nodeId),
        edges: flow.edges.filter(edge => 
          edge.source !== pendingUpdate.nodeId && 
          edge.target !== pendingUpdate.nodeId
        )
      };
      applyFlowChange(updatedFlow);
    } 
    else if (pendingUpdate.type === 'nodeChange' && pendingUpdate.nodes) {
      // Handle position and data updates for nodes
      const updatedFlow = {
        ...flow,
        nodes: flow.nodes.map(node => {
          const updatedNode = pendingUpdate.nodes?.find(n => n.id === node.id);
          if (updatedNode) {
            return {
              ...node,
              position: updatedNode.position,
              data: {
                ...node.data,
                ...updatedNode.data
              }
            };
          }
          return node;
        })
      };
      applyFlowChange(updatedFlow);
    }
    else if (pendingUpdate.type === 'edgeChange' && pendingUpdate.edges) {
      // Handle edge updates
      const updatedFlow = {
        ...flow,
        edges: pendingUpdate.edges.map(edge => {
          const existingEdge = flow.edges.find(e => e.id === edge.id);
          if (existingEdge) {
            return {
              ...existingEdge,
              ...edge
            };
          }
          return edge;
        })
      };
      applyFlowChange(updatedFlow);
    }
    else if (pendingUpdate.type === 'connect' && pendingUpdate.connection) {
      // Handle new connections
      const newEdge = {
        id: generateId(),
        source: pendingUpdate.connection.source || '',
        target: pendingUpdate.connection.target || '',
        sourceHandle: pendingUpdate.connection.sourceHandle ?? null,
        targetHandle: pendingUpdate.connection.targetHandle ?? null
      };
      
      const updatedFlow = {
        ...flow,
        edges: [...flow.edges, newEdge]
      };
      
      applyFlowChange(updatedFlow);
    }
  }, [applyFlowChange, flow, nodes, edges]);
  
  // Convert our Flow format to ReactFlow format
  useEffect(() => {
    // Convert current flow to string for comparison
    const currentFlowStr = JSON.stringify(flow);
    
    // Skip update if flow hasn't changed
    if (prevFlowRef.current === currentFlowStr) {
      return;
    }
    
    // Update ref with current flow
    prevFlowRef.current = currentFlowStr;
    
    // Map nodes
    const rfNodes = flow.nodes.map(node => ({
      id: node.id,
      type: node.type === 'writer' || node.type === 'memory' || node.type === 'artist' 
        ? node.type 
        : node.type === 'start' || node.type === 'end' 
          ? node.type 
          : 'applet',
      position: node.position,
      data: {
        ...node.data,
        label: node.data.label || node.type,
        // Initialize node-specific configuration data if not present
        ...(node.type === 'start' && !node.data.inputData && { inputData: '', parsedInputData: {} }),
        ...(node.type === 'writer' && !node.data.systemPrompt && { systemPrompt: '' }),
        ...(node.type === 'artist' && !node.data.systemPrompt && { systemPrompt: '', generator: 'dall-e' })
      }
    }));
    
    // Map edges
    const rfEdges = flow.edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: edge.animated || false,
    }));
    
    // Update state00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    setNodes(rfNodes);
    setEdges(rfEdges);
  }, [flow]);
  
  // Node changes handler
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      if (readonly) return;
      
      setNodes((nds) => {
        const newNodes = applyNodeChanges(changes, nds);
        
        // Handle node deletion
        const nodeDeleteChange = changes.find(change => change.type === 'remove') as NodeRemoveChange;
        if (nodeDeleteChange) {
          // Get the node ID being deleted
          const nodeId = nodeDeleteChange.id;
          
          // Queue the node deletion update for after render
          pendingFlowUpdateRef.current = {
            type: 'nodeDelete',
            nodeId
          };
          
          return newNodes;
        }
        
        // Queue a regular node change update for after render
        pendingFlowUpdateRef.current = {
          type: 'nodeChange',
          nodes: newNodes
        };
        
        return newNodes;
      });
    },
    [readonly]
  );
  
  // Edge changes handler
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      if (readonly) return;
      
      setEdges((eds) => {
        const newEdges = applyEdgeChanges(changes, eds);
        
        // Queue edge changes for after render
        pendingFlowUpdateRef.current = {
          type: 'edgeChange',
          edges: newEdges
        };
        
        return newEdges;
      });
    },
    [readonly]
  );
  
  // Connection handler
  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (readonly) return;
      
      // Queue the connection for after render
      pendingFlowUpdateRef.current = {
        type: 'connect',
        connection
      };
      
      // Add the edge to the UI immediately
      const newEdge = {
        id: generateId(),
        source: connection.source || '',
        target: connection.target || '',
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null
      };
      
      setEdges((eds) => {
        return addEdge(newEdge as Edge, eds);
      });
    },
    [readonly]
  );
  
  // Subscribe to workflow status updates
  useEffect(() => {
    const unsubscribe = webSocketService.subscribe('workflow.status', (status: WorkflowRunStatus) => {
      // Only update if it's our flow
      if (status.flow_id === flow.id) {
        setRunStatus(status);
        
        // Update completed nodes
        if (status.completed_applets && Array.isArray(status.completed_applets)) {
          setCompletedNodes(status.completed_applets);
          
          // If there's a current applet and it's running, animate it
          if (status.current_applet && status.status === 'running') {
            // Find the node in our nodes array
            const runningNode = nodes.find(node => 
              node.id === status.current_applet
            );
            
            if (runningNode) {
              animateWorkflow(status);
            }
          }
        }
        
        // Update node statuses
        setNodes(prevNodes => {
          return prevNodes.map(node => {
            // Current node is running
            if (status.current_applet && node.id === status.current_applet) {
              return {
                ...node,
                data: { ...node.data, status: 'running' }
              };
            }
            
            // Node is already completed
            if (status.completed_applets && Array.isArray(status.completed_applets) && 
                status.completed_applets.includes(node.id)) {
              return {
                ...node,
                data: { ...node.data, status: 'success' }
              };
            }
            
            // Error state
            if (status.status === 'error' && status.current_applet && node.id === status.current_applet) {
              return {
                ...node,
                data: { ...node.data, status: 'error' }
              };
            }
            
            // Default state
            return {
              ...node,
              data: { ...node.data, status: node.data?.status || 'idle' }
            };
          });
        });
        
        // Update edge animations
        const updatedEdges = edges.map(edge => {
          // Find which node is the target of this edge
          const targetNode = nodes.find(node => node.id === edge.target);
          
          // If target node is the current one or completed, animate the edge
          if (targetNode && status.current_applet && 
              (targetNode.id === status.current_applet || targetNode.data?.status === 'success')) {
            return {
              ...edge,
              animated: true
            };
          }
          
          return edge;
        });
        
        setEdges(updatedEdges);
        
        // Create animations
        animateWorkflow(status);
      }
    });
    
    return () => {
      unsubscribe();
    };
  }, [flow.id, nodes, edges, flow.nodes]);
  
  // Animation function
  const animateWorkflow = (status: WorkflowRunStatus) => {
    // Stop any existing animation
    if (animationRef.current) {
      animationRef.current.pause();
    }
    
    // Create a new animation timeline
    const timeline = anime.timeline({
      easing: 'easeOutElastic(1, .8)',
      duration: 800
    });
    
    // Find the current node element
    if (status.current_applet) {
      const nodeElement = document.querySelector(
        `[data-id="${status.current_applet}"]`
      );
      
      if (nodeElement) {
        // Pulse animation for the current node
        timeline.add({
          targets: nodeElement,
          scale: [1, 1.05, 1],
          opacity: [1, 0.8, 1],
          duration: 1000,
          easing: 'easeInOutQuad',
          loop: true
        });
      }
    }
    
    // Store the timeline reference
    animationRef.current = timeline;
  };
  
  // Handle drop event for new nodes
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      
      if (readonly) return;
      
      if (reactFlowWrapper.current && reactFlowInstance) {
        const nodeData = event.dataTransfer.getData('application/reactflow');
        
        if (!nodeData) return;
        
        const data = JSON.parse(nodeData);
        const position = reactFlowInstance.screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });
        
        const newNodeId = generateId();
        
        // Update the flow with the new node
        const updatedFlow = {
          ...flow,
          nodes: [...flow.nodes, {
            id: newNodeId,
            type: data.type,
            position,
            data: { 
              label: data.data.label,
              status: 'idle'
            }
          }]
        };
        
        applyFlowChange(updatedFlow);
      }
    },
    [applyFlowChange, flow, reactFlowInstance, readonly]
  );
  
  // Handle drag over event
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);
  
  // Delete selected nodes
  const deleteSelectedNodes = useCallback(() => {
    if (readonly) return;
    
    // Get selected nodes
    const selectedNodes = nodes.filter(node => node.selected);
    
    if (selectedNodes.length > 0) {
      // Get IDs of selected nodes
      const selectedNodeIds = selectedNodes.map(node => node.id);
      
      // Filter out selected nodes
      const updatedNodes = flow.nodes.filter(
        node => !selectedNodeIds.includes(node.id)
      );
      
      // Filter out edges connected to selected nodes
      const updatedEdges = flow.edges.filter(
        edge => 
          !selectedNodeIds.includes(edge.source) && 
          !selectedNodeIds.includes(edge.target)
      );
      
      // Update the flow
      const updatedFlow = {
        ...flow,
        nodes: updatedNodes,
        edges: updatedEdges
      };
      
      applyFlowChange(updatedFlow);
    }
  }, [applyFlowChange, flow, nodes, readonly]);
  
  // Delete a specific node by ID
  const deleteNodeById = useCallback((nodeId: string) => {
    if (readonly || !nodeId) return;
    
    // Filter out the node
    const updatedNodes = flow.nodes.filter(node => node.id !== nodeId);
    
    // Filter out edges connected to the node
    const updatedEdges = flow.edges.filter(
      edge => edge.source !== nodeId && edge.target !== nodeId
    );
    
    // Log the changes
    const removedEdges = flow.edges.filter(
      edge => edge.source === nodeId || edge.target === nodeId
    );
    
    if (removedEdges.length > 0) {
      // Optional logging if needed
    }
    
    // Update the flow
    const updatedFlow = {
      ...flow,
      nodes: updatedNodes,
      edges: updatedEdges
    };
    
    applyFlowChange(updatedFlow);
  }, [applyFlowChange, flow, readonly]);
  
  // Define closeContextMenu to close the context menu
  const closeContextMenu = useCallback(() => {
    setContextMenu(prev => ({ ...prev, visible: false, nodeRect: null }));
  }, []);
  
  // Open node configuration modal
  const onOpenConfig = useCallback((nodeId: string) => {
    if (readonly || !nodeId) return;
    
    // Find the node in the flow
    const node = flow.nodes.find(node => node.id === nodeId);
    
    if (node) {
      // Close the context menu first
      closeContextMenu();
      
      // Open the configuration modal with the node data
      setConfigModal({
        isOpen: true,
        nodeId: nodeId,
        nodeType: node.type || '',
        nodeData: node.data || {}
      });
    }
  }, [flow.nodes, readonly, closeContextMenu]);
  
  // Handle saving node configuration
  const handleSaveNodeConfig = useCallback((nodeId: string, updatedData: any) => {
    if (readonly || !nodeId) return;
    
    console.log('Saving node config for node:', nodeId, 'with data:', updatedData);
    
    // Find the existing node to ensure we have the complete data
    const existingNode = flow.nodes.find(node => node.id === nodeId);
    if (!existingNode) {
      console.error('Could not find node with ID:', nodeId);
      return;
    }
    
    // Update the node data in the flow, preserving all existing properties
    const updatedNodes = flow.nodes.map(node => {
      if (node.id === nodeId) {
        return {
          ...node,
          data: {
            ...node.data,
            ...updatedData
          },
          // Ensure position is preserved
          position: node.position
        };
      }
      return node;
    });
    
    // Create a complete updated flow object
    const updatedFlow = {
      ...flow,
      nodes: updatedNodes
    };
    
    console.log('Updated flow:', updatedFlow);
    
    // Update the flow state
    applyFlowChange(updatedFlow);
    
    // Ensure the nodes state is also updated to reflect changes
    setNodes(prevNodes => {
      return prevNodes.map(node => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              ...updatedData
            }
          };
        }
        return node;
      });
    });
  }, [applyFlowChange, flow, readonly]);
  
  // Handle keyboard events for node deletion
  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (readonly) return;
      
      // Delete key or Backspace key
      if (event.key === 'Delete' || event.key === 'Backspace') {
        deleteSelectedNodes();
      }
      
      // Close context menu on Escape key
      if (event.key === 'Escape') {
        setContextMenu({ visible: false, nodeId: null, nodeType: null, nodeRect: null });
      }
    },
    [deleteSelectedNodes, readonly]
  );
  
  // Handle node configuration
  const handleNodeConfig = useCallback((nodeId: string, configData: any) => {
    if (readonly) return;
    
    // Update node data
    const updatedFlow = {
      ...flow,
      nodes: flow.nodes.map(node => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              ...configData
            }
          };
        }
        return node;
      })
    };
    
    applyFlowChange(updatedFlow);
  }, [applyFlowChange, flow, readonly]);
  
  // Open node configuration modal
  const openNodeConfig = useCallback((nodeId: string) => {
    if (readonly) return;
    
    // Find the node
    const node = flow.nodes.find(n => n.id === nodeId);
    if (!node) return;
    

    
    // Find the node element in the DOM
    const nodeElement = document.querySelector(`[data-id="${nodeId}"]`);
    if (!nodeElement) {
      return;
    }
    
    // Find the config button within the node and click it
    const configButton = nodeElement.querySelector('.config-toggle') as HTMLButtonElement;
    if (configButton) {
      // Programmatically click the config button to open the configuration panel
      configButton.click();
    } else {
      console.error(`Node type ${node.type} does not have a configuration panel`);
    }
  }, [flow.nodes, readonly]);
  
  // Handle node context menu
  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      // Prevent default context menu
      event.preventDefault();
      
      if (readonly) return;
      
      // Get the node element from the event
      const nodeElement = event.currentTarget as HTMLElement;
      
      // Get the bounding rectangle of the node element
      const nodeRect = nodeElement.getBoundingClientRect();
      
      // Show our custom context menu with the node's rect
      setContextMenu({
        visible: true,
        nodeId: node.id,
        nodeType: node.type || 'applet',
        nodeRect
      });
    },
    [readonly]
  );
  
  // This is where the duplicate closeContextMenu function was removed

  return (
    <div 
      className="workflow-canvas-container" 
      ref={reactFlowWrapper}
      onKeyDown={onKeyDown}
      tabIndex={0} // Make div focusable to capture keyboard events
    >
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          onInit={setReactFlowInstance}
          onDrop={onDrop}
          onDragOver={onDragOver}
          connectionLineType={ConnectionLineType.SmoothStep}
          onlyRenderVisibleElements={enableVirtualization}
          fitView
          onNodeContextMenu={onNodeContextMenu}
        >
          <Background color="#f8f8f8" gap={16} />
          <Controls />
          <MiniMap
            nodeStrokeColor={(n) => {
              if (n.data?.status === 'running') return '#1a90ff';
              if (n.data?.status === 'success') return '#52c41a';
              if (n.data?.status === 'error') return '#ff4d4f';
              return '#eee';
            }}
            nodeColor={(n) => {
              if (n.type === 'start') return '#d9f7be';
              if (n.type === 'end') return '#fff1f0';
              if (n.type === 'writer') return '#e6f7ff';
              if (n.type === 'memory') return '#f9f0ff';
              if (n.type === 'artist') return '#fff7e6';
              return '#fff';
            }}
            nodeBorderRadius={2}
          />
        </ReactFlow>
      </ReactFlowProvider>
      
      {runStatus && (
        <div className={`workflow-status ${runStatus.status}`}>
          <div className="workflow-status-header">
            <span className="status-label">Status: {runStatus.status}</span>
            <span className="progress-label">
              {runStatus.progress} / {runStatus.total_steps} steps
            </span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ 
                width: `${(runStatus.progress / runStatus.total_steps) * 100}%` 
              }}
            />
          </div>
          {runStatus.error && (
            <div className="error-message">
              Error: {runStatus.error}
            </div>
          )}
        </div>
      )}
      
      {/* Context menu */}
      {contextMenu.visible && contextMenu.nodeRect && (
        <NodeContextMenu
          nodeRect={contextMenu.nodeRect}
          nodeId={contextMenu.nodeId || ''}
          nodeType={contextMenu.nodeType || ''}
          onClose={closeContextMenu}
          onDelete={deleteNodeById}
          onOpenConfig={onOpenConfig}
        />
      )}
      
      {/* Node Configuration Modal */}
      <NodeConfigModal
        isOpen={configModal.isOpen}
        nodeId={configModal.nodeId || ''}
        nodeType={configModal.nodeType || ''}
        nodeData={configModal.nodeData || {}}
        onClose={() => setConfigModal({ isOpen: false, nodeId: null, nodeType: null, nodeData: null })}
        onSave={handleSaveNodeConfig}
      />
    </div>
  );
};

// Wrap the component with ReactFlowProvider to provide the Zustand store
const WorkflowCanvasWithProvider: React.FC<WorkflowCanvasProps> = (props) => {
  return (
    <ReactFlowProvider>
      <WorkflowCanvas {...props} />
    </ReactFlowProvider>
  );
};

export default WorkflowCanvasWithProvider;
