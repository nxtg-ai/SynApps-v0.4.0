/**
 * HistoryPage component
 * Displays workflow run history and details
 */
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import MainLayout from '../../components/Layout/MainLayout';
import { WorkflowRunStatus } from '../../types';
import apiService from '../../services/ApiService';
import './HistoryPage.css';

// Interfaces for node data
interface FlowNode {
  id: string;
  type: string;
  position?: { x: number; y: number };
  data?: {
    label?: string;
    [key: string]: any;
  };
}

interface DisplayNode {
  nodeId: string;
  nodeType: string;
  position: number;
  result: { type: string; output: any; input?: any } | null;
}

const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const runIdParam = queryParams.get('run');
  
  const [runs, setRuns] = useState<WorkflowRunStatus[]>([]);
  const [selectedRun, setSelectedRun] = useState<WorkflowRunStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [flowNames, setFlowNames] = useState<Record<string, string>>({});
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc'); // Default to newest first
  const [flowDetails, setFlowDetails] = useState<Record<string, any>>({});
  
  // Load workflow runs and flow details
  useEffect(() => {
    const loadRuns = async () => {
      setIsLoading(true);
      try {
        const runsData = await apiService.getRuns();
        
        // Sort runs by start_time based on current sort order
        const sortedRuns = [...runsData].sort((a, b) => 
          sortOrder === 'desc' ? b.start_time - a.start_time : a.start_time - b.start_time
        );
        setRuns(sortedRuns);
        
        // Fetch flow details to get names
        const flowIdsMap: Record<string, boolean> = {};
        const uniqueFlowIds: string[] = [];
        
        // Get unique flow IDs without using Set spread operator
        runsData.forEach(run => {
          if (!flowIdsMap[run.flow_id]) {
            flowIdsMap[run.flow_id] = true;
            uniqueFlowIds.push(run.flow_id);
          }
        });
        
        const flowNamesMap: Record<string, string> = {};
        const flowDetailsMap: Record<string, any> = {};
        
        // Fetch flow details in parallel
        await Promise.all(uniqueFlowIds.map(async (flowId) => {
          try {
            const flowData = await apiService.getFlow(flowId);
            flowNamesMap[flowId] = flowData.name || `Flow ${flowId}`;
            flowDetailsMap[flowId] = flowData;
          } catch (error) {
            console.error(`Error loading flow ${flowId}:`, error);
            flowNamesMap[flowId] = `Flow ${flowId}`;
          }
        }));
        
        setFlowNames(flowNamesMap);
        setFlowDetails(flowDetailsMap);
        
        // If run ID is provided in URL, select that run
        if (runIdParam) {
          const run = runsData.find(r => r.run_id === runIdParam);
          if (run) {
            setSelectedRun(run);
          } else {
            // Try to load the specific run
            try {
              const runData = await apiService.getRun(runIdParam);
              setSelectedRun(runData);
            } catch (error) {
              console.error('Error loading specific run:', error);
            }
          }
        } else if (sortedRuns.length > 0) {
          // Select the most recent run (first item in the sorted runs when sorting by desc)
          setSelectedRun(sortedRuns[0]);
        }
      } catch (error) {
        console.error('Error loading runs:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadRuns();
  }, [runIdParam, sortOrder]);
  
  // Format date for display
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };
  
  // Format duration
  const formatDuration = (startTime: number, endTime?: number) => {
    if (!endTime) return 'In progress';
    
    const durationSeconds = endTime - startTime;
    
    if (durationSeconds < 60) {
      return `${durationSeconds.toFixed(2)} seconds`;
    } else {
      const minutes = Math.floor(durationSeconds / 60);
      const seconds = durationSeconds % 60;
      return `${minutes}m ${seconds.toFixed(0)}s`;
    }
  };
  
  // Prepare nodes for display with proper ordering and data
  const prepareDisplayNodes = () => {
    if (!selectedRun || !flowDetails[selectedRun.flow_id]) return [];
    
    const flow = flowDetails[selectedRun.flow_id];
    const allNodes = flow.nodes || [];
    const results = selectedRun.results || {};
    
    // Create a map of node IDs to their positions in the flow
    const nodePositions: Record<string, number> = {};
    allNodes.forEach((node: any, index: number) => {
      nodePositions[node.id] = index;
    });
    
    // Find the last non-End node with results to use for End node output
    let lastNodeOutput: any = null;
    const endNode = allNodes.find((node: FlowNode) => node.type.toLowerCase() === 'end');
    
    if (endNode) {
      // Get all non-End, non-Start nodes sorted by position
      const processingNodes = allNodes
        .filter((node: FlowNode) => !['start', 'end'].includes(node.type.toLowerCase()))
        .sort((a: FlowNode, b: FlowNode) => nodePositions[b.id] - nodePositions[a.id]); // Descending order
      
      // Find the last node that has results
      for (const node of processingNodes) {
        if (results[node.id] && results[node.id].output) {
          lastNodeOutput = results[node.id].output;
          break;
        }
      }
    }
    
    // Create a complete list of nodes to display
    const displayNodes = allNodes.map((node: FlowNode) => {
      const nodeId = node.id;
      const nodeType = node.type;
      const isStartOrEnd = nodeType.toLowerCase() === 'start' || nodeType.toLowerCase() === 'end';
      let resultData = results[nodeId];
      
      // Special handling for End node - use the last node's output if available
      if (nodeType.toLowerCase() === 'end' && !resultData && lastNodeOutput) {
        resultData = {
          type: 'end',
          output: lastNodeOutput,
          input: null // Input will be handled separately
        };
      } else if (isStartOrEnd && !resultData) {
        // For Start node or End node without special handling
        resultData = { type: nodeType, output: {} };
      }
      
      return {
        nodeId,
        nodeType,
        position: nodePositions[nodeId],
        result: resultData
      } as DisplayNode;
    }).filter((item: DisplayNode) => item.result !== null);
    
    // Sort by position in the flow
    return displayNodes.sort((a: DisplayNode, b: DisplayNode) => a.position - b.position);
  };
  
  // Render a node result based on its type
  const renderNodeResult = (nodeResult: any) => {
    if (!nodeResult || !nodeResult.output) return null;
    
    if (typeof nodeResult.output === 'string') {
      return <div className="text-output">{nodeResult.output}</div>;
    } else if (nodeResult.output && nodeResult.output.image) {
      return (
        <div className="image-output">
          <img 
            src={`data:image/png;base64,${nodeResult.output.image}`} 
            alt="Generated output" 
          />
        </div>
      );
    } else {
      return (
        <div className="json-output">
          <pre>{JSON.stringify(nodeResult.output, null, 2)}</pre>
        </div>
      );
    }
  };
  
  return (
    <MainLayout title="Workflow History">
      <div className="history-page">
        {isLoading ? (
          <div className="loading-state">Loading workflow runs...</div>
        ) : runs.length === 0 ? (
          <div className="empty-state">
            <p>No workflow runs found.</p>
            <button 
              className="create-button"
              onClick={() => navigate('/editor')}
            >
              Create a Workflow
            </button>
          </div>
        ) : (
          <div className="history-container">
            <div className="runs-sidebar">
              <div className="runs-header">
                <h3>Recent Runs</h3>
                <button 
                  className="sort-toggle" 
                  onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                  title={sortOrder === 'desc' ? 'Currently: Newest first' : 'Currently: Oldest first'}
                >
                  {sortOrder === 'desc' ? '↓ Newest first' : '↑ Oldest first'}
                </button>
              </div>
              <div className="runs-list">
                {runs.map(run => (
                  <div 
                    key={run.run_id}
                    className={`run-item ${run.status} ${selectedRun?.run_id === run.run_id ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedRun(run);
                      // Update URL without navigating
                      navigate(`/history?run=${run.run_id}`, { replace: true });
                    }}
                  >
                    <div className="run-header">
                      <span className="flow-name">{flowNames[run.flow_id] || 'Loading...'}</span>
                      <span className="flow-id">ID: {run.flow_id}</span>
                      <span className={`status-badge ${run.status}`}>
                        {run.status}
                      </span>
                    </div>
                    <div className="run-time">
                      {formatDate(run.start_time)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="run-details">
              {selectedRun ? (
                <>
                  <div className="details-header">
                    <h2>Run Detail</h2>
                    <span className={`status-badge large ${selectedRun.status}`}>
                      {selectedRun.status}
                    </span>
                  </div>
                  
                  <div className="details-grid">
                    <div className="detail-item">
                      <span className="detail-label">Run ID</span>
                      <span className="detail-value">{selectedRun.run_id}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Flow Name</span>
                      <span className="detail-value">{flowNames[selectedRun.flow_id] || 'Unknown'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Flow ID</span>
                      <span className="detail-value">{selectedRun.flow_id}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Start Time</span>
                      <span className="detail-value">{formatDate(selectedRun.start_time)}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">End Time</span>
                      <span className="detail-value">
                        {selectedRun.end_time ? formatDate(selectedRun.end_time) : 'In progress'}
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Duration</span>
                      <span className="detail-value">
                        {formatDuration(selectedRun.start_time, selectedRun.end_time)}
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Steps</span>
                      <span className="detail-value">
                        {selectedRun.progress} / {selectedRun.total_steps}
                      </span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Load in Editor</span>
                      <span className="detail-value">
                        <button 
                          className="load-in-editor-button"
                          onClick={() => {
                            // Prepare input data for URL parameter
                            let inputDataParam = '';
                            if (selectedRun && selectedRun.input_data) {
                              // Convert input_data to string if it's an object
                              const inputDataStr = typeof selectedRun.input_data === 'string' 
                                ? selectedRun.input_data 
                                : JSON.stringify(selectedRun.input_data);
                              
                              // Base64 encode to safely pass in URL
                              inputDataParam = btoa(inputDataStr);
                            }
                            
                            // Navigate to editor with flow ID and input data
                            navigate(`/editor/${selectedRun.flow_id}?input_data=${inputDataParam}`);
                          }}
                        >
                          Load in Editor
                        </button>
                      </span>
                    </div>
                  </div>
                  
                  {selectedRun.error && (
                    <div className="error-panel">
                      <h3>Error</h3>
                      <div className="error-message">{selectedRun.error}</div>
                    </div>
                  )}
                  
                  <h3>Results</h3>
                  <div className="results-panel">
                    {selectedRun && flowDetails[selectedRun.flow_id] ? (
                      <div className="results-tree">
                        {prepareDisplayNodes().map((item: DisplayNode, index: number) => {
                          const { nodeId, nodeType, result } = item;
                          const stepNumber = index + 1;
                          // Use non-null assertion since we filter out null results in prepareDisplayNodes
                          const nodeResult = result;
                          const isStartOrEnd = nodeType.toLowerCase() === 'start' || nodeType.toLowerCase() === 'end';

                          // Get the custom node label from flowDetails if available
                          let displayNodeName = nodeType;
                          if (selectedRun && flowDetails[selectedRun.flow_id]) {
                            const node = flowDetails[selectedRun.flow_id].nodes?.find((n: any) => n.id === nodeId);
                            if (!isStartOrEnd && node && node.data && node.data.label) {
                              // Use the custom node label instead of generic nodeType
                              displayNodeName = node.data.label;
                            }
                          }

                          // Get input data for this node
                          let inputPrompt = '';
                          
                          // For Start node, use workflow input data from selectedRun
                          if (nodeType.toLowerCase() === 'start') {
                            if (selectedRun && selectedRun.input_data) {
                              // Ensure input_data is properly processed regardless of its type
                              if (typeof selectedRun.input_data === 'string') {
                                inputPrompt = selectedRun.input_data;
                              } else if (typeof selectedRun.input_data === 'object') {
                                inputPrompt = JSON.stringify(selectedRun.input_data, null, 2);
                              } else {
                                // Handle any other type by converting to string
                                inputPrompt = String(selectedRun.input_data);
                              }
                            }
                          }
                          // Special handling for End node - get input from the last processing node
                          else if (nodeType.toLowerCase() === 'end') {
                            // Find the last non-End, non-Start node with results
                            const processingNodes = prepareDisplayNodes()
                              .filter((node: DisplayNode) => !['start', 'end'].includes(node.nodeType.toLowerCase()));
                              
                            if (processingNodes.length > 0) {
                              const lastProcessingNode = processingNodes[processingNodes.length - 1];
                              if (lastProcessingNode && lastProcessingNode.result && lastProcessingNode.result.output) {
                                inputPrompt = typeof lastProcessingNode.result.output === 'string'
                                  ? lastProcessingNode.result.output
                                  : JSON.stringify(lastProcessingNode.result.output, null, 2);
                              }
                            }
                          } 
                          // For other nodes, check multiple possible locations for input data
                          else if (nodeResult) {
                            // Check if input is directly in the result
                            if (nodeResult.input) {
                              // Check if this is an image node and input might contain base64 data
                              if (nodeType.toLowerCase() === 'image') {
                                // If input is an object that might contain base64 data
                                if (typeof nodeResult.input === 'object' && nodeResult.input !== null) {
                                  // Look for common image keys like 'image', 'data', 'base64', etc.
                                  const possibleImageKeys = ['image', 'data', 'base64', 'content', 'img'];
                                  for (const key of possibleImageKeys) {
                                    if (nodeResult.input[key] && typeof nodeResult.input[key] === 'string') {
                                      // Found a potential image string
                                      inputPrompt = nodeResult.input[key];
                                      break;
                                    }
                                  }
                                  
                                  // If we didn't find a specific image key, use the whole input
                                  if (!inputPrompt) {
                                    inputPrompt = JSON.stringify(nodeResult.input, null, 2);
                                  }
                                } else {
                                  // Input is already a string, might be base64 directly
                                  inputPrompt = nodeResult.input;
                                }
                              } else {
                                // Regular non-image node handling
                                inputPrompt = typeof nodeResult.input === 'string'
                                  ? nodeResult.input
                                  : JSON.stringify(nodeResult.input, null, 2);
                              }
                            }
                            // Check if input is in the output object
                            else if (nodeResult.output && typeof nodeResult.output === 'object') {
                              if (nodeResult.output.prompt) {
                                inputPrompt = nodeResult.output.prompt;
                              } else if (nodeResult.output.input) {
                                inputPrompt = typeof nodeResult.output.input === 'string'
                                  ? nodeResult.output.input
                                  : JSON.stringify(nodeResult.output.input, null, 2);
                              }
                            }
                            
                            // For non-start nodes, try to get input from previous node's output
                            if (!inputPrompt && index > 0) {
                              const prevNode = prepareDisplayNodes()[index - 1];
                              if (prevNode && prevNode.result && prevNode.result.output) {
                                inputPrompt = typeof prevNode.result.output === 'string'
                                  ? prevNode.result.output
                                  : JSON.stringify(prevNode.result.output, null, 2);
                              }
                            }
                          }

                          return (
                            <div className="result-node" key={nodeId}>
                              <div className="result-header">
                                <span className="step-number">Step {stepNumber}</span>
                                <span className={`node-type ${isStartOrEnd ? 'system-node-type' : ''}`}>
                                  {displayNodeName}
                                </span>
                                <span className="node-id">{nodeId}</span>
                              </div>

                              {/* Always display input section with proper handling for images */}
                              <div className="result-input">
                                <div className="input-header">Input:</div>
                                <div className="input-content">
                                  {inputPrompt ? (
                                    // Enhanced image detection logic
                                    (nodeType.toLowerCase() === 'image' || 
                                     nodeType.toLowerCase() === 'artist' ||
                                     (inputPrompt.startsWith && 
                                      (inputPrompt.startsWith('data:image') || 
                                       /^[A-Za-z0-9+/=]{100,}$/.test(inputPrompt))) ||
                                     // Check for JSON that might contain base64 image
                                     (inputPrompt.includes('"image":') || 
                                      inputPrompt.includes('"base64":') ||
                                      inputPrompt.includes('"data":"'))) ? (
                                      <div className="image-output">
                                        {(() => {
                                          // Try to extract base64 from JSON if needed
                                          let imageData = inputPrompt;
                                          
                                          // If it looks like JSON, try to parse and extract image data
                                          if ((inputPrompt.startsWith('{') && inputPrompt.includes('":"')) ||
                                              (inputPrompt.trim().startsWith('{') && inputPrompt.includes('"image":'))) {
                                            try {
                                              const jsonData = JSON.parse(inputPrompt);
                                              // Look for common image keys
                                              const possibleKeys = ['image', 'data', 'base64', 'content', 'img'];
                                              for (const key of possibleKeys) {
                                                if (jsonData[key] && typeof jsonData[key] === 'string') {
                                                  // Found a potential image string
                                                  imageData = jsonData[key];
                                                  console.log(`Found image data in key: ${key}`);
                                                  break;
                                                }
                                              }
                                            } catch (e) {
                                              // If JSON parsing fails, try to extract the image key directly
                                              console.log('Failed to parse JSON for image extraction, trying regex extraction');
                                              const imageMatch = inputPrompt.match(/"image"\s*:\s*"([^"]+)"/i);
                                              if (imageMatch && imageMatch[1]) {
                                                imageData = imageMatch[1];
                                                console.log('Extracted image data using regex');
                                              }
                                            }
                                          }
                                          
                                          // Render the image with the extracted or original data
                                          return (
                                            <img 
                                              src={imageData.startsWith('data:image') ? imageData : `data:image/png;base64,${imageData}`} 
                                              alt="Input Image" 
                                              style={{ maxWidth: '100%', maxHeight: '300px' }}
                                              onError={(e) => {
                                                // If image fails to load, fall back to text display
                                                e.currentTarget.style.display = 'none';
                                                const fallbackElement = document.getElementById(`fallback-${nodeId}`);
                                                if (fallbackElement) {
                                                  fallbackElement.style.display = 'block';
                                                }
                                              }}
                                            />
                                          );
                                        })()} 
                                        <div id={`fallback-${nodeId}`} style={{display: 'none'}} className="json-output-pre">{inputPrompt}</div>
                                      </div>
                                    ) : nodeType.toLowerCase() === 'start' ? (
                                      <textarea 
                                        className="text-output"
                                        readOnly
                                        value={inputPrompt}
                                        rows={6}
                                        placeholder={'{\'text\': \'Hello world\'} or plain text'}
                                      />
                                    ) : (
                                      <div className="json-output-pre">{inputPrompt}</div>
                                    )
                                  ) : (
                                    <div className="no-input">No input data available</div>
                                  )}
                                </div>
                              </div>
                              
                              {/* Display output */}
                              {nodeResult && nodeResult.output ? (
                                <div className="result-output">
                                  <div className="output-header">Output:</div>
                                  {renderNodeResult(nodeResult)}
                                </div>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="no-results">
                        {selectedRun.status === 'running' ? 
                          'Workflow is still running...' : 
                          'No results available.'}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="no-run-selected">
                  <p>Select a run from the sidebar to view details.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </MainLayout>
  );
};

export default HistoryPage;
