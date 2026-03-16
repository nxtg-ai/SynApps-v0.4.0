/**
 * DashboardPage component
 * Landing page with workflow templates and recent runs
 */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../../components/Layout/MainLayout';
import TemplateLoader from '../../components/TemplateLoader/TemplateLoader';
import { Flow, WorkflowRunStatus } from '../../types';
import { templates } from '../../templates';
import apiService from '../../services/ApiService';
import './DashboardPage.css';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [isTemplateModalOpen, setIsTemplateModalOpen] = useState<boolean>(false);
  const [recentFlows, setRecentFlows] = useState<Flow[]>([]);
  const [recentRuns, setRecentRuns] = useState<WorkflowRunStatus[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Load recent flows and runs
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        // Load flows
        const flows = await apiService.getFlows();
        setRecentFlows(flows.slice(0, 4)); // Get up to 4 recent flows
        
        // Load runs
        const runs = await apiService.getRuns();
        setRecentRuns(runs.slice(0, 5)); // Get up to 5 recent runs
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadData();
  }, []);
  
  // Handle template selection
  const handleSelectTemplate = async (flow: Flow) => {
    try {
      // Save the new flow
      const response = await apiService.saveFlow(flow);
      
      // Navigate to the editor with the new flow ID
      navigate(`/editor/${response.id}`);
    } catch (error) {
      console.error('Error creating flow from template:', error);
    }
    
    setIsTemplateModalOpen(false);
  };
  
  // Handle workflow import from JSON file
  const handleImportFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const flowData = JSON.parse(text);

      if (!flowData.name || !Array.isArray(flowData.nodes)) {
        alert('Invalid workflow file: missing name or nodes.');
        return;
      }

      const response = await apiService.importFlow(flowData);
      navigate(`/editor/${response.id}`);
    } catch (error) {
      console.error('Import failed:', error);
      alert('Failed to import workflow. Please check the file format.');
    } finally {
      // Reset file input so the same file can be imported again
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Format date for display
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  return (
    <MainLayout title="Dashboard">
      <div className="dashboard">
        <section className="welcome-section">
          <div className="welcome-text">
            <h2>Welcome to SynApps v1.0</h2>
            <p>
              Build modular AI workflows with autonomous agents that work together.
              Get started by creating a new workflow from a template, import an existing one, or open a recent workflow.
            </p>
          </div>
          <div className="welcome-actions">
            <button
              className="new-workflow-button"
              onClick={() => setIsTemplateModalOpen(true)}
            >
              Create New Workflow
            </button>
            <button
              className="import-workflow-button"
              onClick={() => fileInputRef.current?.click()}
            >
              Import Workflow
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.synapps.json"
              style={{ display: 'none' }}
              onChange={handleImportFile}
            />
          </div>
        </section>
        
        <div className="dashboard-grid">
          <section className="templates-section">
            <div className="section-header">
              <h3>Featured Templates</h3>
              <button 
                className="view-all-button"
                onClick={() => setIsTemplateModalOpen(true)}
              >
                View All
              </button>
            </div>
            <div className="templates-grid">
              {templates.slice(0, 3).map(template => (
                <div 
                  key={template.id}
                  className="template-card"
                  onClick={async () => {
                    setIsTemplateModalOpen(true);
                  }}
                >
                  <h4>{template.name}</h4>
                  <p>{template.description}</p>
                  <div className="template-tags">
                    {template.tags.map(tag => (
                      <span key={tag} className="template-tag">{tag}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
          
          <section className="recent-flows-section">
            <div className="section-header">
              <h3>Recent Workflows</h3>
              <button 
                className="view-all-button"
                onClick={() => navigate('/editor')}
              >
                View All
              </button>
            </div>
            {isLoading ? (
              <div className="loading-state">Loading recent workflows...</div>
            ) : recentFlows.length === 0 ? (
              <div className="empty-state">
                <p>No recent workflows found.</p>
                <button 
                  className="create-button"
                  onClick={() => setIsTemplateModalOpen(true)}
                >
                  Create Your First Workflow
                </button>
              </div>
            ) : (
              <div className="flows-list">
                {recentFlows.map(flow => (
                  <div 
                    key={flow.id}
                    className="flow-card"
                    onClick={() => navigate(`/editor/${flow.id}`)}
                  >
                    <h4>{flow.name}</h4>
                    <div className="flow-stats">
                      <span>{flow.nodes.length} nodes</span>
                      <span>{flow.edges.length} connections</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
          
          <section className="recent-runs-section">
            <div className="section-header">
              <h3>Recent Runs</h3>
              <button 
                className="view-all-button"
                onClick={() => navigate('/history')}
              >
                View All
              </button>
            </div>
            {isLoading ? (
              <div className="loading-state">Loading recent runs...</div>
            ) : recentRuns.length === 0 ? (
              <div className="empty-state">
                <p>No workflow runs yet.</p>
                <p>Create and run a workflow to see execution history.</p>
              </div>
            ) : (
              <div className="runs-list">
                {recentRuns.map(run => (
                  <div 
                    key={run.run_id}
                    className={`run-card ${run.status}`}
                    onClick={() => navigate(`/history?run=${run.run_id}`)}
                  >
                    <div className="run-info">
                      <h4>{run.flow_id}</h4>
                      <span className="run-time">
                        {formatDate(run.start_time * 1000)}
                      </span>
                    </div>
                    <div className="run-status">
                      <span className={`status-badge ${run.status}`}>
                        {run.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
      
      {isTemplateModalOpen && (
        <div className="modal-overlay">
          <TemplateLoader 
            onSelectTemplate={handleSelectTemplate}
            onClose={() => setIsTemplateModalOpen(false)}
          />
        </div>
      )}
    </MainLayout>
  );
};

export default DashboardPage;
