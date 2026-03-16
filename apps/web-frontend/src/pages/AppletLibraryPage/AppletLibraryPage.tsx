/**
 * AppletLibraryPage component
 * Displays available applets and their details
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../../components/Layout/MainLayout';
import CodeEditor from '../../components/CodeEditor/CodeEditor';
import { AppletMetadata } from '../../types';
import apiService from '../../services/ApiService';
import './AppletLibraryPage.css';

const AppletLibraryPage: React.FC = () => {
  const navigate = useNavigate();
  
  const [applets, setApplets] = useState<AppletMetadata[]>([]);
  const [selectedApplet, setSelectedApplet] = useState<AppletMetadata | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showCodeEditor, setShowCodeEditor] = useState<boolean>(false);
  const [appletCode, setAppletCode] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // Load applets
  useEffect(() => {
    const loadApplets = async () => {
      setIsLoading(true);
      try {
        const appletsData = await apiService.getApplets();
        setApplets(appletsData);
        
        // Select the first applet by default
        if (appletsData.length > 0) {
          setSelectedApplet(appletsData[0]);
        }
      } catch (error) {
        console.error('Error loading applets:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadApplets();
  }, []);
  
  // Handle applet selection
  const handleSelectApplet = (applet: AppletMetadata) => {
    setSelectedApplet(applet);
  };
  
  // Filter applets based on search query
  const filteredApplets = searchQuery
    ? applets.filter(applet => 
        applet.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        applet.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        applet.capabilities.some(cap => cap.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : applets;
  
  // View applet code
  const handleViewCode = () => {
    if (!selectedApplet) return;
    
    // In a real implementation, we would fetch the actual applet code
    setAppletCode(`
"""
${selectedApplet.name} - ${selectedApplet.description}

Version: ${selectedApplet.version}
"""
import os
import json
import logging
from typing import Dict, Any

from main import BaseApplet, AppletMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("${selectedApplet.type}-applet")

class ${selectedApplet.name}(BaseApplet):
    """
    ${selectedApplet.description}
    
    Capabilities:
    ${selectedApplet.capabilities.map(cap => `- ${cap}`).join('\n    ')}
    """
    
    VERSION = "${selectedApplet.version}"
    
    def __init__(self):
        """Initialize the ${selectedApplet.name}."""
        # Your initialization code here
        pass
    
    async def on_message(self, message: AppletMessage) -> AppletMessage:
        """Process an incoming message."""
        logger.info("${selectedApplet.name} received message")
        
        # Extract content and context
        content = message.content
        context = message.context
        
        # Your applet logic here
        result = self.process_content(content)
        
        # Return the response
        return AppletMessage(
            content=result,
            context=context,
            metadata={"applet": "${selectedApplet.type}"}
        )
    
    def process_content(self, content: Any) -> Any:
        """Process the content and return a result."""
        # Your processing logic here
        return f"Processed by ${selectedApplet.type} applet: {content}"
`);
    
    setShowCodeEditor(true);
  };
  
  return (
    <MainLayout title="Applet Library">
      <div className="applet-library-page">
        {isLoading ? (
          <div className="loading-state">Loading applets...</div>
        ) : applets.length === 0 ? (
          <div className="empty-state">
            <p>No applets found.</p>
          </div>
        ) : (
          <div className="library-container">
            <div className="applets-sidebar">
              <div className="search-container">
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search applets..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              
              <div className="applets-list">
                {filteredApplets.length === 0 ? (
                  <div className="no-results">No applets matching your search</div>
                ) : (
                  filteredApplets.map(applet => (
                    <div 
                      key={applet.type}
                      className={`applet-item ${selectedApplet?.type === applet.type ? 'selected' : ''}`}
                      onClick={() => handleSelectApplet(applet)}
                    >
                      <div className="applet-name">{applet.name}</div>
                      <div className="applet-version">v{applet.version}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
            
            <div className="applet-details">
              {selectedApplet ? (
                <>
                  <div className="details-header">
                    <h2>{selectedApplet.name}</h2>
                    <div className="applet-actions">
                      <button 
                        className="code-button"
                        onClick={handleViewCode}
                      >
                        View Code
                      </button>
                      <button 
                        className="use-button"
                        onClick={() => navigate('/editor')}
                      >
                        Use in Workflow
                      </button>
                    </div>
                  </div>
                  
                  <div className="applet-description">
                    {selectedApplet.description}
                  </div>
                  
                  <div className="applet-meta">
                    <div className="meta-item">
                      <span className="meta-label">Type</span>
                      <span className="meta-value">{selectedApplet.type}</span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Version</span>
                      <span className="meta-value">{selectedApplet.version}</span>
                    </div>
                  </div>
                  
                  <div className="applet-capabilities">
                    <h3>Capabilities</h3>
                    <div className="capabilities-list">
                      {selectedApplet.capabilities.map(capability => (
                        <div key={capability} className="capability-item">
                          {capability}
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="applet-usage">
                    <h3>Usage Example</h3>
                    <div className="usage-content">
                      <pre>
{`# Create a workflow with ${selectedApplet.name}
flow = {
  "nodes": [
    {
      "id": "start",
      "type": "start",
      "position": { "x": 250, "y": 25 },
      "data": { "label": "Start" }
    },
    {
      "id": "${selectedApplet.type}1",
      "type": "${selectedApplet.type}",
      "position": { "x": 250, "y": 125 },
      "data": { "label": "${selectedApplet.name}" }
    },
    {
      "id": "end",
      "type": "end",
      "position": { "x": 250, "y": 225 },
      "data": { "label": "End" }
    }
  ],
  "edges": [
    {
      "id": "start-${selectedApplet.type}",
      "source": "start",
      "target": "${selectedApplet.type}1"
    },
    {
      "id": "${selectedApplet.type}-end",
      "source": "${selectedApplet.type}1",
      "target": "end"
    }
  ]
}`}
                      </pre>
                    </div>
                  </div>
                </>
              ) : (
                <div className="no-applet-selected">
                  <p>Select an applet from the sidebar to view details.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {showCodeEditor && selectedApplet && (
        <div className="modal-overlay">
          <div className="code-editor-modal">
            <CodeEditor 
              appletType={selectedApplet.type}
              initialCode={appletCode}
              readOnly={true}
              onClose={() => setShowCodeEditor(false)}
            />
          </div>
        </div>
      )}
    </MainLayout>
  );
};

export default AppletLibraryPage;
