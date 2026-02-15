import React, { useState, useEffect } from 'react';
import './NodeConfigModal.css';

interface NodeConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  nodeId: string;
  nodeType: string;
  nodeData: any;
  onSave: (nodeId: string, updatedData: any) => void;
}

const NodeConfigModal: React.FC<NodeConfigModalProps> = ({ 
  isOpen, 
  onClose, 
  nodeId, 
  nodeType, 
  nodeData, 
  onSave 
}) => {
  const [formData, setFormData] = useState<any>({});
  
  // Initialize form data when modal opens or node changes
  useEffect(() => {
    if (isOpen && nodeData) {
      setFormData({ ...nodeData });
    }
  }, [isOpen, nodeData]);
  
  if (!isOpen) return null;
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(nodeId, formData);
    onClose();
  };
  
  // Render different form fields based on node type
  const renderFormFields = () => {
    switch (nodeType) {
      case 'writer':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="systemPrompt">System Prompt</label>
              <textarea
                id="systemPrompt"
                name="systemPrompt"
                value={formData.systemPrompt || ''}
                onChange={handleChange}
                placeholder="Enter system prompt"
                rows={5}
              />
            </div>
          </>
        );
      
      case 'artist':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="systemPrompt">System Prompt</label>
              <textarea
                id="systemPrompt"
                name="systemPrompt"
                value={formData.systemPrompt || ''}
                onChange={handleChange}
                placeholder="Enter system prompt"
                rows={5}
              />
            </div>
            <div className="form-group">
              <label htmlFor="generator">Image Generator</label>
              <select
                id="generator"
                name="generator"
                value={formData.generator || 'dall-e'}
                onChange={(e) => handleChange(e as any)}
              >
                <option value="dall-e">DALL-E</option>
                <option value="stability">Stability AI</option>
              </select>
            </div>
          </>
        );
      
      case 'memory':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
          </>
        );
      
      default:
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
          </>
        );
    }
  };
  
  return (
    <div className="node-config-modal-overlay">
      <div className="node-config-modal">
        <div className="modal-header">
          <h2>Configure {nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} Node</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {renderFormFields()}
          </div>
          <div className="modal-footer">
            <button type="button" className="cancel-button" onClick={onClose}>Cancel</button>
            <button type="submit" className="save-button">Save Changes</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NodeConfigModal;
