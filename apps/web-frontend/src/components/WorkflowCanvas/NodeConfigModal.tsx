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
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: parseFloat(value)
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
      case 'llm':
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
              <label htmlFor="provider">Provider</label>
              <select
                id="provider"
                name="provider"
                value={formData.provider || 'openai'}
                onChange={handleChange}
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="google">Google</option>
                <option value="ollama">Ollama</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="model">Model</label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model || ''}
                onChange={handleChange}
                placeholder={formData.provider === 'anthropic' ? 'claude-sonnet-4-20250514' : formData.provider === 'google' ? 'gemini-2.0-flash' : formData.provider === 'ollama' ? 'llama3' : 'gpt-4o'}
              />
            </div>
            <div className="form-group">
              <label htmlFor="system_prompt">System Prompt</label>
              <textarea
                id="system_prompt"
                name="system_prompt"
                value={formData.system_prompt || ''}
                onChange={handleChange}
                placeholder="Enter system instructions for this LLM node"
                rows={5}
              />
            </div>
            <div className="form-group">
              <label htmlFor="temperature">Temperature ({formData.temperature ?? 0.7})</label>
              <input
                type="range"
                id="temperature"
                name="temperature"
                min="0"
                max="2"
                step="0.1"
                value={formData.temperature ?? 0.7}
                onChange={handleNumberChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="max_tokens">Max Tokens</label>
              <input
                type="number"
                id="max_tokens"
                name="max_tokens"
                value={formData.max_tokens ?? 1000}
                onChange={handleNumberChange}
                min={1}
                max={128000}
              />
            </div>
            {(formData.provider === 'custom' || formData.provider === 'ollama') && (
              <div className="form-group">
                <label htmlFor="base_url">Base URL</label>
                <input
                  type="text"
                  id="base_url"
                  name="base_url"
                  value={formData.base_url || ''}
                  onChange={handleChange}
                  placeholder={formData.provider === 'ollama' ? 'http://localhost:11434' : 'https://api.example.com/v1'}
                />
              </div>
            )}
          </>
        );

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
