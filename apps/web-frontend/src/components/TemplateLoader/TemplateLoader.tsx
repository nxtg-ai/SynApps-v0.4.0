/**
 * TemplateLoader component
 * Displays available templates and allows one-click creation
 */
import React, { useState } from 'react';
import { FlowTemplate, Flow } from '../../types';
import { templates } from '../../templates';
import { createFlowFromTemplate } from '../../utils/flowUtils';
import './TemplateLoader.css';

interface TemplateLoaderProps {
  onSelectTemplate: (flow: Flow) => void;
  onClose: () => void;
}

const TemplateLoader: React.FC<TemplateLoaderProps> = ({ onSelectTemplate, onClose }) => {
  const [selectedTemplate, setSelectedTemplate] = useState<FlowTemplate | null>(null);
  
  const handleTemplateClick = (template: FlowTemplate) => {
    setSelectedTemplate(template);
  };
  
  const handleCreateFlow = () => {
    if (selectedTemplate) {
      const newFlow = createFlowFromTemplate(selectedTemplate);
      onSelectTemplate(newFlow);
    }
  };
  
  return (
    <div className="template-loader">
      <div className="template-loader-header">
        <h2>Choose a Template</h2>
        <button className="close-button" onClick={onClose}>×</button>
      </div>
      
      <div className="template-list">
        {templates.map(template => (
          <div 
            key={template.id}
            className={`template-card ${selectedTemplate?.id === template.id ? 'selected' : ''}`}
            onClick={() => handleTemplateClick(template)}
          >
            <h3>{template.name}</h3>
            <p>{template.description}</p>
            <div className="template-tags">
              {template.tags.map(tag => (
                <span key={tag} className="template-tag">{tag}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      <div className="template-loader-footer">
        <button 
          className="create-button"
          disabled={!selectedTemplate}
          onClick={handleCreateFlow}
        >
          Create Flow
        </button>
      </div>
    </div>
  );
};

export default TemplateLoader;
