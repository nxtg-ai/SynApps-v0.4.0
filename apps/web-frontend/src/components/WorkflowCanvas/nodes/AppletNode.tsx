/**
 * AppletNode component
 * Custom node for applets in the workflow canvas
 * Enhanced with modern UI/UX design
 */
import React, { memo, useEffect, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import './Nodes.css';

type AppletNodeData = {
  label?: string;
  description?: string;
  status?: string;
};

type AppletFlowNode = Node<AppletNodeData, string>;

const AppletNode: React.FC<NodeProps<AppletFlowNode>> = ({ data, id, type, selected }) => {
  const appletType = type || 'applet';
  const status = data.status || 'idle';
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Add animation when status changes
  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 500);
    return () => clearTimeout(timer);
  }, [status]);
  
  // Get the icon based on applet type
  const getIcon = () => {
    switch (appletType) {
      case 'writer':
        return <span role="img" aria-label="Writer">✍️</span>;
      case 'artist':
        return <span role="img" aria-label="Artist">🎨</span>;
      case 'memory':
        return <span role="img" aria-label="Memory">🧠</span>;
      case 'researcher':
        return <span role="img" aria-label="Researcher">🔍</span>;
      case 'analyzer':
        return <span role="img" aria-label="Analyzer">📊</span>;
      case 'summarizer':
        return <span role="img" aria-label="Summarizer">📝</span>;
      default:
        return <span role="img" aria-label="Applet">🔌</span>;
    }
  };
  
  // Get the color based on applet type
  const getColor = () => {
    switch (appletType) {
      case 'writer':
        return '#eff6ff';
      case 'artist':
        return '#fff7ed';
      case 'memory':
        return '#faf5ff';
      case 'researcher':
        return '#f0fdfa';
      case 'analyzer':
        return '#fdf2f8';
      case 'summarizer':
        return '#fefce8';
      default:
        return '#f8fafc';
    }
  };
  
  // Get the accent color based on applet type
  const getAccentColor = () => {
    switch (appletType) {
      case 'writer':
        return '#3b82f6';
      case 'artist':
        return '#f97316';
      case 'memory':
        return '#a855f7';
      case 'researcher':
        return '#14b8a6';
      case 'analyzer':
        return '#ec4899';
      case 'summarizer':
        return '#eab308';
      default:
        return '#64748b';
    }
  };
  
  // Get the border color based on status
  const getBorderColor = () => {
    switch (status) {
      case 'running':
        return '#3b82f6';
      case 'success':
        return '#10b981';
      case 'error':
        return '#ef4444';
      default:
        return selected ? 'rgba(59, 130, 246, 0.5)' : 'rgba(226, 232, 240, 0.8)';
    }
  };
  
  return (
    <div 
      className={`applet-node ${status} ${isAnimating ? 'animating' : ''}`} 
      style={{ 
        backgroundColor: getColor(),
        borderColor: getBorderColor(),
        boxShadow: selected ? '0 0 0 2px rgba(59, 130, 246, 0.5), 0 12px 24px -6px rgba(0, 0, 0, 0.25)' : undefined,
        transform: selected ? 'translateY(-2px) scale(1.05)' : undefined
      }}
      data-id={id}
      data-applet-type={appletType}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="handle"
      />
      
      <div className="applet-icon" style={{ color: getAccentColor(), backgroundColor: `${getAccentColor()}15` }}>
        {getIcon()}
      </div>
  
      <div className="applet-content">
        <div className="applet-name">{data.label || appletType.charAt(0).toUpperCase() + appletType.slice(1)}</div>
        {data.description && (
          <div className="applet-description">{data.description}</div>
        )}
        {!data.description && (
          <div className="applet-description">
            {appletType === 'writer' && 'Generates text content using AI'}
            {appletType === 'artist' && 'Creates images using AI models'}
            {appletType === 'memory' && 'Stores and retrieves context'}
            {appletType === 'researcher' && 'Searches for information'}
            {appletType === 'analyzer' && 'Analyzes data and provides insights'}
            {appletType === 'summarizer' && 'Creates concise summaries'}
            {!['writer', 'artist', 'memory', 'researcher', 'analyzer', 'summarizer'].includes(appletType) && 'Custom applet module'}
          </div>
        )}
      </div>
      
      <div className={`applet-status-indicator ${status}`} />
      
      <Handle
        type="source"
        position={Position.Bottom}
        className="handle"
      />
    </div>
  );
};

export default memo(AppletNode);
