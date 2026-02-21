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
      case 'llm':
        return <span role="img" aria-label="LLM">⚡</span>;
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
      case 'merge':
        return <span role="img" aria-label="Merge">🔀</span>;
      case 'for_each':
        return <span role="img" aria-label="For-Each">🔄</span>;
      case 'if_else':
        return <span role="img" aria-label="If/Else">🔀</span>;
      default:
        return <span role="img" aria-label="Applet">🔌</span>;
    }
  };
  
  // Get the color based on applet type
  const getColor = () => {
    switch (appletType) {
      case 'llm':
        return '#ecfdf5';
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
      case 'merge':
        return '#f0f9ff';
      case 'for_each':
        return '#fef3c7';
      case 'if_else':
        return '#fce7f3';
      default:
        return '#f8fafc';
    }
  };
  
  // Get the accent color based on applet type
  const getAccentColor = () => {
    switch (appletType) {
      case 'llm':
        return '#10b981';
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
      case 'merge':
        return '#0284c7';
      case 'for_each':
        return '#d97706';
      case 'if_else':
        return '#db2777';
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
            {appletType === 'llm' && `Universal LLM — ${(data as any).provider || 'openai'}/${(data as any).model || 'gpt-4o'}`}
            {appletType === 'writer' && 'Generates text content using AI'}
            {appletType === 'artist' && 'Creates images using AI models'}
            {appletType === 'memory' && 'Stores and retrieves context'}
            {appletType === 'researcher' && 'Searches for information'}
            {appletType === 'analyzer' && 'Analyzes data and provides insights'}
            {appletType === 'summarizer' && 'Creates concise summaries'}
            {appletType === 'merge' && `Fan-in merge — ${(data as any).strategy || 'array'}`}
            {appletType === 'for_each' && `Loop — ${(data as any).parallel ? 'parallel' : 'sequential'}, max ${(data as any).max_iterations || 1000}`}
            {appletType === 'if_else' && `Conditional — ${(data as any).operation || 'equals'}${(data as any).negate ? ' (negated)' : ''}`}
            {!['llm', 'writer', 'artist', 'memory', 'researcher', 'analyzer', 'summarizer', 'merge', 'for_each', 'if_else'].includes(appletType) && 'Custom applet module'}
          </div>
        )}
      </div>
      
      {/* Spinner overlay for running nodes */}
      {status === 'running' && (
        <div className="node-spinner-overlay">
          <div className="node-spinner" />
        </div>
      )}

      {/* Success/error badge */}
      {status === 'success' && <div className="node-success-badge">&#10003;</div>}
      {status === 'error' && <div className="node-error-badge">!</div>}

      {/* Mini-output preview for completed nodes */}
      {status === 'success' && (data as any).output && (
        <div className="node-output-preview">
          {typeof (data as any).output === 'string'
            ? (data as any).output.slice(0, 120)
            : typeof (data as any).output?.content === 'string'
              ? (data as any).output.content.slice(0, 120)
              : JSON.stringify((data as any).output).slice(0, 120)}
        </div>
      )}

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
