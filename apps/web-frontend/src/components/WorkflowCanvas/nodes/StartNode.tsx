/**
 * StartNode component
 * Custom node for the workflow start point
 * Enhanced with modern UI/UX design
 */
import React, { memo, useState, useEffect } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { Node, NodeProps } from '@xyflow/react';
import './Nodes.css';

type TerminalNodeData = {
  description?: string;
};

type TerminalFlowNode = Node<TerminalNodeData, string>;

const StartNode: React.FC<NodeProps<TerminalFlowNode>> = ({ data, selected }) => {
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Add subtle entrance animation on mount
  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 800);
    return () => clearTimeout(timer);
  }, []);
  return (
    <div 
      className={`start-node ${isAnimating ? 'animating' : ''}`}
      style={{
        boxShadow: selected ? '0 0 0 2px rgba(59, 130, 246, 0.5), 0 8px 16px -4px rgba(0, 0, 0, 0.1)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        transform: selected ? 'translateY(-2px) scale(1.02)' : undefined,
        transition: 'all 0.2s ease'
      }}
    >
      <div className="node-icon">
        <span role="img" aria-label="Start">🚀</span>
      </div>
      <div className="node-label">Start</div>
      {data.description && (
        <div className="node-description">{data.description}</div>
      )}
      
      <Handle
        type="source"
        position={Position.Bottom}
        className="handle"
        style={{
          width: '10px',
          height: '10px',
          background: '#fff',
          border: '2px solid #3b82f6'
        }}
      />
    </div>
  );
};

export default memo(StartNode);
