import React from 'react';
import { BaseEdge, type EdgeProps, getBezierPath } from '@xyflow/react';

const AnimatedEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  animated,
}) => {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  if (!animated) {
    return (
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ ...style, stroke: '#94a3b8', strokeWidth: 2 }}
      />
    );
  }

  return (
    <>
      {/* Base edge path (glow layer) */}
      <path
        id={`${id}-glow`}
        d={edgePath}
        fill="none"
        stroke="rgba(59, 130, 246, 0.2)"
        strokeWidth={8}
        style={{ filter: 'blur(4px)' }}
      />

      {/* Base edge path */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ ...style, stroke: '#3b82f6', strokeWidth: 2.5 }}
      />

      {/* Flowing particles along the edge */}
      <circle r="3.5" fill="#3b82f6" filter="url(#particle-glow)">
        <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} />
      </circle>
      <circle r="3" fill="#60a5fa" opacity="0.7">
        <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} begin="0.5s" />
      </circle>
      <circle r="2.5" fill="#93c5fd" opacity="0.5">
        <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} begin="1s" />
      </circle>

      {/* SVG filter for particle glow effect */}
      <defs>
        <filter id="particle-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
    </>
  );
};

export default AnimatedEdge;
