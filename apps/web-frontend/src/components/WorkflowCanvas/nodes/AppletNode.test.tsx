import React from 'react';
import { render, screen } from '@testing-library/react';
import AppletNode from './AppletNode';
import { NodeProps } from 'reactflow';
import { vi } from 'vitest';

// Mock ReactFlow as it's not easy to test with its full functionality
vi.mock('reactflow', async () => ({
  ...(await vi.importActual('reactflow')),
  Handle: ({ type, position }: { type: string; position: string }) => (
    <div data-testid={`handle-${type}`} data-position={position}></div>
  ),
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
}));

describe('AppletNode Component', () => {
  const defaultProps: NodeProps = {
    id: 'test-node',
    type: 'writer',
    data: { label: 'Test Writer' },
    selected: false,
    dragging: false,
    zIndex: 1,
    xPos: 100,
    yPos: 100,
    isConnectable: true
  };

  test('renders with correct label', () => {
    render(<AppletNode {...defaultProps} />);
    expect(screen.getByText('Test Writer')).toBeInTheDocument();
  });

  test('renders with correct applet type icon', () => {
    render(<AppletNode {...defaultProps} />);
    expect(screen.getByRole('img', { name: 'Writer' })).toBeInTheDocument();
  });

  test('renders with default description when none provided', () => {
    render(<AppletNode {...defaultProps} />);
    expect(screen.getByText('Generates text content using AI')).toBeInTheDocument();
  });

  test('renders with custom description when provided', () => {
    const customProps = {
      ...defaultProps,
      data: { 
        ...defaultProps.data,
        description: 'Custom description for testing'
      }
    };
    render(<AppletNode {...customProps} />);
    expect(screen.getByText('Custom description for testing')).toBeInTheDocument();
  });

  test('renders with different styling when selected', () => {
    const selectedProps = {
      ...defaultProps,
      selected: true,
    };
    const { container } = render(<AppletNode {...selectedProps} />);
    const nodeElement = container.firstChild as HTMLElement;
    
    // Check if the style has transform property when selected
    expect(nodeElement.style.transform).toBeDefined();
    expect(nodeElement.style.transform).not.toBe('');
  });

  test('renders with correct status styling', () => {
    const runningProps = {
      ...defaultProps,
      data: { 
        ...defaultProps.data,
        status: 'running'
      }
    };
    const { container } = render(<AppletNode {...runningProps} />);
    const nodeElement = container.firstChild as HTMLElement;
    
    expect(nodeElement.classList.contains('running')).toBe(true);
  });

  test('renders source and target handles', () => {
    render(<AppletNode {...defaultProps} />);
    expect(screen.getByTestId('handle-target')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source')).toBeInTheDocument();
  });

  test('handles different applet types correctly', () => {
    const artistProps = {
      ...defaultProps,
      type: 'artist',
      data: { label: 'Test Artist' }
    };
    render(<AppletNode {...artistProps} />);
    expect(screen.getByRole('img', { name: 'Artist' })).toBeInTheDocument();
    expect(screen.getByText('Creates images using AI models')).toBeInTheDocument();
  });
});
