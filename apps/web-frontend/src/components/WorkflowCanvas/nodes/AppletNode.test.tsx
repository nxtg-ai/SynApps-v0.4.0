import React from 'react';
import { act, render, screen } from '@testing-library/react';
import AppletNode from './AppletNode';
import type { Node, NodeProps } from '@xyflow/react';
import { vi } from 'vitest';

// Mock ReactFlow as it's not easy to test with its full functionality
vi.mock('@xyflow/react', async () => ({
  ...(await vi.importActual('@xyflow/react')),
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
  type AppletFlowNode = Node<{ label?: string; description?: string; status?: string }, string>;

  const defaultProps = {
    id: 'test-node',
    type: 'writer',
    data: { label: 'Test Writer' },
    selected: false,
    dragging: false,
    zIndex: 1,
    isConnectable: true
  } as unknown as NodeProps<AppletFlowNode>;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

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

  test.each([
    ['memory', 'Memory', 'Stores and retrieves context'],
    ['researcher', 'Researcher', 'Searches for information'],
    ['analyzer', 'Analyzer', 'Analyzes data and provides insights'],
    ['summarizer', 'Summarizer', 'Creates concise summaries'],
    ['custom', 'Applet', 'Custom applet module'],
  ])('supports %s icon/description defaults', (type, ariaLabel, description) => {
    render(
      <AppletNode
        {...defaultProps}
        type={type}
        data={{ label: `Node ${type}` }}
      />,
    );

    expect(screen.getByRole('img', { name: ariaLabel })).toBeInTheDocument();
    expect(screen.getByText(description)).toBeInTheDocument();
  });

  test.each(['success', 'error'])('applies %s status class', (status) => {
    const { container } = render(
      <AppletNode
        {...defaultProps}
        data={{
          ...defaultProps.data,
          status,
        }}
      />,
    );

    expect((container.firstChild as HTMLElement).classList.contains(status)).toBe(true);
  });

  test('removes transient animation class after timeout', () => {
    const { container } = render(<AppletNode {...defaultProps} />);
    const nodeElement = container.firstChild as HTMLElement;

    expect(nodeElement.className).toContain('animating');
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(nodeElement.className).not.toContain('animating');
  });
});
