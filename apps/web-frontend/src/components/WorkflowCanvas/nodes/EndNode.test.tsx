import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Node, NodeProps } from '@xyflow/react';
import EndNode from './EndNode';

vi.mock('@xyflow/react', async () => ({
  ...(await vi.importActual('@xyflow/react')),
  Handle: ({ type, position }: { type: string; position: string }) => (
    <div data-testid={`handle-${type}`} data-position={position} />
  ),
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
}));

describe('EndNode', () => {
  type TerminalFlowNode = Node<{ description?: string }, string>;

  const defaultProps = {
    id: 'end-1',
    type: 'end',
    data: {},
    selected: false,
    dragging: false,
    zIndex: 1,
    isConnectable: true,
  } as unknown as NodeProps<TerminalFlowNode>;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders base end node content and target handle', () => {
    render(<EndNode {...defaultProps} />);

    expect(screen.getByRole('img', { name: 'End' })).toBeInTheDocument();
    expect(screen.getByText('End')).toBeInTheDocument();
    expect(screen.getByTestId('handle-target')).toBeInTheDocument();
  });

  it('renders description and selected styles', () => {
    const { container } = render(
      <EndNode
        {...defaultProps}
        selected
        data={{
          description: 'Terminal node',
        }}
      />,
    );

    expect(screen.getByText('Terminal node')).toBeInTheDocument();
    expect((container.firstChild as HTMLElement).style.transform).toContain('scale');
  });

  it('removes initial animation class after timeout', () => {
    const { container } = render(<EndNode {...defaultProps} />);
    const node = container.firstChild as HTMLElement;

    expect(node.className).toContain('animating');
    act(() => {
      vi.advanceTimersByTime(800);
    });
    expect(node.className).not.toContain('animating');
  });
});
