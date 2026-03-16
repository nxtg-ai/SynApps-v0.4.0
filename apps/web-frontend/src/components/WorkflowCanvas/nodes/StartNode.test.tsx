import React from 'react';
import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Node, NodeProps } from '@xyflow/react';
import StartNode from './StartNode';

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

describe('StartNode', () => {
  type TerminalFlowNode = Node<{ description?: string }, string>;

  const defaultProps = {
    id: 'start-1',
    type: 'start',
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

  it('renders base start node content and source handle', () => {
    render(<StartNode {...defaultProps} />);

    expect(screen.getByRole('img', { name: 'Start' })).toBeInTheDocument();
    expect(screen.getByText('Start')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source')).toBeInTheDocument();
  });

  it('renders description and selected styles', () => {
    const { container } = render(
      <StartNode
        {...defaultProps}
        selected
        data={{
          description: 'Entry point',
        }}
      />,
    );

    expect(screen.getByText('Entry point')).toBeInTheDocument();
    expect((container.firstChild as HTMLElement).style.transform).toContain('scale');
  });

  it('removes initial animation class after timeout', () => {
    const { container } = render(<StartNode {...defaultProps} />);
    const node = container.firstChild as HTMLElement;

    expect(node.className).toContain('animating');
    act(() => {
      vi.advanceTimersByTime(800);
    });
    expect(node.className).not.toContain('animating');
  });
});
