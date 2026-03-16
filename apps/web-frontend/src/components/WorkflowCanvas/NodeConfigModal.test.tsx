import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import NodeConfigModal from './NodeConfigModal';

const baseProps = {
  isOpen: true,
  onClose: vi.fn(),
  nodeId: 'node-1',
  onSave: vi.fn(),
};

describe('NodeConfigModal — Memory Node', () => {
  it('renders operation select with store as default', () => {
    render(
      <NodeConfigModal
        {...baseProps}
        nodeType="memory"
        nodeData={{ label: 'Memory' }}
      />,
    );
    const select = screen.getByLabelText('Operation') as HTMLSelectElement;
    expect(select).toBeDefined();
    expect(select.value).toBe('store');
    const options = Array.from(select.options).map((o) => o.value);
    expect(options).toEqual(['store', 'retrieve', 'delete', 'clear']);
  });

  it('renders backend select with sqlite_fts as default', () => {
    render(
      <NodeConfigModal
        {...baseProps}
        nodeType="memory"
        nodeData={{ label: 'Memory' }}
      />,
    );
    const select = screen.getByLabelText('Backend') as HTMLSelectElement;
    expect(select.value).toBe('sqlite_fts');
    const options = Array.from(select.options).map((o) => o.value);
    expect(options.length).toBeGreaterThanOrEqual(2); // Gate 2: non-empty
    expect(options).toContain('chroma');
  });

  it('hides chroma-specific fields when backend is sqlite_fts', () => {
    render(
      <NodeConfigModal
        {...baseProps}
        nodeType="memory"
        nodeData={{ label: 'Memory', backend: 'sqlite_fts' }}
      />,
    );
    expect(screen.queryByLabelText('Collection Name')).toBeNull();
    expect(screen.queryByLabelText('Persist Path')).toBeNull();
  });

  it('shows chroma-specific fields when backend is chroma', () => {
    render(
      <NodeConfigModal
        {...baseProps}
        nodeType="memory"
        nodeData={{ label: 'Memory', backend: 'chroma' }}
      />,
    );
    expect(screen.getByLabelText('Collection Name')).toBeDefined();
    expect(screen.getByLabelText('Persist Path')).toBeDefined();
  });

  it('reveals chroma fields after switching backend to chroma', () => {
    render(
      <NodeConfigModal
        {...baseProps}
        nodeType="memory"
        nodeData={{ label: 'Memory', backend: 'sqlite_fts' }}
      />,
    );
    const backendSelect = screen.getByLabelText('Backend');
    fireEvent.change(backendSelect, { target: { value: 'chroma' } });
    expect(screen.getByLabelText('Collection Name')).toBeDefined();
  });

  it('saves all memory config fields on submit', () => {
    const onSave = vi.fn();
    render(
      <NodeConfigModal
        {...baseProps}
        onSave={onSave}
        nodeType="memory"
        nodeData={{
          label: 'My Memory',
          operation: 'retrieve',
          backend: 'sqlite_fts',
          namespace: 'test-ns',
          top_k: 10,
          include_metadata: true,
        }}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    expect(onSave).toHaveBeenCalledOnce();
    const [savedNodeId, savedData] = onSave.mock.calls[0];
    expect(savedNodeId).toBe('node-1');
    expect(savedData.operation).toBe('retrieve');
    expect(savedData.backend).toBe('sqlite_fts');
    expect(savedData.namespace).toBe('test-ns');
    expect(savedData.top_k).toBe(10);
    expect(savedData.include_metadata).toBe(true);
  });

  it('saves chroma-specific fields', () => {
    const onSave = vi.fn();
    render(
      <NodeConfigModal
        {...baseProps}
        onSave={onSave}
        nodeType="memory"
        nodeData={{
          label: 'Chroma Memory',
          backend: 'chroma',
          collection: 'my_collection',
          persist_path: '/tmp/chroma',
        }}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /save/i }));
    const [, savedData] = onSave.mock.calls[0];
    expect(savedData.collection).toBe('my_collection');
    expect(savedData.persist_path).toBe('/tmp/chroma');
  });

  it('does not render when isOpen is false', () => {
    const { container } = render(
      <NodeConfigModal
        {...baseProps}
        isOpen={false}
        nodeType="memory"
        nodeData={{}}
      />,
    );
    expect(container.firstChild).toBeNull();
  });
});
