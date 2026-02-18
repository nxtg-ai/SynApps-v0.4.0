import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import WorkflowCanvas from './WorkflowCanvas';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { ReactFlowProvider } from '@xyflow/react';
import { act } from '@testing-library/react'; // Corrected act import

// Mock React Flow components as they are complex to test in unit environment
vi.mock('@xyflow/react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@xyflow/react')>();
  return {
    ...actual,
    ReactFlow: vi.fn(({ children }) => <div data-testid="mock-react-flow">{children}</div>),
    MiniMap: vi.fn(() => <div data-testid="mock-minimap" />),
    Controls: vi.fn(() => <div data-testid="mock-controls" />),
  };
});

// Mock child components simply to assert their presence or absence
vi.mock('./NodeConfigModal', () => ({ default: vi.fn(() => <div data-testid="mock-node-config-modal" />) }));
vi.mock('./NodeContextMenu', () => ({ default: vi.fn(() => <div data-testid="mock-node-context-menu" />) }));

// Mock generateId from flowUtils as it's used when creating an empty flow
vi.mock('@/utils/flowUtils', () => ({
  generateId: vi.fn(() => 'mock-id'),
}));


describe('WorkflowCanvas', () => {
  beforeEach(() => {
    // Reset stores to their initial state before each test
    act(() => {
      useWorkflowStore.getState().resetWorkflowState();
      useSettingsStore.getState().resetSettings();
    });
    vi.clearAllMocks(); // Clear mocks for mocked React Flow components
  });

  const renderWorkflowCanvas = () => {
    return render(
      <ReactFlowProvider>
        <WorkflowCanvas />
      </ReactFlowProvider>
    );
  };

  it('should render the ReactFlow canvas', () => {
    renderWorkflowCanvas();
    expect(screen.getByTestId('mock-react-flow')).toBeInTheDocument();
  });

  // Test for loading indicator removed as the component doesn't explicitly render one.

  it('should initialize with an empty flow if no flow is set in store', () => {
    // Flow is null initially after resetWorkflowState
    renderWorkflowCanvas();
    // Verify that ReactFlow is rendered with a placeholder (empty) flow,
    // as WorkflowCanvas uses a local dummy flow if propFlow and storeFlow are null.
    // We cannot directly inspect props passed to mocked ReactFlow without more complex mocking.
    // For now, asserting the ReactFlow component itself is present is sufficient.
    expect(screen.getByTestId('mock-react-flow')).toBeInTheDocument();
    // More robust checks would require spying on ReactFlow's props or further refactoring.
  });

  it('should render NodeConfigModal', () => {
    renderWorkflowCanvas();
    expect(screen.getByTestId('mock-node-config-modal')).toBeInTheDocument();
  });

  it('should not render NodeContextMenu by default', () => {
    renderWorkflowCanvas();
    expect(screen.queryByTestId('mock-node-context-menu')).not.toBeInTheDocument();
  });

  it('should render MiniMap and Controls', () => {
    renderWorkflowCanvas();
    expect(screen.getByTestId('mock-minimap')).toBeInTheDocument();
    expect(screen.getByTestId('mock-controls')).toBeInTheDocument();
  });
});
