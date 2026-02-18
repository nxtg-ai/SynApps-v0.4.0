import { act } from '@testing-library/react';
import { useWorkflowStore } from './workflowStore';
import { vi } from 'vitest';

// Mock generateId from flowUtils to ensure consistent IDs during tests
vi.mock('@/utils/flowUtils', () => ({
  generateId: vi.fn(() => 'mock-id'),
}));

describe('useWorkflowStore', () => {
  beforeEach(() => {
    // Reset the store to its initial state using the store's own reset action
    act(() => {
      useWorkflowStore.getState().resetWorkflowState();
    });
  });

  it('should have a default initial state after reset', () => {
    const state = useWorkflowStore.getState();
    expect(state.flow).toBeNull();
    expect(state.isLoading).toBe(true);
    expect(state.isSaving).toBe(false);
    expect(state.showTemplates).toBe(false);
    expect(state.showCodeEditor).toBe(false);
    expect(state.selectedApplet).toBe("");
    expect(state.appletCode).toBe("");
    expect(state.inputData).toBe("");
    expect(state.imageGenerator).toBe("stability");
  });

  it('should set flow correctly', () => {
    const { setFlow } = useWorkflowStore.getState();
    const newFlow = {
      id: 'flow-1',
      name: 'Test Flow',
      nodes: [],
      edges: []
    };
    act(() => {
      setFlow(newFlow);
    });
    expect(useWorkflowStore.getState().flow).toEqual(newFlow);
  });

  it('should set isLoading correctly', () => {
    const { setIsLoading } = useWorkflowStore.getState();
    act(() => {
      setIsLoading(false);
    });
    expect(useWorkflowStore.getState().isLoading).toBe(false);
  });

  it('should set isSaving correctly', () => {
    const { setIsSaving } = useWorkflowStore.getState();
    act(() => {
      setIsSaving(true);
    });
    expect(useWorkflowStore.getState().isSaving).toBe(true);
  });

  it('should set showTemplates correctly', () => {
    const { setShowTemplates } = useWorkflowStore.getState();
    act(() => {
      setShowTemplates(true);
    });
    expect(useWorkflowStore.getState().showTemplates).toBe(true);
  });

  it('should set showCodeEditor correctly', () => {
    const { setShowCodeEditor } = useWorkflowStore.getState();
    act(() => {
      setShowCodeEditor(true);
    });
    expect(useWorkflowStore.getState().showCodeEditor).toBe(true);
  });

  it('should set selectedApplet correctly', () => {
    const { setSelectedApplet } = useWorkflowStore.getState();
    act(() => {
      setSelectedApplet("test-applet");
    });
    expect(useWorkflowStore.getState().selectedApplet).toBe("test-applet");
  });

  it('should set appletCode correctly', () => {
    const { setAppletCode } = useWorkflowStore.getState();
    act(() => {
      setAppletCode("console.log('hello');");
    });
    expect(useWorkflowStore.getState().appletCode).toBe("console.log('hello');");
  });

  it('should set inputData correctly', () => {
    const { setInputData } = useWorkflowStore.getState();
    act(() => {
      setInputData("some input");
    });
    expect(useWorkflowStore.getState().inputData).toBe("some input");
  });

  it('should set imageGenerator correctly', () => {
    const { setImageGenerator } = useWorkflowStore.getState();
    act(() => {
      setImageGenerator("dall-e");
    });
    expect(useWorkflowStore.getState().imageGenerator).toBe("dall-e");
  });

  it('should create an empty flow', () => {
    const { createEmptyFlow } = useWorkflowStore.getState();
    act(() => {
      createEmptyFlow();
    });
    const state = useWorkflowStore.getState();
    expect(state.flow).not.toBeNull();
    expect(state.flow?.name).toBe('New Workflow');
    expect(state.flow?.nodes).toHaveLength(2); // Start and End nodes
    expect(state.flow?.nodes[0].id).toBe('mock-id'); // Uses mocked ID
    expect(state.flow?.nodes[1].id).toBe('mock-id'); // Uses mocked ID
    expect(state.flow?.edges).toHaveLength(0);
  });
});

