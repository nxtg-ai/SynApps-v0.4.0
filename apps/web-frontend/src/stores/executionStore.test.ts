import { act } from '@testing-library/react';
import { useExecutionStore } from './executionStore';
import { WorkflowRunStatus } from '@/types';

describe('useExecutionStore', () => {
  beforeEach(() => {
    // Reset the store to its initial state using the store's own reset action
    act(() => {
      useExecutionStore.getState().resetExecutionState();
    });
  });

  it('should have a default initial state after reset', () => {
    const state = useExecutionStore.getState();
    expect(state.isRunning).toBe(false);
    expect(state.workflowResults).toBeNull();
    expect(state.runStatus).toBeNull();
    expect(state.completedNodes).toEqual([]);
  });

  it('should set isRunning correctly', () => {
    const { setIsRunning } = useExecutionStore.getState();
    act(() => {
      setIsRunning(true);
    });
    expect(useExecutionStore.getState().isRunning).toBe(true);
  });

  it('should set workflowResults correctly', () => {
    const { setWorkflowResults } = useExecutionStore.getState();
    const results = { 'node-1': { output: 'test' } };
    act(() => {
      setWorkflowResults(results);
    });
    expect(useExecutionStore.getState().workflowResults).toEqual(results);
  });

  it('should set runStatus correctly', () => {
    const { setRunStatus } = useExecutionStore.getState();
    const status: WorkflowRunStatus = 'running';
    act(() => {
      setRunStatus(status);
    });
    expect(useExecutionStore.getState().runStatus).toBe(status);
  });

  it('should set completedNodes correctly', () => {
    const { setCompletedNodes } = useExecutionStore.getState();
    const nodes = ['node-1', 'node-2'];
    act(() => {
      setCompletedNodes(nodes);
    });
    expect(useExecutionStore.getState().completedNodes).toEqual(nodes);
  });

  it('should clear results when resetExecutionState is called', () => {
    act(() => {
      useExecutionStore.setState({ workflowResults: { 'node-1': { output: 'Test Output' } }, isRunning: true });
    });
    const { resetExecutionState } = useExecutionStore.getState();
    act(() => {
      resetExecutionState();
    });
    const state = useExecutionStore.getState();
    expect(state.workflowResults).toBeNull();
    expect(state.isRunning).toBe(false);
  });
});
