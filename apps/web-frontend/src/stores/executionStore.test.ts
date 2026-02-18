import { vi } from 'vitest';
import { useExecutionStore } from './executionStore';

describe('useExecutionStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useExecutionStore.setState({
      results: {},
      setResults: vi.fn(),
      clearResults: vi.fn(),
    });
  });

  it('should set results', () => {
    const { setResults } = useExecutionStore.getState();
    const newResults = { 'node-1': { output: 'Test Output' } };
    setResults(newResults);
    expect(useExecutionStore.getState().results).toEqual(newResults);
  });

  it('should clear results', () => {
    useExecutionStore.setState({ results: { 'node-1': { output: 'Test Output' } } });
    const { clearResults } = useExecutionStore.getState();
    clearResults();
    expect(useExecutionStore.getState().results).toEqual({});
  });
});
