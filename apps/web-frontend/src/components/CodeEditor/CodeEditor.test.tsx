import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import CodeEditor from './CodeEditor';
import userEvent from '@testing-library/user-event';
import { act } from '@testing-library/react';
import React from 'react';

// Use vi.hoisted to define mockMonacoEditor so it's available when vi.mock is hoisted
const { mockMonacoEditor } = vi.hoisted(() => {
  // A very simple mock for the Monaco Editor component
  // It will just render a div and capture the props it was called with.
  const mockMonacoEditor = vi.fn((props) => {
    return (
      <div
        data-testid="mock-monaco-editor"
        data-value={props.value}
        data-language={props.language}
        data-readonly={props.options?.readOnly ? 'true' : 'false'} // Readonly is inside options
      >
        Mock Monaco Editor Content (Value: {props.value})
      </div>
    );
  });
  return { mockMonacoEditor };
});

// Mock the @monaco-editor/react component using the hoisted mock
vi.mock('@monaco-editor/react', () => ({
  default: mockMonacoEditor,
}));

describe('CodeEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Helper to get the props passed to the mocked Monaco editor from the *latest* call
  const getLatestMonacoEditorProps = () => {
    const calls = mockMonacoEditor.mock.calls;
    expect(calls.length).toBeGreaterThan(0); // Ensure there's at least one call
    return calls[calls.length - 1][0]; // Get props from the last call
  };

  it('should render the mocked Monaco editor', () => {
    render(<CodeEditor appletType="test" initialCode="" />);
    expect(screen.getByTestId('mock-monaco-editor')).toBeInTheDocument();
  });

  it('should pass the correct initialCode to the editor as value', () => {
    const testInitialCode = 'console.log("hello world");';
    render(<CodeEditor appletType="test" initialCode={testInitialCode} />);
    expect(getLatestMonacoEditorProps().value).toBe(testInitialCode);
    expect(screen.getByText(`Mock Monaco Editor Content (Value: ${testInitialCode})`)).toBeInTheDocument();
  });

  it('should pass the hardcoded language, height, and theme to the editor', () => {
    render(<CodeEditor appletType="test" initialCode="" />);
    expect(getLatestMonacoEditorProps().language).toBe('python');
    expect(getLatestMonacoEditorProps().height).toBe('70vh');
    expect(getLatestMonacoEditorProps().theme).toBe('vs-light');
  });

  it('should pass the readOnly prop to the editor options', () => {
    const { rerender } = render(<CodeEditor appletType="test" initialCode="" readOnly={true} />);
    expect(getLatestMonacoEditorProps().options.readOnly).toBe(true);
    expect(screen.getByTestId('mock-monaco-editor')).toHaveAttribute('data-readonly', 'true');

    rerender(<CodeEditor appletType="test" initialCode="" readOnly={false} />);
    expect(getLatestMonacoEditorProps().options.readOnly).toBe(false); // Now showDiff is false by default
    expect(screen.getByTestId('mock-monaco-editor')).toHaveAttribute('data-readonly', 'false');
  });

  it('should call onChange with the new value and update editor value', async () => {
    const onSaveMock = vi.fn();
    const { rerender } = render(<CodeEditor appletType="test" initialCode="initial" onSave={onSaveMock} />);

    const newCode = 'updated code';

    await act(async () => {
      // Simulate the editor's onChange event by directly calling the onChange function
      // that CodeEditor passed to the mocked Monaco editor.
      getLatestMonacoEditorProps().onChange(newCode);
    });
    
    // Rerender to force CodeEditor to pass updated props to the mock after its internal state changes
    rerender(<CodeEditor appletType="test" initialCode="initial" onSave={onSaveMock} />);

    // Verify that the mocked editor's value prop now reflects the newCode
    expect(getLatestMonacoEditorProps().value).toBe(newCode);
  });
});