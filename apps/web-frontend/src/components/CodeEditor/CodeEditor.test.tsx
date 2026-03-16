import React, { useEffect } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import CodeEditor from './CodeEditor';

const { mockApiService } = vi.hoisted(() => ({
  mockApiService: {
    getCodeSuggestion: vi.fn(),
  },
}));

const { mockMonacoEditor, mockEditorInstance, mockMonacoApi } = vi.hoisted(() => {
  const editorInstance = {
    updateOptions: vi.fn(),
    addCommand: vi.fn(),
    setValue: vi.fn(),
  };

  const monacoApi = {
    KeyMod: { CtrlCmd: 1 },
    KeyCode: { KeyS: 2 },
  };

  const editor = vi.fn((props: any) => {
    useEffect(() => {
      props.onMount?.(editorInstance, monacoApi);
    }, [props]);

    return (
      <div data-testid="mock-monaco-editor">
        <button type="button" onClick={() => props.onChange?.('updated-from-editor')}>
          Emit Editor Change
        </button>
        <div data-testid="editor-value">{String(props.value)}</div>
        <div data-testid="editor-readonly">{String(Boolean(props.options?.readOnly))}</div>
      </div>
    );
  });

  return {
    mockMonacoEditor: editor,
    mockEditorInstance: editorInstance,
    mockMonacoApi: monacoApi,
  };
});

vi.mock('@monaco-editor/react', () => ({
  default: mockMonacoEditor,
}));

vi.mock('../../services/ApiService', () => ({
  __esModule: true,
  default: mockApiService,
  apiService: mockApiService,
}));

describe('CodeEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders editor and configures Monaco instance on mount', () => {
    render(<CodeEditor appletType="writer" initialCode="print('hello')" />);

    expect(screen.getByTestId('mock-monaco-editor')).toBeInTheDocument();
    expect(mockEditorInstance.updateOptions).toHaveBeenCalledWith(
      expect.objectContaining({
        fontSize: 14,
        wordWrap: 'on',
        automaticLayout: true,
      }),
    );
    expect(mockEditorInstance.addCommand).toHaveBeenCalledWith(
      mockMonacoApi.KeyMod.CtrlCmd | mockMonacoApi.KeyCode.KeyS,
      expect.any(Function),
    );
  });

  it('saves updated code via editor change and Save Changes button', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();

    render(<CodeEditor appletType="writer" initialCode="initial" onSave={onSave} />);

    await user.click(screen.getByRole('button', { name: 'Emit Editor Change' }));
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(onSave).toHaveBeenCalledWith('updated-from-editor');
  });

  it('supports save keyboard shortcut registered through Monaco command', async () => {
    const onSave = vi.fn();
    render(<CodeEditor appletType="writer" initialCode="shortcut-code" onSave={onSave} />);

    const saveCommand = mockEditorInstance.addCommand.mock.calls[0][1] as () => void;
    saveCommand();

    expect(onSave).toHaveBeenCalledWith('shortcut-code');
  });

  it('shows AI diff and applies suggestion into editor', async () => {
    const user = userEvent.setup();
    const onSave = vi.fn();
    mockApiService.getCodeSuggestion.mockResolvedValue({
      suggestion: 'print("improved")',
    });

    render(<CodeEditor appletType="writer" initialCode="print('raw')" onSave={onSave} />);

    await user.type(
      screen.getByPlaceholderText('Describe what you want to change or ask for help...'),
      'Improve clarity',
    );
    await user.click(screen.getByRole('button', { name: 'Get Suggestion' }));

    await waitFor(() => {
      expect(mockApiService.getCodeSuggestion).toHaveBeenCalledWith({
        code: "print('raw')",
        hint: 'Improve clarity',
      });
    });

    expect(screen.getByRole('button', { name: 'Apply Changes' })).toBeInTheDocument();
    expect(screen.getByTestId('editor-readonly')).toHaveTextContent('true');

    await user.click(screen.getByRole('button', { name: 'Apply Changes' }));
    expect(mockEditorInstance.setValue).toHaveBeenCalledWith('print("improved")');
    expect(screen.queryByRole('button', { name: 'Apply Changes' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Save Changes' }));
    expect(onSave).toHaveBeenCalledWith('print("improved")');
  });

  it('cancels suggestion diff view without applying', async () => {
    const user = userEvent.setup();
    mockApiService.getCodeSuggestion.mockResolvedValue({
      suggestion: 'print("alternative")',
    });

    render(<CodeEditor appletType="writer" initialCode="print('raw')" />);

    await user.type(
      screen.getByPlaceholderText('Describe what you want to change or ask for help...'),
      'Do something',
    );
    await user.click(screen.getByRole('button', { name: 'Get Suggestion' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('button', { name: 'Apply Changes' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Get Suggestion' })).toBeInTheDocument();
  });

  it('handles suggestion generation errors and restores generate state', async () => {
    const user = userEvent.setup();
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    mockApiService.getCodeSuggestion.mockRejectedValue(new Error('suggestion failed'));

    render(<CodeEditor appletType="writer" initialCode="print('x')" />);

    await user.type(
      screen.getByPlaceholderText('Describe what you want to change or ask for help...'),
      'Fail please',
    );
    await user.click(screen.getByRole('button', { name: 'Get Suggestion' }));

    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith('Error generating suggestion:', expect.any(Error));
    });
    expect(screen.getByRole('button', { name: 'Get Suggestion' })).toBeInTheDocument();
  });

  it('hides assistant controls when readOnly and supports close callback', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<CodeEditor appletType="writer" initialCode="print('x')" readOnly onClose={onClose} />);

    expect(screen.queryByText('AI Code Assistant')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Save Changes' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
