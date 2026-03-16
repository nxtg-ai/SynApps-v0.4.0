/**
 * CodeEditor component
 * Monaco-based editor with AI assistance for editing applet code
 */
import React, { useState, useRef, useEffect } from 'react';
import Editor, { Monaco } from '@monaco-editor/react';
import { editor as monacoEditor } from 'monaco-editor';
import apiService from '../../services/ApiService';
import './CodeEditor.css';

interface CodeEditorProps {
  appletType: string;
  initialCode: string;
  readOnly?: boolean;
  onSave?: (code: string) => void;
  onClose?: () => void;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  appletType,
  initialCode,
  readOnly = false,
  onSave,
  onClose
}) => {
  const [code, setCode] = useState<string>(initialCode);
  const [hint, setHint] = useState<string>('');
  const [suggestion, setSuggestion] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [showDiff, setShowDiff] = useState<boolean>(false);
  const editorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);
  
  // Handle editor mounting
  const handleEditorDidMount = (editor: monacoEditor.IStandaloneCodeEditor, monaco: Monaco) => {
    editorRef.current = editor;
    
    // Set editor options
    editor.updateOptions({
      fontSize: 14,
      minimap: {
        enabled: true
      },
      scrollBeyondLastLine: false,
      wordWrap: 'on',
      automaticLayout: true,
      snippetSuggestions: 'top',
      suggestOnTriggerCharacters: true
    });
    
    // Add keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      handleSave();
    });
  };
  
  // Handle code changes
  const handleCodeChange = (value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  };
  
  // Generate AI suggestion
  const handleGenerateSuggestion = async () => {
    if (!code || !hint || isGenerating) return;
    
    setIsGenerating(true);
    
    try {
      const response = await apiService.getCodeSuggestion({
        code,
        hint
      });
      
      setSuggestion(response.suggestion);
      setShowDiff(true);
    } catch (error) {
      console.error('Error generating suggestion:', error);
    } finally {
      setIsGenerating(false);
    }
  };
  
  // Apply suggestion to editor
  const handleApplySuggestion = () => {
    if (suggestion && editorRef.current) {
      editorRef.current.setValue(suggestion);
      setCode(suggestion);
      setSuggestion('');
      setShowDiff(false);
    }
  };
  
  // Save changes
  const handleSave = () => {
    if (onSave) {
      onSave(code);
    }
  };
  
  return (
    <div className="code-editor">
      <div className="code-editor-header">
        <div className="applet-info">
          <span className="applet-type">{appletType}</span>
          <span className="file-name">applet.py</span>
        </div>
        
        <div className="editor-actions">
          {!readOnly && (
            <button 
              className="save-button"
              onClick={handleSave}
            >
              Save Changes
            </button>
          )}
          
          {onClose && (
            <button 
              className="close-button"
              onClick={onClose}
            >
              Close
            </button>
          )}
        </div>
      </div>
      
      <div className="editor-container">
        <Editor
          height="70vh"
          language="python"
          value={showDiff ? suggestion : code}
          onChange={handleCodeChange}
          onMount={handleEditorDidMount}
          options={{
            readOnly: readOnly || showDiff
          }}
          theme="vs-light"
        />
      </div>
      
      {!readOnly && (
        <div className="ai-assistant">
          <div className="ai-header">
            <h3>AI Code Assistant</h3>
            {showDiff && (
              <div className="diff-actions">
                <button 
                  className="apply-button"
                  onClick={handleApplySuggestion}
                >
                  Apply Changes
                </button>
                <button 
                  className="cancel-button"
                  onClick={() => setShowDiff(false)}
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
          
          {!showDiff && (
            <div className="suggestion-input">
              <input
                type="text"
                placeholder="Describe what you want to change or ask for help..."
                value={hint}
                onChange={(e) => setHint(e.target.value)}
                disabled={isGenerating}
              />
              <button
                className="generate-button"
                onClick={handleGenerateSuggestion}
                disabled={!hint || isGenerating}
              >
                {isGenerating ? 'Generating...' : 'Get Suggestion'}
              </button>
            </div>
          )}
          
          <div className="quick-actions">
            <button 
              className="quick-action" 
              onClick={() => {
                setHint('Add error handling with try/except');
                handleGenerateSuggestion();
              }}
              disabled={isGenerating || showDiff}
            >
              Add Error Handling
            </button>
            <button 
              className="quick-action" 
              onClick={() => {
                setHint('Add more detailed comments and docstrings');
                handleGenerateSuggestion();
              }}
              disabled={isGenerating || showDiff}
            >
              Improve Documentation
            </button>
            <button 
              className="quick-action" 
              onClick={() => {
                setHint('Optimize the code for better performance');
                handleGenerateSuggestion();
              }}
              disabled={isGenerating || showDiff}
            >
              Optimize Code
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeEditor;
