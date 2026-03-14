import React, { useState, useEffect } from 'react';
import './NodeConfigModal.css';

interface NodeConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  nodeId: string;
  nodeType: string;
  nodeData: any;
  onSave: (nodeId: string, updatedData: any) => void;
}

const NodeConfigModal: React.FC<NodeConfigModalProps> = ({ 
  isOpen, 
  onClose, 
  nodeId, 
  nodeType, 
  nodeData, 
  onSave 
}) => {
  const [formData, setFormData] = useState<any>({});
  
  // Initialize form data when modal opens or node changes
  useEffect(() => {
    if (isOpen && nodeData) {
      const initialData = { ...nodeData };
      if (nodeType === 'http_request') {
        // Apply defaults for http_request fields
        initialData.method = initialData.method || 'GET';
        initialData.timeout_seconds = initialData.timeout_seconds ?? 30;
        initialData.allow_redirects = initialData.allow_redirects ?? true;
        initialData.verify_ssl = initialData.verify_ssl ?? true;
        initialData.auth_type = initialData.auth_type || 'none';
        initialData.max_retries = initialData.max_retries ?? 0;
        initialData.body_type = initialData.body_type || 'auto';
        // Serialize headers/query_params objects to JSON strings for editing
        initialData.headers_json = initialData.headers && typeof initialData.headers === 'object'
          ? JSON.stringify(initialData.headers, null, 2)
          : '{}';
        initialData.query_params_json = initialData.query_params && typeof initialData.query_params === 'object'
          ? JSON.stringify(initialData.query_params, null, 2)
          : '{}';
      }
      setFormData(initialData);
    }
  }, [isOpen, nodeData, nodeType]);
  
  if (!isOpen) return null;
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  const handleIntChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: parseInt(value, 10)
    }));
  };

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setFormData((prev: Record<string, any>) => ({
      ...prev,
      [name]: checked
    }));
  };
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (nodeType === 'http_request') {
      // Parse JSON text fields back to objects before saving
      const saveData = { ...formData };
      try {
        saveData.headers = JSON.parse(saveData.headers_json || '{}');
      } catch {
        saveData.headers = {};
      }
      try {
        saveData.query_params = JSON.parse(saveData.query_params_json || '{}');
      } catch {
        saveData.query_params = {};
      }
      // Remove the temporary JSON string fields
      delete saveData.headers_json;
      delete saveData.query_params_json;
      onSave(nodeId, saveData);
    } else {
      onSave(nodeId, formData);
    }
    onClose();
  };
  
  // Render different form fields based on node type
  const renderFormFields = () => {
    switch (nodeType) {
      case 'llm':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="provider">Provider</label>
              <select
                id="provider"
                name="provider"
                value={formData.provider || 'openai'}
                onChange={handleChange}
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="google">Google</option>
                <option value="ollama">Ollama</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="model">Model</label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model || ''}
                onChange={handleChange}
                placeholder={formData.provider === 'anthropic' ? 'claude-sonnet-4-20250514' : formData.provider === 'google' ? 'gemini-2.0-flash' : formData.provider === 'ollama' ? 'llama3' : 'gpt-4o'}
              />
            </div>
            <div className="form-group">
              <label htmlFor="system_prompt">System Prompt</label>
              <textarea
                id="system_prompt"
                name="system_prompt"
                value={formData.system_prompt || ''}
                onChange={handleChange}
                placeholder="Enter system instructions for this LLM node"
                rows={5}
              />
            </div>
            <div className="form-group">
              <label htmlFor="temperature">Temperature ({formData.temperature ?? 0.7})</label>
              <input
                type="range"
                id="temperature"
                name="temperature"
                min="0"
                max="2"
                step="0.1"
                value={formData.temperature ?? 0.7}
                onChange={handleNumberChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="max_tokens">Max Tokens</label>
              <input
                type="number"
                id="max_tokens"
                name="max_tokens"
                value={formData.max_tokens ?? 1000}
                onChange={handleNumberChange}
                min={1}
                max={128000}
              />
            </div>
            {(formData.provider === 'custom' || formData.provider === 'ollama') && (
              <div className="form-group">
                <label htmlFor="base_url">Base URL</label>
                <input
                  type="text"
                  id="base_url"
                  name="base_url"
                  value={formData.base_url || ''}
                  onChange={handleChange}
                  placeholder={formData.provider === 'ollama' ? 'http://localhost:11434' : 'https://api.example.com/v1'}
                />
              </div>
            )}
          </>
        );

      case 'writer':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="systemPrompt">System Prompt</label>
              <textarea
                id="systemPrompt"
                name="systemPrompt"
                value={formData.systemPrompt || ''}
                onChange={handleChange}
                placeholder="Enter system prompt"
                rows={5}
              />
            </div>
          </>
        );
      
      case 'artist':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="systemPrompt">System Prompt</label>
              <textarea
                id="systemPrompt"
                name="systemPrompt"
                value={formData.systemPrompt || ''}
                onChange={handleChange}
                placeholder="Enter system prompt"
                rows={5}
              />
            </div>
            <div className="form-group">
              <label htmlFor="generator">Image Generator</label>
              <select
                id="generator"
                name="generator"
                value={formData.generator || 'dall-e'}
                onChange={(e) => handleChange(e as any)}
              >
                <option value="dall-e">DALL-E</option>
                <option value="stability">Stability AI</option>
              </select>
            </div>
          </>
        );
      
      case 'memory':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="operation">Operation</label>
              <select
                id="operation"
                name="operation"
                value={formData.operation || 'store'}
                onChange={handleChange}
              >
                <option value="store">Store</option>
                <option value="retrieve">Retrieve</option>
                <option value="delete">Delete</option>
                <option value="clear">Clear</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="backend">Backend</label>
              <select
                id="backend"
                name="backend"
                value={formData.backend || 'sqlite_fts'}
                onChange={handleChange}
              >
                <option value="sqlite_fts">SQLite FTS (default)</option>
                <option value="chroma">ChromaDB (vector search)</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="namespace">Namespace</label>
              <input
                type="text"
                id="namespace"
                name="namespace"
                value={formData.namespace || ''}
                onChange={handleChange}
                placeholder="default"
              />
            </div>
            <div className="form-group">
              <label htmlFor="top_k">Top K (retrieve results)</label>
              <input
                type="number"
                id="top_k"
                name="top_k"
                value={formData.top_k ?? 5}
                onChange={handleIntChange}
                min={1}
                max={50}
              />
            </div>
            <div className="form-group">
              <label htmlFor="key">Key (optional, for keyed store/retrieve)</label>
              <input
                type="text"
                id="key"
                name="key"
                value={formData.key || ''}
                onChange={handleChange}
                placeholder="Leave blank for auto-generated key"
              />
            </div>
            {formData.backend === 'chroma' && (
              <>
                <div className="form-group">
                  <label htmlFor="collection">Collection Name</label>
                  <input
                    type="text"
                    id="collection"
                    name="collection"
                    value={formData.collection || ''}
                    onChange={handleChange}
                    placeholder="synapps_memory"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="persist_path">Persist Path</label>
                  <input
                    type="text"
                    id="persist_path"
                    name="persist_path"
                    value={formData.persist_path || ''}
                    onChange={handleChange}
                    placeholder="/tmp/chroma (leave blank for default)"
                  />
                </div>
              </>
            )}
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  name="include_metadata"
                  checked={formData.include_metadata ?? false}
                  onChange={handleCheckboxChange}
                />
                {' '}Include metadata in results
              </label>
            </div>
          </>
        );

      case 'merge':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="strategy">Merge Strategy</label>
              <select
                id="strategy"
                name="strategy"
                value={formData.strategy || 'array'}
                onChange={handleChange}
              >
                <option value="array">Array (collect all inputs as list)</option>
                <option value="concatenate">Concatenate (join as text)</option>
                <option value="first_wins">First Wins (use first result)</option>
              </select>
            </div>
            {formData.strategy === 'concatenate' && (
              <div className="form-group">
                <label htmlFor="delimiter">Delimiter</label>
                <input
                  type="text"
                  id="delimiter"
                  name="delimiter"
                  value={formData.delimiter || '\\n'}
                  onChange={handleChange}
                  placeholder="Separator between merged items"
                />
              </div>
            )}
          </>
        );

      case 'for_each':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="array_source">Array Source</label>
              <input
                type="text"
                id="array_source"
                name="array_source"
                value={formData.array_source || '{{input}}'}
                onChange={handleChange}
                placeholder="Template expression for the array to iterate"
              />
            </div>
            <div className="form-group">
              <label htmlFor="max_iterations">Max Iterations</label>
              <input
                type="number"
                id="max_iterations"
                name="max_iterations"
                value={formData.max_iterations ?? 1000}
                onChange={handleNumberChange}
                min={1}
                max={100000}
              />
            </div>
            <div className="form-group">
              <label htmlFor="parallel">
                <input
                  type="checkbox"
                  id="parallel"
                  name="parallel"
                  checked={formData.parallel || false}
                  onChange={(e) => {
                    setFormData((prev: Record<string, any>) => ({
                      ...prev,
                      parallel: e.target.checked
                    }));
                  }}
                />
                {' '}Execute iterations in parallel
              </label>
            </div>
            {formData.parallel && (
              <div className="form-group">
                <label htmlFor="concurrency_limit">Concurrency Limit</label>
                <input
                  type="number"
                  id="concurrency_limit"
                  name="concurrency_limit"
                  value={formData.concurrency_limit ?? 5}
                  onChange={handleNumberChange}
                  min={1}
                  max={50}
                />
              </div>
            )}
          </>
        );

      case 'if_else':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="operation">Condition</label>
              <select
                id="operation"
                name="operation"
                value={formData.operation || 'equals'}
                onChange={handleChange}
              >
                <option value="equals">Equals</option>
                <option value="contains">Contains</option>
                <option value="regex">Regex Match</option>
                <option value="json_path">JSON Path</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="source">Source Expression</label>
              <input
                type="text"
                id="source"
                name="source"
                value={formData.source || '{{content}}'}
                onChange={handleChange}
                placeholder="Template expression to evaluate"
              />
            </div>
            <div className="form-group">
              <label htmlFor="value">Expected Value</label>
              <input
                type="text"
                id="value"
                name="value"
                value={formData.value || ''}
                onChange={handleChange}
                placeholder="Value to compare against"
              />
            </div>
            <div className="form-group">
              <label htmlFor="negate">
                <input
                  type="checkbox"
                  id="negate"
                  name="negate"
                  checked={formData.negate || false}
                  onChange={(e) => {
                    setFormData((prev: Record<string, any>) => ({
                      ...prev,
                      negate: e.target.checked
                    }));
                  }}
                />
                {' '}Negate result (invert true/false)
              </label>
            </div>
            <div className="form-group">
              <label htmlFor="case_sensitive">
                <input
                  type="checkbox"
                  id="case_sensitive"
                  name="case_sensitive"
                  checked={formData.case_sensitive || false}
                  onChange={(e) => {
                    setFormData((prev: Record<string, any>) => ({
                      ...prev,
                      case_sensitive: e.target.checked
                    }));
                  }}
                />
                {' '}Case sensitive
              </label>
            </div>
          </>
        );

      case 'code':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="language">Language</label>
              <select
                id="language"
                name="language"
                value={formData.language || 'python'}
                onChange={handleChange}
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="code">Code</label>
              <textarea
                id="code"
                name="code"
                value={formData.code || ''}
                onChange={handleChange}
                placeholder={formData.language === 'javascript'
                  ? '// Access input via `data`, set output via `result`\nresult = data.toUpperCase();'
                  : '# Access input via `data`, set output via `result`\nresult = data.upper()'}
                rows={12}
                style={{ fontFamily: 'monospace', fontSize: '13px' }}
              />
            </div>
            <div className="form-group">
              <label htmlFor="timeout_seconds">Timeout (seconds)</label>
              <input
                type="number"
                id="timeout_seconds"
                name="timeout_seconds"
                value={formData.timeout_seconds ?? 5}
                onChange={handleNumberChange}
                min={1}
                max={120}
              />
            </div>
            <div className="form-group">
              <label htmlFor="memory_limit_mb">Memory Limit (MB)</label>
              <input
                type="number"
                id="memory_limit_mb"
                name="memory_limit_mb"
                value={formData.memory_limit_mb ?? 256}
                onChange={handleNumberChange}
                min={64}
                max={2048}
              />
            </div>
            <div className="form-group">
              <label htmlFor="cpu_time_seconds">CPU Time Limit (seconds)</label>
              <input
                type="number"
                id="cpu_time_seconds"
                name="cpu_time_seconds"
                value={formData.cpu_time_seconds ?? 3}
                onChange={handleNumberChange}
                min={1}
                max={60}
              />
            </div>
          </>
        );

      case 'http_request':
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
            <div className="form-group">
              <label htmlFor="method">Method</label>
              <select
                id="method"
                name="method"
                value={formData.method || 'GET'}
                onChange={handleChange}
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="PATCH">PATCH</option>
                <option value="DELETE">DELETE</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="url">URL</label>
              <input
                type="text"
                id="url"
                name="url"
                value={formData.url || ''}
                onChange={handleChange}
                required
                placeholder="https://api.example.com/endpoint"
              />
            </div>
            <div className="form-group">
              <label htmlFor="headers_json">Headers (JSON)</label>
              <textarea
                id="headers_json"
                name="headers_json"
                value={formData.headers_json || '{}'}
                onChange={handleChange}
                placeholder='{"Authorization": "Bearer ..."}'
                rows={3}
                style={{ fontFamily: 'monospace', fontSize: '13px' }}
              />
            </div>
            <div className="form-group">
              <label htmlFor="query_params_json">Query Params (JSON)</label>
              <textarea
                id="query_params_json"
                name="query_params_json"
                value={formData.query_params_json || '{}'}
                onChange={handleChange}
                placeholder='{"key": "value"}'
                rows={3}
                style={{ fontFamily: 'monospace', fontSize: '13px' }}
              />
            </div>
            <div className="form-group">
              <label htmlFor="auth_type">Auth Type</label>
              <select
                id="auth_type"
                name="auth_type"
                value={formData.auth_type || 'none'}
                onChange={handleChange}
              >
                <option value="none">None</option>
                <option value="bearer">Bearer Token</option>
                <option value="basic">Basic Auth</option>
                <option value="api_key">API Key</option>
              </select>
            </div>
            {formData.auth_type && formData.auth_type !== 'none' && (
              <div className="form-group">
                <label htmlFor="auth_value">
                  {formData.auth_type === 'bearer' ? 'Bearer Token' :
                   formData.auth_type === 'basic' ? 'Credentials (user:pass)' :
                   'API Key Value'}
                </label>
                <input
                  type="text"
                  id="auth_value"
                  name="auth_value"
                  value={formData.auth_value || ''}
                  onChange={handleChange}
                  placeholder={
                    formData.auth_type === 'bearer' ? 'eyJhbGciOi...' :
                    formData.auth_type === 'basic' ? 'username:password' :
                    'your-api-key'
                  }
                />
              </div>
            )}
            {formData.auth_type === 'api_key' && (
              <div className="form-group">
                <label htmlFor="auth_header_name">Auth Header Name</label>
                <input
                  type="text"
                  id="auth_header_name"
                  name="auth_header_name"
                  value={formData.auth_header_name || ''}
                  onChange={handleChange}
                  placeholder="X-API-Key"
                />
              </div>
            )}
            {['POST', 'PUT', 'PATCH', 'DELETE'].includes(formData.method || 'GET') && (
              <>
                <div className="form-group">
                  <label htmlFor="body_type">Body Type</label>
                  <select
                    id="body_type"
                    name="body_type"
                    value={formData.body_type || 'auto'}
                    onChange={handleChange}
                  >
                    <option value="auto">Auto</option>
                    <option value="json">JSON</option>
                    <option value="text">Text</option>
                    <option value="form">Form</option>
                    <option value="none">None</option>
                  </select>
                </div>
                {formData.body_type !== 'none' && (
                  <div className="form-group">
                    <label htmlFor="body_template">Body</label>
                    <textarea
                      id="body_template"
                      name="body_template"
                      value={formData.body_template || ''}
                      onChange={handleChange}
                      placeholder='{"key": "value"}'
                      rows={6}
                      style={{ fontFamily: 'monospace', fontSize: '13px' }}
                    />
                  </div>
                )}
              </>
            )}
            <div className="form-group">
              <label htmlFor="timeout_seconds">Timeout (seconds)</label>
              <input
                type="number"
                id="timeout_seconds"
                name="timeout_seconds"
                value={formData.timeout_seconds ?? 30}
                onChange={handleIntChange}
                min={1}
                max={600}
              />
            </div>
            <div className="form-group">
              <label htmlFor="max_retries">Max Retries</label>
              <input
                type="number"
                id="max_retries"
                name="max_retries"
                value={formData.max_retries ?? 0}
                onChange={handleIntChange}
                min={0}
                max={5}
              />
            </div>
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  name="allow_redirects"
                  checked={formData.allow_redirects ?? true}
                  onChange={handleCheckboxChange}
                />
                {' '}Allow Redirects
              </label>
            </div>
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  name="verify_ssl"
                  checked={formData.verify_ssl ?? true}
                  onChange={handleCheckboxChange}
                />
                {' '}Verify SSL
              </label>
            </div>
          </>
        );

      default:
        return (
          <>
            <div className="form-group">
              <label htmlFor="label">Node Label</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label || ''}
                onChange={handleChange}
                placeholder="Enter node label"
              />
            </div>
          </>
        );
    }
  };
  
  return (
    <div className="node-config-modal-overlay">
      <div className="node-config-modal">
        <div className="modal-header">
          <h2>Configure {nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} Node</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {renderFormFields()}
          </div>
          <div className="modal-footer">
            <button type="button" className="cancel-button" onClick={onClose}>Cancel</button>
            <button type="submit" className="save-button">Save Changes</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NodeConfigModal;
