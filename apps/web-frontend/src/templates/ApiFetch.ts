/**
 * API Fetch template
 * Fetch JSON from a public REST API, extract a field with Transform, pipe to End
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const apiFetchTemplate: FlowTemplate = {
  id: 'api-fetch-template',
  name: 'Fetch External API → Transform → Display',
  description:
    'Fetch JSON data from a public REST API, extract a field with Transform, and pipe the result to End. Demonstrates the HTTP Request → Transform pipeline using jsonplaceholder.typicode.com.',
  tags: ['http', 'api', 'integration', 'beginner'],
  flow: {
    id: generateId(),
    name: 'Fetch External API → Transform → Display',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 100, y: 150 },
        data: { label: 'Start' },
      },
      {
        id: 'fetch',
        type: 'http_request',
        position: { x: 300, y: 150 },
        data: {
          label: 'Fetch Post',
          method: 'GET',
          url: 'https://jsonplaceholder.typicode.com/posts/1',
          headers: { Accept: 'application/json' },
          timeout_seconds: 30,
          auth_type: 'none',
          max_retries: 0,
          allow_redirects: true,
          verify_ssl: true,
        },
      },
      {
        id: 'extract',
        type: 'transform',
        position: { x: 500, y: 150 },
        data: {
          label: 'Extract Title',
          operation: 'json_path',
          json_path: '$.data.title',
        },
      },
      {
        id: 'end',
        type: 'end',
        position: { x: 700, y: 150 },
        data: { label: 'End' },
      },
    ],
    edges: [
      { id: 'e1', source: 'start', target: 'fetch', animated: false },
      { id: 'e2', source: 'fetch', target: 'extract', animated: false },
      { id: 'e3', source: 'extract', target: 'end', animated: false },
    ],
  },
};
