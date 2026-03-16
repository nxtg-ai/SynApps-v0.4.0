/**
 * Blog Post Writer template
 * Chain of Writer → Memory → Writer for refining content
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const blogPostWriterTemplate: FlowTemplate = {
  id: 'blog-post-writer',
  name: 'Blog Post Writer',
  description: 'Generates a blog post in two stages with memory storage in between',
  tags: ['content', 'writing', 'blog'],
  flow: {
    id: generateId(),
    name: 'Blog Post Writer',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 250, y: 25 },
        data: { label: 'Start' }
      },
      {
        id: 'writer1',
        type: 'writer',
        position: { x: 250, y: 125 },
        data: { 
          label: 'Draft Writer',
          systemPrompt: 'You are a professional blog writer. Generate a first draft of a blog post on the given topic.'
        }
      },
      {
        id: 'memory',
        type: 'memory',
        position: { x: 250, y: 225 },
        data: { 
          label: 'Store Draft', 
          operation: 'store',
          key: 'blog-draft' 
        }
      },
      {
        id: 'writer2',
        type: 'writer',
        position: { x: 250, y: 325 },
        data: { 
          label: 'Refinement Writer',
          systemPrompt: 'You are a professional editor. Take the draft blog post and improve it by enhancing clarity, adding examples, and making the language more engaging.'
        }
      },
      {
        id: 'end',
        type: 'end',
        position: { x: 250, y: 425 },
        data: { label: 'End' }
      }
    ],
    edges: [
      {
        id: 'start-writer1',
        source: 'start',
        target: 'writer1',
        animated: false
      },
      {
        id: 'writer1-memory',
        source: 'writer1',
        target: 'memory',
        animated: false
      },
      {
        id: 'memory-writer2',
        source: 'memory',
        target: 'writer2',
        animated: false
      },
      {
        id: 'writer2-end',
        source: 'writer2',
        target: 'end',
        animated: false
      }
    ]
  }
};
