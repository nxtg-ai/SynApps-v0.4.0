/**
 * Chatbot with Memory template
 * Memory ↔ Writer in a loop configuration
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const chatbotWithMemoryTemplate: FlowTemplate = {
  id: 'chatbot-memory',
  name: 'Chatbot with Memory',
  description: 'A chatbot that remembers previous interactions',
  tags: ['chat', 'memory', 'conversation'],
  flow: {
    id: generateId(),
    name: 'Chatbot with Memory',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 250, y: 25 },
        data: { label: 'Start' }
      },
      {
        id: 'memory-retrieve',
        type: 'memory',
        position: { x: 250, y: 125 },
        data: { 
          label: 'Retrieve Context',
          operation: 'retrieve',
          key: 'chat-history' 
        }
      },
      {
        id: 'writer',
        type: 'writer',
        position: { x: 250, y: 225 },
        data: { 
          label: 'Chatbot',
          systemPrompt: 'You are a friendly and helpful assistant. Provide informative and engaging responses to user questions. Use the chat history for context if available.'
        }
      },
      {
        id: 'memory-store',
        type: 'memory',
        position: { x: 250, y: 325 },
        data: { 
          label: 'Store Response',
          operation: 'store',
          key: 'chat-history' 
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
        id: 'start-memory-retrieve',
        source: 'start',
        target: 'memory-retrieve',
        animated: false
      },
      {
        id: 'memory-retrieve-writer',
        source: 'memory-retrieve',
        target: 'writer',
        animated: false
      },
      {
        id: 'writer-memory-store',
        source: 'writer',
        target: 'memory-store',
        animated: false
      },
      {
        id: 'memory-store-end',
        source: 'memory-store',
        target: 'end',
        animated: false
      }
    ]
  }
};
