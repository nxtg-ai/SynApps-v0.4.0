/**
 * Illustrated Story Generator template
 * Chain of Writer → Artist for creating stories with images
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const illustratedStoryTemplate: FlowTemplate = {
  id: 'illustrated-story',
  name: 'Illustrated Story Generator',
  description: 'Creates a story with accompanying illustrations',
  tags: ['creative', 'story', 'image'],
  flow: {
    id: generateId(),
    name: 'Illustrated Story',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 250, y: 25 },
        data: { label: 'Start' }
      },
      {
        id: 'writer',
        type: 'writer',
        position: { x: 250, y: 125 },
        data: { 
          label: 'Story Writer',
          systemPrompt: 'You are a creative storyteller. Create a short story based on the provided theme or idea. Make it vivid and descriptive, suitable for illustration.'
        }
      },
      {
        id: 'artist',
        type: 'artist',
        position: { x: 250, y: 225 },
        data: { 
          label: 'Illustrator',
          style: 'digital art' 
        }
      },
      {
        id: 'end',
        type: 'end',
        position: { x: 250, y: 325 },
        data: { label: 'End' }
      }
    ],
    edges: [
      {
        id: 'start-writer',
        source: 'start',
        target: 'writer',
        animated: false
      },
      {
        id: 'writer-artist',
        source: 'writer',
        target: 'artist',
        animated: false
      },
      {
        id: 'artist-end',
        source: 'artist',
        target: 'end',
        animated: false
      }
    ]
  }
};
