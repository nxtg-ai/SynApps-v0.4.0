/**
 * 2Brain Inbox Triage template
 * Captures raw text, classifies it via Ollama, structures the result,
 * and stores it in memory for later recall.
 *
 * Pipeline: Start → LLM Classifier → Code Structurer → Memory Store → End
 *
 * This is the first dogfood workflow — validates SynApps with
 * 2Brain's capture→classify→store pipeline (PI-001).
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const twoBrainInboxTemplate: FlowTemplate = {
  id: '2brain-inbox-triage',
  name: '2Brain Inbox Triage',
  description: 'Drop in any thought or note. Ollama classifies it (idea / task / reference / note) and stores it in memory with structured metadata.',
  tags: ['2brain', 'ollama', 'classification', 'memory', 'local-ai'],
  flow: {
    id: generateId(),
    name: '2Brain Inbox Triage',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 300, y: 25 },
        data: { label: 'Raw Inbox Item' }
      },
      {
        id: 'classifier',
        type: 'llm',
        position: { x: 300, y: 150 },
        data: {
          label: 'Ollama Classifier',
          provider: 'ollama',
          model: 'llama3.1',
          base_url: 'http://localhost:11434',
          temperature: 0.1,
          max_tokens: 64,
          system_prompt:
            'You are a triage assistant for a personal knowledge base called 2Brain. ' +
            'Classify the user\'s input into exactly one of these categories: idea, task, reference, note. ' +
            'Respond with ONLY the single category word — no explanation, no punctuation, no extra text.'
        }
      },
      {
        id: 'structurer',
        type: 'code',
        position: { x: 300, y: 300 },
        data: {
          label: 'Structure Output',
          language: 'python',
          timeout_seconds: 5,
          memory_limit_mb: 256,
          code:
            'import json\n' +
            'import datetime\n' +
            '\n' +
            '# data = LLM output (category string)\n' +
            '# context["input"] = original user input\n' +
            'raw_input = context.get("input", {})\n' +
            'if isinstance(raw_input, dict):\n' +
            '    original_text = raw_input.get("text", str(raw_input))\n' +
            'else:\n' +
            '    original_text = str(raw_input)\n' +
            '\n' +
            'category = str(data).strip().lower() if data else "note"\n' +
            'if category not in {"idea", "task", "reference", "note"}:\n' +
            '    category = "note"\n' +
            '\n' +
            'result = {\n' +
            '    "category": category,\n' +
            '    "content": original_text,\n' +
            '    "captured_at": datetime.datetime.utcnow().isoformat() + "Z",\n' +
            '    "tags": [category],\n' +
            '}\n'
        }
      },
      {
        id: 'memory-store',
        type: 'memory',
        position: { x: 300, y: 440 },
        data: {
          label: 'Store in 2Brain',
          operation: 'store',
          key: '2brain-inbox',
          namespace: '2brain'
        }
      },
      {
        id: 'end',
        type: 'end',
        position: { x: 300, y: 560 },
        data: { label: 'Triaged Item' }
      }
    ],
    edges: [
      {
        id: 'start-classifier',
        source: 'start',
        target: 'classifier',
        animated: false
      },
      {
        id: 'classifier-structurer',
        source: 'classifier',
        target: 'structurer',
        animated: false
      },
      {
        id: 'structurer-memory',
        source: 'structurer',
        target: 'memory-store',
        animated: false
      },
      {
        id: 'memory-end',
        source: 'memory-store',
        target: 'end',
        animated: false
      }
    ]
  }
};
