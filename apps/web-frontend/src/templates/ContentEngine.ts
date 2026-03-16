/**
 * Content Engine Pipeline template
 * Fetches content from a web source, summarizes via LLM,
 * formats as a markdown article, and stores the result.
 *
 * Pipeline: Start → HTTP (Research) → LLM (Enrich) → Code (Format) → Memory (Store) → End
 *
 * Portfolio consumer: nxtg-content-engine (P-14).
 * Second dogfood workflow after 2Brain Inbox Triage (N-16).
 */
import { FlowTemplate } from '../types';
import { generateId } from '../utils/flowUtils';

export const contentEngineTemplate: FlowTemplate = {
  id: 'content-engine-pipeline',
  name: 'Content Engine Pipeline',
  description: 'Fetch content from a web source, summarize it with an LLM, format as a structured markdown article, and store the result.',
  tags: ['content-engine', 'research', 'summarization', 'markdown', 'portfolio'],
  flow: {
    id: generateId(),
    name: 'Content Engine Pipeline',
    nodes: [
      {
        id: 'start',
        type: 'start',
        position: { x: 300, y: 25 },
        data: { label: 'Research Topic' }
      },
      {
        id: 'research',
        type: 'http_request',
        position: { x: 300, y: 150 },
        data: {
          label: 'Fetch Source',
          method: 'GET',
          url: '{{input.url}}',
          headers: { Accept: 'application/json' },
          timeout_seconds: 30
        }
      },
      {
        id: 'enrich',
        type: 'llm',
        position: { x: 300, y: 300 },
        data: {
          label: 'Summarize Content',
          provider: 'ollama',
          model: 'llama3.1',
          base_url: 'http://localhost:11434',
          temperature: 0.3,
          max_tokens: 1024,
          system_prompt:
            'You are a research assistant for a content engine. ' +
            'Summarize the provided content into clear, concise key points. ' +
            'Focus on facts, insights, and actionable information. ' +
            'Output a structured summary with bullet points.'
        }
      },
      {
        id: 'format',
        type: 'code',
        position: { x: 300, y: 450 },
        data: {
          label: 'Format Article',
          language: 'python',
          timeout_seconds: 5,
          memory_limit_mb: 256,
          code:
            'import json\n' +
            'import datetime\n' +
            '\n' +
            '# data = LLM summary output\n' +
            '# context["input"] = original user input with topic/url\n' +
            'raw_input = context.get("input", {})\n' +
            'if isinstance(raw_input, dict):\n' +
            '    topic = raw_input.get("topic", "Untitled")\n' +
            '    source_url = raw_input.get("url", "")\n' +
            'else:\n' +
            '    topic = str(raw_input)\n' +
            '    source_url = ""\n' +
            '\n' +
            'summary = str(data).strip() if data else "No summary available."\n' +
            '\n' +
            'article = f"# {topic}\\n\\n"\n' +
            'article += f"*Generated: {datetime.datetime.utcnow().strftime(\'%Y-%m-%d %H:%M UTC\')}*\\n\\n"\n' +
            'if source_url:\n' +
            '    article += f"**Source:** {source_url}\\n\\n"\n' +
            'article += "---\\n\\n"\n' +
            'article += f"## Summary\\n\\n{summary}\\n"\n' +
            '\n' +
            'result = {\n' +
            '    "title": topic,\n' +
            '    "content": article,\n' +
            '    "source_url": source_url,\n' +
            '    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",\n' +
            '    "format": "markdown",\n' +
            '}\n'
        }
      },
      {
        id: 'store',
        type: 'memory',
        position: { x: 300, y: 600 },
        data: {
          label: 'Store Article',
          operation: 'store',
          key: 'content-engine-article',
          namespace: 'content-engine'
        }
      },
      {
        id: 'end',
        type: 'end',
        position: { x: 300, y: 720 },
        data: { label: 'Published Article' }
      }
    ],
    edges: [
      {
        id: 'start-research',
        source: 'start',
        target: 'research',
        animated: false
      },
      {
        id: 'research-enrich',
        source: 'research',
        target: 'enrich',
        animated: false
      },
      {
        id: 'enrich-format',
        source: 'enrich',
        target: 'format',
        animated: false
      },
      {
        id: 'format-store',
        source: 'format',
        target: 'store',
        animated: false
      },
      {
        id: 'store-end',
        source: 'store',
        target: 'end',
        animated: false
      }
    ]
  }
};
