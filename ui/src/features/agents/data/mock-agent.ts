import type { AgentRun } from '@/features/agents/types';

export const mockSupportAgent: AgentRun = {
  id: 'support-bot',
  agentName: 'Support Agent',
  version: 'v1',
  runLabel: 'Run 1',
  nodes: [
    {
      id: 'n1',
      type: 'agent',
      title: 'Agent Node',
      subtitle: 'customer_support_v1',
    },
    {
      id: 'n2',
      type: 'tool',
      title: 'Tool: KB Search',
      subtitle: 'vector_retrieval(top_k=5)',
      badge: 'NEW',
    },
    {
      id: 'n3',
      type: 'llm',
      title: 'LLM Prompt',
      subtitle: '"You are a helpful support assistant"',
      meta: 'gpt-4o-mini',
    },
    {
      id: 'n4',
      type: 'router',
      title: 'Router',
      subtitle: 'escalate | resolve | clarify',
    },
    {
      id: 'n5',
      type: 'output',
      title: 'Output',
      subtitle: 'response → user',
    },
  ],
  metrics: [
    { name: 'Factuality', value: 62 },
    { name: 'Relevance', value: 71 },
    { name: 'Safety', value: 'pass' },
    { name: 'Completeness', value: 48 },
  ],
  overall: 67,
  warning: 'Agent relies on general knowledge. Add retrieval step for KB articles.',
};
