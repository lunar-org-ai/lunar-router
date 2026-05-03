import type { AgentNode, EvalMetric } from '@/features/agents/types';

export type AgentTemplate = {
  id: string;
  nodes: AgentNode[];
  metrics: EvalMetric[];
  overall: number;
  warning?: string;
};

export const SUPPORT_TEMPLATE: AgentTemplate = {
  id: 'support-bot',
  nodes: [
    {
      id: 's1',
      type: 'agent',
      title: 'Agent Node',
      subtitle: 'customer_support_v1',
      latency: '12ms',
    },
    {
      id: 's2',
      type: 'tool',
      title: 'Tool: KB Search',
      subtitle: 'vector_retrieval(top_k=5)',
      badge: 'NEW',
      cost: '$0.0001',
      latency: '180ms p95',
    },
    {
      id: 's3',
      type: 'llm',
      title: 'LLM Prompt',
      subtitle: '"You are a helpful support assistant"',
      meta: 'gpt-4o-mini',
      cost: '$0.0008',
      latency: '1.2s p95',
    },
    {
      id: 's4',
      type: 'router',
      title: 'Router',
      subtitle: 'escalate | resolve | clarify',
      latency: '8ms',
    },
    {
      id: 's5',
      type: 'output',
      title: 'Output',
      subtitle: 'response → user',
      latency: '4ms',
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
