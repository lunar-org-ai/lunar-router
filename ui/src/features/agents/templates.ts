import type { AgentNode, EvalMetric } from '@/features/agents/types';

export type AgentTemplate = {
  id: string;
  name: string;
  description: string;
  nodes: AgentNode[];
  metrics: EvalMetric[];
  overall: number;
  warning?: string;
};

export const AGENT_TEMPLATES: AgentTemplate[] = [
  {
    id: 'support-bot',
    name: 'Support Agent',
    description: 'Customer support flow with KB retrieval and routing across resolution paths.',
    nodes: [
      { id: 's1', type: 'agent', title: 'Agent Node', subtitle: 'customer_support_v1' },
      {
        id: 's2',
        type: 'tool',
        title: 'Tool: KB Search',
        subtitle: 'vector_retrieval(top_k=5)',
        badge: 'NEW',
      },
      {
        id: 's3',
        type: 'llm',
        title: 'LLM Prompt',
        subtitle: '"You are a helpful support assistant"',
        meta: 'gpt-4o-mini',
      },
      { id: 's4', type: 'router', title: 'Router', subtitle: 'escalate | resolve | clarify' },
      { id: 's5', type: 'output', title: 'Output', subtitle: 'response → user' },
    ],
    metrics: [
      { name: 'Factuality', value: 62 },
      { name: 'Relevance', value: 71 },
      { name: 'Safety', value: 'pass' },
      { name: 'Completeness', value: 48 },
    ],
    overall: 67,
    warning: 'Agent relies on general knowledge. Add retrieval step for KB articles.',
  },
  {
    id: 'rag-agent',
    name: 'RAG Agent',
    description: 'Retrieve-then-answer pipeline grounded on a vector store.',
    nodes: [
      {
        id: 'r1',
        type: 'tool',
        title: 'Tool: Vector Search',
        subtitle: 'pinecone(top_k=8)',
      },
      {
        id: 'r2',
        type: 'llm',
        title: 'LLM Prompt',
        subtitle: '"Answer using only the provided context"',
        meta: 'gpt-4o-mini',
      },
      { id: 'r3', type: 'output', title: 'Output', subtitle: 'response → user' },
    ],
    metrics: [
      { name: 'Factuality', value: 84 },
      { name: 'Relevance', value: 79 },
      { name: 'Safety', value: 'pass' },
      { name: 'Completeness', value: 71 },
    ],
    overall: 78,
    warning: 'Queries without retrieval hits fall back to the LLM. Add a query rewriter.',
  },
  {
    id: 'research-agent',
    name: 'Research Agent',
    description: 'Multi-step research pipeline with planning, search, reading, and synthesis.',
    nodes: [
      { id: 'rs1', type: 'agent', title: 'Planner', subtitle: 'plan_research_steps' },
      {
        id: 'rs2',
        type: 'tool',
        title: 'Tool: Web Search',
        subtitle: 'serpapi(num=10)',
      },
      { id: 'rs3', type: 'tool', title: 'Tool: Reader', subtitle: 'fetch_and_chunk' },
      {
        id: 'rs4',
        type: 'llm',
        title: 'Synthesizer',
        subtitle: '"Synthesize findings with citations"',
        meta: 'claude-sonnet-4-6',
      },
      { id: 'rs5', type: 'output', title: 'Output', subtitle: 'response → user' },
    ],
    metrics: [
      { name: 'Factuality', value: 91 },
      { name: 'Depth', value: 88 },
      { name: 'Recency', value: 65 },
      { name: 'Citation', value: 92 },
    ],
    overall: 84,
  },
];

export const SUPPORT_TEMPLATE_ID = 'support-bot';

export function findTemplate(id: string | null | undefined): AgentTemplate | null {
  if (!id) return null;
  return AGENT_TEMPLATES.find((template) => template.id === id) ?? null;
}
