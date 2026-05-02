export type NodeType = 'agent' | 'tool' | 'llm' | 'router' | 'output';

export type AgentNode = {
  id: string;
  type: NodeType;
  title: string;
  subtitle?: string;
  meta?: string;
  badge?: 'NEW';
  cost?: string;
  latency?: string;
};

export type MetricStatus = 'pass' | 'fail';

export type EvalMetric = {
  name: string;
  value: number | MetricStatus;
};

export type AgentRun = {
  id: string;
  agentName: string;
  version: string;
  runLabel: string;
  nodes: AgentNode[];
  metrics: EvalMetric[];
  overall: number;
  warning?: string;
};
