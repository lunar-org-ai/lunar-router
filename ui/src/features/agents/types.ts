export type AgentFramework = 'langchain' | 'langgraph' | 'crewai' | 'openai-agents';

export type MetricStatus = 'pass' | 'fail';

export type EvalMetric = {
  name: string;
  value: number | MetricStatus;
};

export type AgentStatus = 'healthy' | 'degraded' | 'silent';

export type RecentTrace = {
  id: string;
  status: 'ok' | 'error';
  durationMs: number;
  costUsd: number;
  agoLabel: string;
  model?: string;
  tokensIn?: number;
  tokensOut?: number;
};

export type StackKind = 'model' | 'tool' | 'vectorstore' | 'memory' | 'guardrail';

export type StackComponent = {
  kind: StackKind;
  label: string;
  callCount?: number;
  successRate?: number;
  avgLatencyMs?: number;
  costShare?: number;
};

export type AgentSummary = {
  id: string;
  name: string;
  framework: AgentFramework;
  importedAt: string;
  lastSeenAt: string;
  status: AgentStatus;
  traces24h: number;
  p95LatencyMs: number;
  errorRate: number;
  costPerTrace: number;
  evalScore: number;
  traceVolume: number[];
  metrics: EvalMetric[];
  recentTraces: RecentTrace[];
  stack: StackComponent[];
  isMock: boolean;
};
