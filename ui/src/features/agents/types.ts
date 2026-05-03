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
  isMock: boolean;
};
