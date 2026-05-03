import type {
  AgentFramework,
  AgentStatus,
  AgentSummary,
  EvalMetric,
  RecentTrace,
} from '@/features/agents/types';

const buildVolume = (
  buckets: number,
  base: number,
  peak: number,
  shape: 'business' | 'even' | 'spiky'
): number[] => {
  return Array.from({ length: buckets }, (_, i) => {
    const t = i / (buckets - 1);
    if (shape === 'business') {
      const wave = Math.sin((t - 0.1) * Math.PI * 1.6);
      const noise = Math.sin(t * Math.PI * 11) * 0.08;
      return Math.max(0, Math.round(base + (peak - base) * Math.max(0, wave + noise)));
    }
    if (shape === 'even') {
      const wave = 0.6 + Math.sin(t * Math.PI * 4) * 0.18 + Math.sin(t * Math.PI * 9) * 0.08;
      return Math.max(0, Math.round(base + (peak - base) * wave));
    }
    const spike = Math.pow(Math.sin(t * Math.PI * 5.7), 12);
    const baseline = 0.2 + Math.sin(t * Math.PI * 2.3) * 0.12;
    return Math.max(0, Math.round(base + (peak - base) * (baseline + spike)));
  });
};

type MockSeed = {
  id: string;
  name: string;
  framework: AgentFramework;
  status: AgentStatus;
  traces24h: number;
  p95LatencyMs: number;
  errorRate: number;
  costPerTrace: number;
  evalScore: number;
  importedDaysAgo: number;
  lastSeenMinutesAgo: number;
  metrics: EvalMetric[];
  recentTraces: RecentTrace[];
  volume: { base: number; peak: number; shape: 'business' | 'even' | 'spiky' };
};

const SEEDS: MockSeed[] = [
  {
    id: 'support-bot',
    name: 'support-bot',
    framework: 'langchain',
    status: 'healthy',
    traces24h: 12847,
    p95LatencyMs: 1240,
    errorRate: 0.003,
    costPerTrace: 0.0008,
    evalScore: 67,
    importedDaysAgo: 4,
    lastSeenMinutesAgo: 1,
    metrics: [
      { name: 'Factuality', value: 62 },
      { name: 'Relevance', value: 71 },
      { name: 'Safety', value: 'pass' },
      { name: 'Completeness', value: 48 },
    ],
    recentTraces: [
      { id: 't_8a4f', status: 'ok', durationMs: 1182, costUsd: 0.00091, agoLabel: '12s' },
      { id: 't_7b21', status: 'ok', durationMs: 1056, costUsd: 0.00083, agoLabel: '34s' },
      { id: 't_6d9c', status: 'ok', durationMs: 894, costUsd: 0.00072, agoLabel: '1m' },
      { id: 't_5e02', status: 'error', durationMs: 412, costUsd: 0.00031, agoLabel: '2m' },
      { id: 't_4f8a', status: 'ok', durationMs: 1310, costUsd: 0.00098, agoLabel: '3m' },
    ],
    volume: { base: 80, peak: 720, shape: 'business' },
  },
  {
    id: 'rag-pipeline',
    name: 'rag-pipeline',
    framework: 'langgraph',
    status: 'healthy',
    traces24h: 3214,
    p95LatencyMs: 1820,
    errorRate: 0.001,
    costPerTrace: 0.0042,
    evalScore: 84,
    importedDaysAgo: 11,
    lastSeenMinutesAgo: 4,
    metrics: [
      { name: 'Factuality', value: 91 },
      { name: 'Relevance', value: 79 },
      { name: 'Safety', value: 'pass' },
      { name: 'Citation', value: 92 },
    ],
    recentTraces: [
      { id: 't_b14e', status: 'ok', durationMs: 1714, costUsd: 0.00412, agoLabel: '1m' },
      { id: 't_a02d', status: 'ok', durationMs: 1880, costUsd: 0.00451, agoLabel: '2m' },
      { id: 't_9c7b', status: 'ok', durationMs: 1622, costUsd: 0.00388, agoLabel: '4m' },
      { id: 't_8d49', status: 'ok', durationMs: 1985, costUsd: 0.00472, agoLabel: '5m' },
      { id: 't_7e10', status: 'ok', durationMs: 1543, costUsd: 0.00367, agoLabel: '7m' },
    ],
    volume: { base: 60, peak: 220, shape: 'even' },
  },
  {
    id: 'research-assistant',
    name: 'research-assistant',
    framework: 'openai-agents',
    status: 'degraded',
    traces24h: 587,
    p95LatencyMs: 4210,
    errorRate: 0.021,
    costPerTrace: 0.0156,
    evalScore: 71,
    importedDaysAgo: 2,
    lastSeenMinutesAgo: 18,
    metrics: [
      { name: 'Factuality', value: 88 },
      { name: 'Depth', value: 85 },
      { name: 'Recency', value: 65 },
      { name: 'Citation', value: 92 },
    ],
    recentTraces: [
      { id: 't_f309', status: 'ok', durationMs: 3920, costUsd: 0.01489, agoLabel: '18m' },
      { id: 't_e88c', status: 'error', durationMs: 1240, costUsd: 0.00412, agoLabel: '24m' },
      { id: 't_d712', status: 'ok', durationMs: 4584, costUsd: 0.01712, agoLabel: '31m' },
      { id: 't_c521', status: 'ok', durationMs: 3680, costUsd: 0.01345, agoLabel: '47m' },
      { id: 't_b440', status: 'ok', durationMs: 4120, costUsd: 0.01589, agoLabel: '1h' },
    ],
    volume: { base: 4, peak: 38, shape: 'spiky' },
  },
];

const minutesAgoIso = (mins: number): string =>
  new Date(Date.now() - mins * 60_000).toISOString();
const daysAgoIso = (days: number): string =>
  new Date(Date.now() - days * 24 * 60 * 60_000).toISOString();

export const MOCK_AGENTS: AgentSummary[] = SEEDS.map((seed) => ({
  id: seed.id,
  name: seed.name,
  framework: seed.framework,
  importedAt: daysAgoIso(seed.importedDaysAgo),
  lastSeenAt: minutesAgoIso(seed.lastSeenMinutesAgo),
  status: seed.status,
  traces24h: seed.traces24h,
  p95LatencyMs: seed.p95LatencyMs,
  errorRate: seed.errorRate,
  costPerTrace: seed.costPerTrace,
  evalScore: seed.evalScore,
  traceVolume: buildVolume(48, seed.volume.base, seed.volume.peak, seed.volume.shape),
  metrics: seed.metrics,
  recentTraces: seed.recentTraces,
  isMock: true,
}));

export function findMockAgent(id: string): AgentSummary | null {
  return MOCK_AGENTS.find((agent) => agent.id === id) ?? null;
}

const FRESH_METRICS: EvalMetric[] = [
  { name: 'Factuality', value: 70 },
  { name: 'Relevance', value: 74 },
  { name: 'Safety', value: 'pass' },
  { name: 'Completeness', value: 62 },
];

const FRESH_TRACES: RecentTrace[] = [
  { id: 't_new5', status: 'ok', durationMs: 1110, costUsd: 0.00084, agoLabel: '8s' },
  { id: 't_new4', status: 'ok', durationMs: 980, costUsd: 0.00074, agoLabel: '22s' },
  { id: 't_new3', status: 'ok', durationMs: 1245, costUsd: 0.00094, agoLabel: '41s' },
  { id: 't_new2', status: 'ok', durationMs: 875, costUsd: 0.00067, agoLabel: '1m' },
  { id: 't_new1', status: 'ok', durationMs: 1320, costUsd: 0.00102, agoLabel: '1m' },
];

export function buildFreshAgent(slug: string, name: string, framework: AgentFramework): AgentSummary {
  const now = new Date().toISOString();
  return {
    id: slug,
    name,
    framework,
    importedAt: now,
    lastSeenAt: now,
    status: 'healthy',
    traces24h: 248,
    p95LatencyMs: 1180,
    errorRate: 0.004,
    costPerTrace: 0.00086,
    evalScore: 70,
    traceVolume: buildVolume(48, 8, 64, 'business'),
    metrics: FRESH_METRICS,
    recentTraces: FRESH_TRACES,
    isMock: false,
  };
}
