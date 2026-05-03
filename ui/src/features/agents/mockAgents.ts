import type {
  AgentFramework,
  AgentStatus,
  AgentSummary,
  EvalMetric,
  RecentTrace,
  StackComponent,
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
  stack: StackComponent[];
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
      {
        id: 't_8a4f',
        status: 'ok',
        durationMs: 1182,
        costUsd: 0.00091,
        agoLabel: '12s',
        model: 'gpt-4o-mini',
        tokensIn: 482,
        tokensOut: 184,
      },
      {
        id: 't_7b21',
        status: 'ok',
        durationMs: 1056,
        costUsd: 0.00083,
        agoLabel: '34s',
        model: 'gpt-4o-mini',
        tokensIn: 421,
        tokensOut: 162,
      },
      {
        id: 't_6d9c',
        status: 'ok',
        durationMs: 894,
        costUsd: 0.00072,
        agoLabel: '1m',
        model: 'gpt-4o-mini',
        tokensIn: 376,
        tokensOut: 138,
      },
      {
        id: 't_5e02',
        status: 'error',
        durationMs: 412,
        costUsd: 0.00031,
        agoLabel: '2m',
        model: 'gpt-4o-mini',
        tokensIn: 198,
        tokensOut: 0,
      },
      {
        id: 't_4f8a',
        status: 'ok',
        durationMs: 1310,
        costUsd: 0.00098,
        agoLabel: '3m',
        model: 'gpt-4o-mini',
        tokensIn: 524,
        tokensOut: 198,
      },
    ],
    stack: [
      {
        kind: 'model',
        label: 'gpt-4o-mini',
        callCount: 12847,
        successRate: 0.998,
        avgLatencyMs: 920,
        costShare: 0.62,
      },
      {
        kind: 'tool',
        label: 'KB Search',
        callCount: 11240,
        successRate: 0.94,
        avgLatencyMs: 180,
        costShare: 0,
      },
      {
        kind: 'vectorstore',
        label: 'Pinecone',
        callCount: 11240,
        successRate: 1.0,
        avgLatencyMs: 165,
        costShare: 0.18,
      },
      {
        kind: 'tool',
        label: 'Slack escalation',
        callCount: 128,
        successRate: 1.0,
        avgLatencyMs: 240,
        costShare: 0,
      },
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
      {
        id: 't_b14e',
        status: 'ok',
        durationMs: 1714,
        costUsd: 0.00412,
        agoLabel: '1m',
        model: 'gpt-4o-mini',
        tokensIn: 2412,
        tokensOut: 318,
      },
      {
        id: 't_a02d',
        status: 'ok',
        durationMs: 1880,
        costUsd: 0.00451,
        agoLabel: '2m',
        model: 'gpt-4o-mini',
        tokensIn: 2680,
        tokensOut: 342,
      },
      {
        id: 't_9c7b',
        status: 'ok',
        durationMs: 1622,
        costUsd: 0.00388,
        agoLabel: '4m',
        model: 'gpt-4o-mini',
        tokensIn: 2284,
        tokensOut: 296,
      },
      {
        id: 't_8d49',
        status: 'ok',
        durationMs: 1985,
        costUsd: 0.00472,
        agoLabel: '5m',
        model: 'gpt-4o-mini',
        tokensIn: 2812,
        tokensOut: 358,
      },
      {
        id: 't_7e10',
        status: 'ok',
        durationMs: 1543,
        costUsd: 0.00367,
        agoLabel: '7m',
        model: 'gpt-4o-mini',
        tokensIn: 2148,
        tokensOut: 274,
      },
    ],
    stack: [
      {
        kind: 'model',
        label: 'gpt-4o-mini',
        callCount: 3214,
        successRate: 0.999,
        avgLatencyMs: 1080,
        costShare: 0.4,
      },
      {
        kind: 'tool',
        label: 'Vector Search',
        callCount: 3214,
        successRate: 0.96,
        avgLatencyMs: 520,
        costShare: 0,
      },
      {
        kind: 'vectorstore',
        label: 'Pinecone',
        callCount: 3214,
        successRate: 1.0,
        avgLatencyMs: 280,
        costShare: 0.42,
      },
      {
        kind: 'tool',
        label: 'Query Rewriter',
        callCount: 3214,
        successRate: 0.998,
        avgLatencyMs: 180,
        costShare: 0.18,
      },
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
      {
        id: 't_f309',
        status: 'ok',
        durationMs: 3920,
        costUsd: 0.01489,
        agoLabel: '18m',
        model: 'o3-mini',
        tokensIn: 3548,
        tokensOut: 812,
      },
      {
        id: 't_e88c',
        status: 'error',
        durationMs: 1240,
        costUsd: 0.00412,
        agoLabel: '24m',
        model: 'o3-mini',
        tokensIn: 1124,
        tokensOut: 0,
      },
      {
        id: 't_d712',
        status: 'ok',
        durationMs: 4584,
        costUsd: 0.01712,
        agoLabel: '31m',
        model: 'o3-mini',
        tokensIn: 4012,
        tokensOut: 924,
      },
      {
        id: 't_c521',
        status: 'ok',
        durationMs: 3680,
        costUsd: 0.01345,
        agoLabel: '47m',
        model: 'o3-mini',
        tokensIn: 3198,
        tokensOut: 720,
      },
      {
        id: 't_b440',
        status: 'ok',
        durationMs: 4120,
        costUsd: 0.01589,
        agoLabel: '1h',
        model: 'o3-mini',
        tokensIn: 3742,
        tokensOut: 856,
      },
    ],
    stack: [
      {
        kind: 'model',
        label: 'o3-mini',
        callCount: 1180,
        successRate: 0.971,
        avgLatencyMs: 2780,
        costShare: 0.55,
      },
      {
        kind: 'tool',
        label: 'Web Search',
        callCount: 2410,
        successRate: 0.92,
        avgLatencyMs: 1240,
        costShare: 0.28,
      },
      {
        kind: 'tool',
        label: 'Reader',
        callCount: 8200,
        successRate: 0.96,
        avgLatencyMs: 320,
        costShare: 0.12,
      },
      {
        kind: 'tool',
        label: 'Citation Validator',
        callCount: 580,
        successRate: 0.99,
        avgLatencyMs: 180,
        costShare: 0.04,
      },
      {
        kind: 'guardrail',
        label: 'PII redaction',
        callCount: 580,
        successRate: 1.0,
        avgLatencyMs: 24,
        costShare: 0.01,
      },
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
  stack: seed.stack,
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
  {
    id: 't_new5',
    status: 'ok',
    durationMs: 1110,
    costUsd: 0.00084,
    agoLabel: '8s',
    model: 'gpt-4o-mini',
    tokensIn: 412,
    tokensOut: 156,
  },
  {
    id: 't_new4',
    status: 'ok',
    durationMs: 980,
    costUsd: 0.00074,
    agoLabel: '22s',
    model: 'gpt-4o-mini',
    tokensIn: 368,
    tokensOut: 142,
  },
  {
    id: 't_new3',
    status: 'ok',
    durationMs: 1245,
    costUsd: 0.00094,
    agoLabel: '41s',
    model: 'gpt-4o-mini',
    tokensIn: 478,
    tokensOut: 188,
  },
  {
    id: 't_new2',
    status: 'ok',
    durationMs: 875,
    costUsd: 0.00067,
    agoLabel: '1m',
    model: 'gpt-4o-mini',
    tokensIn: 332,
    tokensOut: 124,
  },
  {
    id: 't_new1',
    status: 'ok',
    durationMs: 1320,
    costUsd: 0.00102,
    agoLabel: '1m',
    model: 'gpt-4o-mini',
    tokensIn: 512,
    tokensOut: 204,
  },
];

const FRESH_STACK: StackComponent[] = [
  {
    kind: 'model',
    label: 'gpt-4o-mini',
    callCount: 248,
    successRate: 0.996,
    avgLatencyMs: 980,
    costShare: 0.7,
  },
  {
    kind: 'tool',
    label: 'Custom tool',
    callCount: 124,
    successRate: 0.98,
    avgLatencyMs: 210,
    costShare: 0,
  },
];

export function buildFreshAgent(
  slug: string,
  name: string,
  framework: AgentFramework
): AgentSummary {
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
    stack: FRESH_STACK,
    isMock: false,
  };
}
