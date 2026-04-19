import type { ReactNode } from 'react';
import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  useMemo,
} from 'react';
import { getProviderIconByBackend, formatProviderName } from '../utils/modelUtils';
import { MODEL_ICONS } from '../constants/models';
import { useAnalyticsMetricsService } from '../services/analyticsMetricsService';
import type { TraceItem, AnalyticsMetrics } from '../types/analyticsType';
import type {
  OverviewData,
  CostAnalysisData,
  PerformanceData,
  UsageByModel,
  CostByProvider,
  CostByTask,
  LatencyData,
  ExpensiveRequest,
} from '../types/dashboardTypes';

interface MetricsState {
  allTraces: TraceItem[];
  rawMetrics: AnalyticsMetrics | null;

  isLoading: boolean;
  isInitialized: boolean;
  error: string | null;
  lastFetchedAt: number | null;

  selectedDays: number;
}

interface MetricsContextType {
  isLoading: boolean;
  isInitialized: boolean;
  error: string | null;
  selectedDays: number;

  allTraces: TraceItem[];
  overviewData: OverviewData | null;
  costData: CostAnalysisData | null;
  performanceData: PerformanceData | null;
  analyticsData: {
    metrics: AnalyticsMetrics | null;
    traces: TraceItem[];
    totalTraces: number;
  };

  setSelectedDays: (days: number) => void;
  refreshData: () => Promise<void>;
  findTraceById: (traceId: string) => TraceItem | null;
}

function isOpentracyProvider(provider: string): boolean {
  return provider?.toLowerCase() === 'opentracy';
}

function formatModelName(modelId: string, backend: string): string {
  if (isOpentracyProvider(backend)) {
    return `opentracy/${modelId}`;
  }
  if (modelId.includes('/')) {
    return modelId.split('/')[1];
  }
  return modelId;
}

function calculateP95(values: number[]): number {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];

  const sorted = [...values].sort((a, b) => a - b);
  const rank = 0.95 * (sorted.length - 1);
  const lowerIndex = Math.floor(rank);
  const upperIndex = Math.ceil(rank);
  const weight = rank - lowerIndex;

  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

function formatCostValue(cost: number): string {
  if (cost === 0) return '$0.00';
  if (cost < 0.0001) {
    const str = cost.toFixed(10);
    const match = str.match(/0\.(0*)([1-9])/);
    if (match) {
      const zeros = match[1].length;
      const decimals = Math.min(zeros + 2, 10);
      return `$${cost.toFixed(decimals).replace(/0+$/, '')}`;
    }
  }
  if (cost < 0.01) {
    return `$${cost.toFixed(6).replace(/0+$/, '').replace(/\.$/, '')}`;
  }
  if (cost < 1) {
    return `$${cost.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')}`;
  }
  return `$${cost.toFixed(2)}`;
}

function formatLatencyValue(latencyMs: number): string {
  if (latencyMs === 0) return '0ms';
  if (latencyMs >= 1000) {
    const seconds = latencyMs / 1000;
    return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
  }
  return `${latencyMs.toFixed(0)}ms`;
}

function filterTracesByDays(traces: TraceItem[], days: number): TraceItem[] {
  if (days >= 90) return traces;

  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return traces.filter((trace) => {
    const traceDate = new Date(trace.created_at);
    return traceDate >= cutoffDate;
  });
}

interface ModelStats {
  modelId: string;
  backend: string;
  requests: number;
  totalCost: number;
  totalLatency: number;
  latencies: number[];
  isOpentracy: boolean;
}

function aggregateByModel(traces: TraceItem[]): Map<string, ModelStats> {
  const modelMap = new Map<string, ModelStats>();

  for (const trace of traces) {
    const key = `${trace.backend}:${trace.model_id}`;
    const existing = modelMap.get(key);

    if (existing) {
      existing.requests++;
      existing.totalCost += trace.cost_usd;
      existing.totalLatency += trace.latency_s;
      existing.latencies.push(trace.latency_s);
    } else {
      modelMap.set(key, {
        modelId: trace.model_id,
        backend: trace.backend,
        requests: 1,
        totalCost: trace.cost_usd,
        totalLatency: trace.latency_s,
        latencies: [trace.latency_s],
        isOpentracy: isOpentracyProvider(trace.backend),
      });
    }
  }

  return modelMap;
}

interface ProviderStats {
  provider: string;
  requests: number;
  totalCost: number;
  totalLatency: number;
  latencies: number[];
  isOpentracy: boolean;
}

function aggregateByProvider(traces: TraceItem[]): Map<string, ProviderStats> {
  const providerMap = new Map<string, ProviderStats>();

  for (const trace of traces) {
    const provider = trace.backend?.toLowerCase() || 'unknown';
    const existing = providerMap.get(provider);

    if (existing) {
      existing.requests++;
      existing.totalCost += trace.cost_usd;
      existing.totalLatency += trace.latency_s;
      existing.latencies.push(trace.latency_s);
    } else {
      providerMap.set(provider, {
        provider,
        requests: 1,
        totalCost: trace.cost_usd,
        totalLatency: trace.latency_s,
        latencies: [trace.latency_s],
        isOpentracy: isOpentracyProvider(provider),
      });
    }
  }

  return providerMap;
}

function transformToModelUsage(modelStats: Map<string, ModelStats>): UsageByModel[] {
  return Array.from(modelStats.values())
    .map((stats) => {
      const displayName = formatModelName(stats.modelId, stats.backend);
      const icon = stats.isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(stats.backend);

      return {
        model: displayName,
        name: displayName,
        requests: stats.requests,
        icon,
        isOpentracy: stats.isOpentracy,
        cost: stats.totalCost,
        latency: calculateP95(stats.latencies) * 1000,
      };
    })
    .sort((a, b) => b.requests - a.requests);
}

function transformToProviderCost(providerStats: Map<string, ProviderStats>): CostByProvider[] {
  return Array.from(providerStats.values())
    .filter((stats) => !stats.isOpentracy)
    .map((stats) => {
      const displayName = formatProviderName(stats.provider);
      const icon = getProviderIconByBackend(stats.provider);

      return {
        provider: displayName,
        cost: stats.totalCost,
        icon,
        isOpentracy: stats.isOpentracy,
      };
    })
    .sort((a, b) => b.cost - a.cost);
}

function transformToCostByTask(traces: TraceItem[]): CostByTask[] {
  const successfulTraces = traces.filter((t) => t.is_success);
  const modelStats = aggregateByModel(successfulTraces);

  return Array.from(modelStats.values())
    .filter((stats) => stats.requests > 0 && stats.totalCost > 0)
    .map((stats) => {
      const displayName = formatModelName(stats.modelId, stats.backend);
      const icon = stats.isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(stats.backend);

      return {
        task: displayName,
        name: displayName,
        cost: stats.totalCost,
        icon,
        isOpentracy: stats.isOpentracy,
      };
    })
    .sort((a, b) => b.cost - a.cost);
}

function transformToLatencyData(modelStats: Map<string, ModelStats>): LatencyData[] {
  return Array.from(modelStats.values())
    .map((stats) => {
      const displayName = formatModelName(stats.modelId, stats.backend);
      const icon = stats.isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(stats.backend);
      const p95Latency = calculateP95(stats.latencies);

      return {
        key: displayName,
        name: displayName,
        value: p95Latency * 1000,
        icon,
        isOpentracy: stats.isOpentracy,
      };
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);
}

function transformToExpensiveRequests(traces: TraceItem[]): ExpensiveRequest[] {
  return traces
    .filter((t) => t.cost_usd > 0 && !isOpentracyProvider(t.backend))
    .sort((a, b) => b.cost_usd - a.cost_usd)
    .slice(0, 10)
    .map((trace, index) => {
      const displayName = formatModelName(trace.model_id, trace.backend);
      const icon = getProviderIconByBackend(trace.backend);

      return {
        id: trace.id || `req_${index}`,
        cost: trace.cost_usd,
        model: displayName,
        icon,
        promptSize: trace.input_tokens,
        date: new Date(trace.created_at).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
        isOpentracy: false,
      };
    });
}

function calculateKPIs(traces: TraceItem[]): {
  totalCost: number;
  totalRequests: number;
  avgLatencyP95: number;
  errorRate: number;
} {
  if (traces.length === 0) {
    return { totalCost: 0, totalRequests: 0, avgLatencyP95: 0, errorRate: 0 };
  }

  const totalCost = traces.reduce((sum, t) => sum + t.cost_usd, 0);
  const totalRequests = traces.length;
  const latencies = traces.map((t) => t.latency_s);
  const avgLatencyP95 = calculateP95(latencies);
  const errorCount = traces.filter((t) => !t.is_success).length;
  const errorRate = (errorCount / totalRequests) * 100;

  return { totalCost, totalRequests, avgLatencyP95, errorRate };
}

function buildTimeSeries(traces: TraceItem[], days: number): { date: string; cost: number }[] {
  const dateMap = new Map<string, number>();

  const now = new Date();
  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];
    dateMap.set(dateStr, 0);
  }

  for (const trace of traces) {
    if (!trace.created_at) continue;
    const dateStr = new Date(trace.created_at).toISOString().split('T')[0];
    if (dateMap.has(dateStr)) {
      dateMap.set(dateStr, (dateMap.get(dateStr) || 0) + trace.cost_usd);
    }
  }

  return Array.from(dateMap.entries())
    .map(([date, cost]) => ({ date, cost }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function buildLatencyHistogram(traces: TraceItem[]): { bucket: string; count: number }[] {
  const buckets = {
    '<1s': 0,
    '1-2s': 0,
    '2-5s': 0,
    '5-10s': 0,
    '>10s': 0,
  };

  for (const trace of traces) {
    const latency = trace.latency_s;
    if (latency < 1) buckets['<1s']++;
    else if (latency < 2) buckets['1-2s']++;
    else if (latency < 5) buckets['2-5s']++;
    else if (latency < 10) buckets['5-10s']++;
    else buckets['>10s']++;
  }

  return Object.entries(buckets).map(([bucket, count]) => ({ bucket, count }));
}

function buildErrorsOverTime(
  traces: TraceItem[],
  days: number
): { date: string; errors: number }[] {
  const dateMap = new Map<string, number>();

  const now = new Date();
  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];
    dateMap.set(dateStr, 0);
  }

  for (const trace of traces) {
    if (!trace.is_success && trace.created_at) {
      const dateStr = new Date(trace.created_at).toISOString().split('T')[0];
      if (dateMap.has(dateStr)) {
        dateMap.set(dateStr, (dateMap.get(dateStr) || 0) + 1);
      }
    }
  }

  return Array.from(dateMap.entries())
    .map(([date, errors]) => ({ date, errors }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function buildErrorsTable(traces: TraceItem[]): {
  date: string;
  model: string;
  reason: string;
  requestId: string;
}[] {
  return traces
    .filter((t) => !t.is_success)
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 10)
    .map((trace) => {
      const displayName = formatModelName(trace.model_id, trace.backend);
      let reason = 'Unknown error';
      if (trace.error_code) {
        reason = trace.error_code;
      } else if (trace.latency_s > 30) {
        reason = 'Timeout';
      }

      return {
        date: trace.created_at,
        model: displayName,
        reason,
        requestId:
          trace.id || trace.event_id || `req_${Math.random().toString(36).substring(2, 8)}`,
      };
    });
}

const MetricsContext = createContext<MetricsContextType | undefined>(undefined);

export function MetricsProvider({ children }: { children: ReactNode }) {
  const tenantId = 'default';
  const userLoading = false;
  const { getAnalyticsMetrics } = useAnalyticsMetricsService();

  const [state, setState] = useState<MetricsState>({
    allTraces: [],
    rawMetrics: null,
    isLoading: false,
    isInitialized: false,
    error: null,
    lastFetchedAt: null,
    selectedDays: 30,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const parseJsonField = (value: unknown): any => {
    if (value == null) return null;
    if (typeof value === 'string') {
      try {
        return JSON.parse(value);
      } catch {
        return null;
      }
    }
    if (typeof value === 'object') return value;
    return null;
  };

  const transformSample = useCallback((sample: any): TraceItem => {
    const latency = sample.latency_s || 0;
    const outputTokens = sample.output_tokens || 0;
    const inputTokens = sample.input_tokens || 0;
    const backend = sample.backend || sample.provider || '';

    return {
      id: sample.id || sample.event_id || crypto.randomUUID(),
      event_id: sample.event_id || '',
      model_id: sample.model_id || sample.model || '',
      backend,
      provider: formatProviderName(backend),
      endpoint: sample.endpoint || '',
      created_at: sample.created_at || sample.timestamp || '',
      latency_s: latency,
      ttft_s: sample.ttft_s || 0,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
      tokens_per_s: latency > 0 ? outputTokens / latency : 0,
      cost_usd: sample.total_cost_usd || sample.cost_usd || 0,
      is_success: Boolean(sample.is_success ?? sample.success),
      is_stream: Boolean(sample.is_stream),
      status: (sample.is_success ?? sample.success) ? 'Success' : 'Error',
      input_preview: sample.input_preview || '',
      output_preview: sample.output_preview || '',
      output_text: sample.output_text || '',
      deployment_id: sample.deployment_id ?? null,
      error_code: sample.error_code ?? null,
      history: sample.history ?? null,
      input_messages: parseJsonField(sample.input_messages),
      output_message: parseJsonField(sample.output_message),
      finish_reason: sample.finish_reason ?? null,
      request_tools: parseJsonField(sample.request_tools),
      has_tool_calls: Boolean(sample.has_tool_calls),
      tool_calls_count: sample.tool_calls_count ?? undefined,
      tool_calls: sample.tool_calls ?? null,
      execution_timeline: parseJsonField(sample.execution_timeline) ?? undefined,
    };
  }, []);

  const fetchAllData = useCallback(async () => {
    if (!tenantId || userLoading) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await getAnalyticsMetrics('', tenantId, 90, {
        trace_limit: 100000,
      });

      const allTraces = (response.raw_sample || []).map(transformSample);
      const { raw_sample, ...rest } = response;

      setState((prev) => ({
        ...prev,
        allTraces,
        rawMetrics: rest,
        isLoading: false,
        isInitialized: true,
        error: null,
        lastFetchedAt: Date.now(),
      }));
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;

      console.error('[Metrics] Error fetching data:', err);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        isInitialized: true,
        error: err instanceof Error ? err.message : 'Unknown error',
      }));
    }
  }, [tenantId, userLoading, getAnalyticsMetrics, transformSample]);

  useEffect(() => {
    if (!state.isInitialized && !state.isLoading && tenantId && !userLoading) {
      fetchAllData();
    }
  }, [state.isInitialized, state.isLoading, tenantId, userLoading, fetchAllData]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const filteredTraces = useMemo(() => {
    return filterTracesByDays(state.allTraces, state.selectedDays);
  }, [state.allTraces, state.selectedDays]);

  const overviewData = useMemo((): OverviewData | null => {
    if (!state.isInitialized || filteredTraces.length === 0) return null;

    const modelStats = aggregateByModel(filteredTraces);
    const providerStats = aggregateByProvider(filteredTraces);
    const kpis = calculateKPIs(filteredTraces);
    const externalTraces = filteredTraces.filter((t) => !isOpentracyProvider(t.backend));
    const externalCost = externalTraces.reduce((sum, t) => sum + t.cost_usd, 0);

    const models = transformToModelUsage(modelStats);
    const providers = transformToProviderCost(providerStats);
    const opentracyModels = models.filter((m) => m.isOpentracy);
    const externalModels = models.filter((m) => !m.isOpentracy);

    return {
      kpis: [
        {
          label: `External API Cost (${state.selectedDays}d)`,
          value: formatCostValue(externalCost),
          change: 'N/A',
          isPositive: true,
          icon: 'dollar' as const,
        },
        {
          label: 'Total Requests',
          value: kpis.totalRequests.toLocaleString(),
          change: 'N/A',
          isPositive: true,
          icon: 'activity' as const,
        },
        {
          label: 'Latency (P95)',
          value: formatLatencyValue(kpis.avgLatencyP95 * 1000),
          change: 'N/A',
          isPositive: true,
          icon: 'trending' as const,
        },
        {
          label: 'Error Rate',
          value: `${kpis.errorRate.toFixed(1)}%`,
          change: 'N/A',
          isPositive: kpis.errorRate < 5,
          icon: 'alert' as const,
        },
      ],
      providers,
      models,
      alerts: [],
      opentracy: {
        totalCost: opentracyModels.reduce((sum, m) => sum + (m.cost || 0), 0),
        totalRequests: opentracyModels.reduce((sum, m) => sum + m.requests, 0),
        providers: [],
        models: opentracyModels,
      },
      external: {
        totalCost: externalModels.reduce((sum, m) => sum + (m.cost || 0), 0),
        totalRequests: externalModels.reduce((sum, m) => sum + m.requests, 0),
        providers,
        models: externalModels,
      },
    };
  }, [state.isInitialized, filteredTraces, state.selectedDays]);

  const costData = useMemo((): CostAnalysisData | null => {
    if (!state.isInitialized || filteredTraces.length === 0) return null;

    const costByTask = transformToCostByTask(filteredTraces);
    const expensiveRequests = transformToExpensiveRequests(filteredTraces);
    const opentracyCosts = costByTask.filter((c) => c.isOpentracy);
    const externalCosts = costByTask.filter((c) => !c.isOpentracy);

    return {
      timeSeries: buildTimeSeries(filteredTraces, state.selectedDays),
      costByTask,
      expensiveRequests,
      opentracyCosts,
      externalCosts,
    };
  }, [state.isInitialized, filteredTraces, state.selectedDays]);

  const performanceData = useMemo((): PerformanceData | null => {
    if (!state.isInitialized || filteredTraces.length === 0) return null;

    const modelStats = aggregateByModel(filteredTraces);

    return {
      latencyBy: transformToLatencyData(modelStats),
      latencyHistogram: buildLatencyHistogram(filteredTraces),
      errors: buildErrorsOverTime(filteredTraces, state.selectedDays),
      errorsTable: buildErrorsTable(filteredTraces),
    };
  }, [state.isInitialized, filteredTraces, state.selectedDays]);

  const analyticsData = useMemo(() => {
    return {
      metrics: state.rawMetrics,
      traces: filteredTraces,
      totalTraces: filteredTraces.length,
    };
  }, [state.rawMetrics, filteredTraces]);

  const setSelectedDays = useCallback((days: number) => {
    setState((prev) => ({ ...prev, selectedDays: days }));
  }, []);

  const refreshData = useCallback(async () => {
    await fetchAllData();
  }, [fetchAllData]);

  const findTraceById = useCallback(
    (traceId: string): TraceItem | null => {
      return state.allTraces.find((t) => t.id === traceId || t.event_id === traceId) || null;
    },
    [state.allTraces]
  );

  const value: MetricsContextType = {
    isLoading: state.isLoading,
    isInitialized: state.isInitialized,
    error: state.error,
    selectedDays: state.selectedDays,
    allTraces: state.allTraces,
    overviewData,
    costData,
    performanceData,
    analyticsData,
    setSelectedDays,
    refreshData,
    findTraceById,
  };

  return <MetricsContext.Provider value={value}>{children}</MetricsContext.Provider>;
}

export function useMetrics(): MetricsContextType {
  const context = useContext(MetricsContext);
  if (!context) {
    throw new Error('useMetrics must be used within MetricsProvider');
  }
  return context;
}
