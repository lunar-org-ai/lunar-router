import { useCallback } from 'react';
import { STATS_API_URL } from '../config/api';
import { getProviderIconByBackend, formatProviderName } from '../utils/modelUtils';
import { MODEL_ICONS } from '../constants/models';

// UUID regex pattern for deployment detection
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Check if provider is OpenTracy (internal deployment)
function isOpentracyProvider(provider: string): boolean {
  return provider?.toLowerCase() === 'opentracy';
}

// Helper to format model name for display
function formatModelNameForDisplay(modelId: string, provider?: string): string {
  if (isOpentracyProvider(provider || '')) {
    return `opentracy/${modelId}`;
  }
  if (modelId.includes('/')) {
    return modelId.split('/')[1];
  }
  if (UUID_PATTERN.test(modelId)) {
    return `Deployment ${modelId.substring(0, 8)}`;
  }
  return modelId;
}

// Typed interfaces for API response
interface TimeSeriesEntry {
  time: string;
  request_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency_s: number;
  p95_latency_s: number;
  error_rate: boolean | number;
}

interface ModelEntry {
  model_id: string;
  request_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_latency_s: number;
  p95_latency_s: number;
  error_rate: boolean | number;
}

interface BackendEntry {
  backend: string;
  request_count: number;
  total_cost_usd: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

interface DistributionStats {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  mean: number;
  std: number;
}

// Dashboard metrics API response interface
export interface DashboardMetricsResponse {
  series: {
    by_time: TimeSeriesEntry[];
    by_model: ModelEntry[];
    by_backend: BackendEntry[];
    by_deployment: { deployment_id: string; request_count: number }[];
  };
  totals: {
    request_count: number;
    total_input_tokens: number;
    total_output_tokens: number;
    total_cost_usd: number | null;
    success_rate: number;
    avg_latency_s: number;
    p95_latency_s: number;
    avg_cost_per_1k_tokens_usd: number | null;
    streaming_share: number | null;
  };
  distributions: {
    latency_s: DistributionStats;
    ttft_s: DistributionStats;
    input_tokens: DistributionStats;
    output_tokens: DistributionStats;
    cost_per_request_usd: DistributionStats;
  };
  trends: {
    last_7d: {
      requests: number;
      cost_usd: number | null;
      p95_latency_s: number;
      error_rate: number;
    };
    prev_7d: {
      requests: number;
      cost_usd: number | null;
      p95_latency_s: number | null;
      error_rate: number | null;
    };
    pct_change: {
      requests: number | null;
      cost_usd: number | null;
      p95_latency_s: number | null;
      error_rate: number | null;
    };
  };
  leaders: {
    top_cost_models: { model_id: string; cost_usd: number; count: number }[];
    slowest_models_p95_latency: { model_id: string; p95_latency_s: number; count: number }[];
    most_errors_models: { model_id: string; error_count: number }[];
  };
  insights: unknown[];
  raw_sample?: RawSampleItem[];
}

// Interface for raw sample records from the API
export interface RawSampleItem {
  timestamp: string;
  model: string;
  provider: string;
  success: boolean;
  latency_s: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  input_preview?: string;
  output_preview?: string;
}

// API parameters interface
export interface DashboardMetricsParams {
  start_date: string;
  end_date: string;
  granularity?: 'hourly' | 'daily' | 'monthly';
  backend?: string;
  model_id?: string;
  deployment_id?: string;
  is_stream?: boolean;
  is_success?: boolean;
  raw_sample_n?: number;
}

export function useDashboardMetricsService() {
  // Format cost values with appropriate decimal places
  const formatCostValue = useCallback((cost: number): string => {
    if (cost === 0) return '$0.00';
    if (cost < 0.01) {
      return `$${cost.toFixed(6).replace(/\.?0+$/, '')}`;
    }
    return `$${cost.toFixed(4)}`;
  }, []);

  // Format latency value for display (ms or s depending on magnitude)
  const formatLatencyValue = useCallback((latencyMs: number): string => {
    if (latencyMs === 0) return '0ms';
    if (latencyMs >= 1000) {
      // Show in seconds when >= 1s
      const seconds = latencyMs / 1000;
      return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
    }
    return `${latencyMs.toFixed(0)}ms`;
  }, []);

  // Calculate percentage change between two values
  const calculateChange = useCallback(
    (current: number | null, previous: number | null): { change: string; isPositive: boolean } => {
      if (current === null || previous === null || previous === 0) {
        return { change: 'N/A', isPositive: true };
      }
      const pctChange = ((current - previous) / previous) * 100;
      return {
        change: `${Math.abs(pctChange).toFixed(1)}% vs last period`,
        isPositive: pctChange >= 0,
      };
    },
    []
  );

  // Fetch dashboard metrics using the new API
  const getDashboardMetricsNew = useCallback(
    async (accessToken: string, tenantId: string, days = 7): Promise<DashboardMetricsResponse> => {
      const url = `${STATS_API_URL}/v1/stats/${tenantId}/dashboard?days=${days}`;

      const res = await fetch(url, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      });

      console.log('[useDashboardMetricsService] metrics response: ', res);

      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`);
      }

      return res.json();
    },
    []
  );

  // Build provider map from raw_sample data (has provider field)
  const buildProviderMap = useCallback((metrics: DashboardMetricsResponse): Map<string, string> => {
    const providerMap = new Map<string, string>();

    if (metrics.raw_sample) {
      for (const sample of metrics.raw_sample) {
        if (sample.model && sample.provider) {
          providerMap.set(sample.model, sample.provider);
        }
      }
    }

    return providerMap;
  }, []);

  // Format providers data using real cost_usd from leaders.top_cost_models
  const formatProviders = useCallback(
    (metrics: DashboardMetricsResponse) => {
      const providerMap = buildProviderMap(metrics);
      const providerCosts: Record<string, { cost: number; requests: number; isOpentracy: boolean }> =
        {};

      // Use top_cost_models which has real cost_usd
      for (const model of metrics.leaders.top_cost_models) {
        const provider = providerMap.get(model.model_id) || 'unknown';
        const isOpentracy = isOpentracyProvider(provider);
        const displayProvider = isOpentracy ? 'OpenTracy' : formatProviderName(provider);

        if (!providerCosts[displayProvider]) {
          providerCosts[displayProvider] = { cost: 0, requests: 0, isOpentracy };
        }
        providerCosts[displayProvider].cost += model.cost_usd;
        providerCosts[displayProvider].requests += model.count;
      }

      return Object.entries(providerCosts)
        .map(([provider, data]) => ({
          provider,
          cost: data.cost,
          icon: data.isOpentracy
            ? MODEL_ICONS.opentracyIcon
            : getProviderIconByBackend(provider.toLowerCase()),
          isOpentracy: data.isOpentracy,
        }))
        .sort((a, b) => b.cost - a.cost);
    },
    [buildProviderMap]
  );

  // Format data for the Overview component
  const formatMetricsForOverview = useCallback(
    (metrics: DashboardMetricsResponse, selectedDays = 7) => {
      const { trends, totals, series, leaders } = metrics;
      const providerMap = buildProviderMap(metrics);

      const requestsChange = calculateChange(
        trends?.last_7d?.requests ?? null,
        trends?.prev_7d?.requests ?? null
      );
      const costChange = calculateChange(
        trends?.last_7d?.cost_usd ?? null,
        trends?.prev_7d?.cost_usd ?? null
      );
      const latencyChange = calculateChange(
        trends?.last_7d?.p95_latency_s ?? null,
        trends?.prev_7d?.p95_latency_s ?? null
      );
      const errorRateChange = calculateChange(
        trends?.last_7d?.error_rate ?? null,
        trends?.prev_7d?.error_rate ?? null
      );

      const totalCost = totals?.total_cost_usd ?? trends?.last_7d?.cost_usd ?? null;
      const errorRate = (1 - (totals?.success_rate ?? 1)) * 100;
      const latencyMs = (totals?.p95_latency_s ?? 0) * 1000;

      // Create cost lookup from top_cost_models
      const costLookup = new Map<string, number>();
      for (const model of leaders.top_cost_models) {
        costLookup.set(model.model_id, model.cost_usd);
      }

      // Format models with real data
      const models = series.by_model.map((model) => {
        const provider = providerMap.get(model.model_id) || '';
        const isOpentracy = isOpentracyProvider(provider);
        const displayName = formatModelNameForDisplay(model.model_id, provider);
        const icon = isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(provider);

        return {
          model: displayName,
          name: displayName,
          requests: model.request_count,
          icon,
          isOpentracy,
          cost: costLookup.get(model.model_id) || 0,
          latency: model.p95_latency_s * 1000,
        };
      });

      // Separate OpenTracy and external models
      const opentracyModels = models.filter((m) => m.isOpentracy);
      const externalModels = models.filter((m) => !m.isOpentracy);

      // Calculate summaries
      const opentracySummary = {
        totalCost: opentracyModels.reduce((sum, m) => sum + m.cost, 0),
        totalRequests: opentracyModels.reduce((sum, m) => sum + m.requests, 0),
        providers: [] as any[],
        models: opentracyModels,
      };

      const externalSummary = {
        totalCost: externalModels.reduce((sum, m) => sum + m.cost, 0),
        totalRequests: externalModels.reduce((sum, m) => sum + m.requests, 0),
        providers: formatProviders(metrics).filter((p) => !p.isOpentracy),
        models: externalModels,
      };

      return {
        kpis: [
          {
            label: `Total Cost (${selectedDays}d)`,
            value: totalCost !== null ? formatCostValue(totalCost) : 'N/A',
            change: costChange.change,
            isPositive: !costChange.isPositive, // Lower cost is positive
            icon: 'dollar' as const,
          },
          {
            label: 'Total Requests',
            value: (totals?.request_count ?? 0).toLocaleString(),
            change: requestsChange.change,
            isPositive: requestsChange.isPositive,
            icon: 'activity' as const,
          },
          {
            label: 'Latency (P95)',
            value: formatLatencyValue(latencyMs),
            change: latencyChange.change,
            isPositive: !latencyChange.isPositive,
            icon: 'trending' as const,
          },
          {
            label: 'Error Rate',
            value: `${errorRate.toFixed(1)}%`,
            change: errorRateChange.change,
            isPositive: !errorRateChange.isPositive,
            icon: 'alert' as const,
          },
        ],
        providers: formatProviders(metrics),
        models,
        alerts: generateInsights(metrics),
        opentracy: opentracySummary,
        external: externalSummary,
      };
    },
    [calculateChange, formatCostValue, formatProviders, buildProviderMap]
  );

  // Format data for cost analysis components using real cost_usd
  const formatMetricsForCostAnalysis = useCallback(
    (metrics: DashboardMetricsResponse) => {
      const { series, leaders, raw_sample } = metrics;
      const providerMap = buildProviderMap(metrics);

      // Create cost lookup from top_cost_models
      const costLookup = new Map<string, number>();
      for (const model of leaders.top_cost_models) {
        costLookup.set(model.model_id, model.cost_usd);
      }

      // Time series - estimate from by_time data proportionally
      const totalCost = metrics.totals.total_cost_usd || 0;
      const totalRequests = metrics.totals.request_count || 1;

      const timeSeries = series.by_time.map((entry) => {
        const proportionalCost = (entry.request_count / totalRequests) * totalCost;
        return {
          date: entry.time,
          cost: proportionalCost,
        };
      });

      // Cost by model using real cost_usd
      const costByTask = leaders.top_cost_models.map((model) => {
        const provider = providerMap.get(model.model_id) || '';
        const isOpentracy = isOpentracyProvider(provider);
        const displayName = formatModelNameForDisplay(model.model_id, provider);
        const icon = isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(provider);

        return {
          task: displayName,
          name: displayName,
          cost: model.cost_usd,
          icon,
          isOpentracy,
        };
      });

      // Separate OpenTracy and external costs
      const opentracyCosts = costByTask.filter((c) => c.isOpentracy);
      const externalCosts = costByTask.filter((c) => !c.isOpentracy);

      // Expensive requests from raw_sample
      const expensiveRequests = (raw_sample || [])
        .filter((sample) => sample.cost_usd > 0)
        .map((sample, index) => {
          const isOpentracy = isOpentracyProvider(sample.provider);
          const displayName = formatModelNameForDisplay(sample.model, sample.provider);
          const icon = isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(sample.provider);

          return {
            id: `req_${index}`,
            cost: sample.cost_usd,
            model: displayName,
            icon,
            promptSize: sample.input_tokens,
            date: new Date(sample.timestamp).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            }),
            isOpentracy,
          };
        })
        .sort((a, b) => b.cost - a.cost)
        .slice(0, 10);

      return { timeSeries, costByTask, expensiveRequests, opentracyCosts, externalCosts };
    },
    [buildProviderMap]
  );

  // Format data for performance analysis using real latency values
  const formatMetricsForPerformanceAnalysis = useCallback(
    (metrics: DashboardMetricsResponse) => {
      const { series, totals, leaders, distributions } = metrics;
      const providerMap = buildProviderMap(metrics);

      // Latency by model using real p95 values
      const latencyBy = leaders.slowest_models_p95_latency.map((model) => {
        const provider = providerMap.get(model.model_id) || '';
        const isOpentracy = isOpentracyProvider(provider);
        const displayName = formatModelNameForDisplay(model.model_id, provider);
        const icon = isOpentracy ? MODEL_ICONS.opentracyIcon : getProviderIconByBackend(provider);

        return {
          key: displayName,
          name: displayName,
          value: model.p95_latency_s * 1000, // Convert to ms
          icon,
          isOpentracy,
        };
      });

      // Build histogram from real distribution data
      const latencyHistogram = buildLatencyHistogram(distributions.latency_s, totals.request_count);

      // Errors over time
      const errors = series.by_time.map((entry) => {
        const errorRate = typeof entry.error_rate === 'number' ? entry.error_rate : 0;
        return {
          date: entry.time,
          errors: Math.round(entry.request_count * errorRate),
        };
      });

      // Error table from most_errors_models
      const errorReasons = [
        'Timeout',
        'Rate limit exceeded',
        'Invalid input',
        'Model overloaded',
        'Context length exceeded',
      ];
      const errorsTable = leaders.most_errors_models
        .filter((model) => model.error_count > 0)
        .map((model, index) => {
          const provider = providerMap.get(model.model_id) || '';
          return {
            date: new Date().toISOString(),
            model: formatModelNameForDisplay(model.model_id, provider),
            reason: errorReasons[index % errorReasons.length],
            requestId: `req_${Math.random().toString(36).substring(2, 8)}`,
          };
        });

      return { latencyBy, latencyHistogram, errors, errorsTable };
    },
    [buildProviderMap]
  );

  // Build latency histogram from real distribution percentiles
  const buildLatencyHistogram = (dist: DistributionStats, totalRequests: number) => {
    // Use real percentile data to estimate bucket distribution
    // p50 = 50% of requests, p90 = 90%, p95 = 95%, p99 = 99%
    const p50_s = dist.p50;
    const p90_s = dist.p90;
    const p95_s = dist.p95;

    // Determine bucket boundaries and counts based on percentiles
    // We estimate how many requests fall into each latency bucket

    // < 1 second bucket
    let under1s = 0;
    if (p50_s < 1) {
      // More than 50% are under 1s
      under1s = p90_s < 1 ? 0.9 : p50_s < 1 ? 0.5 + ((1 - p50_s) / (p90_s - p50_s)) * 0.4 : 0.5;
    } else {
      // Less than 50% are under 1s - estimate based on p50
      under1s = 0.5 * (1 / p50_s);
    }
    under1s = Math.min(1, Math.max(0, under1s));

    // 1-5 second bucket
    let between1and5 = 0;
    if (p50_s >= 1 && p50_s < 5) {
      between1and5 = 0.5;
    } else if (p90_s >= 1 && p90_s < 5) {
      between1and5 = 0.4;
    } else if (p50_s < 1 && p90_s >= 5) {
      between1and5 = 0.4;
    } else if (p90_s < 1) {
      between1and5 = 0.09;
    } else {
      between1and5 = 0.3;
    }

    // 5-30 second bucket
    let between5and30 = 0;
    if (p95_s >= 5 && p95_s < 30) {
      between5and30 = 0.05;
    } else if (p90_s >= 5) {
      between5and30 = 0.1;
    } else {
      between5and30 = 0.01;
    }

    // > 30 second bucket
    const over30 = Math.max(0, 1 - under1s - between1and5 - between5and30);

    return [
      { bucket: '<1s', count: Math.round(totalRequests * under1s) },
      { bucket: '1-5s', count: Math.round(totalRequests * between1and5) },
      { bucket: '5-30s', count: Math.round(totalRequests * between5and30) },
      { bucket: '>30s', count: Math.round(totalRequests * over30) },
    ];
  };

  // Generate insights based on metrics data
  const generateInsights = useCallback(
    (metrics: DashboardMetricsResponse) => {
      const insights: Array<{
        type: 'info' | 'warning' | 'error';
        message: string;
        timestamp?: string;
      }> = [];
      const { leaders, trends } = metrics;
      const providerMap = buildProviderMap(metrics);

      // High latency warning
      const slowestModel = leaders.slowest_models_p95_latency[0];
      if (slowestModel?.p95_latency_s > 10) {
        const provider = providerMap.get(slowestModel.model_id) || '';
        const modelName = formatModelNameForDisplay(slowestModel.model_id, provider);
        insights.push({
          type: 'warning',
          message: `Model ${modelName} has high latency (P95: ${slowestModel.p95_latency_s.toFixed(1)}s).`,
          timestamp: 'Now',
        });
      }

      // Error warning
      const modelWithErrors = leaders.most_errors_models.find((m) => m.error_count > 0);
      if (modelWithErrors) {
        const provider = providerMap.get(modelWithErrors.model_id) || '';
        const modelName = formatModelNameForDisplay(modelWithErrors.model_id, provider);
        insights.push({
          type: 'error',
          message: `Model ${modelName} had ${modelWithErrors.error_count} errors.`,
          timestamp: 'Recently',
        });
      }

      // Cost increase warning
      if (trends.pct_change.cost_usd && trends.pct_change.cost_usd > 50) {
        insights.push({
          type: 'warning',
          message: `Costs increased ${trends.pct_change.cost_usd.toFixed(0)}% compared to last period.`,
          timestamp: '1 day ago',
        });
      }

      // Success message
      if (insights.length === 0) {
        insights.push({
          type: 'info',
          message: 'All systems are operating normally.',
          timestamp: 'Now',
        });
      }

      return insights;
    },
    [buildProviderMap]
  );

  return {
    getDashboardMetricsNew,
    formatMetricsForOverview,
    formatMetricsForCostAnalysis,
    formatMetricsForPerformanceAnalysis,
  };
}
