import { useState, useCallback, useEffect, useMemo } from 'react';
import { useMetrics } from '@/contexts/MetricsContext';
import {
  fetchEfficiencyData,
  fetchModelPerformanceData,
  fetchTrainingActivityData,
  fetchRoutingIntelligenceData,
} from '@/features/router-intelligence/api/routerIntelligenceService';
import type {
  EfficiencyData,
  ModelPerformanceData,
  TrainingActivityData,
  RoutingIntelligenceData,
} from '@/features/router-intelligence/types';
import type {
  OverviewData,
  CostAnalysisData,
  PerformanceData,
} from '@/features/observability/types';
import type {
  Period,
  UnifiedModelRow,
  RoutingDecision,
  TrainingRunDetail,
  ModelCapability,
} from '../types';
import { PERIOD_TO_DAYS } from '../types';

export interface IntelligenceData {
  loading: boolean;
  error: string | null;

  overviewData: OverviewData | null;
  costData: CostAnalysisData | null;
  performanceData: PerformanceData | null;

  efficiency: EfficiencyData | null;
  models: ModelPerformanceData | null;
  training: TrainingActivityData | null;
  routingIntelligence: RoutingIntelligenceData | null;

  selectedDays: number;
  unifiedModelRows: UnifiedModelRow[];
  routingDecisions: RoutingDecision[];
  trainingRuns: TrainingRunDetail[];
  modelCapabilities: ModelCapability[];

  refreshData: () => void;
}

function deriveProvider(model: string): string {
  const lower = model.toLowerCase();
  if (lower.includes('gpt') || lower.includes('o1') || lower.includes('o3') || lower.includes('o4'))
    return 'OpenAI';
  if (lower.includes('claude')) return 'Anthropic';
  if (lower.includes('llama') || lower.includes('meta')) return 'Meta';
  if (lower.includes('gemma') || lower.includes('gemini')) return 'Google';
  if (lower.includes('mixtral') || lower.includes('mistral')) return 'Mistral';
  if (lower.includes('deepseek')) return 'DeepSeek';
  if (lower.includes('qwen')) return 'Qwen';
  if (lower.startsWith('lunar/')) return 'Lunar';
  return 'Other';
}

// Reference data for model capabilities — factual, not mock
const KNOWN_CAPABILITIES: Record<string, Omit<ModelCapability, 'model' | 'provider'>> = {
  'gpt-4o': {
    contextWindow: 128000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'premium',
  },
  'gpt-4o-mini': {
    contextWindow: 128000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'medium',
  },
  'gpt-4-turbo': {
    contextWindow: 128000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'high',
  },
  'gpt-3.5-turbo': {
    contextWindow: 16385,
    supportsVision: false,
    supportsFunctionCalling: true,
    costTier: 'low',
  },
  'claude-3.5-sonnet': {
    contextWindow: 200000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'high',
  },
  'claude-3-opus': {
    contextWindow: 200000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'premium',
  },
  'claude-3-haiku': {
    contextWindow: 200000,
    supportsVision: true,
    supportsFunctionCalling: true,
    costTier: 'low',
  },
  'llama-3.1-70b': {
    contextWindow: 128000,
    supportsVision: false,
    supportsFunctionCalling: false,
    costTier: 'medium',
  },
  'llama-3.1-8b': {
    contextWindow: 128000,
    supportsVision: false,
    supportsFunctionCalling: false,
    costTier: 'low',
  },
  'mixtral-8x7b': {
    contextWindow: 32768,
    supportsVision: false,
    supportsFunctionCalling: true,
    costTier: 'low',
  },
  'mistral-large': {
    contextWindow: 128000,
    supportsVision: false,
    supportsFunctionCalling: true,
    costTier: 'high',
  },
  'deepseek-v2': {
    contextWindow: 128000,
    supportsVision: false,
    supportsFunctionCalling: true,
    costTier: 'low',
  },
};

function buildModelCapabilities(rows: UnifiedModelRow[]): ModelCapability[] {
  return rows.map((row): ModelCapability => {
    const lowerModel = row.model.toLowerCase();
    const match = Object.entries(KNOWN_CAPABILITIES).find(([key]) => lowerModel.includes(key));
    const defaults = match?.[1] ?? {
      contextWindow: 32768,
      supportsVision: false,
      supportsFunctionCalling: false,
      costTier: 'medium' as const,
    };
    return {
      model: row.model,
      provider: row.provider,
      ...defaults,
    };
  });
}

/**
 * Build routing decisions from real API data.
 */
function buildRoutingDecisions(ri: RoutingIntelligenceData | null): RoutingDecision[] {
  if (!ri?.decisions?.length) return [];
  return ri.decisions.map((d) => ({
    requestId: d.request_id,
    cluster: d.cluster,
    modelChosen: d.model_chosen,
    reason: d.reason,
    cost: d.cost,
    latency: d.latency,
    outcome: d.outcome,
    timestamp: d.timestamp,
  }));
}

/**
 * Build training run details from real API data (training_runs_detail).
 * Falls back to training_history + distillation_jobs if detail not available.
 */
function buildTrainingRuns(
  training: TrainingActivityData | null,
  efficiency: EfficiencyData | null
): TrainingRunDetail[] {
  // Prefer the new training_runs_detail with real confidence/duration
  if (training?.training_runs_detail?.length) {
    return training.training_runs_detail.map((r) => ({
      runId: r.run_id,
      name: r.name || 'Training run',
      date: r.date,
      outcome: r.outcome === 'promoted' ? 'promoted' : 'rejected',
      confidence: r.confidence,
      cost: r.cost,
      duration: r.duration,
      reason: r.reason,
      teacherModel: r.teacher_model ?? '',
      studentModel: r.student_model ?? '',
      qualityScore: r.quality_score ?? 0,
      status: r.status ?? r.outcome,
    })) as TrainingRunDetail[];
  }

  // Fallback: build from training_history + distillation_jobs
  const runs: TrainingRunDetail[] = [];

  if (training?.training_history) {
    for (let i = 0; i < training.training_history.length; i++) {
      const h = training.training_history[i];
      runs.push({
        runId: `run_${String(i + 1).padStart(3, '0')}`,
        name: 'Training run',
        date: h.date,
        outcome: h.promoted ? 'promoted' : 'rejected',
        confidence: 0,
        cost: 0,
        duration: '\u2014',
        reason: h.reason,
        teacherModel: '',
        studentModel: '',
        qualityScore: 0,
        status: h.promoted ? 'completed' : 'failed',
      });
    }
  }

  if (efficiency?.distillation_jobs) {
    for (const job of efficiency.distillation_jobs) {
      const alreadyAdded = runs.some(
        (r) => Math.abs(new Date(r.date).getTime() - new Date(job.created_at).getTime()) < 60000
      );
      if (!alreadyAdded) {
        let duration = '\u2014';
        if (job.completed_at && job.created_at) {
          const mins = Math.round(
            (new Date(job.completed_at).getTime() - new Date(job.created_at).getTime()) / 60000
          );
          duration = mins >= 60 ? `${Math.floor(mins / 60)}h ${mins % 60}m` : `${mins}m`;
        }
        runs.push({
          runId: job.job_id,
          name: job.name || 'Distillation run',
          date: job.created_at,
          outcome: job.status === 'completed' ? 'promoted' : 'rejected',
          confidence: 0,
          cost: job.cost_accrued,
          duration,
          reason: 'Distillation run',
          teacherModel: '',
          studentModel: '',
          qualityScore: 0,
          status: job.status,
        });
      }
    }
  }

  return runs.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function useIntelligenceData(period: Period): IntelligenceData {
  const days = PERIOD_TO_DAYS[period];

  const {
    isLoading: metricsLoading,
    error: metricsError,
    overviewData,
    costData,
    performanceData,
    setSelectedDays,
    refreshData: refreshMetrics,
  } = useMetrics();

  const [efficiency, setEfficiency] = useState<EfficiencyData | null>(null);
  const [models, setModels] = useState<ModelPerformanceData | null>(null);
  const [training, setTraining] = useState<TrainingActivityData | null>(null);
  const [routingIntelligence, setRoutingIntelligence] = useState<RoutingIntelligenceData | null>(
    null
  );
  const [riLoading, setRiLoading] = useState(false);
  const [riError, setRiError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedDays(days);
  }, [days, setSelectedDays]);

  const fetchRouterIntelligence = useCallback(async () => {
    setRiLoading(true);
    setRiError(null);
    try {
      const [eff, mod, trn, ri] = await Promise.all([
        fetchEfficiencyData(days),
        fetchModelPerformanceData(),
        fetchTrainingActivityData(days),
        fetchRoutingIntelligenceData(days),
      ]);
      setEfficiency(eff);
      setModels(mod);
      setTraining(trn);
      setRoutingIntelligence(ri);
    } catch (err) {
      setRiError(err instanceof Error ? err.message : 'Failed to fetch intelligence data');
    } finally {
      setRiLoading(false);
    }
  }, [days]);

  useEffect(() => {
    fetchRouterIntelligence();
  }, [fetchRouterIntelligence]);

  const refreshData = useCallback(() => {
    refreshMetrics();
    fetchRouterIntelligence();
  }, [refreshMetrics, fetchRouterIntelligence]);

  const unifiedModelRows = useMemo((): UnifiedModelRow[] => {
    const rowMap = new Map<string, UnifiedModelRow>();

    if (efficiency?.model_breakdown) {
      const totalRequests = efficiency.model_breakdown.reduce((s, r) => s + r.requests, 0);
      for (const row of efficiency.model_breakdown) {
        rowMap.set(row.model, {
          model: row.model,
          provider: deriveProvider(row.model),
          requests: row.requests,
          trafficPct: totalRequests > 0 ? (row.requests / totalRequests) * 100 : 0,
          accuracy: row.accuracy,
          avgCost: row.avg_cost,
          totalCost: row.avg_cost * row.requests,
        });
      }
    }

    if (overviewData?.models) {
      for (const m of overviewData.models) {
        const name = m.model || m.name || 'Unknown';
        const existing = rowMap.get(name);
        if (existing) {
          existing.totalCost = m.cost ?? existing.totalCost;
        } else {
          const totalRequests = overviewData.models.reduce((s, x) => s + x.requests, 0);
          rowMap.set(name, {
            model: name,
            provider: deriveProvider(name),
            requests: m.requests,
            trafficPct: totalRequests > 0 ? (m.requests / totalRequests) * 100 : 0,
            accuracy: null,
            avgCost: m.cost && m.requests > 0 ? m.cost / m.requests : 0,
            totalCost: m.cost ?? 0,
          });
        }
      }
    }

    return Array.from(rowMap.values()).sort((a, b) => b.requests - a.requests);
  }, [efficiency, overviewData]);

  const routingDecisions = useMemo(
    () => buildRoutingDecisions(routingIntelligence),
    [routingIntelligence]
  );

  const trainingRuns = useMemo(
    () => buildTrainingRuns(training, efficiency),
    [training, efficiency]
  );

  const modelCapabilities = useMemo(
    () => buildModelCapabilities(unifiedModelRows),
    [unifiedModelRows]
  );

  const loading = metricsLoading || riLoading;
  const error = metricsError || riError;

  return {
    loading,
    error,
    overviewData,
    costData,
    performanceData,
    efficiency,
    models,
    training,
    routingIntelligence,
    selectedDays: days,
    unifiedModelRows,
    routingDecisions,
    trainingRuns,
    modelCapabilities,
    refreshData,
  };
}
