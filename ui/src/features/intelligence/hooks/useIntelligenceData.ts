import { useState, useCallback, useEffect, useMemo } from 'react';
import { useMetrics } from '@/contexts/MetricsContext';
import {
  fetchEfficiencyData,
  fetchModelPerformanceData,
  fetchTrainingActivityData,
} from '@/features/router-intelligence/api/routerIntelligenceService';
import type {
  EfficiencyData,
  ModelPerformanceData,
  TrainingActivityData,
} from '@/features/router-intelligence/types';
import type {
  OverviewData,
  CostAnalysisData,
  PerformanceData,
} from '@/features/observability/types';
import type { Period, UnifiedModelRow } from '../types';
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

  selectedDays: number;
  unifiedModelRows: UnifiedModelRow[];

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
  const [riLoading, setRiLoading] = useState(false);
  const [riError, setRiError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedDays(days);
  }, [days, setSelectedDays]);

  const fetchRouterIntelligence = useCallback(async () => {
    setRiLoading(true);
    setRiError(null);
    try {
      const [eff, mod, trn] = await Promise.all([
        fetchEfficiencyData(days),
        fetchModelPerformanceData(),
        fetchTrainingActivityData(days),
      ]);
      setEfficiency(eff);
      setModels(mod);
      setTraining(trn);
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
    selectedDays: days,
    unifiedModelRows,
    refreshData,
  };
}
