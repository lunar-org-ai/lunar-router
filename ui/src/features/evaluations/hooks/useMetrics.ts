import { useState, useCallback, useEffect, useRef } from 'react';
import { useEvaluationsService } from '../api/evaluationsService';
import type { EvaluationMetric, CreateCustomMetricRequest } from '../types';

interface MetricsState {
  builtin: EvaluationMetric[];
  custom: EvaluationMetric[];
}

export function useMetrics() {
  const accessToken = 'no-auth';
  const service = useEvaluationsService();
  const loaded = useRef(false);

  const [metrics, setMetrics] = useState<MetricsState>({ builtin: [], custom: [] });
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await service.listMetrics(accessToken);
      setMetrics(data);
    } catch (err) {
      console.error('[useMetrics] Failed to fetch:', err);
    } finally {
      setLoading(false);
    }
  }, [accessToken, service]);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    refresh();
  }, [refresh]);

  const createCustomMetric = useCallback(
    async (request: CreateCustomMetricRequest): Promise<EvaluationMetric | null> => {
      try {
        const metric = await service.createCustomMetric(accessToken, request);
        setMetrics((prev) => ({ ...prev, custom: [...prev.custom, metric] }));
        return metric;
      } catch (err) {
        console.error('[useMetrics] Failed to create:', err);
        return null;
      }
    },
    [accessToken, service]
  );

  const deleteCustomMetric = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        await service.deleteCustomMetric(accessToken, id);
        setMetrics((prev) => ({
          ...prev,
          custom: prev.custom.filter((m) => m.metric_id !== id),
        }));
        return true;
      } catch (err) {
        console.error('[useMetrics] Failed to delete:', err);
        return false;
      }
    },
    [accessToken, service]
  );

  return { metrics, loading, refresh, createCustomMetric, deleteCustomMetric };
}
