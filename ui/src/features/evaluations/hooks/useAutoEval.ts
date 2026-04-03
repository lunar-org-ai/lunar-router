import { useState, useCallback, useEffect } from 'react';
import { useEvaluationsService } from '../api/evaluationsService';
import type { AutoEvalConfig, AutoEvalRun } from '../types/evaluationsTypes';

function normalizeConfig(c: Record<string, unknown>): AutoEvalConfig {
  return { ...c, id: (c.config_id as string) || (c.id as string) } as AutoEvalConfig;
}

function normalizeRun(r: Record<string, unknown>): AutoEvalRun {
  return { ...r, id: (r.run_id as string) || (r.id as string) } as AutoEvalRun;
}

export function useAutoEval() {
  const accessToken = '';
  const service = useEvaluationsService();

  const [configs, setConfigs] = useState<AutoEvalConfig[]>([]);
  const [runs, setRuns] = useState<AutoEvalRun[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    service
      .listAutoEvalConfigs(accessToken)
      .then((res) => {
        if (!cancelled) setConfigs((res.configs || []).map(normalizeConfig));
      })
      .catch((err) => console.error('[useAutoEval] load failed:', err))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [service]);

  const createConfig = useCallback(
    async (
      data: Omit<
        AutoEvalConfig,
        'id' | 'created_at' | 'updated_at' | 'last_run_at' | 'last_run_score'
      >
    ) => {
      const result = await service.createAutoEvalConfig(accessToken, data);
      setConfigs((prev) => [normalizeConfig(result), ...prev]);
    },
    [service]
  );

  const updateConfig = useCallback(
    async (id: string, updates: Partial<AutoEvalConfig>) => {
      const result = await service.updateAutoEvalConfig(accessToken, id, updates);
      const updated = normalizeConfig(result);
      setConfigs((prev) => prev.map((c) => (c.id === id ? { ...c, ...updated } : c)));
    },
    [service]
  );

  const deleteConfig = useCallback(
    async (id: string) => {
      await service.deleteAutoEvalConfig(accessToken, id);
      setConfigs((prev) => prev.filter((c) => c.id !== id));
    },
    [service]
  );

  const triggerRun = useCallback(
    async (configId: string) => {
      setLoading(true);
      try {
        const result = await service.triggerAutoEvalRun(accessToken, configId);
        const newRun: AutoEvalRun = {
          id: result.run_id || result.id,
          config_id: configId,
          status: 'running',
          started_at: result.started_at || new Date().toISOString(),
          scores: {},
          regression_detected: false,
          evaluation_id: result.evaluation_id,
        };
        setRuns((prev) => [newRun, ...prev]);
      } finally {
        setLoading(false);
      }
    },
    [service]
  );

  const loadRuns = useCallback(
    async (configId: string) => {
      const res = await service.listAutoEvalRuns(accessToken, configId);
      setRuns((res.runs || []).map(normalizeRun));
    },
    [service]
  );

  const suggestMetrics = useCallback(
    async (datasetId: string) => {
      const res = await service.suggestMetrics(accessToken, datasetId);
      return res.suggestions || [];
    },
    [service]
  );

  return {
    configs,
    runs,
    loading,
    createConfig,
    updateConfig,
    deleteConfig,
    triggerRun,
    loadRuns,
    suggestMetrics,
  };
}
