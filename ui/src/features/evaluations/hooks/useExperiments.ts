import { useState, useCallback, useEffect, useRef } from 'react';
import { useEvaluationsService } from '../api/evaluationsService';
import type { Experiment, ExperimentComparison } from '../types/evaluationsTypes';

export function useExperiments() {
  const accessToken = '';
  const service = useEvaluationsService();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const initialLoadDone = useRef(false);

  const fetchExperiments = useCallback(async () => {
    setLoading(true);
    try {
      const data = await service.listExperiments(accessToken);
      setExperiments(data);
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  }, [service]);

  const createExperiment = useCallback(
    async (name: string, description: string, datasetId: string, evaluationIds: string[]) => {
      try {
        const newExperiment = await service.createExperiment(accessToken, {
          name,
          description,
          dataset_id: datasetId,
          evaluation_ids: evaluationIds,
        });
        setExperiments((prev) => [newExperiment, ...prev]);
      } catch (error) {
        console.error('Failed to create experiment:', error);
        throw error;
      }
    },
    [service]
  );

  const deleteExperiment = useCallback(
    async (id: string) => {
      try {
        await service.deleteExperiment(accessToken, id);
        setExperiments((prev) => prev.filter((e) => e.id !== id));
      } catch (error) {
        console.error('Failed to delete experiment:', error);
        throw error;
      }
    },
    [service]
  );

  const getComparison = useCallback(
    async (experimentId: string): Promise<ExperimentComparison | null> => {
      try {
        return await service.getExperimentComparison(accessToken, experimentId);
      } catch (error) {
        console.error('Failed to get comparison:', error);
        return null;
      }
    },
    [service]
  );

  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      fetchExperiments();
    }
  }, [fetchExperiments]);

  return {
    experiments,
    loading,
    createExperiment,
    deleteExperiment,
    getComparison,
    refresh: fetchExperiments,
  };
}
