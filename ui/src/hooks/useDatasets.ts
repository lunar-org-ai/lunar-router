import { useState, useCallback, useEffect, useRef } from 'react';
import { useEvaluationsService } from '../features/evaluations/api/evaluationsService';
import type {
  Dataset,
  DatasetSample,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  CreateFromInstructionResponse,
  GenerateDatasetRequest,
  GenerateDatasetResponse,
} from '../features/evaluations/types/evaluationsTypes';

export function useDatasets() {
  const accessToken = 'no-auth';
  const service = useEvaluationsService();
  const loaded = useRef(false);

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await service.listDatasets(accessToken);
      setDatasets(data);
    } catch (err) {
      console.error('[useDatasets] refresh failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch datasets');
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, [accessToken, service]);

  const getDataset = useCallback(
    async (
      id: string,
      options?: { include_samples?: boolean; samples_limit?: number; samples_offset?: number }
    ): Promise<{ dataset: Dataset; samples: DatasetSample[]; samples_total?: number } | null> => {
      try {
        return await service.getDataset(accessToken, id, options);
      } catch (err) {
        console.error('[useDatasets] getDataset failed:', err);
        return null;
      }
    },
    [accessToken, service]
  );

  const addDatasetLocally = useCallback((dataset: Dataset) => {
    setDatasets((prev) => [dataset, ...prev]);
  }, []);

  const createDataset = useCallback(
    async (request: CreateDatasetRequest): Promise<Dataset | null> => {
      try {
        const dataset = await service.createDataset(accessToken, request);
        addDatasetLocally(dataset);
        return dataset;
      } catch (err) {
        console.error('[useDatasets] create failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to create dataset');
        return null;
      }
    },
    [accessToken, service, addDatasetLocally]
  );

  const deleteDataset = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        await service.deleteDataset(accessToken, id);
        setDatasets((prev) => prev.filter((d) => d.id !== id));
        return true;
      } catch (err) {
        console.error('[useDatasets] delete failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to delete dataset');
        return false;
      }
    },
    [accessToken, service]
  );

  const importDataset = useCallback(
    async (file: File, name: string): Promise<Dataset | null> => {
      setLoading(true);
      try {
        const dataset = await service.importDataset(accessToken, file, name);
        addDatasetLocally(dataset);
        return dataset;
      } catch (err) {
        console.error('[useDatasets] import failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to import dataset');
        return null;
      } finally {
        setLoading(false);
      }
    },
    [accessToken, service, addDatasetLocally]
  );

  const createDatasetFromTraces = useCallback(
    async (name: string, traceIds: string[]): Promise<Dataset | null> => {
      try {
        const dataset = await service.createDatasetFromTraces(accessToken, name, traceIds);
        if (!dataset?.id) {
          setError('Failed to create dataset: invalid response from server');
          return null;
        }
        addDatasetLocally(dataset);
        return dataset;
      } catch (err) {
        console.error('[useDatasets] createFromTraces failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to create dataset');
        return null;
      }
    },
    [accessToken, service, addDatasetLocally]
  );

  const createDatasetFromInstruction = useCallback(
    async (
      request: CreateFromInstructionRequest
    ): Promise<CreateFromInstructionResponse | null> => {
      try {
        const result = await service.createDatasetFromInstruction(accessToken, request);
        addDatasetLocally({
          id: result.dataset_id,
          name: result.name,
          source: 'instruction',
          samples_count: result.samples_count,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        return result;
      } catch (err) {
        console.error('[useDatasets] createFromInstruction failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to create dataset');
        return null;
      }
    },
    [accessToken, service, addDatasetLocally]
  );

  const generateDataset = useCallback(
    async (request: GenerateDatasetRequest): Promise<GenerateDatasetResponse | null> => {
      try {
        const result = await service.generateDataset(accessToken, request);
        addDatasetLocally({
          id: result.dataset_id,
          name: result.name,
          source: 'synthetic',
          samples_count: result.samples_count,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
        return result;
      } catch (err) {
        console.error('[useDatasets] generate failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to generate dataset');
        return null;
      }
    },
    [accessToken, service, addDatasetLocally]
  );

  const clearError = useCallback(() => setError(null), []);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    refreshDatasets();
  }, [refreshDatasets]);

  return {
    datasets,
    loading,
    error,
    refreshDatasets,
    getDataset,
    createDataset,
    deleteDataset,
    importDataset,
    createDatasetFromTraces,
    createDatasetFromInstruction,
    generateDataset,
    clearError,
  };
}
