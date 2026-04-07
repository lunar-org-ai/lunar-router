/**
 * Hook for the Datasets page — connects to the clustering pipeline API.
 * Replaces the old useDatasetsPage that used the evaluations gateway.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { toast } from 'sonner';
import {
  useClusteringService,
  type ClusterDataset,
  type ClusteringRunResult,
} from '@/services/clusteringService';

interface UseClusterDatasetsReturn {
  // Data
  datasets: ClusterDataset[];
  runId: string | null;
  loading: boolean;
  running: boolean;
  error: string | null;

  // Filters
  searchTerm: string;
  setSearchTerm: (v: string) => void;
  statusFilter: string;
  setStatusFilter: (v: string) => void;
  filteredDatasets: ClusterDataset[];

  // Actions
  triggerClustering: (days?: number) => Promise<void>;
  refreshDatasets: () => Promise<void>;
  qualifyDataset: (clusterId: number, status: 'qualified' | 'rejected' | 'candidate') => Promise<void>;
  exportDataset: (clusterId: number) => Promise<void>;
}

export function useClusterDatasets(): UseClusterDatasetsReturn {
  const service = useClusteringService();

  const [datasets, setDatasets] = useState<ClusterDataset[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  // Load datasets on mount
  const refreshDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await service.listDatasets();
      setDatasets(result.datasets);
      setRunId(result.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Trigger clustering run
  const triggerClustering = useCallback(
    async (days = 30) => {
      setRunning(true);
      setError(null);
      toast.info('Clustering pipeline started...');

      try {
        const result: ClusteringRunResult = await service.triggerRun(days);
        toast.success(
          `Clustering complete: ${result.summary.qualified} qualified, ` +
            `${result.summary.candidate} candidate, ${result.summary.rejected} rejected`
        );
        await refreshDatasets();
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Clustering failed';
        setError(msg);
        toast.error(msg);
      } finally {
        setRunning(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [refreshDatasets]
  );

  // Qualify/reject a dataset
  const qualifyDataset = useCallback(
    async (clusterId: number, status: 'qualified' | 'rejected' | 'candidate') => {
      if (!runId) return;
      try {
        await service.qualifyDataset(runId, clusterId, status);
        toast.success(`Dataset ${clusterId} set to ${status}`);
        await refreshDatasets();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to update status');
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runId, refreshDatasets]
  );

  // Export dataset as JSONL
  const exportDataset = useCallback(
    async (clusterId: number) => {
      if (!runId) return;
      try {
        const blob = await service.exportDataset(runId, clusterId);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dataset_${runId}_${clusterId}.jsonl`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Dataset exported');
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Export failed');
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runId]
  );

  // Filtered datasets
  const filteredDatasets = useMemo(() => {
    let result = datasets;
    if (statusFilter !== 'all') {
      result = result.filter((d) => d.status === statusFilter);
    }
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      result = result.filter(
        (d) =>
          d.domain_label.toLowerCase().includes(q) ||
          d.short_description.toLowerCase().includes(q) ||
          d.top_models.some((m) => m.toLowerCase().includes(q))
      );
    }
    return result;
  }, [datasets, statusFilter, searchTerm]);

  return {
    datasets,
    runId,
    loading,
    running,
    error,
    searchTerm,
    setSearchTerm,
    statusFilter,
    setStatusFilter,
    filteredDatasets,
    triggerClustering,
    refreshDatasets,
    qualifyDataset,
    exportDataset,
  };
}
