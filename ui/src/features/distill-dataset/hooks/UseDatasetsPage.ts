import { useState, useCallback, useRef, useEffect, useMemo, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useDatasets } from '@/hooks/useDatasets';
import { useMetrics } from '@/contexts/MetricsContext';
import { useEvaluationsService } from '@/features/evaluations/api/evaluationsService';
import { useClusteringService, type ClusterDataset } from '@/services/clusteringService';
import { DEFAULT_AUTO_COLLECT } from '../constants';
import type {
  Dataset,
  Trace,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  GenerateDatasetRequest,
} from '../types';

export function useDatasetsPage() {
  const navigate = useNavigate();
  const accessToken = '';

  const [searchTerm, setSearchTerm] = useState('');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [deletingDatasetId, setDeletingDatasetId] = useState<string | null>(null);

  const service = useEvaluationsService();
  const clusteringService = useClusteringService();
  const { allTraces, isLoading: metricsLoading, isInitialized: metricsInitialized } = useMetrics();
  const datasetsHook = useDatasets();
  const bgPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cluster datasets
  const [clusters, setClusters] = useState<ClusterDataset[]>([]);
  const [clustersLoading, setClustersLoading] = useState(false);

  const {
    datasets,
    loading: datasetsLoading,
    deleteDataset,
    createDataset,
    createDatasetFromTraces,
    createDatasetFromInstruction,
    generateDataset,
    getDataset,
    refreshDatasets,
    importDataset,
  } = datasetsHook;

  useEffect(
    () => () => {
      if (bgPollRef.current) clearInterval(bgPollRef.current);
    },
    []
  );

  // Load cluster datasets on mount
  useEffect(() => {
    let cancelled = false;
    setClustersLoading(true);
    clusteringService
      .listDatasets()
      .then((result) => {
        if (!cancelled) setClusters(result.datasets);
      })
      .catch(() => {
        // silently fail — clusters are optional
      })
      .finally(() => {
        if (!cancelled) setClustersLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const traces: Trace[] = useMemo(
    () =>
      allTraces.map((t) => ({
        id: t.id || t.event_id || t.created_at,
        input: t.input_preview ?? '',
        output: t.output_preview ?? '',
        model_id: t.model_id,
        latency_ms: t.latency_s * 1000,
        cost_usd: t.cost_usd,
        source: t.endpoint ?? '',
        created_at: t.created_at,
      })),
    [allTraces]
  );

  const filteredDatasets = useMemo(
    () =>
      datasets.filter((d) => {
        const matchesSearch =
          !searchTerm ||
          d.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          d.description?.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSource = sourceFilter === 'all' || (d.source ?? 'manual') === sourceFilter;
        return matchesSearch && matchesSource;
      }),
    [datasets, searchTerm, sourceFilter]
  );

  const setupAutoCollect = useCallback(
    async (datasetId: string, instruction: string, maxSamples?: number) => {
      try {
        await service.putAutoCollectConfig(accessToken, datasetId, {
          enabled: true,
          instruction,
          ...DEFAULT_AUTO_COLLECT,
          ...(maxSamples ? { max_samples: maxSamples } : {}),
        });
        await service.triggerAutoCollect(accessToken, datasetId);
      } catch (err) {
        console.warn('Auto-collect setup failed:', err);
      }
    },
    [accessToken, service]
  );

  const handleGenerateBackground = useCallback(
    (datasetId: string, name: string, requested: number) => {
      // Clear any existing poll for this session before starting a new one
      if (bgPollRef.current) {
        clearInterval(bgPollRef.current);
        bgPollRef.current = null;
      }

      const toastId = `generate-${datasetId}`;
      toast.loading(`Generating ${requested} samples for "${name}"…`, {
        id: toastId,
        duration: Infinity,
        action: {
          label: 'Dismiss',
          onClick: () => {
            toast.dismiss(toastId);
          },
        },
      });
      let stablePolls = 0;
      let lastCount = 0;

      bgPollRef.current = setInterval(async () => {
        try {
          const result = await getDataset(datasetId, { include_samples: false });
          const currentCount = result?.samples_total ?? result?.dataset?.samples_count ?? 0;

          if (currentCount > lastCount) {
            stablePolls = 0;
            lastCount = currentCount;
            if (currentCount >= requested) {
              clearInterval(bgPollRef.current!);
              bgPollRef.current = null;
              toast.dismiss(toastId);
              toast.success(`Generated ${currentCount} samples for "${name}"`, {
                closeButton: true,
              });
              refreshDatasets();
            }
          } else if (currentCount > 0) {
            stablePolls++;
            if (stablePolls >= 5) {
              clearInterval(bgPollRef.current!);
              bgPollRef.current = null;
              toast.dismiss(toastId);
              toast.success(`Generated ${currentCount} samples for "${name}"`, {
                closeButton: true,
              });
              refreshDatasets();
            }
          }
        } catch {
          /* retry next interval */
        }
      }, 4000);
    },
    [getDataset, refreshDatasets]
  );

  const handleDelete = useCallback(
    async (dataset: Dataset, e: MouseEvent) => {
      e.stopPropagation();
      setDeletingDatasetId(dataset.id);
      const success = await deleteDataset(dataset.id);
      setDeletingDatasetId(null);
      if (success) toast.success(`Deleted "${dataset.name}"`);
      else toast.error('Failed to delete dataset');
    },
    [deleteDataset]
  );

  const handleCreate = useCallback(
    async (request: CreateDatasetRequest) => {
      const dataset = await createDataset(request);
      if (!dataset) return;
      toast.success(`Created "${dataset.name}"`);
      setShowCreateModal(false);
      if (request.auto_collect_instruction) {
        await setupAutoCollect(dataset.id, request.auto_collect_instruction);
      }
    },
    [createDataset, setupAutoCollect]
  );

  const handleCreateFromTraces = useCallback(
    async (name: string, traceIds: string[]) => {
      const dataset = await createDatasetFromTraces(name, traceIds);
      if (!dataset) return;
      toast.success(`Created "${dataset.name}" from traces`);
    },
    [createDatasetFromTraces]
  );

  const refreshClusters = useCallback(async () => {
    setClustersLoading(true);
    try {
      const result = await clusteringService.listDatasets();
      setClusters(result.datasets);
    } catch {
      // silently fail
    } finally {
      setClustersLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusteringService]);

  const handleTriggerClustering = useCallback(
    async (days = 30) => {
      setClustersLoading(true);
      toast.info('Clustering pipeline started...');
      try {
        const result = await clusteringService.triggerRun(days);
        toast.success(
          `Clustering complete: ${result.summary.qualified} qualified, ` +
            `${result.summary.candidate} candidate, ${result.summary.rejected} rejected`
        );
        await refreshClusters();
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Clustering failed';
        toast.error(msg);
      } finally {
        setClustersLoading(false);
      }
    },
    [clusteringService, refreshClusters]
  );

  const handleCreateFromCluster = useCallback(
    async (name: string, runId: string, clusterId: number) => {
      // Export the cluster traces then create a dataset via the evaluations pipeline
      const blob = await clusteringService.exportDataset(runId, clusterId);
      const file = new File([blob], `cluster_${clusterId}.jsonl`, {
        type: 'application/jsonl',
      });
      const dataset = await importDataset(file, name);
      if (!dataset) {
        throw new Error('Failed to create dataset from cluster');
      }
      toast.success(`Created "${dataset.name}" from cluster`);
      refreshDatasets();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [clusteringService, importDataset, refreshDatasets]
  );

  const handleCreateFromTopic = useCallback(
    async (request: CreateFromInstructionRequest) => {
      const result = await createDatasetFromInstruction(request);
      if (!result) return result;
      toast.success(`Created "${result.name}" with ${result.samples_count} samples`);
      setShowCreateModal(false);
      if (request.max_samples && request.max_samples > result.samples_count) {
        await setupAutoCollect(result.dataset_id, request.instruction, request.max_samples);
      }
      return result;
    },
    [createDatasetFromInstruction, setupAutoCollect]
  );

  const handleGenerate = useCallback(
    async (request: GenerateDatasetRequest) => {
      const result = await generateDataset(request);
      if (!result) return result;
      if (request.auto_collect_instruction) {
        await setupAutoCollect(result.dataset_id, request.auto_collect_instruction);
      }
      return result;
    },
    [generateDataset, setupAutoCollect]
  );

  const handleImport = useCallback(
    async (file: File, name: string, autoCollectInstruction?: string) => {
      const dataset = await importDataset(file, name);
      if (!dataset) {
        toast.error('Failed to import dataset');
        return null;
      }
      toast.success(`Imported "${dataset.name}"`);
      setShowCreateModal(false);
      if (autoCollectInstruction) {
        await setupAutoCollect(dataset.id, autoCollectInstruction);
      }
      return dataset;
    },
    [importDataset, setupAutoCollect]
  );

  const handleAnalyzeTraces = useCallback(
    async (data: any[]) => {
      return await service.analyzeTraces(accessToken, data);
    },
    [accessToken, service]
  );

  const handleImportTraces = useCallback(
    async (name: string, data: any[], mapping: any, description?: string) => {
      const result = await service.importTraces(accessToken, name, data, mapping, description);
      toast.success(`Imported "${result.name}" with ${result.samples_count} samples`);
      refreshDatasets();
      return result;
    },
    [accessToken, service, refreshDatasets]
  );

  const handlePollGenerate = useCallback(
    async (datasetId: string) => {
      const result = await getDataset(datasetId, { include_samples: false });
      return result?.samples_total ?? result?.dataset?.samples_count ?? 0;
    },
    [getDataset]
  );

  const handleCloseCreateModal = useCallback(() => {
    setShowCreateModal(false);
    refreshDatasets();
  }, [refreshDatasets]);

  const navigateToDataset = useCallback(
    (id: string) => navigate(`/distill-datasets/${id}`),
    [navigate]
  );

  return {
    searchTerm,
    setSearchTerm,
    sourceFilter,
    setSourceFilter,
    showCreateModal,
    setShowCreateModal,
    deletingDatasetId,

    datasets,
    datasetsLoading,
    filteredDatasets,
    traces,
    tracesLoading: metricsLoading && !metricsInitialized,
    clusters,
    clustersLoading,
    isEmpty: datasets.length === 0,

    handleDelete,
    handleCreate,
    handleCreateFromTraces,
    handleCreateFromCluster,
    handleTriggerClustering,
    handleCreateFromTopic,
    handleGenerate,
    handleImport,
    handleAnalyzeTraces,
    handleImportTraces,
    handlePollGenerate,
    handleGenerateBackground,
    handleCloseCreateModal,

    navigateToDataset,
  };
}
