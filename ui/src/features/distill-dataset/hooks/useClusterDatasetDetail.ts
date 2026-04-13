/**
 * Hook for the cluster dataset detail page.
 * Fetches traces from /v1/clustering/datasets/{runId}/{clusterId} and
 * adapts them to the DatasetSample format used by SamplesExplorer.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useClusteringService, type ClusterDataset } from '@/services/clusteringService';
import type { DatasetSample } from '@/features/evaluations/types/evaluationsTypes';

interface LengthFilter {
  min: number;
  max: number;
}

export function useClusterDatasetDetail() {
  const { runId, clusterId } = useParams<{ runId: string; clusterId: string }>();
  const navigate = useNavigate();
  const service = useClusteringService();

  const [dataset, setDataset] = useState<ClusterDataset | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [samplesPerPage, setSamplesPerPage] = useState(25);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [inputLengthFilter, setInputLengthFilter] = useState<LengthFilter | null>(null);
  const [outputLengthFilter, setOutputLengthFilter] = useState<LengthFilter | null>(null);

  // Fetch dataset metadata + traces
  useEffect(() => {
    if (!runId || !clusterId) return;

    const load = async () => {
      setLoading(true);
      try {
        // Get dataset info
        const runData = await service.getRun(runId);
        const ds = runData.datasets.find((d) => d.cluster_id === Number(clusterId));
        if (ds) setDataset(ds);

        // Get traces
        const tracesData = await service.getDatasetTraces(runId, Number(clusterId), 1000, 0);
        const adapted: DatasetSample[] = (tracesData.traces || []).map(
          (t: Record<string, unknown>, idx: number) => ({
            id: (t.request_id as string) || `trace-${idx}`,
            input: (t.input_text as string) || '',
            expected_output: (t.output_text as string) || '',
            output: (t.output_text as string) || '',
            metadata: {
              model: t.selected_model as string,
              provider: t.provider as string,
              latency_ms: t.latency_ms as number,
              cost_usd: t.total_cost_usd as number,
              is_error: t.is_error as boolean,
            },
          })
        );
        setSamples(adapted);
      } catch (err) {
        toast.error('Failed to load dataset');
      } finally {
        setLoading(false);
      }
    };

    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, clusterId]);

  // Filtered samples
  const filteredSamples = useMemo(() => {
    let result = samples;
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      result = result.filter(
        (s) =>
          s.input?.toLowerCase().includes(q) ||
          s.output?.toLowerCase().includes(q) ||
          s.expected_output?.toLowerCase().includes(q)
      );
    }
    if (inputLengthFilter) {
      result = result.filter((s) => {
        const len = s.input?.length || 0;
        return len >= inputLengthFilter.min && len <= inputLengthFilter.max;
      });
    }
    if (outputLengthFilter) {
      result = result.filter((s) => {
        const len = (s.expected_output || s.output || '').length;
        return len >= outputLengthFilter.min && len <= outputLengthFilter.max;
      });
    }
    return result;
  }, [samples, searchTerm, inputLengthFilter, outputLengthFilter]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredSamples.length / samplesPerPage));
  const startIndex = (currentPage - 1) * samplesPerPage;
  const paginatedSamples = filteredSamples.slice(startIndex, startIndex + samplesPerPage);

  const hasActiveFilters = !!searchTerm || !!inputLengthFilter || !!outputLengthFilter;

  const toggleRowExpand = useCallback((id: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const clearFilters = useCallback(() => {
    setSearchTerm('');
    setInputLengthFilter(null);
    setOutputLengthFilter(null);
    setCurrentPage(1);
  }, []);

  const handleExportJSON = useCallback(() => {
    if (!runId || !clusterId) return;
    service
      .exportDataset(runId, Number(clusterId))
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dataset_${runId}_${clusterId}.jsonl`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Dataset exported');
      })
      .catch(() => toast.error('Export failed'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, clusterId]);

  // Stats
  const stats = useMemo(() => {
    if (samples.length === 0) return null;
    const inputLengths = samples.map((s) => s.input?.length || 0);
    const outputLengths = samples.map((s) => (s.expected_output || s.output || '').length);
    return {
      totalSamples: samples.length,
      avgInputLength: Math.round(inputLengths.reduce((a, b) => a + b, 0) / inputLengths.length),
      avgOutputLength: Math.round(outputLengths.reduce((a, b) => a + b, 0) / outputLengths.length),
      maxInputLength: Math.max(...inputLengths),
      maxOutputLength: Math.max(...outputLengths),
    };
  }, [samples]);

  // Add traces to this dataset
  const handleAddTraces = useCallback(
    async (traces: Array<{ input: string; output: string; model?: string }>) => {
      if (!runId || !clusterId) return;
      try {
        const result = await service.addTracesToDataset(runId, Number(clusterId), traces);
        toast.success(`Added ${result.ingested} traces to dataset`);
        // Reload traces
        const tracesData = await service.getDatasetTraces(runId, Number(clusterId), 1000, 0);
        const adapted: DatasetSample[] = (tracesData.traces || []).map(
          (t: Record<string, unknown>, idx: number) => ({
            id: (t.request_id as string) || `trace-${idx}`,
            input: (t.input_text as string) || '',
            expected_output: (t.output_text as string) || '',
            output: (t.output_text as string) || '',
            metadata: {
              model: t.selected_model as string,
              provider: t.provider as string,
              latency_ms: t.latency_ms as number,
              cost_usd: t.total_cost_usd as number,
              is_error: t.is_error as boolean,
            },
          })
        );
        setSamples(adapted);
        if (dataset) {
          setDataset({ ...dataset, trace_count: dataset.trace_count + result.ingested });
        }
      } catch (err) {
        toast.error('Failed to add traces');
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [runId, clusterId, dataset]
  );

  return {
    dataset,
    loading,
    handleAddTraces,
    samples: {
      loading,
      stats,
      filteredSamples,
      paginatedSamples,
      totalPages,
      currentPage,
      startIndex,
      samplesPerPage,
      searchTerm,
      hasActiveFilters,
      expandedRows,
      inputLengthFilter,
      outputLengthFilter,
      setSearchTerm,
      setCurrentPage,
      setSamplesPerPage,
      toggleRowExpand,
      clearFilters,
      setInputLengthFilter,
      setOutputLengthFilter,
      handleHistogramClick: () => {},
    },
    handleExportJSON,
    navigate,
    runId,
    clusterId,
  };
}
