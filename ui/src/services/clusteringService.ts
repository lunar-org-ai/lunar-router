/**
 * Clustering API service — connects UI to the clustering pipeline endpoints.
 */

import { useCallback } from 'react';
import { API_BASE_URL } from '../config/api';

const BASE = API_BASE_URL;

// --- Types ---

export interface ClusterLabel {
  domain_label: string;
  short_description: string;
  inclusion_rule: string;
  exclusion_rule: string;
  confidence: number;
}

export interface ClusterDataset {
  run_id: string;
  cluster_id: number;
  status: 'candidate' | 'qualified' | 'rejected';
  domain_label: string;
  short_description: string;
  inclusion_rule: string;
  exclusion_rule: string;
  label_confidence: number;
  trace_count: number;
  coherence_score: number;
  diversity_score: number;
  noise_rate: number;
  avg_success_rate: number;
  avg_latency_ms: number;
  avg_cost_usd: number;
  top_models: string[];
  top_providers: string[];
  sample_prompts: string[];
  version: string;
}

export interface ClusteringRun {
  run_id: string;
  created_at: string;
  strategy: string;
  num_clusters: number;
  silhouette_score: number;
  source_window_start: string;
  source_window_end: string;
  total_traces: number;
  embedding_model: string;
  labeler_model: string;
  config: string;
}

export interface MergeSuggestion {
  cluster_a: number;
  cluster_b: number;
  similarity_score: number;
  llm_agrees: boolean;
  reason: string;
}

export interface ClusteringRunResult {
  version: {
    run_id: string;
    created_at: string;
    source_window_start: string | null;
    source_window_end: string | null;
    embedding_model: string;
    clustering_config: Record<string, unknown>;
    labeler_model: string;
    trace_count: number;
    num_clusters: number;
    silhouette_score: number;
  };
  datasets: Array<{
    cluster_id: number;
    label: ClusterLabel;
    trace_count: number;
    status: string;
    coherence_score: number;
    diversity_score: number;
    noise_rate: number;
    avg_success_rate: number;
    avg_latency_ms: number;
    avg_cost_usd: number;
    top_models: string[];
    top_providers: string[];
    sample_prompts: string[];
  }>;
  merge_suggestions: MergeSuggestion[];
  summary: {
    total: number;
    qualified: number;
    candidate: number;
    rejected: number;
  };
}

// --- Service Hook ---

export function useClusteringService() {
  const triggerRun = useCallback(
    async (days = 30, minTraces = 50, strategy = 'auto'): Promise<ClusteringRunResult> => {
      const params = new URLSearchParams({
        days: days.toString(),
        min_traces: minTraces.toString(),
        strategy,
      });
      const res = await fetch(`${BASE}/v1/clustering/run?${params}`, { method: 'POST' });
      if (!res.ok) throw new Error(`Clustering run failed: ${res.status}`);
      return res.json();
    },
    []
  );

  const listRuns = useCallback(async (): Promise<ClusteringRun[]> => {
    const res = await fetch(`${BASE}/v1/clustering/runs`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.runs || [];
  }, []);

  const getRun = useCallback(
    async (runId: string): Promise<{ run: ClusteringRun; datasets: ClusterDataset[] }> => {
      const res = await fetch(`${BASE}/v1/clustering/runs/${runId}`);
      if (!res.ok) throw new Error(`Run not found: ${res.status}`);
      return res.json();
    },
    []
  );

  const listDatasets = useCallback(
    async (status?: string): Promise<{ datasets: ClusterDataset[]; run_id: string | null }> => {
      const params = status ? `?status=${status}` : '';
      const res = await fetch(`${BASE}/v1/clustering/datasets${params}`);
      if (!res.ok) return { datasets: [], run_id: null };
      return res.json();
    },
    []
  );

  const getDatasetTraces = useCallback(
    async (
      runId: string,
      clusterId: number,
      limit = 100,
      offset = 0
    ): Promise<{ traces: Record<string, unknown>[]; total: number }> => {
      const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
      const res = await fetch(`${BASE}/v1/clustering/datasets/${runId}/${clusterId}?${params}`);
      if (!res.ok) throw new Error(`Failed to fetch traces: ${res.status}`);
      return res.json();
    },
    []
  );

  const exportDataset = useCallback(async (runId: string, clusterId: number): Promise<Blob> => {
    const res = await fetch(`${BASE}/v1/clustering/datasets/${runId}/${clusterId}/export`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Export failed: ${res.status}`);
    return res.blob();
  }, []);

  const qualifyDataset = useCallback(
    async (runId: string, clusterId: number, status: 'qualified' | 'rejected' | 'candidate') => {
      const res = await fetch(
        `${BASE}/v1/clustering/datasets/${runId}/${clusterId}/qualify?status=${status}`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error(`Qualify failed: ${res.status}`);
      return res.json();
    },
    []
  );

  const assignTracesToDataset = useCallback(
    async (
      runId: string,
      clusterId: number,
      requestIds: string[]
    ): Promise<{ assigned: number; message: string }> => {
      const res = await fetch(
        `${BASE}/v1/clustering/datasets/${runId}/${clusterId}/assign`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ request_ids: requestIds }),
        }
      );
      if (!res.ok) throw new Error(`Assign traces failed: ${res.status}`);
      return res.json();
    },
    []
  );

  const addTracesToDataset = useCallback(
    async (
      runId: string,
      clusterId: number,
      traces: Array<{ input?: string; output?: string; messages?: any[]; model?: string; source?: string }>
    ): Promise<{ ingested: number; message: string }> => {
      const res = await fetch(
        `${BASE}/v1/clustering/datasets/${runId}/${clusterId}/traces`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ traces }),
        }
      );
      if (!res.ok) throw new Error(`Add traces failed: ${res.status}`);
      return res.json();
    },
    []
  );

  return {
    triggerRun,
    listRuns,
    getRun,
    listDatasets,
    getDatasetTraces,
    exportDataset,
    qualifyDataset,
    addTracesToDataset,
    assignTracesToDataset,
  };
}
