import { useCallback, useMemo } from 'react';
import type {
  Dataset,
  DatasetSample,
  Evaluation,
  EvaluationMetric,
  EvaluationResults,
  EvaluationStatus,
  EvaluationStatusResponse,
  EvaluationProgress,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  CreateFromInstructionResponse,
  CreateEvaluationRequest,
  CreateCustomMetricRequest,
  UpdateCustomMetricRequest,
  ValidateScriptRequest,
  ValidateScriptResponse,
  Trace,
  AutoCollectConfig,
  CollectRun,
  Experiment,
  ExperimentComparison,
  AnnotationQueue,
  AnnotationItem,
  AnnotationAnalytics,
} from '../types/evaluationsTypes';
import { BUILTIN_METRICS } from '../types/evaluationsTypes';
import { API_BASE_URL } from '@/config/api';

const API_BASE = API_BASE_URL;
const DEFAULT_TENANT_ID = 'default';

// Helper function for API calls
async function apiCall<T>(
  endpoint: string,
  accessToken: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'X-Tenant-Id': DEFAULT_TENANT_ID,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`[EvaluationsService] API Error: ${response.status}`, errorText);
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  return response.json();
}

// Variant that accepts 202 Accepted responses (for async operations)
async function apiCallWithAccepted<T>(
  endpoint: string,
  accessToken: string,
  options: RequestInit = {}
): Promise<{ data: T; status: number }> {
  const url = `${API_BASE}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    'X-Tenant-Id': DEFAULT_TENANT_ID,
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    ...options.headers,
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  // Accept 200, 201, 202 as success
  if (!response.ok && response.status !== 202) {
    const errorText = await response.text();
    console.error(`[EvaluationsService] API Error: ${response.status}`, errorText);
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  return { data, status: response.status };
}

// Normalize backend annotation queue to frontend type
function normalizeQueue(q: Record<string, any>): AnnotationQueue {
  const total = q.total_items || 0;
  const completed = q.completed_items || 0;
  const skipped = q.skipped_items || 0;
  const pending = total - completed - skipped;

  // Derive status from counts
  let status: AnnotationQueue['status'] = 'active';
  if (total > 0 && pending === 0) status = 'completed';

  // Normalize rubric: backend stores as array, frontend expects {criteria, instructions?}
  let rubric = q.rubric;
  if (Array.isArray(rubric)) {
    rubric = {
      criteria: rubric.map((c: any) => ({
        id: c.name || c.id,
        name: c.name,
        description: c.description || '',
        scale: { min: c.scale_min ?? 1, max: c.scale_max ?? 5 },
      })),
    };
  }

  return {
    id: q.queue_id || q.id,
    name: q.name || '',
    description: q.description,
    dataset_id: q.dataset_id || '',
    dataset_name: q.dataset_name,
    rubric: rubric || { criteria: [] },
    status,
    total_items: total,
    completed_items: completed,
    skipped_items: skipped,
    annotators: q.annotators || [],
    created_at: q.created_at || new Date().toISOString(),
    updated_at: q.updated_at || new Date().toISOString(),
  };
}

// Normalize backend annotation item to frontend type
function normalizeItem(item: Record<string, any>): AnnotationItem {
  return {
    id: item.item_id || item.id,
    queue_id: item.queue_id || '',
    sample_id: item.sample_id || '',
    input: item.input || '',
    output: item.expected_output || item.output || '',
    expected_output: item.expected_output,
    model_id: item.model_id,
    ai_pre_scores: item.ai_pre_scores,
    human_scores: item.scores || item.human_scores,
    human_notes: item.notes || item.human_notes,
    status: item.status || 'pending',
    annotated_by: item.annotated_by,
    annotated_at: item.annotated_at,
  };
}

export function useEvaluationsService() {
  // ============================================================
  // DATASETS
  // ============================================================

  const listDatasets = useCallback(async (accessToken: string): Promise<Dataset[]> => {
    console.log('[EvaluationsService] Listing datasets');

    try {
      const data = await apiCall<{ datasets: Array<Record<string, unknown>>; total: number }>(
        '/v1/datasets',
        accessToken
      );

      // Normalize datasets - handle different field names from API
      const normalizedDatasets: Dataset[] = (data.datasets || []).map((d, index) => ({
        id: String(d.id || d.dataset_id || d.uuid || `dataset-${index}`),
        name: String(d.name || d.dataset_name || 'Unnamed Dataset'),
        description: d.description as string | undefined,
        source: (d.source as Dataset['source']) || 'manual',
        samples_count: Number(d.samples_count || d.sample_count || d.count || 0),
        created_at: String(d.created_at || new Date().toISOString()),
        updated_at: String(d.updated_at || d.created_at || new Date().toISOString()),
        schema: d.schema as Dataset['schema'],
      }));

      console.log('[EvaluationsService] Normalized datasets:', normalizedDatasets);
      return normalizedDatasets;
    } catch (error) {
      console.error('[EvaluationsService] Failed to list datasets:', error);
      throw error;
    }
  }, []);

  const getDataset = useCallback(
    async (
      accessToken: string,
      datasetId: string,
      options?: { include_samples?: boolean; samples_limit?: number; samples_offset?: number }
    ): Promise<{ dataset: Dataset; samples: DatasetSample[]; samples_total?: number }> => {
      console.log('[EvaluationsService] Getting dataset:', datasetId);

      try {
        const params = new URLSearchParams();
        if (options?.include_samples !== undefined) {
          params.append('include_samples', String(options.include_samples));
        }
        if (options?.samples_limit !== undefined) {
          params.append('samples_limit', String(options.samples_limit));
        }
        if (options?.samples_offset !== undefined) {
          params.append('samples_offset', String(options.samples_offset));
        }

        const queryString = params.toString();
        const endpoint = `/v1/datasets/${datasetId}${queryString ? `?${queryString}` : ''}`;

        const data = await apiCall<{
          dataset: Dataset;
          samples: DatasetSample[];
          samples_total?: number;
        }>(endpoint, accessToken);

        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to get dataset:', error);
        throw error;
      }
    },
    []
  );

  const createDataset = useCallback(
    async (accessToken: string, request: CreateDatasetRequest): Promise<Dataset> => {
      console.log('[EvaluationsService] Creating dataset:', request.name);
      console.log('[EvaluationsService] Request includes samples:', request.samples?.length || 0);
      if (request.samples && request.samples.length > 0) {
        console.log('[EvaluationsService] First sample:', request.samples[0]);
      }

      try {
        console.log('[EvaluationsService] Making API call to create dataset...');
        const rawData = await apiCall<any>('/v1/datasets', accessToken, {
          method: 'POST',
          body: JSON.stringify(request),
        }).catch((err) => {
          console.error('[EvaluationsService] API call failed with error:', err);
          console.error('[EvaluationsService] Error details:', {
            message: err.message,
            stack: err.stack,
            response: err.response,
          });
          throw err;
        });

        console.log('[EvaluationsService] API response for dataset creation:', rawData);

        // Backend returns dataset directly (not wrapped in { dataset: {...} })
        // Normalize the response to match our Dataset type
        const datasetData = rawData.dataset || rawData;

        // Normalize field names: dataset_id -> id
        const created: Dataset = {
          id: datasetData.dataset_id || datasetData.id,
          name: datasetData.name,
          description: datasetData.description,
          source: datasetData.source || 'manual',
          samples_count: datasetData.samples_count || 0,
          created_at: datasetData.created_at,
          updated_at: datasetData.updated_at,
          schema: datasetData.schema,
        };

        console.log('[EvaluationsService] Normalized dataset with ID:', created.id);
        console.log('[EvaluationsService] Initial samples_count:', created.samples_count);

        // If manual samples were provided, add them to the new dataset
        if (request.samples && request.samples.length > 0) {
          try {
            console.log(
              '[EvaluationsService] Adding',
              request.samples.length,
              'samples to dataset',
              created.id
            );
            const samplesPayload = { samples: request.samples };
            console.log(
              '[EvaluationsService] Samples payload:',
              JSON.stringify(samplesPayload, null, 2)
            );

            const result = await apiCall<{ message: string; count: number }>(
              `/v1/datasets/${created.id}/samples`,
              accessToken,
              {
                method: 'POST',
                body: JSON.stringify(samplesPayload),
              }
            );
            console.log('[EvaluationsService] Samples added successfully:', result);

            // Best effort: bump samples_count locally so UI reflects it without refetch
            created.samples_count =
              typeof created.samples_count === 'number'
                ? created.samples_count + request.samples.length
                : request.samples.length;
            console.log('[EvaluationsService] Updated samples_count to:', created.samples_count);
          } catch (err) {
            console.error(
              '[EvaluationsService] Failed to add samples after creating dataset:',
              err
            );
            throw err;
          }
        }

        return created;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create dataset:', error);
        throw error;
      }
    },
    []
  );

  const deleteDataset = useCallback(
    async (accessToken: string, datasetId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting dataset:', datasetId);

      try {
        await apiCall<{ success: boolean }>(`/v1/datasets/${datasetId}`, accessToken, {
          method: 'DELETE',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete dataset:', error);
        throw error;
      }
    },
    []
  );

  const importDataset = useCallback(
    async (
      accessToken: string,
      file: File,
      name: string,
      options?: { description?: string; input_column?: string; output_column?: string }
    ): Promise<Dataset> => {
      console.log('[EvaluationsService] Importing dataset from file:', file.name);

      try {
        // Read file content
        const fileContent = await file.text();
        console.log('[EvaluationsService] File content length:', fileContent.length);
        console.log('[EvaluationsService] First 200 chars:', fileContent.substring(0, 200));

        // Parse based on file type
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        let samples: Array<{ input: string; expected_output?: string; output?: string }> = [];

        if (fileExt === 'json') {
          console.log('[EvaluationsService] Parsing JSON file');
          const parsed = JSON.parse(fileContent);

          if (!Array.isArray(parsed)) {
            throw new Error('JSON file must contain an array of samples');
          }

          // Map to our expected format
          samples = parsed.map((item: any) => ({
            input: item.input || '',
            expected_output: item.expected_output || item.output,
            output: item.output || item.expected_output,
          }));
        } else if (fileExt === 'csv') {
          console.log('[EvaluationsService] Parsing CSV file');
          // Simple CSV parser - handles quoted fields
          const lines = fileContent.split('\n').filter((l) => l.trim());
          if (lines.length < 2) {
            throw new Error('CSV file must have a header row and at least one data row');
          }

          const header = lines[0].split(',').map((h) => h.trim().replace(/"/g, ''));
          const inputCol = options?.input_column || 'input';
          const outputCol = options?.output_column || 'expected_output';

          const inputIdx = header.findIndex((h) => h.toLowerCase() === inputCol.toLowerCase());
          const outputIdx = header.findIndex(
            (h) => h.toLowerCase() === outputCol.toLowerCase() || h.toLowerCase() === 'output'
          );

          if (inputIdx === -1) {
            throw new Error(
              `Column "${inputCol}" not found in CSV. Available columns: ${header.join(', ')}`
            );
          }

          for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map((v) => v.trim().replace(/^"|"$/g, ''));
            if (values[inputIdx]) {
              samples.push({
                input: values[inputIdx],
                expected_output: outputIdx !== -1 ? values[outputIdx] : undefined,
                output: outputIdx !== -1 ? values[outputIdx] : undefined,
              });
            }
          }
        } else {
          throw new Error('Unsupported file format. Please use JSON or CSV.');
        }

        console.log('[EvaluationsService] Parsed', samples.length, 'samples');
        console.log('[EvaluationsService] First sample:', samples[0]);

        // Create dataset with parsed samples
        const request: CreateDatasetRequest = {
          name,
          description: options?.description,
          source: 'imported',
          samples: samples.map((s) => ({
            input: s.input,
            expected_output: s.expected_output,
            output: s.output,
            raw: JSON.stringify(s),
          })) as Omit<DatasetSample, 'id' | 'created_at'>[],
        };

        console.log(
          '[EvaluationsService] Creating imported dataset with',
          request.samples?.length,
          'samples'
        );

        // Use the regular createDataset function
        return await createDataset(accessToken, request);
      } catch (error) {
        console.error('[EvaluationsService] Failed to import dataset:', error);
        throw error;
      }
    },
    [createDataset]
  );

  const analyzeTraces = useCallback(
    async (
      accessToken: string,
      data: any[]
    ): Promise<{
      mapping: any;
      preview: any[];
      source_format: string;
      total_records: number;
    }> => {
      console.log('[EvaluationsService] Analyzing traces schema:', data.length, 'records');

      try {
        const result = await apiCall<{
          mapping: any;
          preview: any[];
          source_format: string;
          total_records: number;
        }>('/v1/datasets/analyze-traces', accessToken, {
          method: 'POST',
          body: JSON.stringify({ data }),
        });
        console.log('[EvaluationsService] Schema detected:', result.source_format);
        return result;
      } catch (error) {
        console.error('[EvaluationsService] Failed to analyze traces:', error);
        throw error;
      }
    },
    []
  );

  const importTraces = useCallback(
    async (
      accessToken: string,
      name: string,
      data: any[],
      mapping?: any,
      description?: string
    ): Promise<{
      dataset_id: string;
      name: string;
      source: string;
      samples_count: number;
      skipped_count?: number;
    }> => {
      console.log('[EvaluationsService] Importing traces:', data.length, 'records');

      try {
        const body: any = { name, data };
        if (mapping) body.mapping = mapping;
        if (description) body.description = description;

        const result = await apiCall<{
          dataset_id: string;
          name: string;
          source: string;
          samples_count: number;
          skipped_count?: number;
        }>('/v1/datasets/import-traces', accessToken, {
          method: 'POST',
          body: JSON.stringify(body),
        });
        console.log('[EvaluationsService] Imported:', result.samples_count, 'samples');
        return result;
      } catch (error) {
        console.error('[EvaluationsService] Failed to import traces:', error);
        throw error;
      }
    },
    []
  );

  const addSamplesToDataset = useCallback(
    async (
      accessToken: string,
      datasetId: string,
      samples: Omit<DatasetSample, 'id' | 'created_at'>[]
    ): Promise<{ added: number }> => {
      console.log('[EvaluationsService] Adding samples to dataset:', datasetId);

      try {
        const data = await apiCall<{ added: number }>(
          `/v1/datasets/${datasetId}/samples`,
          accessToken,
          {
            method: 'POST',
            body: JSON.stringify({ samples }),
          }
        );
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to add samples:', error);
        throw error;
      }
    },
    []
  );

  const deleteSample = useCallback(
    async (accessToken: string, datasetId: string, sampleId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting sample:', sampleId);

      try {
        await apiCall<{ success: boolean }>(
          `/v1/datasets/${datasetId}/samples/${sampleId}`,
          accessToken,
          { method: 'DELETE' }
        );
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete sample:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // TRACES (Auto-collection)
  // ============================================================

  const listTraces = useCallback(
    async (
      accessToken: string,
      options?: {
        limit?: number;
        offset?: number;
        start_date?: string;
        end_date?: string;
        source?: string;
        model_id?: string;
      }
    ): Promise<{ traces: Trace[]; total: number; has_more?: boolean }> => {
      console.log('[EvaluationsService] Listing traces');

      try {
        const params = new URLSearchParams();
        if (options?.limit !== undefined) {
          params.append('limit', String(options.limit));
        }
        if (options?.offset !== undefined) {
          params.append('offset', String(options.offset));
        }
        if (options?.start_date) {
          params.append('start_date', options.start_date);
        }
        if (options?.end_date) {
          params.append('end_date', options.end_date);
        }
        if (options?.source) {
          params.append('source', options.source);
        }
        if (options?.model_id) {
          params.append('model_id', options.model_id);
        }

        const queryString = params.toString();
        const endpoint = `/v1/traces${queryString ? `?${queryString}` : ''}`;

        const data = await apiCall<{
          traces: Array<Record<string, unknown>>;
          total: number;
          has_more?: boolean;
        }>(endpoint, accessToken);

        // Normalize traces - handle different field names from API
        const normalizedTraces: Trace[] = (data.traces || []).map((t, index) => {
          // CRITICAL FIX: Handle trace_id properly
          // Backend returns IDs in format "timestamp#uuid", but API might send different field names
          // Try multiple possible field names for the trace ID
          const traceId = t.trace_id || t.traceId || t.id || t.uuid || t.pk || t.sk;

          // Handle latency - could be in ms, seconds, or already formatted
          let latencyMs = 0;
          const rawLatency =
            t.latency_ms || t.latency || t.duration_ms || t.duration || t.response_time || 0;
          if (typeof rawLatency === 'number') {
            // If value is less than 100, it's probably in seconds, convert to ms
            latencyMs = rawLatency < 100 ? Math.round(rawLatency * 1000) : Math.round(rawLatency);
          } else if (typeof rawLatency === 'string') {
            const parsed = parseFloat(rawLatency);
            latencyMs = parsed < 100 ? Math.round(parsed * 1000) : Math.round(parsed);
          }

          // Handle cost - normalize to USD
          let costUsd = 0;
          const rawCost = t.cost_usd || t.cost || t.total_cost || t.price || 0;
          if (typeof rawCost === 'number') {
            costUsd = rawCost;
          } else if (typeof rawCost === 'string') {
            costUsd = parseFloat(rawCost) || 0;
          }

          return {
            id: String(traceId || `trace-${index}-${Date.now()}`),
            input: String(t.input || t.prompt || t.request || t.message || ''),
            output: String(t.output || t.response || t.completion || t.result || ''),
            model_id: String(t.model_id || t.model || t.model_name || 'unknown'),
            latency_ms: latencyMs,
            cost_usd: costUsd,
            source: (t.source as Trace['source']) || 'api',
            created_at: String(
              t.created_at || t.timestamp || t.created || new Date().toISOString()
            ),
            metadata: t.metadata as Record<string, unknown> | undefined,
          };
        });

        console.log('[EvaluationsService] Normalized traces:', normalizedTraces.length, 'traces');
        if (normalizedTraces.length > 0) {
          console.log(
            '[EvaluationsService] Sample normalized trace IDs:',
            normalizedTraces.slice(0, 3).map((t) => ({ id: t.id, created_at: t.created_at }))
          );
        }
        return {
          traces: normalizedTraces,
          total: data.total || normalizedTraces.length,
          has_more: data.has_more,
        };
      } catch (error) {
        console.error('[EvaluationsService] Failed to list traces:', error);
        throw error;
      }
    },
    []
  );

  const createDatasetFromTraces = useCallback(
    async (
      accessToken: string,
      name: string,
      traceIds: string[],
      description?: string
    ): Promise<Dataset> => {
      console.log('[EvaluationsService] Creating dataset from traces:', traceIds);
      console.log('[EvaluationsService] First 3 trace IDs:', traceIds.slice(0, 3));

      try {
        const data = await apiCall<{ dataset: Dataset }>('/v1/datasets/from-traces', accessToken, {
          method: 'POST',
          body: JSON.stringify({
            name,
            description,
            trace_ids: traceIds,
          }),
        });
        return data.dataset;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create dataset from traces:', error);
        throw error;
      }
    },
    []
  );

  const createDatasetFromInstruction = useCallback(
    async (
      accessToken: string,
      request: CreateFromInstructionRequest
    ): Promise<CreateFromInstructionResponse> => {
      console.log('[EvaluationsService] Creating dataset from instruction:', request.instruction);

      try {
        const data = await apiCall<CreateFromInstructionResponse>(
          '/v1/datasets/from-instruction',
          accessToken,
          {
            method: 'POST',
            body: JSON.stringify(request),
          }
        );
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create dataset from instruction:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // EVALUATIONS
  // ============================================================

  const listEvaluations = useCallback(
    async (
      accessToken: string,
      options?: { status?: string; limit?: number; offset?: number }
    ): Promise<Evaluation[]> => {
      console.log('[EvaluationsService] Listing evaluations');

      try {
        const params = new URLSearchParams();
        if (options?.status) {
          params.append('status', options.status);
        }
        if (options?.limit !== undefined) {
          params.append('limit', String(options.limit));
        }
        if (options?.offset !== undefined) {
          params.append('offset', String(options.offset));
        }

        const queryString = params.toString();
        const endpoint = `/v1/evaluations${queryString ? `?${queryString}` : ''}`;

        const data = await apiCall<{ evaluations: Array<Record<string, any>>; total: number }>(
          endpoint,
          accessToken
        );

        // Normalize evaluation IDs - backend might return evaluation_id instead of id
        const normalizedEvaluations: Evaluation[] = (data.evaluations || []).map(
          (e) =>
            ({
              ...e,
              id: e.id || e.evaluation_id || e.pk || e.sk,
            }) as Evaluation
        );

        console.log('[EvaluationsService] Normalized evaluations:', normalizedEvaluations.length);
        if (normalizedEvaluations.length > 0) {
          console.log(
            '[EvaluationsService] Sample evaluation IDs:',
            normalizedEvaluations.slice(0, 3).map((e) => e.id)
          );
        }

        return normalizedEvaluations;
      } catch (error) {
        console.error('[EvaluationsService] Failed to list evaluations:', error);
        throw error;
      }
    },
    []
  );

  const getEvaluation = useCallback(
    async (accessToken: string, evaluationId: string): Promise<Evaluation> => {
      console.log('[EvaluationsService] Getting evaluation:', evaluationId);

      try {
        const data = await apiCall<{ evaluation: Record<string, any> }>(
          `/v1/evaluations/${evaluationId}`,
          accessToken
        );
        // Normalize ID
        const normalized: Evaluation = {
          ...data.evaluation,
          id:
            data.evaluation.id ||
            data.evaluation.evaluation_id ||
            data.evaluation.pk ||
            data.evaluation.sk,
        } as Evaluation;
        return normalized;
      } catch (error) {
        console.error('[EvaluationsService] Failed to get evaluation:', error);
        throw error;
      }
    },
    []
  );

  const createEvaluation = useCallback(
    async (accessToken: string, request: CreateEvaluationRequest): Promise<Evaluation> => {
      console.log('[EvaluationsService] Creating evaluation:', request.name);

      try {
        const { data, status } = await apiCallWithAccepted<Record<string, unknown>>(
          '/v1/evaluations',
          accessToken,
          {
            method: 'POST',
            body: JSON.stringify(request),
          }
        );

        console.log('[EvaluationsService] Create evaluation response:', data, 'status:', status);

        // Handle 202 Accepted (async processing)
        if (status === 202) {
          console.log('[EvaluationsService] Evaluation queued for async processing');
          console.log('[EvaluationsService] 202 Response data:', JSON.stringify(data, null, 2));

          // Try multiple field names for evaluation ID
          const evaluationId = (data.evaluation_id ||
            data.id ||
            data.evaluationId ||
            (data.evaluation as Record<string, unknown>)?.id ||
            (data.evaluation as Record<string, unknown>)?.evaluation_id) as string;

          if (!evaluationId) {
            console.error('[EvaluationsService] No evaluation_id in 202 response:', data);
            throw new Error('API returned 202 but no evaluation_id was provided');
          }

          // Return a minimal Evaluation object with queued status
          const queuedEvaluation: Evaluation = {
            id: evaluationId,
            name: request.name,
            description: request.description,
            dataset_id: request.dataset_id,
            models: request.models,
            metrics: request.metrics,
            status: 'queued' as EvaluationStatus,
            progress: { total_samples: 0, completed_samples: 0, failed_samples: 0 },
            created_at: new Date().toISOString(),
          };

          console.log('[EvaluationsService] Created queued evaluation:', queuedEvaluation);
          return queuedEvaluation;
        }

        // Handle synchronous response (200/201)
        // Normalize response - API may return { evaluation: {...} } or direct object
        let evalData: Record<string, unknown>;
        if (data.evaluation && typeof data.evaluation === 'object') {
          evalData = data.evaluation as Record<string, unknown>;
        } else if (data.evaluation_id || data.id) {
          // API returned the evaluation directly
          evalData = data;
        } else {
          console.error('[EvaluationsService] Unexpected response format:', data);
          throw new Error('Unexpected API response format');
        }

        // Normalize ID and progress
        const normalized: Evaluation = {
          ...evalData,
          id: (evalData.id || evalData.evaluation_id || evalData.pk || evalData.sk) as string,
          progress: evalData.progress as EvaluationProgress | undefined,
        } as Evaluation;

        console.log('[EvaluationsService] Normalized evaluation:', normalized);
        return normalized;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create evaluation:', error);
        throw error;
      }
    },
    []
  );

  const getEvaluationStatus = useCallback(
    async (accessToken: string, evaluationId: string): Promise<EvaluationStatusResponse> => {
      console.log('[EvaluationsService] Getting evaluation status:', evaluationId);

      try {
        const data = await apiCall<Record<string, unknown>>(
          `/v1/evaluations/${evaluationId}/status`,
          accessToken
        );

        // Normalize the response to EvaluationStatusResponse
        const statusResponse: EvaluationStatusResponse = {
          evaluation_id: (data.evaluation_id || evaluationId) as string,
          status: (data.status || 'queued') as EvaluationStatus,
          progress: {
            total_samples:
              (data.progress as Record<string, number>)?.total_samples ??
              (data.total_samples as number) ??
              0,
            completed_samples:
              (data.progress as Record<string, number>)?.completed_samples ??
              (data.completed_samples as number) ??
              (data.current_sample as number) ??
              0,
            failed_samples:
              (data.progress as Record<string, number>)?.failed_samples ??
              (data.failed_samples as number) ??
              0,
          },
          started_at: data.started_at as string | undefined,
          updated_at: data.updated_at as string | undefined,
          error: data.error as EvaluationStatusResponse['error'],
        };

        return statusResponse;
      } catch (error) {
        console.error('[EvaluationsService] Failed to get evaluation status:', error);
        throw error;
      }
    },
    []
  );

  const getEvaluationResults = useCallback(
    async (
      accessToken: string,
      evaluationId: string,
      options?: {
        include_samples?: boolean;
        samples_limit?: number;
        samples_offset?: number;
        sort_by?: string;
        filter_model?: string;
      }
    ): Promise<{ results: EvaluationResults; samples_total?: number }> => {
      console.log('[EvaluationsService] Getting evaluation results:', evaluationId);

      try {
        const params = new URLSearchParams();
        if (options?.include_samples !== undefined) {
          params.append('include_samples', String(options.include_samples));
        }
        if (options?.samples_limit !== undefined) {
          params.append('samples_limit', String(options.samples_limit));
        }
        if (options?.samples_offset !== undefined) {
          params.append('samples_offset', String(options.samples_offset));
        }
        if (options?.sort_by) {
          params.append('sort_by', options.sort_by);
        }
        if (options?.filter_model) {
          params.append('filter_model', options.filter_model);
        }

        const queryString = params.toString();
        const endpoint = `/v1/evaluations/${evaluationId}/results${queryString ? `?${queryString}` : ''}`;

        const rawData = await apiCall<Record<string, unknown>>(endpoint, accessToken);

        console.log('[EvaluationsService] Raw API response:', rawData);

        // Normalize the response - API may return results directly or wrapped in { results: ... }
        let results: EvaluationResults;

        if (rawData.results && typeof rawData.results === 'object') {
          // Response is wrapped: { results: { summary: ..., samples: ... } }
          results = rawData.results as EvaluationResults;
        } else if (rawData.summary && typeof rawData.summary === 'object') {
          // Response is direct: { summary: ..., samples: ... }
          results = {
            evaluation_id: evaluationId,
            summary: rawData.summary as EvaluationResults['summary'],
            samples: (rawData.samples as EvaluationResults['samples']) || [],
            winner: rawData.winner as EvaluationResults['winner'],
          };
        } else {
          // Fallback: treat the entire response as results
          results = {
            evaluation_id: evaluationId,
            summary: {
              models: (rawData.models as EvaluationResults['summary']['models']) || {},
              metrics: (rawData.metrics as EvaluationResults['summary']['metrics']) || {},
            },
            samples: [],
          };
        }

        // Ensure samples is always an array
        if (!Array.isArray(results.samples)) {
          results.samples = [];
        }

        // Ensure summary has required structure
        if (!results.summary) {
          results.summary = { models: {}, metrics: {} };
        }

        const summaryObj = results.summary as Record<string, unknown>;
        if (!summaryObj.models && !summaryObj.metrics) {
          const modelsMap: Record<
            string,
            {
              total_latency: number;
              avg_latency: number;
              total_cost: number;
              avg_cost: number;
              avg_scores: Record<string, number>;
            }
          > = {};
          const metricsMap: Record<string, { avg_by_model: Record<string, number> }> = {};

          for (const [metricName, metricData] of Object.entries(summaryObj)) {
            if (typeof metricData !== 'object' || metricData === null) continue;
            const modelEntries = metricData as Record<
              string,
              { average?: number; count?: number; min?: number; max?: number }
            >;
            metricsMap[metricName] = { avg_by_model: {} };
            for (const [modelId, stats] of Object.entries(modelEntries)) {
              if (typeof stats !== 'object' || stats === null) continue;
              const avg = stats.average ?? 0;
              metricsMap[metricName].avg_by_model[modelId] = avg;
              if (!modelsMap[modelId]) {
                modelsMap[modelId] = {
                  total_latency: 0,
                  avg_latency: 0,
                  total_cost: 0,
                  avg_cost: 0,
                  avg_scores: {},
                };
              }
              modelsMap[modelId].avg_scores[metricName] = avg;
            }
          }

          // Fill latency/cost from samples if available
          if (Array.isArray(results.samples)) {
            const modelStats: Record<string, { latencies: number[]; costs: number[] }> = {};
            for (const sample of results.samples) {
              if (!sample.outputs) continue;
              for (const [modelId, output] of Object.entries(sample.outputs)) {
                if (!modelStats[modelId]) modelStats[modelId] = { latencies: [], costs: [] };
                if (output.latency) modelStats[modelId].latencies.push(output.latency);
                if (output.cost) modelStats[modelId].costs.push(output.cost);
              }
            }
            for (const [modelId, stats] of Object.entries(modelStats)) {
              if (!modelsMap[modelId]) {
                modelsMap[modelId] = {
                  total_latency: 0,
                  avg_latency: 0,
                  total_cost: 0,
                  avg_cost: 0,
                  avg_scores: {},
                };
              }
              const totalLat = stats.latencies.reduce((a, b) => a + b, 0);
              const totalCost = stats.costs.reduce((a, b) => a + b, 0);
              modelsMap[modelId].total_latency = totalLat;
              modelsMap[modelId].avg_latency = stats.latencies.length
                ? totalLat / stats.latencies.length
                : 0;
              modelsMap[modelId].total_cost = totalCost;
              modelsMap[modelId].avg_cost = stats.costs.length ? totalCost / stats.costs.length : 0;
            }
          }

          results.summary = { models: modelsMap, metrics: metricsMap };
        }

        if (!results.summary.models) {
          results.summary.models = {};
        }
        if (!results.summary.metrics) {
          results.summary.metrics = {};
        }

        console.log('[EvaluationsService] Normalized results:', results);

        return {
          results,
          samples_total: rawData.samples_total as number | undefined,
        };
      } catch (error) {
        console.error('[EvaluationsService] Failed to get evaluation results:', error);
        throw error;
      }
    },
    []
  );

  const cancelEvaluation = useCallback(
    async (accessToken: string, evaluationId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Canceling evaluation:', evaluationId);

      try {
        await apiCall<{ success: boolean }>(`/v1/evaluations/${evaluationId}/cancel`, accessToken, {
          method: 'POST',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to cancel evaluation:', error);
        throw error;
      }
    },
    []
  );

  const rerunEvaluation = useCallback(
    async (
      accessToken: string,
      evaluationId: string,
      options?: { failed_only?: boolean; models?: string[] }
    ): Promise<Evaluation> => {
      console.log('[EvaluationsService] Rerunning evaluation:', evaluationId);

      try {
        const data = await apiCall<{ evaluation: Evaluation }>(
          `/v1/evaluations/${evaluationId}/rerun`,
          accessToken,
          {
            method: 'POST',
            body: JSON.stringify(options || {}),
          }
        );
        return data.evaluation;
      } catch (error) {
        console.error('[EvaluationsService] Failed to rerun evaluation:', error);
        throw error;
      }
    },
    []
  );

  const deleteEvaluation = useCallback(
    async (accessToken: string, evaluationId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting evaluation:', evaluationId);

      try {
        await apiCall<{ success: boolean }>(`/v1/evaluations/${evaluationId}`, accessToken, {
          method: 'DELETE',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete evaluation:', error);
        throw error;
      }
    },
    []
  );

  const exportEvaluationResults = useCallback(
    async (
      accessToken: string,
      evaluationId: string,
      format: 'csv' | 'json' | 'xlsx'
    ): Promise<Blob> => {
      console.log('[EvaluationsService] Exporting evaluation results:', evaluationId);

      try {
        const url = `${API_BASE}/v1/evaluations/${evaluationId}/export?format=${format}`;
        const response = await fetch(url, {
          headers: {
            ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
          },
        });

        if (!response.ok) {
          throw new Error(`Export failed: ${response.status}`);
        }

        return response.blob();
      } catch (error) {
        console.error('[EvaluationsService] Failed to export results:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // METRICS
  // ============================================================

  const listMetrics = useCallback(
    async (
      accessToken: string
    ): Promise<{ builtin: EvaluationMetric[]; custom: EvaluationMetric[] }> => {
      console.log('[EvaluationsService] Listing metrics');

      try {
        const data = await apiCall<{ metrics: EvaluationMetric[]; count: number }>(
          '/v1/metrics',
          accessToken
        );

        const allMetrics = data.metrics || [];
        const builtin = allMetrics.filter((m) => m.is_builtin);
        const custom = allMetrics.filter((m) => !m.is_builtin);

        console.log(
          '[EvaluationsService] Loaded',
          builtin.length,
          'builtin and',
          custom.length,
          'custom metrics'
        );

        return {
          builtin: builtin.length > 0 ? builtin : BUILTIN_METRICS,
          custom,
        };
      } catch (error) {
        console.error('[EvaluationsService] Failed to list metrics, using defaults:', error);
        // Return builtin metrics if API fails
        return {
          builtin: BUILTIN_METRICS,
          custom: [],
        };
      }
    },
    []
  );

  const getMetric = useCallback(
    async (accessToken: string, metricId: string): Promise<EvaluationMetric> => {
      console.log('[EvaluationsService] Getting metric:', metricId);

      try {
        const data = await apiCall<EvaluationMetric>(`/v1/metrics/${metricId}`, accessToken);
        // API returns the metric directly
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to get metric:', error);
        throw error;
      }
    },
    []
  );

  const createCustomMetric = useCallback(
    async (accessToken: string, request: CreateCustomMetricRequest): Promise<EvaluationMetric> => {
      console.log('[EvaluationsService] Creating custom metric:', request.name);

      try {
        const data = await apiCall<EvaluationMetric>('/v1/metrics', accessToken, {
          method: 'POST',
          body: JSON.stringify(request),
        });
        // API returns the metric directly
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create custom metric:', error);
        throw error;
      }
    },
    []
  );

  const updateCustomMetric = useCallback(
    async (
      accessToken: string,
      metricId: string,
      request: UpdateCustomMetricRequest
    ): Promise<EvaluationMetric> => {
      console.log('[EvaluationsService] Updating custom metric:', metricId);

      try {
        const data = await apiCall<EvaluationMetric>(`/v1/metrics/${metricId}`, accessToken, {
          method: 'PUT',
          body: JSON.stringify(request),
        });
        // API returns the metric directly according to the docs
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to update custom metric:', error);
        throw error;
      }
    },
    []
  );

  const deleteCustomMetric = useCallback(
    async (accessToken: string, metricId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting custom metric:', metricId);

      try {
        await apiCall<{ success: boolean }>(`/v1/metrics/${metricId}`, accessToken, {
          method: 'DELETE',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete custom metric:', error);
        throw error;
      }
    },
    []
  );

  const validatePythonScript = useCallback(
    async (
      accessToken: string,
      request: ValidateScriptRequest
    ): Promise<ValidateScriptResponse> => {
      console.log('[EvaluationsService] Validating Python script');

      try {
        const data = await apiCall<ValidateScriptResponse>(
          '/v1/metrics/validate-script',
          accessToken,
          {
            method: 'POST',
            body: JSON.stringify(request),
          }
        );
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to validate script:', error);
        // If API returns 400 with validation error, parse it
        if (error instanceof Error && error.message.includes('400')) {
          try {
            const errorMatch = error.message.match(/\{.*\}/);
            if (errorMatch) {
              const errorData = JSON.parse(errorMatch[0]);
              return {
                valid: false,
                error: errorData.error || 'Validation failed',
                details: errorData.details,
              };
            }
          } catch (parseError) {
            // Ignore parse errors
          }
        }
        throw error;
      }
    },
    []
  );

  // ============================================================
  // MODELS (for evaluation selection)
  // ============================================================

  const listAvailableModels = useCallback(
    async (
      accessToken: string
    ): Promise<
      Array<{
        id: string;
        name: string;
        provider: string;
        type: 'external' | 'deployment' | 'registered' | 'deployed';
        available: boolean;
        status?: string;
        providers?: string[];
      }>
    > => {
      console.log('[EvaluationsService] Listing available models');

      try {
        const data = await apiCall<{
          models: Array<{
            id: string;
            name: string;
            provider: string;
            type: 'external' | 'deployment' | 'registered' | 'deployed';
            available: boolean;
            status?: string;
            providers?: string[];
          }>;
        }>('/v1/models/available', accessToken);
        return data.models;
      } catch (error) {
        console.error('[EvaluationsService] Failed to list available models:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // AUTO-COLLECT
  // ============================================================

  const getAutoCollectConfig = useCallback(
    async (accessToken: string, datasetId: string): Promise<AutoCollectConfig | null> => {
      console.log('[EvaluationsService] Getting auto-collect config for dataset:', datasetId);

      try {
        const data = await apiCall<AutoCollectConfig>(
          `/v1/datasets/${datasetId}/auto-collect`,
          accessToken
        );
        return data;
      } catch (error) {
        // 404 means no config exists yet — not an error
        if (error instanceof Error && error.message.includes('404')) {
          return null;
        }
        console.error('[EvaluationsService] Failed to get auto-collect config:', error);
        throw error;
      }
    },
    []
  );

  const putAutoCollectConfig = useCallback(
    async (
      accessToken: string,
      datasetId: string,
      config: Omit<
        AutoCollectConfig,
        'dataset_id' | 'created_at' | 'updated_at' | 'total_collected' | 'last_collected_at'
      >
    ): Promise<AutoCollectConfig> => {
      console.log('[EvaluationsService] Saving auto-collect config for dataset:', datasetId);

      try {
        const data = await apiCall<AutoCollectConfig>(
          `/v1/datasets/${datasetId}/auto-collect`,
          accessToken,
          {
            method: 'PUT',
            body: JSON.stringify(config),
          }
        );
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to save auto-collect config:', error);
        throw error;
      }
    },
    []
  );

  const deleteAutoCollectConfig = useCallback(
    async (accessToken: string, datasetId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting auto-collect config for dataset:', datasetId);

      try {
        await apiCall<{ success: boolean }>(`/v1/datasets/${datasetId}/auto-collect`, accessToken, {
          method: 'DELETE',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete auto-collect config:', error);
        throw error;
      }
    },
    []
  );

  const getAutoCollectHistory = useCallback(
    async (accessToken: string, datasetId: string, limit?: number): Promise<CollectRun[]> => {
      console.log('[EvaluationsService] Getting auto-collect history for dataset:', datasetId);

      try {
        const params = new URLSearchParams();
        if (limit !== undefined) {
          params.append('limit', String(limit));
        }
        const queryString = params.toString();
        const endpoint = `/v1/datasets/${datasetId}/auto-collect/history${queryString ? `?${queryString}` : ''}`;

        const data = await apiCall<{ runs: CollectRun[] }>(endpoint, accessToken);
        return data.runs || [];
      } catch (error) {
        console.error('[EvaluationsService] Failed to get auto-collect history:', error);
        throw error;
      }
    },
    []
  );

  const triggerAutoCollect = useCallback(
    async (accessToken: string, datasetId: string): Promise<{ run_id: string }> => {
      console.log('[EvaluationsService] Triggering auto-collect for dataset:', datasetId);

      try {
        const { data } = await apiCallWithAccepted<{ run_id: string }>(
          `/v1/datasets/${datasetId}/auto-collect/run`,
          accessToken,
          { method: 'POST' }
        );
        return data;
      } catch (error) {
        console.error('[EvaluationsService] Failed to trigger auto-collect:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // TRACE ISSUES
  // ============================================================

  const listTraceIssues = useCallback(
    async (
      accessToken: string,
      options?: { severity?: string; type?: string; resolved?: boolean }
    ): Promise<{ issues: Array<Record<string, any>>; count: number }> => {
      const params = new URLSearchParams();
      if (options?.severity) {
        params.append('severity', options.severity);
      }
      if (options?.type) {
        params.append('type', options.type);
      }
      if (options?.resolved !== undefined) {
        params.append('resolved', String(options.resolved));
      }
      const queryString = params.toString();
      const endpoint = `/v1/trace-issues${queryString ? `?${queryString}` : ''}`;

      const data = await apiCall<{ issues: Array<Record<string, any>>; count: number }>(
        endpoint,
        accessToken
      );
      return data;
    },
    []
  );

  const triggerTraceScan = useCallback(
    async (accessToken: string): Promise<{ scan_id: string; status: string }> => {
      const { data } = await apiCallWithAccepted<{ scan_id: string; status: string }>(
        '/v1/trace-issues/scan',
        accessToken,
        { method: 'POST' }
      );
      return data;
    },
    []
  );

  const getTraceScanStatus = useCallback(
    async (accessToken: string, scanId: string): Promise<Record<string, any>> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/trace-issues/scan/${scanId}`,
        accessToken
      );
      return data;
    },
    []
  );

  const resolveTraceIssue = useCallback(
    async (accessToken: string, issueId: string): Promise<Record<string, any>> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/trace-issues/${issueId}/resolve`,
        accessToken,
        { method: 'PUT' }
      );
      return data;
    },
    []
  );

  const dismissTraceIssue = useCallback(
    async (accessToken: string, issueId: string, reason?: string): Promise<Record<string, any>> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/trace-issues/${issueId}/dismiss`,
        accessToken,
        {
          method: 'PUT',
          ...(reason ? { body: JSON.stringify({ reason }) } : {}),
        }
      );
      return data;
    },
    []
  );

  const getScheduleConfig = useCallback(
    async (accessToken: string): Promise<Record<string, any>> => {
      const data = await apiCall<Record<string, any>>(
        '/v1/trace-issues/schedule',
        accessToken
      );
      return data;
    },
    []
  );

  const updateScheduleConfig = useCallback(
    async (
      accessToken: string,
      config: { enabled?: boolean; interval_seconds?: number; days_lookback?: number; trace_limit?: number }
    ): Promise<Record<string, any>> => {
      const data = await apiCall<Record<string, any>>(
        '/v1/trace-issues/schedule',
        accessToken,
        { method: 'PUT', body: JSON.stringify(config) }
      );
      return data;
    },
    []
  );

  // ============================================================
  // EXPERIMENTS
  // ============================================================

  const listExperiments = useCallback(
    async (accessToken: string, options?: { status?: string }): Promise<Experiment[]> => {
      console.log('[EvaluationsService] Listing experiments');

      try {
        const params = new URLSearchParams();
        if (options?.status) {
          params.append('status', options.status);
        }
        const queryString = params.toString();
        const endpoint = `/v1/experiments${queryString ? `?${queryString}` : ''}`;

        const data = await apiCall<{ experiments: Array<Record<string, any>>; count: number }>(
          endpoint,
          accessToken
        );

        const normalized: Experiment[] = (data.experiments || []).map(
          (e) =>
            ({
              ...e,
              id: e.id || e.experiment_id,
            }) as Experiment
        );

        console.log('[EvaluationsService] Normalized experiments:', normalized.length);
        return normalized;
      } catch (error) {
        console.error('[EvaluationsService] Failed to list experiments:', error);
        throw error;
      }
    },
    []
  );

  const createExperiment = useCallback(
    async (
      accessToken: string,
      request: {
        name: string;
        description?: string;
        dataset_id: string;
        evaluation_ids: string[];
        tags?: string[];
      }
    ): Promise<Experiment> => {
      console.log('[EvaluationsService] Creating experiment:', request.name);

      try {
        const rawData = await apiCall<Record<string, any>>('/v1/experiments', accessToken, {
          method: 'POST',
          body: JSON.stringify(request),
        });

        // Backend returns experiment directly
        const expData = rawData.experiment || rawData;
        const normalized: Experiment = {
          ...expData,
          id: expData.id || expData.experiment_id,
        } as Experiment;

        console.log('[EvaluationsService] Created experiment:', normalized.id);
        return normalized;
      } catch (error) {
        console.error('[EvaluationsService] Failed to create experiment:', error);
        throw error;
      }
    },
    []
  );

  const getExperiment = useCallback(
    async (accessToken: string, experimentId: string): Promise<Experiment> => {
      console.log('[EvaluationsService] Getting experiment:', experimentId);

      try {
        const rawData = await apiCall<Record<string, any>>(
          `/v1/experiments/${experimentId}`,
          accessToken
        );

        const expData = rawData.experiment || rawData;
        return {
          ...expData,
          id: expData.id || expData.experiment_id,
        } as Experiment;
      } catch (error) {
        console.error('[EvaluationsService] Failed to get experiment:', error);
        throw error;
      }
    },
    []
  );

  const deleteExperiment = useCallback(
    async (accessToken: string, experimentId: string): Promise<boolean> => {
      console.log('[EvaluationsService] Deleting experiment:', experimentId);

      try {
        await apiCall<{ message: string }>(`/v1/experiments/${experimentId}`, accessToken, {
          method: 'DELETE',
        });
        return true;
      } catch (error) {
        console.error('[EvaluationsService] Failed to delete experiment:', error);
        throw error;
      }
    },
    []
  );

  const getExperimentComparison = useCallback(
    async (accessToken: string, experimentId: string): Promise<ExperimentComparison | null> => {
      console.log('[EvaluationsService] Getting experiment comparison:', experimentId);

      try {
        const data = await apiCall<Record<string, any>>(
          `/v1/experiments/${experimentId}/comparison?force=true`,
          accessToken
        );

        console.log('[EvaluationsService] Raw comparison data:', data);

        // Extract evaluation names and normalize metric_summary
        // Backend returns { eval_id: { evaluation_name, models, metrics: { metric_id: number } } }
        // Frontend expects { eval_id: { metric_id: number } }
        const rawSummary = data.metric_summary || {};
        const evaluationNames: { [evalId: string]: string } = {};
        const normalizedSummary: { [evalId: string]: { [metricId: string]: number } } = {};
        for (const [evalId, evalData] of Object.entries(rawSummary)) {
          const ed = evalData as Record<string, any>;
          evaluationNames[evalId] = ed.evaluation_name || evalId;
          normalizedSummary[evalId] = ed.metrics || ed;
        }

        // Normalize samples: backend returns samples[].evaluations with nested structure
        // Frontend expects samples[].scores with { eval_id: { metric_id: { score, delta? } } }
        const rawSamples = data.samples || [];
        const normalizedSamples = rawSamples.map((sample: Record<string, any>) => {
          const evaluations = sample.evaluations || {};
          const firstEvalData = Object.values(evaluations)[0] as Record<string, any> | undefined;

          const scores: {
            [evalId: string]: { [metricId: string]: { score: number; delta?: number } };
          } = {};
          for (const [evalId, evalData] of Object.entries(evaluations)) {
            const ed = evalData as Record<string, any>;
            if (ed.missing) continue;
            const evalScores = ed.scores || {};
            const deltas = ed.deltas || {};
            scores[evalId] = {};
            for (const [metricId, metricData] of Object.entries(evalScores)) {
              const md = metricData as any;
              const scoreVal = typeof md === 'number' ? md : (md?.score ?? md);
              scores[evalId][metricId] = {
                score: typeof scoreVal === 'number' ? scoreVal : 0,
                ...(deltas[metricId] != null ? { delta: deltas[metricId] } : {}),
              };
            }
          }

          return {
            sample_id: sample.sample_id,
            input: firstEvalData?.input || '',
            expected: firstEvalData?.expected || '',
            scores,
          };
        });

        const result: ExperimentComparison = {
          experiment_id: data.experiment_id || experimentId,
          evaluation_names: evaluationNames,
          metric_summary: normalizedSummary,
          samples: normalizedSamples,
        };

        console.log('[EvaluationsService] Normalized comparison:', result);
        return result;
      } catch (error) {
        // 400 means experiment not ready, 404 means no comparison yet
        if (
          error instanceof Error &&
          (error.message.includes('400') || error.message.includes('404'))
        ) {
          console.log('[EvaluationsService] No comparison available yet');
          return null;
        }
        console.error('[EvaluationsService] Failed to get experiment comparison:', error);
        throw error;
      }
    },
    []
  );

  // ============================================================
  // ANNOTATIONS
  // ============================================================

  const listAnnotationQueues = useCallback(
    async (accessToken: string): Promise<AnnotationQueue[]> => {
      const data = await apiCall<{ queues: Array<Record<string, any>>; count: number }>(
        '/v1/annotations/queues',
        accessToken
      );

      return (data.queues || []).map((q) => normalizeQueue(q));
    },
    []
  );

  const createAnnotationQueue = useCallback(
    async (
      accessToken: string,
      request: { name: string; dataset_id: string; rubric?: Array<Record<string, any>> }
    ): Promise<AnnotationQueue> => {
      const rawData = await apiCall<Record<string, any>>('/v1/annotations/queues', accessToken, {
        method: 'POST',
        body: JSON.stringify(request),
      });

      return normalizeQueue(rawData);
    },
    []
  );

  const deleteAnnotationQueue = useCallback(
    async (accessToken: string, queueId: string): Promise<boolean> => {
      await apiCall<{ message: string }>(`/v1/annotations/queues/${queueId}`, accessToken, {
        method: 'DELETE',
      });
      return true;
    },
    []
  );

  const listAnnotationItems = useCallback(
    async (accessToken: string, queueId: string, status?: string): Promise<AnnotationItem[]> => {
      const params = new URLSearchParams();
      if (status) params.append('status', status);
      const qs = params.toString();
      const endpoint = `/v1/annotations/queues/${queueId}/items${qs ? `?${qs}` : ''}`;

      const data = await apiCall<{ items: Array<Record<string, any>>; count: number }>(
        endpoint,
        accessToken
      );

      return (data.items || []).map((item) => normalizeItem(item));
    },
    []
  );

  const getNextAnnotationItem = useCallback(
    async (accessToken: string, queueId: string): Promise<AnnotationItem | null> => {
      try {
        const data = await apiCall<Record<string, any>>(
          `/v1/annotations/queues/${queueId}/next`,
          accessToken
        );
        return normalizeItem(data);
      } catch (error) {
        if (error instanceof Error && error.message.includes('404')) {
          return null;
        }
        throw error;
      }
    },
    []
  );

  const submitAnnotationItem = useCallback(
    async (
      accessToken: string,
      queueId: string,
      itemId: string,
      scores: Record<string, any>,
      notes?: string
    ): Promise<AnnotationItem> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/annotations/queues/${queueId}/items/${itemId}/submit`,
        accessToken,
        {
          method: 'POST',
          body: JSON.stringify({ scores, notes: notes || '' }),
        }
      );
      return normalizeItem(data);
    },
    []
  );

  const skipAnnotationItem = useCallback(
    async (accessToken: string, queueId: string, itemId: string): Promise<AnnotationItem> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/annotations/queues/${queueId}/items/${itemId}/skip`,
        accessToken,
        { method: 'POST' }
      );
      return normalizeItem(data);
    },
    []
  );

  const exportAnnotations = useCallback(
    async (
      accessToken: string,
      queueId: string,
      format: 'json' | 'csv' = 'json'
    ): Promise<{
      csv?: string;
      filename?: string;
      queue?: Record<string, any>;
      annotations?: Array<Record<string, any>>;
      count?: number;
    }> => {
      const data = await apiCall<Record<string, any>>(
        `/v1/annotations/queues/${queueId}/export?format=${format}`,
        accessToken
      );
      return data;
    },
    []
  );

  const getAnnotationAnalytics = useCallback(
    async (accessToken: string, queueId: string): Promise<AnnotationAnalytics> => {
      return apiCall<AnnotationAnalytics>(
        `/v1/annotations/queues/${queueId}/analytics`,
        accessToken
      );
    },
    []
  );

  // =========================================================================
  // Auto-Eval
  // =========================================================================

  const listAutoEvalConfigs = useCallback(
    async (accessToken: string): Promise<{ configs: any[]; count: number }> => {
      return apiCall('/v1/auto-eval/configs', accessToken);
    },
    []
  );

  const createAutoEvalConfig = useCallback(
    async (accessToken: string, data: Record<string, any>): Promise<any> => {
      return apiCall('/v1/auto-eval/configs', accessToken, {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    []
  );

  const updateAutoEvalConfig = useCallback(
    async (accessToken: string, configId: string, data: Record<string, any>): Promise<any> => {
      return apiCall(`/v1/auto-eval/configs/${configId}`, accessToken, {
        method: 'PUT',
        body: JSON.stringify(data),
      });
    },
    []
  );

  const deleteAutoEvalConfig = useCallback(
    async (accessToken: string, configId: string): Promise<any> => {
      return apiCall(`/v1/auto-eval/configs/${configId}`, accessToken, {
        method: 'DELETE',
      });
    },
    []
  );

  const triggerAutoEvalRun = useCallback(
    async (accessToken: string, configId: string): Promise<any> => {
      const { data } = await apiCallWithAccepted(
        `/v1/auto-eval/configs/${configId}/trigger`,
        accessToken,
        { method: 'POST' }
      );
      return data;
    },
    []
  );

  const listAutoEvalRuns = useCallback(
    async (accessToken: string, configId: string): Promise<{ runs: any[]; count: number }> => {
      return apiCall(`/v1/auto-eval/configs/${configId}/runs`, accessToken);
    },
    []
  );

  const suggestMetrics = useCallback(
    async (accessToken: string, datasetId: string): Promise<{ suggestions: any[] }> => {
      return apiCall('/v1/auto-eval/suggest-metrics', accessToken, {
        method: 'POST',
        body: JSON.stringify({ dataset_id: datasetId }),
      });
    },
    []
  );

  // =========================================================================
  // Proposals (Decision Engine)
  // =========================================================================

  const listProposals = useCallback(
    async (accessToken: string, status?: string): Promise<{ proposals: any[]; count: number }> => {
      const params = new URLSearchParams();
      if (status) params.append('status', status);
      const qs = params.toString();
      return apiCall(`/v1/proposals${qs ? `?${qs}` : ''}`, accessToken);
    },
    []
  );

  const approveProposal = useCallback(
    async (
      accessToken: string,
      proposalId: string
    ): Promise<{ message: string; proposal_id: string; execution_result: any }> => {
      return apiCall(`/v1/proposals/${proposalId}/approve`, accessToken, { method: 'POST' });
    },
    []
  );

  const rejectProposal = useCallback(
    async (
      accessToken: string,
      proposalId: string,
      reason?: string
    ): Promise<{ message: string }> => {
      return apiCall(`/v1/proposals/${proposalId}/reject`, accessToken, {
        method: 'POST',
        body: JSON.stringify(reason ? { reason } : {}),
      });
    },
    []
  );

  const getAnnotationStats = useCallback(
    async (
      accessToken: string,
      queueId: string
    ): Promise<{
      total_items: number;
      completed_items: number;
      skipped_items: number;
      pending_items: number;
    }> => {
      return apiCall(`/v1/annotations/queues/${queueId}/stats`, accessToken);
    },
    []
  );

  return useMemo(
    () => ({
      // Datasets
      listDatasets,
      getDataset,
      createDataset,
      deleteDataset,
      importDataset,
      analyzeTraces,
      importTraces,
      addSamplesToDataset,
      deleteSample,

      // Traces
      listTraces,
      createDatasetFromTraces,
      createDatasetFromInstruction,

      // Evaluations
      listEvaluations,
      getEvaluation,
      createEvaluation,
      getEvaluationStatus,
      getEvaluationResults,
      cancelEvaluation,
      rerunEvaluation,
      deleteEvaluation,
      exportEvaluationResults,

      // Metrics
      listMetrics,
      getMetric,
      createCustomMetric,
      updateCustomMetric,
      deleteCustomMetric,
      validatePythonScript,

      // Models
      listAvailableModels,

      // Auto-Collect
      getAutoCollectConfig,
      putAutoCollectConfig,
      deleteAutoCollectConfig,
      getAutoCollectHistory,
      triggerAutoCollect,

      // Trace Issues
      listTraceIssues,
      triggerTraceScan,
      getTraceScanStatus,
      resolveTraceIssue,
      dismissTraceIssue,
      getScheduleConfig,
      updateScheduleConfig,

      // Experiments
      listExperiments,
      createExperiment,
      getExperiment,
      deleteExperiment,
      getExperimentComparison,

      // Auto-Eval
      listAutoEvalConfigs,
      createAutoEvalConfig,
      updateAutoEvalConfig,
      deleteAutoEvalConfig,
      triggerAutoEvalRun,
      listAutoEvalRuns,
      suggestMetrics,

      // Proposals
      listProposals,
      approveProposal,
      rejectProposal,

      // Annotations
      listAnnotationQueues,
      createAnnotationQueue,
      deleteAnnotationQueue,
      listAnnotationItems,
      getNextAnnotationItem,
      submitAnnotationItem,
      skipAnnotationItem,
      getAnnotationStats,
      exportAnnotations,
      getAnnotationAnalytics,
    }),
    [
      listDatasets,
      getDataset,
      createDataset,
      deleteDataset,
      importDataset,
      analyzeTraces,
      importTraces,
      addSamplesToDataset,
      deleteSample,
      listTraces,
      createDatasetFromTraces,
      createDatasetFromInstruction,
      listEvaluations,
      getEvaluation,
      createEvaluation,
      getEvaluationStatus,
      getEvaluationResults,
      cancelEvaluation,
      rerunEvaluation,
      deleteEvaluation,
      exportEvaluationResults,
      listMetrics,
      getMetric,
      createCustomMetric,
      updateCustomMetric,
      deleteCustomMetric,
      validatePythonScript,
      listAvailableModels,
      getAutoCollectConfig,
      putAutoCollectConfig,
      deleteAutoCollectConfig,
      getAutoCollectHistory,
      triggerAutoCollect,
      listTraceIssues,
      triggerTraceScan,
      getTraceScanStatus,
      resolveTraceIssue,
      listExperiments,
      createExperiment,
      getExperiment,
      deleteExperiment,
      getExperimentComparison,
      listAutoEvalConfigs,
      createAutoEvalConfig,
      updateAutoEvalConfig,
      deleteAutoEvalConfig,
      triggerAutoEvalRun,
      listAutoEvalRuns,
      suggestMetrics,
      listProposals,
      approveProposal,
      rejectProposal,
      listAnnotationQueues,
      createAnnotationQueue,
      deleteAnnotationQueue,
      listAnnotationItems,
      getNextAnnotationItem,
      submitAnnotationItem,
      skipAnnotationItem,
      getAnnotationStats,
      exportAnnotations,
      getAnnotationAnalytics,
    ]
  );
}
