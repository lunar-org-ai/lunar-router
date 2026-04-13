export type {
  Dataset,
  DatasetSample,
  CreateDatasetRequest,
  CreateFromInstructionRequest,
  CreateFromInstructionResponse,
  GenerateDatasetRequest,
  GenerateDatasetResponse,
  CollectRun,
  Trace,
} from '@/features/evaluations/types/evaluationsTypes';

export type ViewTab = 'general' | 'data-pipeline' | 'models' | 'evaluate' | 'settings';

export type CreateMode =
  | 'manual'
  | 'import'
  | 'smart-import'
  | 'topic'
  | 'generate'
  | 'traces'
  | 'cluster';

export interface TraceMapping {
  input: { path: string; transform: string };
  output: { path: string; transform: string };
  metadata: Record<string, string>;
}

export interface AnalyzeTracesResponse {
  mapping: TraceMapping;
  preview: Array<{ input: string; expected_output: string; metadata: Record<string, any> }>;
  source_format: string;
  total_records: number;
}

export interface ImportTracesResponse {
  dataset_id: string;
  name: string;
  source: string;
  samples_count: number;
  skipped_count?: number;
  skipped_reasons?: Array<{ index: number; reason: string }>;
}

export type GeneratePhase =
  | 'idle'
  | 'preparing'
  | 'generating'
  | 'reviewing'
  | 'building'
  | 'done'
  | 'error';

export type TopicPhase =
  | 'idle'
  | 'scanning'
  | 'analyzing'
  | 'matching'
  | 'building'
  | 'done'
  | 'no-match'
  | 'error';
