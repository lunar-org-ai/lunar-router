export interface Dataset {
  id: string;
  name: string;
  description?: string;
  source: 'manual' | 'imported' | 'auto_collected' | 'instruction' | 'synthetic';
  samples_count: number;
  created_at: string;
  updated_at: string;
  schema?: DatasetSchema;
}

export interface DatasetSample {
  id: string;
  input: string;
  output?: string;
  expected_output?: string;
  metadata?: Record<string, unknown>;
  raw?: string;
  created_at: string;
}

export interface DatasetSchema {
  input_field: string;
  output_field?: string;
  metadata_fields?: string[];
}

// Metric Types
export interface EvaluationMetric {
  metric_id: string;
  name: string;
  type: MetricType;
  description: string;
  config?: MetricConfig;
  is_builtin: boolean;
  python_script?: string;
  created_at: string;
  updated_at?: string;
}

export type MetricType =
  | 'exact_match'
  | 'contains'
  | 'semantic_sim'
  | 'hf_similarity'
  | 'llm_judge'
  | 'latency'
  | 'cost'
  | 'python';

export interface MetricConfig {
  ignore_case?: boolean;
  ignore_whitespace?: boolean;
  normalize?: boolean;

  all_must_match?: boolean;
  expected_values?: string[];

  judge_model?: string;
  criteria?: string[];
  prompt_template?: string;
  scale?: { min: number; max: number };

  model?: string;
  embedding_model?: string;
  threshold?: number;
  similarity_threshold?: number;

  max_acceptable?: number;
}

// Evaluation Types
export interface Evaluation {
  id: string;
  name: string;
  description?: string;
  dataset_id: string;
  dataset_name?: string;
  models: string[];
  metrics: string[];
  status: EvaluationStatus;
  progress?: EvaluationProgress;
  error?: EvaluationError;
  created_at: string;
  started_at?: string;
  updated_at?: string;
  completed_at?: string;
  results?: EvaluationResults;
}

export type EvaluationStatus =
  | 'queued'
  | 'starting'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface EvaluationProgress {
  total_samples: number;
  completed_samples: number;
  failed_samples: number;
}

export type EvaluationErrorCode =
  | 'WORKER_TIMEOUT'
  | 'MODEL_ERROR'
  | 'QUEUE_ERROR'
  | 'INSUFFICIENT_CREDITS';

export interface EvaluationError {
  code: EvaluationErrorCode;
  message: string;
  details?: Record<string, unknown>;
}

export interface CreateEvaluationResponse {
  evaluation_id: string;
  status: 'queued';
  message: string;
}

export interface EvaluationStatusResponse {
  evaluation_id: string;
  status: EvaluationStatus;
  progress: EvaluationProgress;
  started_at?: string;
  updated_at?: string;
  error?: EvaluationError;
}

export interface EvaluationResults {
  evaluation_id: string;
  samples: SampleResult[];
  summary: {
    models: {
      [model_id: string]: ModelSummary;
    };
    metrics: {
      [metric_id: string]: {
        avg_by_model: {
          [model_id: string]: number;
        };
      };
    };
  };
  winner?: {
    model: string;
    overall_score: number;
    scores_by_model: {
      [model_id: string]: number;
    };
  };
  entity_type?: string;
  created_at?: string;
  pk?: string;
  sk?: string;
}

export interface ModelSummary {
  total_latency: number;
  avg_latency: number;
  total_cost: number;
  avg_cost: number;
  avg_scores: {
    [metric_id: string]: number;
  };
}

export interface SampleResult {
  sample_id: string;
  input: string;
  expected?: string; // Note: API uses "expected" not "expected_output"
  outputs: {
    [model_id: string]: ModelOutput;
  };
  scores: {
    [metric_id: string]: {
      [model_id: string]: MetricScore;
    };
  };
}

export interface ModelOutput {
  output: string;
  error?: string;
  latency: number;
  cost?: number;
}

export interface MetricScore {
  score: number;
  passed?: boolean;
  match?: boolean;
  latency_seconds?: number;
  max_acceptable?: number;
  [key: string]: any; // Allow additional metric-specific fields
}

// API Request Types
export interface CreateDatasetRequest {
  name: string;
  description?: string;
  source?: 'manual' | 'imported' | 'auto_collected' | 'instruction' | 'synthetic';
  samples?: Omit<DatasetSample, 'id' | 'created_at'>[];
  auto_collect_instruction?: string;
}

export interface CreateFromInstructionRequest {
  name: string;
  instruction: string;
  description?: string;
  model_id?: string;
  limit?: number;
  max_samples?: number;
}

export interface CreateFromInstructionResponse {
  dataset_id: string;
  name: string;
  source: 'instruction';
  samples_count: number;
  traces_scanned: number;
  traces_matched: number;
}

export interface GenerateDatasetRequest {
  name: string;
  instruction: string;
  description?: string;
  count?: number;
  auto_collect_instruction?: string;
}

export interface GenerateDatasetResponse {
  dataset_id: string;
  name: string;
  source: 'synthetic';
  samples_count: number;
  samples_requested: number;
}

export interface CreateEvaluationRequest {
  name: string;
  description?: string;
  dataset_id: string;
  models: string[];
  metrics: string[]; // Array of metric_ids
  config?: {
    max_tokens?: number;
    temperature?: number;
    [key: string]: unknown;
  };
}

export interface CreateCustomMetricRequest {
  name: string;
  type: MetricType;
  description: string;
  config?: MetricConfig;
  python_script?: string;
  requirements?: string[]; // Pip packages to install (e.g., ['textstat', 'numpy'])
}

export interface UpdateCustomMetricRequest {
  name?: string;
  description?: string;
  python_script?: string;
  config?: MetricConfig;
}

export interface ValidateScriptRequest {
  python_script: string;
  test_data?: {
    output: string;
    expected: string;
    input_text: string;
  };
}

export interface ValidateScriptResponse {
  valid: boolean;
  message?: string;
  error?: string;
  details?: string;
  test_result?: {
    score: number;
    [key: string]: unknown;
  };
}

// Trace Types (for auto-collection)
export interface Trace {
  id: string;
  input: string;
  output: string;
  model_id: string;
  latency_ms: number;
  cost_usd: number;
  source: string;
  created_at: string;
  metadata?: Record<string, unknown>;
}

// Auto-Collect Types
export interface AutoCollectConfig {
  dataset_id: string;
  enabled: boolean;
  instruction?: string;
  source_model?: string;
  max_samples: number;
  collection_interval_minutes: number;
  curation_config: {
    quality_threshold: number;
    selection_rate: number;
    agent_weights: { quality: number; diversity: number; difficulty: number };
  };
  last_collected_at?: string;
  total_collected: number;
  created_at: string;
  updated_at: string;
}

export interface CollectRun {
  run_id: string;
  dataset_id: string;
  created_at: string;
  traces_found: number;
  traces_after_dedup: number;
  traces_scored: number;
  traces_selected: number;
  samples_added: number;
}

// Built-in Metrics (default/fallback - should be fetched from API)
export const BUILTIN_METRICS: EvaluationMetric[] = [
  {
    metric_id: 'exact_match',
    name: 'Exact Match',
    type: 'exact_match',
    description: 'Compares output exactly with expected output',
    is_builtin: true,
    config: {
      ignore_case: false,
      ignore_whitespace: false,
      normalize: false,
    },
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'contains',
    name: 'Contains',
    type: 'contains',
    description: 'Check if output contains expected text',
    is_builtin: true,
    config: {
      ignore_case: true,
      all_must_match: true,
    },
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'hf_similarity',
    name: 'HuggingFace-Style Similarity',
    type: 'hf_similarity',
    description:
      'Calculates semantic similarity using Cohere embeddings (similar quality to all-MiniLM-L6-v2)',
    is_builtin: true,
    config: {
      model: 'cohere.embed-english-v3',
      threshold: 0.8,
    },
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'semantic_sim',
    name: 'Semantic Similarity (Bedrock Titan)',
    type: 'semantic_sim',
    description: 'Embedding-based similarity using AWS Bedrock Titan embeddings',
    is_builtin: true,
    config: {
      model: 'amazon.titan-embed-text-v1',
      threshold: 0.8,
    },
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'llm_judge',
    name: 'LLM-as-Judge',
    type: 'llm_judge',
    description: 'Use an LLM to evaluate quality of responses',
    config: {
      judge_model: 'gpt-4',
      criteria: ['accuracy', 'relevance', 'completeness'],
      scale: { min: 1, max: 10 },
    },
    is_builtin: true,
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'latency',
    name: 'Latency',
    type: 'latency',
    description: 'Measures response time in seconds',
    is_builtin: true,
    config: {
      max_acceptable: 5.0,
    },
    created_at: new Date().toISOString(),
  },
  {
    metric_id: 'cost',
    name: 'Cost',
    type: 'cost',
    description: 'Measures cost in USD',
    is_builtin: true,
    config: {
      max_acceptable: 0.01,
    },
    created_at: new Date().toISOString(),
  },
];

// Template for Python custom metrics
export const PYTHON_METRIC_TEMPLATE = `import re
import json
import math

def evaluate(output, expected, input_text):
    """
    Evaluate the model's response.

    Args:
        output: Text generated by the model
        expected: Expected text (can be None)
        input_text: Original prompt/input

    Returns:
        dict with at least the 'score' key (float 0.0-1.0)
        Can include other keys for metadata
    """
    # Your logic here
    score = 1.0

    return {
        "score": score,
        "passed": score >= 0.8,
        # Add other fields if needed
    }
`;

// ============================================================================
// Experiment Types (Phase 1)
// ============================================================================

export type ExperimentStatus = 'draft' | 'running' | 'completed' | 'failed';

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  evaluation_ids: string[];
  dataset_id: string;
  dataset_name?: string;
  status: ExperimentStatus;
  created_at: string;
  updated_at: string;
  tags?: string[];
}

export interface ExperimentComparison {
  experiment_id: string;
  evaluation_names: { [eval_id: string]: string };
  samples: ExperimentComparisonRow[];
  metric_summary: { [eval_id: string]: { [metric_id: string]: number } };
}

export interface ExperimentComparisonRow {
  sample_id: string;
  input: string;
  expected?: string;
  scores: { [eval_id: string]: { [metric_id: string]: { score: number; delta?: number } } };
}

// ============================================================================
// Trace Issue Types (Phase 2)
// ============================================================================

export type IssueSeverity = 'high' | 'medium' | 'low';
export type IssueType =
  | 'hallucination'
  | 'refusal'
  | 'safety'
  | 'quality_regression'
  | 'latency_spike'
  | 'cost_anomaly'
  | 'format_violation'
  | 'incomplete_response';

export interface TraceIssue {
  id: string;
  trace_id: string;
  type: IssueType;
  severity: IssueSeverity;
  title: string;
  description: string;
  ai_confidence: number;
  model_id: string;
  trace_input: string;
  trace_output: string;
  detected_at: string;
  resolved: boolean;
  dismissed?: boolean;
  suggested_action?: string;
  suggested_eval_config?: EvalPrefillConfig;
}

export interface TraceScan {
  id: string;
  status: 'running' | 'completed' | 'failed';
  traces_scanned: number;
  issues_found: number;
  started_at: string;
  completed_at?: string;
}

// Re-export for convenience — matches AISuggestionsPanel's EvalPrefillConfig
export interface EvalPrefillConfig {
  name?: string;
  datasetId?: string;
  models?: string[];
  metrics?: string[];
}

// ============================================================================
// Annotation Types (Phase 4)
// ============================================================================

export type AnnotationQueueStatus = 'active' | 'paused' | 'completed';

export interface AnnotationQueue {
  id: string;
  name: string;
  description?: string;
  dataset_id: string;
  dataset_name?: string;
  evaluation_id?: string;
  rubric: AnnotationRubric;
  status: AnnotationQueueStatus;
  total_items: number;
  completed_items: number;
  skipped_items: number;
  annotators: string[];
  created_at: string;
  updated_at: string;
}

export interface AnnotationRubric {
  criteria: AnnotationCriterion[];
  instructions?: string;
}

export interface AnnotationCriterion {
  id: string;
  name: string;
  description: string;
  scale: { min: number; max: number; labels?: Record<number, string> };
}

export interface AnnotationItem {
  id: string;
  queue_id: string;
  sample_id: string;
  input: string;
  output: string;
  expected_output?: string;
  model_id?: string;
  ai_pre_scores?: { [criterion_id: string]: { score: number; reasoning: string } };
  human_scores?: { [criterion_id: string]: number };
  human_notes?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  annotated_by?: string;
  annotated_at?: string;
}

export interface AnnotationCriterionStats {
  mean: number | null;
  median: number | null;
  std_dev: number | null;
  min: number | null;
  max: number | null;
  distribution: Record<string, number>;
}

export interface AnnotationAgreement {
  compared_queues: string[];
  overlapping_samples: number;
  cohens_kappa: Record<string, number>;
  percent_agreement: Record<string, number>;
}

export interface AnnotationAnalytics {
  queue_id: string;
  total_annotated: number;
  criteria: Record<string, AnnotationCriterionStats>;
  agreement: AnnotationAgreement | null;
}

// ============================================================================
// Auto-Evaluation Types (Phase 5)
// ============================================================================

export type AutoEvalSchedule = 'hourly' | 'daily' | 'weekly' | 'on_deploy';

export interface AutoEvalConfig {
  id: string;
  name: string;
  enabled: boolean;
  schedule: AutoEvalSchedule;
  dataset_id: string;
  dataset_name?: string;
  models: string[];
  metrics: string[];
  topic_filter?: string;
  alert_on_regression: boolean;
  regression_threshold: number;
  last_run_at?: string;
  last_run_score?: number;
  created_at: string;
  updated_at: string;
}

export interface AutoEvalRun {
  id: string;
  config_id: string;
  status: 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  scores: Record<string, number>;
  regression_detected: boolean;
  evaluation_id?: string;
}

// ============================================================================
// Proposal Types (Decision Engine)
// ============================================================================

export type ProposalStatus =
  | 'pending'
  | 'approved'
  | 'rejected'
  | 'expired'
  | 'executed'
  | 'failed';
export type ProposalType = 'create_evaluation' | 'setup_auto_eval' | 'create_experiment';
export type ProposalPriority = 'low' | 'medium' | 'high';

export interface Proposal {
  id: string;
  proposal_type: ProposalType;
  event_type: string;
  status: ProposalStatus;
  priority: ProposalPriority;
  title: string;
  description: string;
  reason: string;
  dataset_id: string;
  config_id?: string;
  evaluation_id?: string;
  dedup_key: string;
  action_payload: Record<string, unknown>;
  execution_result?: {
    success: boolean;
    created_id?: string;
    error?: string;
  };
  created_at: string;
  expires_at: string;
  resolved_at?: string;
}
