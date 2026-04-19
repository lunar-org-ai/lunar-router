import type { SVGProps } from 'react';

export type ModelIcon = React.ComponentType<SVGProps<SVGSVGElement>>;

export interface DeploymentModel {
  id: string;
  name: string;
  description: string;
  icon: ModelIcon | string | null;
  features: string[];
  availableInstances?: string[];
  recommendedInstance?: string;
  // VRAM requirements in GB
  vramRequired?: {
    fp16: number; // Full precision (FP16/BF16)
    fp8?: number; // FP8 quantization
    int4?: number; // INT4/AWQ quantization
  };
  // Model parameters in billions
  parameters?: number;
  // Quantization type if pre-quantized
  quantization?: 'fp16' | 'fp8' | 'int4' | 'awq' | 'gptq';
}

// API error codes from backend
export type DeploymentErrorCode =
  | 'model_too_large'
  | 'no_spot_capacity'
  | 'gpu_unavailable'
  | 'startup_crash'
  | 'image_pull_error'
  | 'node_not_ready'
  | 'network_error'
  | 'unknown';

export interface DeploymentError {
  error_code: DeploymentErrorCode;
  error_message: string;
  user_message: string;
  retryable: boolean;
}

export interface ModelOptions {
  // Engine arguments (for model initialization)
  maxTokens: number; // max-model-len: Model context length
  dtype: 'auto' | 'bfloat16' | 'float16' | 'float32'; // Data type for model weights
  gpuMemoryUtilization: number; // GPU memory utilization (0.0-1.0)
  maxNumSeqs: number; // Maximum number of sequences in a batch
  blockSize: number; // Token block size for memory management
  swapSpace: number; // CPU swap space in GB

  // Inference parameters (for requests - stored but not used in engine args)
  temperature: number;
  topP: number;
  topK: number;
}

export interface AutoscalingConfig {
  enabled: boolean;
  maxReplicas: number;
  versionComment: string;
}

export interface DeploymentData {
  id: string;
  name: string;
  selectedModel: string;
  selectedInstance: string;
  deployment_id?: string;
  modelOptions: ModelOptions;
  autoscalingConfig: AutoscalingConfig;
  createdAt: string;
  status:
    | 'active'
    | 'inactive'
    | 'pending'
    | 'in_service'
    | 'starting'
    | 'creating'
    | 'updating'
    | 'stopped'
    | 'failed'
    | 'deleting'
    | 'paused'
    | 'pausing'
    | 'resuming';
  error_message?: string;
  error_code?: string;
  endpoint_url?: string;
}

export interface GPUInstanceSpecs {
  gpu: string;
  vram: string;
  vCPUs: number;
  ram: string;
  storage: string;
  network: string;
  spotPrice: string;
  modelSize: string;
  gpuCount: number;
  tensorParallelSize: number;
}

export interface GPUInstanceType {
  id: string;
  name: string;
  tier: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL';
  ec2Instance: string;
  gpus: string;
  price?: string;
  memory: string;
  recommended?: boolean;
  description: string;
  specs: GPUInstanceSpecs;
}

export interface ModelCategory {
  id: string;
  label: string;
  count: number;
}

// Metrics API Types
export interface MetricsLatest {
  cpu_utilization: number;
  memory_utilization: number;
  gpu_utilization: number;
  gpu_memory_utilization: number;
  model_latency_ms: number;
  invocations: number;
  // EKS/vLLM specific fields
  gpu_cache_usage?: number;
  cpu_cache_usage?: number;
  requests_running?: number;
  requests_waiting?: number;
  timestamp?: string;
}

export interface InferenceStats {
  total_inferences: number;
  successful: number;
  failed: number;
  success_rate: number;
  avg_latency_ms: number;
  total_tokens: number;
  total_cost_usd: number;
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

// EKS time series uses 'time' instead of 'timestamp'
export interface EKSTimeSeriesPoint {
  time: string;
  value: number;
}

export interface MetricsTimeSeries {
  cpu_utilization: TimeSeriesPoint[];
  memory_utilization: TimeSeriesPoint[];
  gpu_utilization: TimeSeriesPoint[];
  gpu_memory_utilization: TimeSeriesPoint[];
  model_latency: TimeSeriesPoint[];
  invocations: TimeSeriesPoint[];
  avg_ttft?: (TimeSeriesPoint | EKSTimeSeriesPoint)[];
  // EKS/vLLM specific fields
  gpu_cache_usage?: (TimeSeriesPoint | EKSTimeSeriesPoint)[];
  cpu_cache_usage?: (TimeSeriesPoint | EKSTimeSeriesPoint)[];
  requests_running?: (TimeSeriesPoint | EKSTimeSeriesPoint)[];
  requests_waiting?: (TimeSeriesPoint | EKSTimeSeriesPoint)[];
}

export interface DeploymentMetricsData {
  deployment_id: string;
  latest: MetricsLatest;
  inference_stats: InferenceStats;
  time_series: MetricsTimeSeries;
}
