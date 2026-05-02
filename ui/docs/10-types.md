# Types e Interfaces

Definicoes TypeScript do projeto.

## Arquivo Principal

```
src/types/
├── deploymentTypes.ts    # Types de deployment e metricas
└── toast.ts              # Types de notificacoes
```

## Deployment Types

### DeploymentData

Dados completos de um deployment.

```typescript
interface DeploymentData {
  id: string;
  name: string;
  selectedModel: string;       // ID do modelo
  selectedInstance: string;    // Tipo de instancia
  deployment_id?: string;      // ID do deployment na API
  modelOptions: ModelOptions;
  autoscalingConfig: AutoscalingConfig;
  createdAt: string;           // ISO 8601
  status: DeploymentStatus;
}

type DeploymentStatus =
  | 'active'
  | 'inactive'
  | 'pending'
  | 'in_service'
  | 'starting'
  | 'creating'
  | 'updating'
  | 'stopped'
  | 'failed'
  | 'deleting';
```

### DeploymentModel

Modelo disponivel para deployment.

```typescript
interface DeploymentModel {
  id: string;                    // ID unico (ex: "Llama-3.2-1B")
  name: string;                  // Nome de exibicao
  description: string;
  icon: string;                  // SVG import
  features: string[];            // Ex: ["LLM"]
  availableInstances?: string[]; // Instancias compativeis
  recommendedInstance?: string;  // Instancia recomendada
}
```

### GPUInstanceType

Tipo de instancia GPU.

```typescript
interface GPUInstanceType {
  id: string;           // Ex: "ml.g5.xlarge"
  name: string;         // Nome de exibicao
  gpus: string;         // Ex: "1x GPU"
  price?: string;
  memory: string;       // Ex: "24GB"
  recommended?: boolean;
  description: string;
}
```

### ModelOptions

Opcoes de configuracao do modelo VLLM.

```typescript
interface ModelOptions {
  // Engine arguments
  maxTokens: number;           // max-model-len
  dtype: 'auto' | 'bfloat16' | 'float16' | 'float32';
  gpuMemoryUtilization: number; // 0.0-1.0
  maxNumSeqs: number;
  blockSize: number;
  swapSpace: number;           // GB

  // Inference parameters (para requests)
  temperature: number;
  topP: number;
  topK: number;
}
```

### AutoscalingConfig

Configuracao de autoscaling.

```typescript
interface AutoscalingConfig {
  enabled: boolean;
  maxReplicas: number;
  versionComment: string;
}
```

## Metrics Types

### DeploymentMetricsData

Resposta da API de metricas.

```typescript
interface DeploymentMetricsData {
  deployment_id: string;
  latest: MetricsLatest;
  inference_stats: InferenceStats;
  time_series: MetricsTimeSeries;
}
```

### MetricsLatest

Metricas atuais de utilizacao.

```typescript
interface MetricsLatest {
  cpu_utilization: number;       // Percentual
  memory_utilization: number;    // Percentual
  gpu_utilization: number;       // Percentual
  gpu_memory_utilization: number; // Percentual
  model_latency_ms: number;
  invocations: number;
}
```

### InferenceStats

Estatisticas de inferencia.

```typescript
interface InferenceStats {
  total_inferences: number;
  successful: number;
  failed: number;
  success_rate: number;      // 0-100
  avg_latency_ms: number;
  total_tokens: number;
  total_cost_usd: number;
}
```

### MetricsTimeSeries

Series temporais para graficos.

```typescript
interface MetricsTimeSeries {
  cpu_utilization: TimeSeriesPoint[];
  memory_utilization: TimeSeriesPoint[];
  gpu_utilization: TimeSeriesPoint[];
  gpu_memory_utilization: TimeSeriesPoint[];
  model_latency: TimeSeriesPoint[];
  invocations: TimeSeriesPoint[];
}

interface TimeSeriesPoint {
  timestamp: string;  // ISO 8601
  value: number;
}
```

### ProcessedDeploymentMetrics

Metricas processadas para UI (Dashboard).

```typescript
interface ProcessedDeploymentMetrics {
  deploymentId: string;
  endpoint: string;

  // Metricas basicas
  totalInvocations: number;
  successRate: number;
  avgLatency: number;
  p95Latency: number;
  avgOverhead: number;
  errorRate: number;

  // Latencia end-to-end
  e2eLatency: number;
  modelLatencyPercentage: number;
  overheadPercentage: number;

  // Throughput
  rps: number;
  rpm: number;
  theoreticalRps: number;
  utilizationPercentage: number;
  headroomPercentage: number;

  // Erros
  errors4xx: number;
  errors5xx: number;
  errorsModel: number;

  // SLO
  apdexScore: number;
  sloStatus: 'pass' | 'fail';

  // Saturacao
  invocationsPerInstance: number;

  // Anomalias
  hasLatencyAnomaly: boolean;
  hasErrorAnomaly: boolean;

  // Dados temporais
  timePoints: {
    timestamp: string;
    invocations: number;
    latency: number;
    overhead: number;
    errors: number;
    isLatencyAnomaly?: boolean;
    isErrorAnomaly?: boolean;
  }[];
}
```

## API Response Types

### DeploymentResponse

Resposta da API de deployment.

```typescript
interface DeploymentResponse {
  deployment_id: string;
  endpoint_name?: string;
  status: string;
  model_id?: string;
  instance_type?: string;
  updated_at?: string;
  tenant_id?: string;
  scaling?: {
    min: number;
    max: number;
  };
}
```

## Toast Types

```typescript
interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  onClose: () => void;
}
```

## Utility Types

### ModelCategory

```typescript
interface ModelCategory {
  id: string;
  label: string;
  count: number;
}
```

## Exports

```typescript
// deploymentTypes.ts
export type {
  DeploymentModel,
  ModelOptions,
  AutoscalingConfig,
  DeploymentData,
  GPUInstanceType,
  ModelCategory,
  MetricsLatest,
  InferenceStats,
  TimeSeriesPoint,
  MetricsTimeSeries,
  DeploymentMetricsData
};
```
