# Metricas de Deployment

Sistema de monitoramento de metricas para deployments ativos.

## Arquivos Relacionados

```
src/
├── components/Deployment/
│   └── DeploymentMetricsCharts.tsx       # Componente de graficos
├── services/
│   ├── DeploymentService.ts              # getDeploymentMetrics()
│   └── deploymentMetricsService.ts       # Servico de metricas (Dashboard)
├── hooks/
│   └── useDeploymentMetrics.ts           # Hook para Dashboard
└── types/
    └── deploymentTypes.ts                # Types de metricas
```

## API Endpoint

### GET /v1/deployments/{deployment_id}/metrics

Retorna metricas de utilizacao e inferencia do deployment.

```http
GET /v1/deployments/{deployment_id}/metrics
Authorization: Bearer {accessToken}
```

### Response (Novo Formato)

```json
{
  "deployment_id": "95b38a70-8d54-4a18-84bc-88d4ed0b2832",
  "latest": {
    "cpu_utilization": 0.34,
    "memory_utilization": 4.87,
    "gpu_utilization": 0.0,
    "gpu_memory_utilization": 83.67,
    "model_latency_ms": 0.0,
    "invocations": 0.0
  },
  "inference_stats": {
    "total_inferences": 2,
    "successful": 2,
    "failed": 0,
    "success_rate": 100.0,
    "avg_latency_ms": 2002.45,
    "total_tokens": 250,
    "total_cost_usd": 0.000025
  },
  "time_series": {
    "cpu_utilization": [
      { "timestamp": "2024-01-01T00:00:00Z", "value": 0.34 }
    ],
    "memory_utilization": [...],
    "gpu_utilization": [...],
    "gpu_memory_utilization": [...],
    "model_latency": [...],
    "invocations": [...]
  }
}
```

## Types

### DeploymentMetricsData

```typescript
interface DeploymentMetricsData {
  deployment_id: string;
  latest: MetricsLatest;
  inference_stats: InferenceStats;
  time_series: MetricsTimeSeries;
}
```

### MetricsLatest

```typescript
interface MetricsLatest {
  cpu_utilization: number;      // Percentual (0-100)
  memory_utilization: number;   // Percentual (0-100)
  gpu_utilization: number;      // Percentual (0-100)
  gpu_memory_utilization: number; // Percentual (0-100)
  model_latency_ms: number;     // Latencia em ms
  invocations: number;          // Invocacoes atuais
}
```

### InferenceStats

```typescript
interface InferenceStats {
  total_inferences: number;   // Total de inferencias
  successful: number;         // Inferencias bem sucedidas
  failed: number;             // Inferencias com erro
  success_rate: number;       // Taxa de sucesso (0-100)
  avg_latency_ms: number;     // Latencia media em ms
  total_tokens: number;       // Total de tokens processados
  total_cost_usd: number;     // Custo total em USD
}
```

### MetricsTimeSeries

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

## DeploymentMetricsCharts

Componente React para visualizacao de metricas.

### Props

```typescript
interface DeploymentMetricsChartsProps {
  deploymentId: string;   // ID do deployment
  isVisible: boolean;     // Se esta visivel (para lazy loading)
}
```

### Uso

```tsx
import { DeploymentMetricsCharts } from './DeploymentMetricsCharts';

<DeploymentMetricsCharts
  deploymentId={deployment.deployment_id}
  isVisible={activeTab === 'metrics'}
/>
```

### Funcionalidades

1. **Cards de Estatisticas**
   - CPU Utilization (azul)
   - Memory Utilization (verde)
   - GPU Utilization (roxo)
   - GPU Memory (amarelo)
   - Clicaveis para alternar grafico

2. **Painel de Inferencia**
   - Total de inferencias
   - Taxa de sucesso (com icone)
   - Latencia media
   - Custo total
   - Tokens processados

3. **Grafico de Serie Temporal**
   - Area chart com gradiente
   - Tabs: GPU Mem, Latency, Invocations
   - Tooltip com valores
   - Responsivo

4. **Botao Refresh**
   - Atualiza metricas manualmente

## Integracao no DeploymentDetailsModal

O modal de detalhes inclui abas quando o deployment esta ativo:

```tsx
// DeploymentDetailsModal.tsx

const isActive = deployment.status === 'in_service' || deployment.status === 'active';

// Renderiza tabs se ativo
{isActive && (
  <div className="flex border-b">
    <button onClick={() => setActiveTab('details')}>Details</button>
    <button onClick={() => setActiveTab('metrics')}>Metrics</button>
  </div>
)}

// Conteudo da aba Metrics
{activeTab === 'metrics' && isActive && (
  <DeploymentMetricsCharts
    deploymentId={deployment.deployment_id || deployment.id}
    isVisible={activeTab === 'metrics'}
  />
)}
```

## deploymentMetricsService (Dashboard)

Servico usado pelo Dashboard com suporte a dois formatos de API.

### Formatos Suportados

1. **Novo Formato**: `latest`, `inference_stats`, `time_series`
2. **Formato Legado**: `table` com dados tabulares

### Type Guard

```typescript
function isNewMetricsFormat(metrics: DeploymentMetricsResponse): metrics is NewMetricsResponse {
  return 'latest' in metrics && 'inference_stats' in metrics && 'time_series' in metrics;
}
```

### ProcessedDeploymentMetrics

Interface processada para uso na UI:

```typescript
interface ProcessedDeploymentMetrics {
  deploymentId: string;
  endpoint: string;
  totalInvocations: number;
  successRate: number;
  avgLatency: number;
  p95Latency: number;
  avgOverhead: number;
  errorRate: number;
  e2eLatency: number;
  modelLatencyPercentage: number;
  overheadPercentage: number;
  rps: number;
  rpm: number;
  theoreticalRps: number;
  utilizationPercentage: number;
  headroomPercentage: number;
  errors4xx: number;
  errors5xx: number;
  errorsModel: number;
  apdexScore: number;
  sloStatus: 'pass' | 'fail';
  invocationsPerInstance: number;
  hasLatencyAnomaly: boolean;
  hasErrorAnomaly: boolean;
  timePoints: TimePoint[];
}
```

## Dashboard Metrics

O Dashboard usa `useDeploymentMetrics` hook que:

1. Lista deployments ativos
2. Busca metricas para o deployment selecionado
3. Processa metricas (detecta formato automaticamente)
4. Retorna dados formatados para os componentes

### Uso

```typescript
import { useDeploymentMetrics } from '../hooks/useDeploymentMetrics';

const {
  loading,
  error,
  deployments,
  selectedDeploymentId,
  setSelectedDeploymentId,
  setTimeRange,
  refreshData,
  formatNumber,
  formatPercent,
  formatLatency
} = useDeploymentMetrics();
```

## Cores e Visualizacao

| Metrica | Cor | Gradiente |
|---------|-----|-----------|
| CPU | Azul (#3B82F6) | cpuGradient |
| Memory | Verde (#10B981) | memoryGradient |
| GPU | Roxo (#8B5CF6) | gpuGradient |
| GPU Memory | Amarelo (#F59E0B) | gpuMemGradient |
| Latency | Rosa (#EC4899) | latencyGradient |
| Invocations | Ciano (#06B6D4) | invocationsGradient |
