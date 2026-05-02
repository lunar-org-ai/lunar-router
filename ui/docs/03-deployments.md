# Deployments

Sistema de deploy de modelos LLM em instancias GPU via AWS SageMaker.

## Arquivos Relacionados

```
src/
├── views/DeploymentsTab.tsx              # View principal
├── components/Deployment/
│   ├── DeploymentCard.tsx                # Card de deployment
│   ├── DeploymentDetailsModal.tsx        # Modal de detalhes
│   ├── DeploymentMetricsCharts.tsx       # Graficos de metricas
│   ├── DeploymentModal.tsx               # Modal de criacao
│   ├── HowToUseModal.tsx                 # Modal de instrucoes
│   ├── ModelSpecsModal.tsx               # Specs do modelo
│   └── ExploreModelsSection.tsx          # Secao de explorar modelos
├── services/DeploymentService.ts         # Servico de API
├── hooks/useDeployments.ts               # Hook de deployments
├── constants/deployment.ts               # Constantes (modelos, instancias)
└── types/deploymentTypes.ts              # Types/Interfaces
```

## API Endpoints

### Listar Deployments

```http
GET /v1/deployments
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "deployments": [
    {
      "deployment_id": "uuid",
      "endpoint_name": "ten-xxx-mdl-modelname-xxx",
      "model_id": "Llama-3.2-1B",
      "instance_type": "ml.g5.xlarge",
      "status": "in_service",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Criar Deployment

```http
POST /v1/deployments
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "model_id": "Llama-3.2-1B",
  "instance_type": "ml.g5.xlarge",
  "config": {
    "vllm_args": "--dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.92"
  }
}
```

**Response:**
```json
{
  "deployment_id": "uuid",
  "status": "creating"
}
```

### Deletar Deployment

```http
DELETE /v1/deployments/{deployment_id}
Authorization: Bearer {accessToken}
```

### Status do Deployment

```http
GET /v1/deployments/{deployment_id}/status
Authorization: Bearer {accessToken}
```

## Status do Deployment

| Status | Descricao | Cor |
|--------|-----------|-----|
| `creating` | Criando endpoint SageMaker | Amarelo |
| `starting` | Iniciando modelo | Amarelo |
| `pending` | Aguardando recursos | Amarelo |
| `updating` | Atualizando configuracao | Amarelo |
| `in_service` | Ativo e pronto | Verde |
| `active` | Ativo (alias) | Verde |
| `failed` | Erro na criacao | Vermelho |
| `stopped` | Parado | Vermelho |
| `deleting` | Sendo deletado | Laranja |

## DeploymentService

Servico para interacao com a API de deployments.

```typescript
import { useDeploymentService } from '../services/DeploymentService';

const {
  createDeployment,
  getDeploymentStatus,
  listDeployments,
  deleteDeployment,
  getDeploymentMetrics  // NOVO - metricas
} = useDeploymentService();
```

### Metodos

#### createDeployment
```typescript
createDeployment(
  idToken: string,
  payload: { model_id: string; instance_type: string },
  data: Partial<DeploymentData>
): Promise<DeploymentResponse>
```

#### listDeployments
```typescript
listDeployments(
  accessToken: string,
  statuses?: string[]
): Promise<DeploymentResponse[]>
```

#### deleteDeployment
```typescript
deleteDeployment(
  accessToken: string,
  id: string
): Promise<void>
```

#### getDeploymentMetrics
```typescript
getDeploymentMetrics(
  accessToken: string,
  deploymentId: string
): Promise<DeploymentMetricsData>
```

## VLLM Args

Argumentos de configuracao do VLLM gerados automaticamente:

| Argumento | Descricao | Default |
|-----------|-----------|---------|
| `--port` | Porta do servidor | 8080 |
| `--dtype` | Tipo de dados | bfloat16 |
| `--tensor-parallel-size` | GPUs paralelas | Auto (baseado na instancia) |
| `--gpu-memory-utilization` | Uso de memoria GPU | 0.92 |
| `--max-model-len` | Contexto maximo | Configuravel |
| `--max-num-seqs` | Sequences em batch | Configuravel |

## DeploymentCard

Componente que exibe informacoes do deployment:

- Nome do modelo
- Data de criacao
- Status com indicador visual
- Model ID
- Instance type
- Botoes: View Details, How to Use, Delete

## DeploymentDetailsModal

Modal com detalhes completos e abas:

### Aba Details
- Status atual
- Data de criacao
- Deployment ID (copiavel)
- Instance type
- Model name

### Aba Metrics (NOVO)
- Disponivel apenas para deployments ativos (`in_service`/`active`)
- Mostra graficos de CPU, memoria, GPU
- Estatisticas de inferencia
- Ver [Metricas de Deployment](./04-deployment-metrics.md)

## Fluxo de Criacao

```
1. Usuario clica "Create Deployment"
2. Seleciona modelo e instancia
3. Configura opcoes do VLLM (opcional)
4. Sistema envia POST /v1/deployments
5. Status muda: creating -> starting -> in_service
6. Toast notifica quando pronto
```

## Tratamento de Erros

### Deployment duplicado (409)
```
"Voce nao pode ter dois modelos identicos com a mesma configuracao e maquina."
```

### Erro generico
```
"Falha ao criar deployment. Por favor, tente novamente."
```
