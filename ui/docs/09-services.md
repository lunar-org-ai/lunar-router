# Services

Camada de servicos para comunicacao com APIs backend.

## Estrutura

```
src/services/
├── DeploymentService.ts          # Deployments CRUD + Metricas
├── deploymentMetricsService.ts   # Metricas processadas (Dashboard)
├── dashboardMetricsService.ts    # Metricas do dashboard
├── profileService.ts             # Perfil do usuario
├── keyService.ts                 # API keys
├── routerAPIService.ts           # Router API (Playground)
└── fineTuningService.ts          # Fine tuning
```

## Base URL

```typescript
const API_BASE = "https://qqf2ajs1b7.execute-api.us-east-1.amazonaws.com";
```

## DeploymentService

Servico principal para operacoes de deployment.

```typescript
import { useDeploymentService } from '../services/DeploymentService';

const {
  createDeployment,
  getDeploymentStatus,
  listDeployments,
  deleteDeployment,
  getDeploymentMetrics
} = useDeploymentService();
```

### Metodos

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| `createDeployment` | POST /v1/deployments | Cria novo deployment |
| `listDeployments` | GET /v1/deployments | Lista deployments |
| `getDeploymentStatus` | GET /v1/deployments/{id}/status | Status do deployment |
| `deleteDeployment` | DELETE /v1/deployments/{id} | Deleta deployment |
| `getDeploymentMetrics` | GET /v1/deployments/{id}/metrics | Metricas do deployment |

## deploymentMetricsService

Servico para metricas do Dashboard com processamento avancado.

```typescript
import { useDeploymentMetricsService } from '../services/deploymentMetricsService';

const { getDeploymentMetrics, processMetrics } = useDeploymentMetricsService();
```

### Formatos Suportados

**Novo Formato:**
```json
{
  "deployment_id": "...",
  "latest": { "cpu_utilization": 0.34, ... },
  "inference_stats": { "total_inferences": 2, ... },
  "time_series": { "cpu_utilization": [...], ... }
}
```

**Formato Legado:**
```json
{
  "deployment_id": "...",
  "table": [{ "timestamp": "...", "invocations": 10, ... }],
  "series": [...]
}
```

### Processamento

O `processMetrics` calcula automaticamente:
- Total de invocacoes
- Taxa de sucesso
- Latencia media e P95
- Apdex score
- Deteccao de anomalias
- RPS/RPM
- Breakdown de erros

## dashboardMetricsService

Metricas gerais do dashboard.

```typescript
import { useDashboardMetrics } from '../hooks/useDashboardMetrics';

const {
  metrics,
  loading,
  error,
  timeRange,
  setTimeRange
} = useDashboardMetrics();
```

### Endpoint

```http
GET /v1/stats?days=7
Authorization: Bearer {accessToken}
x-tenant-id: {tenantId}
```

## profileService

Gerenciamento do perfil do usuario.

```typescript
import { useProfileService } from '../services/profileService';

const { profile, error, fetchProfile } = useProfileService();
```

### Resposta

```typescript
interface ProfileData {
  id: string;
  email: string;
  tenant_id?: string;
  tenant?: {
    id: string;
    name: string;
  };
}
```

## keyService

Gerenciamento de API keys.

```typescript
import { keyService } from '../services/keyService';

const { createKey, listKeys, deleteKey } = keyService(accessToken);
```

### Metodos

```typescript
// Criar chave
const key = await createKey("my-key-name");

// Listar chaves
const keys = await listKeys();

// Deletar chave
await deleteKey(keyId);
```

## routerAPIService

Servico para chamadas ao Router (Playground).

```typescript
import { useRouterAPI } from '../services/routerAPIService';

const { callProvider, callInfer, isProviderConfigured } = useRouterAPI();
```

### Chamada Direta

```typescript
// Chamada para provedor especifico
const response = await callProvider({
  provider: 'openai',
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }],
  temperature: 0.7
});
```

### Chamada via Router

```typescript
// Inferencia via router (selecao automatica)
const response = await callInfer({
  messages: [{ role: 'user', content: 'Hello' }],
  profile: 'QUALITY'
});
```

## Padrao de Autenticacao

Todos os servicos seguem o padrao:

```typescript
const response = await fetch(url, {
  method: 'GET',  // ou POST, DELETE
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json'
  }
});
```

## Tratamento de Erros

```typescript
try {
  const response = await fetch(url, options);

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');

    if (response.status === 403) {
      throw new Error('Authentication failed');
    }
    if (response.status === 409) {
      throw new Error('already_exists: Resource already exists');
    }

    throw new Error(`Request failed: ${response.status}`);
  }

  return await response.json();
} catch (error) {
  console.error('Service error:', error);
  throw error;
}
```

## CORS

Configuracao para chamadas cross-origin:

```typescript
const response = await fetch(url, {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  mode: 'cors',
  credentials: 'omit'  // Evita problemas de CORS com cookies
});
```
