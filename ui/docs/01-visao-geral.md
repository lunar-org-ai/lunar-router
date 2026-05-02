# Visao Geral

O PureAI Console e uma plataforma para gerenciamento de modelos de IA, incluindo:

- **Deployments**: Deploy de modelos LLM em GPUs (SageMaker)
- **Playground**: Teste de modelos de diversos provedores
- **Fine Tuning**: Ajuste fino de modelos
- **Analytics**: Dashboard com metricas de uso e custos
- **Integracoes**: Configuracao de API keys de provedores

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                     PureAI Console (React)                   │
├─────────────────────────────────────────────────────────────┤
│  Views: Dashboard | Deployments | Playground | FineTuning   │
├─────────────────────────────────────────────────────────────┤
│  Contexts: UserContext | AuthContext                         │
├─────────────────────────────────────────────────────────────┤
│  Services: DeploymentService | MetricsService | KeyService   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AWS API Gateway                           │
├─────────────────────────────────────────────────────────────┤
│  /v1/deployments | /v1/profile | /v1/stats | /v1/metrics    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Lambda  │   │ DynamoDB │   │ SageMaker│
        └──────────┘   └──────────┘   └──────────┘
```

## Fluxo de Autenticacao

1. Usuario faz login via Cognito
2. Frontend recebe `idToken` e `accessToken`
3. `accessToken` e usado para chamadas de API
4. `UserContext` armazena dados do usuario e `tenantId`

## Modelos Suportados

| Modelo | ID | Instance Recomendada |
|--------|-----|---------------------|
| Llama 4 Scout 17B | `Llama-4-Scout-17B-16E-Instruct` | ml.g5.4xlarge |
| DeepSeek R1 8B | `DeepSeek-R1-Distill-Llama-8B` | ml.g5.2xlarge |
| LLaMA 3.2 1B | `Llama-3.2-1B` | ml.g5.xlarge |
| Qwen3 30B | `Qwen3-30B-A3B-Instruct-2507` | ml.g5.4xlarge |
| Qwen3 4B Instruct | `Qwen3-4B-Instruct-2507` | ml.g5.4xlarge |
| Qwen3 4B Thinking | `Qwen3-4B-Thinking-2507` | ml.g5.4xlarge |
| DeepSeek R1 Qwen 7B | `DeepSeek-R1-Distill-Qwen-7B` | ml.g5.4xlarge |
| GPT OSS 20B | `gpt-oss-20b` | ml.g5.2xlarge |
| Gemma 3 4B | `gemma-3-4b-it` | ml.g5.4xlarge |

## Tipos de Instancia GPU

| Instance | GPUs | Memoria | Descricao |
|----------|------|---------|-----------|
| ml.g4dn.xlarge | 1x T4 | 16GB | Economico para modelos pequenos |
| ml.g5.xlarge | 1x A10G | 24GB | Balanceado |
| ml.g5.2xlarge | 1x A10G | 24GB | Mais CPU/memoria |
| ml.g5.4xlarge | 1x A10G | 24GB | Alto desempenho |
| ml.g5.12xlarge | 4x A10G | 96GB | Multi-GPU para modelos grandes |
