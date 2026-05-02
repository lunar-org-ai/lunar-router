# PureAI Console - Documentacao

Esta documentacao descreve a arquitetura, APIs e funcionalidades do PureAI Console.

## Indice

1. [Visao Geral](./01-visao-geral.md)
2. [Autenticacao e Contexto](./02-autenticacao.md)
3. [Deployments](./03-deployments.md)
4. [Metricas de Deployment](./04-deployment-metrics.md)
5. [Dashboard e Analytics](./05-dashboard.md)
6. [Playground](./06-playground.md)
7. [Fine Tuning](./07-fine-tuning.md)
8. [Integracoes](./08-integracoes.md)
9. [Services](./09-services.md)
10. [Types e Interfaces](./10-types.md)

## Stack Tecnologico

- **Frontend**: React 19, TypeScript, Vite
- **UI**: Tailwind CSS 4, Lucide Icons, Framer Motion
- **Charts**: Recharts 3
- **Auth**: AWS Amplify 6, Amazon Cognito
- **Backend**: AWS API Gateway, Lambda, DynamoDB, SageMaker
- **Pagamentos**: Stripe

## Estrutura do Projeto

```
src/
├── assets/          # Icones e imagens
├── components/      # Componentes React
│   ├── Dashboard/   # Componentes do dashboard
│   ├── Deployment/  # Componentes de deployment
│   ├── FineTuning/  # Componentes de fine tuning
│   └── UI/          # Componentes reutilizaveis
├── constants/       # Constantes e configuracoes
├── contexts/        # React Contexts
├── hooks/           # Custom hooks
├── services/        # Servicos de API
├── types/           # TypeScript types/interfaces
├── utils/           # Funcoes utilitarias
└── views/           # Paginas/Views principais
```

## Endpoints de API

Base URL: `https://qqf2ajs1b7.execute-api.us-east-1.amazonaws.com`

| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/v1/deployments` | GET | Lista deployments |
| `/v1/deployments` | POST | Cria deployment |
| `/v1/deployments/{id}` | DELETE | Deleta deployment |
| `/v1/deployments/{id}/status` | GET | Status do deployment |
| `/v1/deployments/{id}/metrics` | GET | Metricas do deployment |
| `/v1/profile` | GET | Perfil do usuario |
| `/v1/stats` | GET | Estatisticas do dashboard |
