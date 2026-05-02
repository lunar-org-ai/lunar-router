# Autenticacao e Contexto

Sistema de autenticacao usando AWS Cognito via Amplify.

## Arquivos Relacionados

```
src/
├── contexts/
│   └── UserContext.tsx           # Contexto do usuario
├── hooks/
│   └── useUserService.ts         # Hook de autenticacao
├── services/
│   ├── profileService.ts         # Servico de perfil
│   └── keyService.ts             # Servico de API keys
└── amplify/
    └── auth/resource.ts          # Configuracao Cognito
```

## Configuracao Cognito

```typescript
// amplify/auth/resource.ts
export const auth = referenceAuth({
  userPoolId: "us-east-1_XXXXXXXXX",
  identityPoolId: "us-east-1:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  authRoleArn: "arn:aws:iam::...",
  unauthRoleArn: "arn:aws:iam::...",
  userPoolClientId: "xxxxxxxxxxxxxxxxxxxxxxxxxx"
});
```

## UserContext

Contexto global que fornece dados do usuario autenticado.

### Interface

```typescript
interface UserContextType {
  user: User | null;           // Dados do usuario Cognito
  idToken: string | null;      // Token de identidade
  accessToken: string | null;  // Token de acesso (usar para APIs)
  tenantId: string | null;     // ID do tenant
  loading: boolean;            // Estado de carregamento
  error: string | null;        // Erro se houver
  refetchUser: () => Promise<void>;  // Recarregar dados
  apiKey: KeyResponse | null;  // Chave de API do usuario
}
```

### Uso

```typescript
import { useUser } from '../contexts/UserContext';

function MyComponent() {
  const { user, accessToken, tenantId, loading } = useUser();

  if (loading) return <Loading />;
  if (!accessToken) return <LoginRequired />;

  // Usar accessToken para chamadas de API
  const response = await fetch(url, {
    headers: { 'Authorization': `Bearer ${accessToken}` }
  });
}
```

## Fluxo de Autenticacao

```
1. Usuario faz login via Cognito (Amplify UI)
2. UserProvider inicializa
3. useUserService busca tokens do Cognito
4. fetchProfile busca dados do perfil na API
5. Extrai tenantId do perfil
6. keyService cria/recupera API key
7. Contexto disponivel para toda aplicacao
```

## profileService

Servico para buscar perfil do usuario.

### Endpoint

```http
GET /v1/profile
Authorization: Bearer {accessToken}
```

### Response

```json
{
  "profile": {
    "id": "user-uuid",
    "email": "user@example.com",
    "tenant_id": "tenant-uuid",
    "tenant": {
      "id": "tenant-uuid",
      "name": "Tenant Name"
    }
  }
}
```

### Uso

```typescript
import { useProfileService } from '../services/profileService';

const { profile, error, fetchProfile } = useProfileService();

// Buscar perfil
const profileData = await fetchProfile(accessToken);
const tenantId = profileData?.tenant_id || profileData?.tenant?.id;
```

## keyService

Servico para gerenciar API keys.

### Criar Key

```typescript
const { createKey } = keyService(accessToken);
const key = await createKey("key-name");
```

### Response

```typescript
interface KeyResponse {
  id: string;
  name: string;
  key: string;          // A chave de API
  created_at: string;
  last_used?: string;
}
```

## Tokens

### idToken
- Token JWT de identidade
- Contem claims do usuario (email, sub, etc)
- Usado para identificacao

### accessToken
- Token JWT de acesso
- **USAR ESTE PARA CHAMADAS DE API**
- Contem permissoes e scopes

### Exemplo de Header

```typescript
const headers = {
  'Authorization': `Bearer ${accessToken}`,
  'Content-Type': 'application/json'
};
```

## Tenant ID

O `tenantId` e extraido do perfil do usuario e usado para:

- Filtrar dados por tenant
- Autorizacao em endpoints
- Isolamento multi-tenant

### Extracao

```typescript
// O perfil pode vir em diferentes formatos
const tenantId = profile?.tenant_id || profile?.tenant?.id || null;
```

## Tratamento de Erros

### Token expirado
- Amplify renova automaticamente
- Se falhar, redireciona para login

### Perfil nao encontrado
```
"Tenant ID nao encontrado"
```

### Erro de autenticacao (403)
```
"Authentication failed: Please check your credentials or permissions"
```

## Provider Setup

```tsx
// App.tsx ou main.tsx
import { UserProvider } from './contexts/UserContext';

function App() {
  return (
    <UserProvider>
      <Router>
        <Routes />
      </Router>
    </UserProvider>
  );
}
```
