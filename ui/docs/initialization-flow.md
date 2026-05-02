# Application Initialization Flow

This document describes how the application boots, from the HTML entry point to the first rendered view. The architecture follows the [Bulletproof React](https://github.com/alan2207/bulletproof-react) pattern.

## File Structure

```Text
src/
  main.tsx                              # Vite entry point
  app/
    App.tsx                             # Application root component
    provider.tsx                        # Global + authenticated providers
    routes.tsx                          # Route table + lazy imports
    AppLayout.tsx                       # Authenticated shell (sidebar + outlet)
  features/auth/
    AuthGate.tsx                        # Pure auth boundary (render prop)
    LoginPage.tsx                       # Login / sign-up UI
  components/shared/
    FullScreenSpinner.tsx               # Shared loading indicator
  contexts/
    UserContext.tsx                      # User profile, tokens, tenant
    MetricsContext.tsx                   # Analytics metrics data
    WorkspaceContext.tsx                 # Workspace selection & switching
```

## Boot Sequence

```Text
main.tsx
  │  FOUC prevention (sync, before React)
  │  Amplify.configure()
  │  createRoot().render(<App />)
  │
  ▼
app/App.tsx
  │  AppProvider → AuthGate → AuthenticatedProviders → AppRoutes
  │
  ▼
app/provider.tsx — AppProvider
  │  Wraps the entire tree with global providers:
  │    PostHogProvider (conditional — skipped if no API key)
  │    ThemeProvider
  │    BrowserRouter
  │    ErrorBoundary
  │    Toaster
  │
  ▼
features/auth/AuthGate.tsx
  │  Calls getCurrentUser() on mount
  │  Listens for OAuth redirects via Amplify Hub
  │  Identifies user in PostHog
  │  Exposes signOut via render-prop children(signOut)
  │
  ├─ loading  → FullScreenSpinner
  ├─ no user  → LoginPage
  └─ user ok  → children(signOut)
                    │
                    ▼
app/provider.tsx — AuthenticatedProviders
  │  Mounts data-layer providers (only when authenticated):
  │    UserProvider   → profile, tokens, tenant, API key
  │    MetricsProvider → dashboard & analytics data
  │    WorkspaceProvider → personal + org workspaces
  │
  ▼
app/routes.tsx — AppRoutes
  │  PostHogPageView (captures pageview on route change)
  │  Suspense (fallback: FullScreenSpinner)
  │  Route table with lazy-loaded views
  │
  ▼
app/AppLayout.tsx
  │  Reads workspace loading state
  │  If workspace not ready → FullScreenSpinner
  │  Otherwise → Sidebar + OnboardingBar + Outlet + WelcomeModal
  │
  ▼
  Rendered view (CommandCenter, Billing, etc.)
```

## Loading States

A single `FullScreenSpinner` component is reused across all loading phases for a consistent experience:

| Phase | Trigger | What happens |
| ----- | ------- | ------------ |
| **Auth check** | `AuthGate` calls `getCurrentUser()` | Spinner while resolving the Amplify session |
| **Route chunk** | `Suspense` in `AppRoutes` | Spinner while the lazy-imported view JS loads |
| **Workspace init** | `AppLayout` reads `useWorkspace()` | Spinner while workspace data loads |

## Layer Responsibilities

| Layer | File | Single Responsibility |
| ----- | ---- | --------------------- |
| **Entry** | `main.tsx` | FOUC fix, Amplify config, render `<App />` |
| **Root** | `app/App.tsx` | Compose providers → auth gate → routes |
| **Global providers** | `app/provider.tsx` (`AppProvider`) | Theme, analytics, routing, error boundary, toasts |
| **Auth gate** | `features/auth/AuthGate.tsx` | Resolve session, render login or pass through children |
| **Auth providers** | `app/provider.tsx` (`AuthenticatedProviders`) | User, metrics, and workspace contexts |
| **Routes** | `app/routes.tsx` | Lazy route definitions, Suspense boundary |
| **Layout** | `app/AppLayout.tsx` | Sidebar shell, workspace loading guard |

## Key Design Decisions

- **`src/app/` is the orchestration layer** — it wires providers and routes but contains no business logic. This follows the Bulletproof React convention where `app/` is the application shell, not a feature.
- **`features/auth/` owns only auth concerns** — `AuthGate` (session resolution) and `LoginPage` (login UI). Provider composition and routing live in `app/`.
- **AuthGate uses a render prop** — `children(signOut)` lets the parent thread `signOut` into whichever component needs it, without coupling AuthGate to the app structure.
- **PostHog is conditionally mounted** — if `VITE_PUBLIC_POSTHOG_KEY` is not set, the provider is skipped entirely (no empty provider wrapper).
- **One ErrorBoundary** — placed in `AppProvider`, catches everything below including auth, routes, and views.
- **One Suspense boundary** — placed in `AppRoutes`, covers all lazy-loaded route chunks.
- **Theme is applied before React** — an IIFE in `main.tsx` reads `localStorage` and sets the CSS class on `<html>` synchronously to prevent flash of unstyled content.
