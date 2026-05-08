/**
 * Router — code-based TanStack Router config for the entire UI.
 *
 * Tree shape:
 *   __root (sidebar + topbar + AgentSheet)
 *     ├── /                       Evolution
 *     ├── /lesson/$id             LessonDetail   (search: tab)
 *     ├── /review                 Review
 *     ├── /versions               Versions
 *     ├── /talk                   TalkToAgent
 *     ├── /policies               Policies
 *     ├── /technical              Outlet
 *     │   ├── /technical/traces   Traces
 *     │   ├── /technical/evals    EvalSuites
 *     │   ├── /technical/router   RouterConfig
 *     │   └── /technical/datasets Datasets
 *     └── /lab                    Outlet (chat library experiments)
 *         ├── /lab/assistant-ui   ChatAssistantUi
 *         └── /lab/copilot-kit    ChatCopilotKit
 *
 * `view` (simple | technical, on root) and `tab` (lesson tab, on /lesson/$id)
 * are URL search params with loose `?:` validators so cross-route navigation
 * typechecks without requiring every Link to redeclare them.
 *
 * Unknown URLs → redirect to /.
 *
 * Scroll-reset on navigation lives in main.tsx via router.subscribe('onResolved').
 */

import { createRootRoute, createRoute, createRouter, Outlet, redirect } from '@tanstack/react-router';
import { RootLayout } from './routes/__root';
import { Evolution } from './screens/Evolution';
import { LessonDetail } from './screens/LessonDetail';
import { Review } from './screens/Review';
import { Versions } from './screens/Versions';
import { TalkToAgent } from './screens/TalkToAgent';
import { Policies } from './screens/Policies';
import { Traces, EvalSuites, RouterConfig, Datasets } from './screens/Technical';
import { ChatAssistantUi } from './screens/chat-assistant-ui';
import { ChatCopilotKit } from './screens/chat-copilot-kit';

export type View = 'simple' | 'technical';
type LessonTab = 'story' | 'traces' | 'evals' | 'diff' | 'decision';

const LESSON_TABS: readonly LessonTab[] = ['story', 'traces', 'evals', 'diff', 'decision'];

const rootRoute = createRootRoute({
  component: RootLayout,
  validateSearch: (search: Record<string, unknown>): { view?: View } => {
    const v = search.view;
    return v === 'technical' || v === 'simple' ? { view: v } : {};
  },
  notFoundComponent: () => {
    throw redirect({ to: '/' });
  },
});

const evolutionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Evolution,
});

const lessonRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/lesson/$id',
  validateSearch: (search: Record<string, unknown>): { tab?: LessonTab } => {
    const t = search.tab;
    return LESSON_TABS.includes(t as LessonTab) ? { tab: t as LessonTab } : {};
  },
  component: LessonDetail,
});

const reviewRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/review',
  component: Review,
});

const versionsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/versions',
  component: Versions,
});

const talkRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/talk',
  component: TalkToAgent,
});

const policiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/policies',
  component: Policies,
});

const technicalLayoutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/technical',
  component: Outlet,
});

const tracesRoute = createRoute({
  getParentRoute: () => technicalLayoutRoute,
  path: 'traces',
  component: Traces,
});

const evalsRoute = createRoute({
  getParentRoute: () => technicalLayoutRoute,
  path: 'evals',
  component: EvalSuites,
});

const routerConfigRoute = createRoute({
  getParentRoute: () => technicalLayoutRoute,
  path: 'router',
  component: RouterConfig,
});

const datasetsRoute = createRoute({
  getParentRoute: () => technicalLayoutRoute,
  path: 'datasets',
  component: Datasets,
});

const labLayoutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/lab',
  component: Outlet,
});

const labAssistantUiRoute = createRoute({
  getParentRoute: () => labLayoutRoute,
  path: 'assistant-ui',
  component: ChatAssistantUi,
});

const labCopilotKitRoute = createRoute({
  getParentRoute: () => labLayoutRoute,
  path: 'copilot-kit',
  component: ChatCopilotKit,
});

const routeTree = rootRoute.addChildren([
  evolutionRoute,
  lessonRoute,
  reviewRoute,
  versionsRoute,
  talkRoute,
  policiesRoute,
  technicalLayoutRoute.addChildren([tracesRoute, evalsRoute, routerConfigRoute, datasetsRoute]),
  labLayoutRoute.addChildren([labAssistantUiRoute, labCopilotKitRoute]),
]);

export const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

/**
 * Pass-through search reducer — preserves all current search params on
 * navigation. Pass to <Link search={preserveSearch} /> or
 * navigate({ search: preserveSearch }).
 */
export const preserveSearch = (prev: Record<string, unknown>) => prev;
