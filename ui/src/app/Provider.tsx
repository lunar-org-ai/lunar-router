import type { ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import posthog from 'posthog-js';
import { PostHogProvider } from 'posthog-js/react';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';

import { ThemeProvider } from '@/components/ThemeProvider';
import ErrorFallback from '@/components/shared/ErrorFallback';

const POSTHOG_KEY = import.meta.env.VITE_PUBLIC_POSTHOG_KEY as string | undefined;
const POSTHOG_HOST =
  (import.meta.env.VITE_PUBLIC_POSTHOG_HOST as string | undefined) ?? 'https://us.i.posthog.com';

const posthogClient = (() => {
  if (!POSTHOG_KEY || typeof window === 'undefined') return null;
  posthog.init(POSTHOG_KEY, {
    api_host: POSTHOG_HOST,
    person_profiles: 'identified_only',
    capture_pageview: false,
    capture_pageleave: true,
    autocapture: true,
    enable_recording_console_log: true,
    session_recording: {
      maskAllInputs: true,
      maskTextSelector: '[data-private]',
    },
    disable_session_recording: false,
  });
  posthog.register({ app: 'console' });
  return posthog;
})();

export function AppProvider({ children }: { children: ReactNode }) {
  const tree = (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <TooltipProvider>
        <BrowserRouter>
          <ErrorBoundary FallbackComponent={ErrorFallback}>
            <Toaster position="bottom-right" />
            {children}
          </ErrorBoundary>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  );
  return posthogClient ? <PostHogProvider client={posthogClient}>{tree}</PostHogProvider> : tree;
}
