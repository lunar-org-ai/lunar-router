/**
 * Harness page — three tabs over the same underlying ledger:
 *
 *   Objectives  what the system is optimizing for
 *   Ledger      what the system did (newest first) + chain drill-down
 *   Proposals   pending writes awaiting approval (Phase 1.5)
 *
 * Two orchestrators share the same provider key + critic gate:
 *   - autonomous loop (Anthropic API, trigger engine drives recipes)
 *   - Claude Code (operator-in-the-loop via MCP)
 *
 * The status alert above the tabs reflects whichever prereqs still need
 * attention; the "Configure" button opens the full SetupGuide walkthrough.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, KeyRound, Settings2 } from 'lucide-react';
import { PageHeader } from '@/components/shared/PageHeader';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  useHarnessService,
  type HarnessSetupStatus,
} from '@/services/harnessService';
import { SetupGuide } from './components/SetupGuide';
import { LedgerTab } from './tabs/LedgerTab';
import { ObjectivesTab } from './tabs/ObjectivesTab';
import { ProposalsTab } from './tabs/ProposalsTab';

export default function HarnessPage() {
  const service = useHarnessService();
  const [status, setStatus] = useState<HarnessSetupStatus | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState<number | null>(null);

  const refreshStatus = useCallback(() => {
    service.getSetupStatus().then(setStatus);
  }, [service]);

  const refreshPending = useCallback(() => {
    service
      .listProposals({ status: 'pending', limit: 100 })
      .then((list) => setPendingCount(list.length))
      .catch(() => setPendingCount(null));
  }, [service]);

  useEffect(() => {
    refreshStatus();
    refreshPending();
  }, [refreshStatus, refreshPending]);

  const ready = status?.critic.ready ?? false;
  const criticModel = status?.critic.model ?? undefined;
  const criticProvider = status?.critic.provider ?? 'provider';

  const subtitle = useMemo(() => {
    if (!status) return 'Loading harness…';
    if (ready)
      return `Critic on ${criticModel} · autonomous loop running · Claude Code can attach via MCP`;
    return `Add a ${criticProvider} key to bring the harness online`;
  }, [status, ready, criticModel, criticProvider]);

  const defaultTab = ready ? 'objectives' : 'proposals';

  return (
    <div>
      <PageHeader title="Harness" />

      <div className="mx-auto max-w-6xl px-6 py-6 space-y-4">
        <p className="text-sm text-muted-foreground">{subtitle}</p>

        <StatusAlert
          ready={ready}
          provider={criticProvider}
          model={criticModel}
          onConfigure={() => setSetupOpen(true)}
        />

        <Tabs defaultValue={defaultTab} className="space-y-4">
          <TabsList>
            <TabsTrigger value="objectives">Objectives</TabsTrigger>
            <TabsTrigger value="ledger">Ledger</TabsTrigger>
            <TabsTrigger value="proposals" className="gap-2">
              Proposals
              {pendingCount !== null && pendingCount > 0 && (
                <Badge
                  variant="secondary"
                  className="h-4 min-w-4 px-1 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-400 font-mono text-[10px] tabular-nums"
                >
                  {pendingCount}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="objectives">
            <ObjectivesTab />
          </TabsContent>

          <TabsContent value="ledger">
            <LedgerTab />
          </TabsContent>

          <TabsContent value="proposals">
            <ProposalsTab
              onSetupChange={refreshStatus}
              onProposalsChange={refreshPending}
            />
          </TabsContent>
        </Tabs>
      </div>

      <Dialog open={setupOpen} onOpenChange={setSetupOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Set up the harness</DialogTitle>
            <DialogDescription>
              Wire a provider key, then drive the harness through the autonomous
              loop, Claude Code (MCP), or both.
            </DialogDescription>
          </DialogHeader>
          <SetupGuide onConfigured={refreshStatus} />
        </DialogContent>
      </Dialog>
    </div>
  );
}

function StatusAlert({
  ready,
  provider,
  model,
  onConfigure,
}: {
  ready: boolean;
  provider: string;
  model?: string;
  onConfigure: () => void;
}) {
  if (ready) {
    return (
      <Alert className="border-emerald-500/30 bg-emerald-500/5">
        <CheckCircle2 className="text-emerald-600 dark:text-emerald-400" />
        <AlertTitle className="flex items-center gap-2">
          Harness is live
          <Badge
            variant="outline"
            className="font-mono text-[10px] bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
          >
            {model}
          </Badge>
        </AlertTitle>
        <AlertDescription className="flex items-center justify-between gap-4">
          <span>
            Both orchestrators share the same critic gate and ledger. Approve
            proposals on the Proposals tab or let the loop run.
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={onConfigure}
            className="shrink-0"
          >
            <Settings2 />
            Reconfigure
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Alert className="border-amber-500/40 bg-amber-500/5">
      <KeyRound className="text-amber-600 dark:text-amber-400" />
      <AlertTitle>Add a provider key to start the harness</AlertTitle>
      <AlertDescription className="flex items-center justify-between gap-4">
        <span>
          The critic agent needs a {provider} key. Once it's set, the
          autonomous loop runs and Claude Code can connect via MCP.
        </span>
        <Button size="sm" onClick={onConfigure} className="shrink-0">
          <Settings2 />
          Configure
        </Button>
      </AlertDescription>
    </Alert>
  );
}
