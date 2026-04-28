/**
 * Harness page — four tabs over the same underlying ledger:
 *
 *   Objectives  what the system is optimizing for
 *   Ledger      what the system did (newest first) + chain drill-down
 *   Proposals   pending writes awaiting approval (Phase 1.5)
 *   Agents      browse + manually trigger a single agent (existing UX)
 *
 * Objectives + Ledger are the read side; Proposals is the operator
 * surface for the critic-gated write paths.
 *
 * Two orchestrators are accepted: the autonomous loop (Anthropic API,
 * trigger engine drives recipes itself) and Claude Code (operator-in-
 * the-loop via MCP). Either or both can be on — both share the same
 * provider key + critic gate. The status strip above the tabs reflects
 * whichever prereqs still need attention; clicking it opens the full
 * SetupGuide walkthrough (also rendered inline as the empty state of
 * Proposals).
 */

import { useCallback, useEffect, useState } from 'react';
import { ChevronRight, Sparkles } from 'lucide-react';
import { PageHeader } from '@/components/shared/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  useHarnessService,
  type HarnessSetupStatus,
} from '@/services/harnessService';
import { SetupGuide } from './components/SetupGuide';
import { AgentsTab } from './tabs/AgentsTab';
import { LedgerTab } from './tabs/LedgerTab';
import { ObjectivesTab } from './tabs/ObjectivesTab';
import { ProposalsTab } from './tabs/ProposalsTab';

export default function HarnessPage() {
  const service = useHarnessService();
  const [status, setStatus] = useState<HarnessSetupStatus | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);

  const refreshStatus = useCallback(() => {
    service.getSetupStatus().then(setStatus);
  }, [service]);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  const ready = status?.critic.ready ?? false;

  return (
    <div>
      <PageHeader title="Harness" />

      <div className="mx-auto max-w-6xl px-6 py-6 space-y-4">
        <Card
          role="button"
          onClick={() => setSetupOpen(true)}
          className={`cursor-pointer transition-colors ${
            ready
              ? 'border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10'
              : 'border-amber-500/30 bg-amber-500/5 hover:bg-amber-500/10'
          }`}
        >
          <CardContent className="flex items-center gap-3 p-3 text-sm">
            <Sparkles
              className={`size-4 ${
                ready ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'
              }`}
            />
            <div className="flex-1">
              <div className="font-medium">
                {ready
                  ? 'Harness ready — autonomous loop and Claude Code can drive'
                  : 'Add a provider key to start the harness'}
              </div>
              <div className="text-xs text-muted-foreground">
                {ready
                  ? `Critic agent on ${status?.critic.model} — both orchestrators share the same gate and ledger.`
                  : `The critic agent needs a ${status?.critic.provider ?? 'provider'} key. Once it's set, the autonomous loop runs and Claude Code can connect via MCP.`}
              </div>
            </div>
            <ChevronRight className="size-4 text-muted-foreground" />
          </CardContent>
        </Card>

        <Tabs defaultValue={ready ? 'objectives' : 'proposals'} className="space-y-4">
          <TabsList>
            <TabsTrigger value="objectives">Objectives</TabsTrigger>
            <TabsTrigger value="ledger">Ledger</TabsTrigger>
            <TabsTrigger value="proposals">Proposals</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
          </TabsList>

          <TabsContent value="objectives">
            <ObjectivesTab />
          </TabsContent>

          <TabsContent value="ledger">
            <LedgerTab />
          </TabsContent>

          <TabsContent value="proposals">
            <ProposalsTab onSetupChange={refreshStatus} />
          </TabsContent>

          <TabsContent value="agents">
            <AgentsTab />
          </TabsContent>
        </Tabs>
      </div>

      <Dialog open={setupOpen} onOpenChange={setSetupOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Set up the harness</DialogTitle>
          </DialogHeader>
          <SetupGuide onConfigured={refreshStatus} />
        </DialogContent>
      </Dialog>
    </div>
  );
}
