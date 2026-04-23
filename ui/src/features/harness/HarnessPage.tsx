/**
 * Harness page — three tabs over the same underlying ledger:
 *
 *   Objectives  what the system is optimizing for
 *   Ledger      what the system did (newest first) + chain drill-down
 *   Agents      browse + manually trigger a single agent (existing UX)
 *
 * The Objectives + Ledger tabs are the read side of Phase 1 of the
 * harness redesign: they exist so a user can answer "why did X move?"
 * from the ledger alone, without opening SQLite or writing queries.
 */

import { PageHeader } from '@/components/shared/PageHeader';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AgentsTab } from './tabs/AgentsTab';
import { LedgerTab } from './tabs/LedgerTab';
import { ObjectivesTab } from './tabs/ObjectivesTab';

export default function HarnessPage() {
  return (
    <div>
      <PageHeader title="Harness" />

      <div className="mx-auto max-w-6xl px-6 py-6">
        <Tabs defaultValue="objectives" className="space-y-4">
          <TabsList>
            <TabsTrigger value="objectives">Objectives</TabsTrigger>
            <TabsTrigger value="ledger">Ledger</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
          </TabsList>

          <TabsContent value="objectives">
            <ObjectivesTab />
          </TabsContent>

          <TabsContent value="ledger">
            <LedgerTab />
          </TabsContent>

          <TabsContent value="agents">
            <AgentsTab />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
