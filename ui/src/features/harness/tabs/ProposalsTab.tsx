/**
 * Proposals tab — operator surface for the harness write paths.
 *
 * Each row is a `type='proposal'` ledger entry; status is derived
 * server-side from descendant decision/action rows. Click a row to
 * open the shared ChainDrawer (so the same drill-down view as the
 * Ledger tab is reused). Approve/Reject buttons trigger the gated
 * write endpoints; on a successful approve the critic re-runs (cache
 * usually hits) and either resolves the proposal or flips its status
 * to `rejected_by_critic`.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Filter, XCircle } from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  useHarnessService,
  type HarnessSetupStatus,
  type Proposal,
  type ProposalStatus,
} from '@/services/harnessService';
import { ChainDrawer } from '../components/ChainDrawer';
import { ProposalCard, StatusBadge, formatUsd } from '../components/ProposalCard';
import { SetupGuide } from '../components/SetupGuide';
import { formatTs } from '../components/ledgerFormat';

const STATUS_OPTIONS: { value: ProposalStatus | ''; label: string }[] = [
  { value: 'pending', label: 'Pending' },
  { value: '', label: 'All' },
  { value: 'approved', label: 'Approved' },
  { value: 'executed', label: 'Executed' },
  { value: 'failed', label: 'Failed' },
  { value: 'rejected', label: 'Rejected' },
  { value: 'rejected_by_critic', label: 'Rejected by critic' },
];

type ConfirmAction = 'approve' | 'reject' | null;

interface ProposalsTabProps {
  onSetupChange?: () => void;
}

export function ProposalsTab({ onSetupChange }: ProposalsTabProps = {}) {
  const service = useHarnessService();
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<ProposalStatus | ''>('pending');
  const [objectiveFilter, setObjectiveFilter] = useState('');
  const [drawerId, setDrawerId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [setup, setSetup] = useState<HarnessSetupStatus | null>(null);

  useEffect(() => {
    service.getSetupStatus().then(setSetup);
  }, [service]);

  const handleSetupConfigured = useCallback(() => {
    service.getSetupStatus().then(setSetup);
    onSetupChange?.();
  }, [service, onSetupChange]);

  const [confirm, setConfirm] = useState<{
    action: ConfirmAction;
    proposal: Proposal | null;
    reason: string;
    busy: boolean;
  }>({ action: null, proposal: null, reason: '', busy: false });

  const load = useCallback(() => {
    setLoading(true);
    service
      .listProposals({
        status: statusFilter,
        objective_id: objectiveFilter || undefined,
        limit: 100,
      })
      .then((list) => {
        setProposals(list);
        setLoading(false);
      });
  }, [service, statusFilter, objectiveFilter]);

  useEffect(() => {
    load();
  }, [load]);

  const openConfirm = (action: ConfirmAction, proposal: Proposal) => {
    setConfirm({ action, proposal, reason: '', busy: false });
  };
  const closeConfirm = () => {
    if (confirm.busy) return;
    setConfirm({ action: null, proposal: null, reason: '', busy: false });
  };

  const submitConfirm = async () => {
    if (!confirm.proposal || !confirm.action) return;
    setConfirm((c) => ({ ...c, busy: true }));
    const id = confirm.proposal.id;
    try {
      if (confirm.action === 'approve') {
        const res = await service.approveProposal(id);
        if (res.decision === 'rejected') {
          toast.warning('Critic re-check rejected the proposal', {
            description: res.verdict?.rationale ?? 'No rationale returned.',
          });
        } else {
          toast.success('Proposal approved');
        }
      } else {
        await service.rejectProposal(id, confirm.reason);
        toast.success('Proposal rejected');
      }
      setConfirm({ action: null, proposal: null, reason: '', busy: false });
      load();
    } catch (err) {
      toast.error('Action failed', {
        description: err instanceof Error ? err.message : String(err),
      });
      setConfirm((c) => ({ ...c, busy: false }));
    }
  };

  const summary = useMemo(() => {
    const counts: Record<ProposalStatus, number> = {
      pending: 0,
      approved: 0,
      rejected: 0,
      rejected_by_critic: 0,
      executed: 0,
      failed: 0,
    };
    for (const p of proposals) counts[p.status] = (counts[p.status] ?? 0) + 1;
    return counts;
  }, [proposals]);

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="p-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Filter className="size-4 text-muted-foreground" />

            <Select
              value={statusFilter || '__all__'}
              onValueChange={(v) =>
                setStatusFilter((v === '__all__' ? '' : v) as ProposalStatus | '')
              }
            >
              <SelectTrigger className="h-8 w-44">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                {STATUS_OPTIONS.map((o) => (
                  <SelectItem key={o.value || '__all__'} value={o.value || '__all__'}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Input
              className="h-8 w-56"
              placeholder="objective_id…"
              value={objectiveFilter}
              onChange={(e) => setObjectiveFilter(e.target.value)}
            />

            <Button size="sm" variant="outline" onClick={load} className="h-8">
              Refresh
            </Button>

            <span className="ml-auto text-xs text-muted-foreground tabular-nums">
              {proposals.length} {proposals.length === 1 ? 'proposal' : 'proposals'}
              {summary.pending > 0 && statusFilter !== 'pending' && (
                <span className="ml-2 rounded-md bg-amber-500/15 px-1.5 py-0.5 text-amber-700 dark:text-amber-400">
                  {summary.pending} pending
                </span>
              )}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-32">Kind</TableHead>
                <TableHead>Summary</TableHead>
                <TableHead className="w-44">Objective</TableHead>
                <TableHead className="w-32">Status</TableHead>
                <TableHead className="w-20 text-right">Cost</TableHead>
                <TableHead className="w-40">Created</TableHead>
                <TableHead className="w-40 text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                Array.from({ length: 4 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell colSpan={7}>
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  </TableRow>
                ))
              ) : proposals.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-6">
                    {setup && !setup.critic.ready ? (
                      <SetupGuide onConfigured={handleSetupConfigured} />
                    ) : (
                      <div className="text-center text-muted-foreground">
                        No proposals match these filters. Once the autonomous loop
                        or Claude Code (via MCP) posts to{' '}
                        <code className="font-mono">/v1/harness/proposals</code>,
                        they'll appear here for approval.
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ) : (
                proposals.map((p) => {
                  const kind =
                    ((p.data ?? {}) as Record<string, unknown>).kind ?? '—';
                  const summary =
                    ((p.data ?? {}) as Record<string, unknown>).summary ?? '';
                  return (
                    <TableRow
                      key={p.id}
                      onClick={() => {
                        setDrawerId(p.id);
                        setDrawerOpen(true);
                      }}
                      className="cursor-pointer"
                    >
                      <TableCell className="font-mono text-xs">{String(kind)}</TableCell>
                      <TableCell className="max-w-xs truncate text-sm">
                        {String(summary) || (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </TableCell>
                      <TableCell className="font-mono text-[11px] text-muted-foreground">
                        {p.objective_id ?? '—'}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={p.status} />
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs tabular-nums">
                        {formatUsd(p.cost_usd)}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground tabular-nums">
                        {formatTs(p.ts)}
                      </TableCell>
                      <TableCell
                        className="text-right"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {p.status === 'pending' ? (
                          <div className="flex items-center justify-end gap-1">
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2"
                              onClick={() => openConfirm('approve', p)}
                            >
                              <CheckCircle2 className="size-3.5 mr-1 text-emerald-500" />
                              Approve
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2"
                              onClick={() => openConfirm('reject', p)}
                            >
                              <XCircle className="size-3.5 mr-1 text-rose-500" />
                              Reject
                            </Button>
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <ChainDrawer
        open={drawerOpen}
        rootEntryId={drawerId}
        onClose={() => setDrawerOpen(false)}
      />

      <Dialog
        open={confirm.action !== null}
        onOpenChange={(open) => !open && closeConfirm()}
      >
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>
              {confirm.action === 'approve' ? 'Approve proposal' : 'Reject proposal'}
            </DialogTitle>
            <DialogDescription>
              {confirm.action === 'approve'
                ? 'The critic will re-check before the proposal is resolved. If the objective recovered since the proposal was created, the critic may reject it.'
                : 'Manual rejection always succeeds. Optionally provide a reason recorded in the ledger.'}
            </DialogDescription>
          </DialogHeader>

          {confirm.proposal && <ProposalCard proposal={confirm.proposal} />}

          {confirm.action === 'reject' && (
            <Input
              placeholder="Reason (optional)"
              value={confirm.reason}
              onChange={(e) =>
                setConfirm((c) => ({ ...c, reason: e.target.value }))
              }
            />
          )}

          <DialogFooter>
            <Button variant="outline" onClick={closeConfirm} disabled={confirm.busy}>
              Cancel
            </Button>
            <Button
              onClick={submitConfirm}
              disabled={confirm.busy}
              variant={confirm.action === 'reject' ? 'destructive' : 'default'}
            >
              {confirm.busy
                ? 'Working…'
                : confirm.action === 'approve'
                  ? 'Approve'
                  : 'Reject'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
