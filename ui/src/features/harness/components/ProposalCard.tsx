/**
 * ProposalCard — shared display fragment for a harness proposal.
 *
 * Used by the confirm dialogs in ProposalsTab so the operator sees the
 * critic's verdict + cost + payload before acting. Pure presentation:
 * no fetching, no actions.
 */

import { Badge } from '@/components/ui/badge';
import type { Proposal, ProposalStatus } from '@/services/harnessService';

const STATUS_STYLES: Record<ProposalStatus, string> = {
  pending: 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30',
  approved: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30',
  rejected: 'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30',
  rejected_by_critic:
    'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30',
  executed: 'bg-sky-500/10 text-sky-700 dark:text-sky-400 border-sky-500/30',
  failed: 'bg-rose-500/15 text-rose-700 dark:text-rose-400 border-rose-500/40',
};

const STATUS_LABEL: Record<ProposalStatus, string> = {
  pending: 'pending',
  approved: 'approved',
  rejected: 'rejected',
  rejected_by_critic: 'rejected by critic',
  executed: 'executed',
  failed: 'failed',
};

export function StatusBadge({ status }: { status: ProposalStatus }) {
  return (
    <Badge
      variant="outline"
      className={`font-mono text-[10px] uppercase tracking-wide ${STATUS_STYLES[status]}`}
    >
      {STATUS_LABEL[status]}
    </Badge>
  );
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null) return '—';
  if (value === 0) return '$0';
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function ProposalCard({ proposal }: { proposal: Proposal }) {
  const data = (proposal.data ?? {}) as Record<string, unknown>;
  const verdict = (data.verdict ?? {}) as {
    decision?: string;
    rationale?: string;
    estimated_cost_usd?: number;
    estimated_benefit?: string;
  };
  const kind = (data.kind as string | undefined) ?? '—';
  const summary = (data.summary as string | undefined) ?? '';
  const payload = (data.payload as Record<string, unknown> | undefined) ?? {};

  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="font-mono text-[10px]">
          {kind}
        </Badge>
        <StatusBadge status={proposal.status} />
        {proposal.objective_id && (
          <span className="font-mono text-[11px] text-muted-foreground">
            {proposal.objective_id}
          </span>
        )}
      </div>

      {summary && <p className="text-sm text-foreground">{summary}</p>}

      <div className="rounded-md border bg-muted/30 p-3 space-y-1">
        <div className="text-xs font-medium uppercase text-muted-foreground tracking-wide">
          Critic verdict
        </div>
        <div className="text-xs">
          <span className="font-mono">{verdict.decision ?? '—'}</span>
          <span className="text-muted-foreground"> · est. </span>
          <span className="font-mono">{formatUsd(verdict.estimated_cost_usd)}</span>
        </div>
        {verdict.rationale && (
          <p className="text-xs text-muted-foreground">{verdict.rationale}</p>
        )}
        {verdict.estimated_benefit && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Benefit:</span> {verdict.estimated_benefit}
          </p>
        )}
      </div>

      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
          Payload
        </summary>
        <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted/40 p-2 text-[11px] leading-tight">
          {JSON.stringify(payload, null, 2)}
        </pre>
      </details>
    </div>
  );
}
