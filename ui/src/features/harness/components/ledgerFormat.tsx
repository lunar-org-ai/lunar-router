/**
 * Shared formatting primitives for ledger entries: type badges (color-
 * coded), outcome icons, short ids, human timestamps. Used by both the
 * Ledger table and the Chain drawer so visual language stays consistent
 * across places the user sees ledger entries.
 */

import { AlertTriangle, CheckCircle2, SkipForward, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import type {
  LedgerEntry,
  LedgerEntryType,
  LedgerOutcome,
} from '@/services/harnessService';

export const TYPE_STYLES: Record<LedgerEntryType, string> = {
  signal: 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30',
  run: 'bg-sky-500/10 text-sky-700 dark:text-sky-400 border-sky-500/30',
  observation: 'bg-violet-500/10 text-violet-700 dark:text-violet-400 border-violet-500/30',
  decision: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30',
  action: 'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30',
  proposal: 'bg-slate-500/10 text-slate-700 dark:text-slate-400 border-slate-500/30',
  lesson: 'bg-teal-500/10 text-teal-700 dark:text-teal-400 border-teal-500/30',
};

export function TypeBadge({ type }: { type: LedgerEntryType }) {
  return (
    <Badge
      variant="outline"
      className={`font-mono text-[10px] uppercase tracking-wide ${TYPE_STYLES[type]}`}
    >
      {type}
    </Badge>
  );
}

export function OutcomeIcon({ outcome }: { outcome: LedgerOutcome | null }) {
  if (outcome === null) return null;
  if (outcome === 'ok') return <CheckCircle2 className="size-3.5 text-emerald-500" />;
  if (outcome === 'failed') return <XCircle className="size-3.5 text-rose-500" />;
  if (outcome === 'skipped')
    return <SkipForward className="size-3.5 text-muted-foreground" />;
  if (outcome === 'rolled_back')
    return <AlertTriangle className="size-3.5 text-amber-500" />;
  return null;
}

export function shortId(id: string): string {
  return id.length > 10 ? `${id.slice(0, 8)}…` : id;
}

export function formatTs(ts: string): string {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

export function markerColorFor(entry: Pick<LedgerEntry, 'type' | 'outcome'>): string {
  // Dots on the objective plot. Fail > regression > neutral.
  if (entry.outcome === 'failed') return '#f43f5e'; // rose
  if (entry.type === 'signal') return '#f59e0b'; // amber
  if (entry.type === 'decision') return '#10b981'; // emerald
  if (entry.type === 'action') return '#e11d48'; // rose dark
  return '#64748b';
}
