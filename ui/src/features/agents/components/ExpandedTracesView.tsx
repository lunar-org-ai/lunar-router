import { motion } from 'framer-motion';
import { X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { RecentTrace } from '@/features/agents/types';

const SPRING = { type: 'spring' as const, stiffness: 280, damping: 32 };
const EASE = [0.16, 1, 0.3, 1] as const;

const formatDuration = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
const formatCost = (n: number): string => `$${n.toFixed(4)}`;
const formatTokens = (trace: RecentTrace): string => {
  if (trace.tokensIn === undefined || trace.tokensOut === undefined) return '—';
  return `${trace.tokensIn}+${trace.tokensOut}`;
};

type Props = {
  traces: RecentTrace[];
  agentName: string;
  totalToday: number;
  onClose: () => void;
};

export function ExpandedTracesView({ traces, agentName, totalToday, onClose }: Props) {
  return (
    <motion.div
      key="traces-panel"
      layoutId="agent-traces-panel"
      transition={SPRING}
      className="absolute inset-0 flex flex-col overflow-hidden rounded-xl border border-border/50 bg-background"
    >
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3, delay: 0.18, ease: EASE }}
          className="flex items-center justify-between border-b border-border/40 px-6 py-4"
        >
          <div className="flex flex-col gap-0.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
              Traces · {agentName}
            </span>
            <h2 className="text-lg font-medium tracking-tight tabular-nums">
              {totalToday.toLocaleString()}{' '}
              <span className="font-normal text-muted-foreground/60">today</span>
            </h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close"
            className="size-8"
          >
            <X className="size-4" />
          </Button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, delay: 0.22, ease: EASE }}
          className="flex-1 overflow-y-auto px-6 py-6"
        >
          <div className="flex flex-col gap-4">
            <FilterBar />
            <TraceTable traces={traces} />
            <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/40 bg-card/20 px-5 py-6 text-center">
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground/60">
                Trace inspector
              </span>
              <span className="font-mono text-[11px] text-muted-foreground/50">
                Click a trace to see prompt → response → spans → eval verdicts. Coming soon.
              </span>
            </div>
          </div>
        </motion.div>
    </motion.div>
  );
}

function FilterBar() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <FilterPill active label="All" />
      <FilterPill label="Errors only" />
      <FilterPill label="High latency" />
      <FilterPill label="Last 1h" />
    </div>
  );
}

type FilterPillProps = {
  label: string;
  active?: boolean;
};

function FilterPill({ label, active }: FilterPillProps) {
  return (
    <button
      type="button"
      disabled
      className={cn(
        'cursor-not-allowed rounded-full border px-3 py-1 font-mono text-[11px] tracking-wide transition-colors',
        active
          ? 'border-foreground/40 bg-foreground/10 text-foreground'
          : 'border-border/40 bg-card/30 text-muted-foreground/70'
      )}
    >
      {label}
    </button>
  );
}

const COLUMN_TEMPLATE =
  'grid-cols-[1fr_5rem_8rem_5rem_5rem_5rem_3rem]';

type TraceTableProps = {
  traces: RecentTrace[];
};

function TraceTable({ traces }: TraceTableProps) {
  return (
    <div className="flex flex-col rounded-xl border border-border/40 bg-card/30">
      <div
        className={cn(
          'grid items-center gap-3 border-b border-border/40 px-4 py-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60',
          COLUMN_TEMPLATE
        )}
      >
        <span>Trace ID</span>
        <span>Status</span>
        <span>Model</span>
        <span className="text-right">Tokens</span>
        <span className="text-right">Duration</span>
        <span className="text-right">Cost</span>
        <span className="text-right">Ago</span>
      </div>
      {traces.map((trace, i) => (
        <motion.div
          key={trace.id}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.26 + i * 0.04, ease: 'easeOut' }}
          className={cn(
            'grid items-center gap-3 border-b border-border/30 px-4 py-2.5 transition-colors last:border-0 hover:bg-card/50',
            COLUMN_TEMPLATE
          )}
        >
          <code className="truncate font-mono text-[12px] text-foreground/80">{trace.id}</code>
          <span
            className={cn(
              'inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider',
              trace.status === 'ok' ? 'text-emerald-500' : 'text-rose-500'
            )}
          >
            <span
              className={cn(
                'size-1.5 rounded-full',
                trace.status === 'ok' ? 'bg-emerald-500' : 'bg-rose-500'
              )}
            />
            {trace.status}
          </span>
          <span className="truncate font-mono text-[11px] text-muted-foreground/80">
            {trace.model ?? '—'}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground/80">
            {formatTokens(trace)}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground/80">
            {formatDuration(trace.durationMs)}
          </span>
          <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground/80">
            {formatCost(trace.costUsd)}
          </span>
          <span className="text-right font-mono text-[11px] text-muted-foreground/60">
            {trace.agoLabel}
          </span>
        </motion.div>
      ))}
    </div>
  );
}
