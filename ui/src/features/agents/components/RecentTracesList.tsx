import { motion } from 'framer-motion';

import { cn } from '@/lib/utils';
import type { RecentTrace } from '@/features/agents/types';

type RecentTracesListProps = {
  traces: RecentTrace[];
  delay?: number;
};

const formatDuration = (ms: number): string => {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
};

const formatCost = (n: number): string => `$${n.toFixed(4)}`;

export function RecentTracesList({ traces, delay = 0 }: RecentTracesListProps) {
  return (
    <div className="flex flex-col gap-px p-1">
      {traces.map((trace, i) => (
        <motion.div
          key={trace.id}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25, delay: delay + 0.08 + i * 0.04, ease: 'easeOut' }}
          className="flex items-center justify-between gap-3 rounded-lg px-3 py-2"
        >
          <div className="flex min-w-0 items-center gap-3">
            <span
              className={cn(
                'size-1.5 shrink-0 rounded-full',
                trace.status === 'ok' ? 'bg-emerald-500' : 'bg-rose-500'
              )}
            />
            <code className="truncate font-mono text-[11px] text-muted-foreground/80">
              {trace.id}
            </code>
          </div>
          <div className="flex shrink-0 items-center gap-4 font-mono text-[11px] tabular-nums text-muted-foreground">
            <span className="w-14 text-right">{formatDuration(trace.durationMs)}</span>
            <span className="w-16 text-right">{formatCost(trace.costUsd)}</span>
            <span className="w-10 text-right text-muted-foreground/60">{trace.agoLabel}</span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
