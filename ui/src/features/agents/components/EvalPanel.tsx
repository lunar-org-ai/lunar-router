import { AlertTriangle } from 'lucide-react';

import { MetricBar } from '@/features/agents/components/MetricBar';
import { cn } from '@/lib/utils';
import type { AgentRun } from '@/features/agents/types';

type EvalPanelProps = {
  run: AgentRun;
};

function overallColor(value: number) {
  if (value >= 80) return 'text-emerald-500';
  if (value >= 50) return 'text-amber-500';
  return 'text-rose-500';
}

export function EvalPanel({ run }: EvalPanelProps) {
  return (
    <aside className="flex w-[360px] shrink-0 flex-col gap-6 border-l border-border/40 px-6 py-8">
      <div className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
          Evaluation
        </span>
        <span className="font-mono text-xs text-muted-foreground">{run.runLabel}</span>
      </div>

      <div className="flex flex-col gap-5">
        {run.metrics.map((metric, index) => (
          <MetricBar key={metric.name} metric={metric} index={index} />
        ))}
      </div>

      <div className="flex items-baseline justify-between border-t border-border/40 pt-5">
        <span className="text-sm font-medium">Overall</span>
        <span className={cn('font-mono text-xl font-medium', overallColor(run.overall))}>
          {run.overall}%
        </span>
      </div>

      {run.warning ? (
        <div className="flex gap-2 rounded-md border border-border/30 bg-card/30 px-3 py-2.5 text-xs">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-amber-500" />
          <span className="font-mono leading-relaxed text-muted-foreground">{run.warning}</span>
        </div>
      ) : null}
    </aside>
  );
}
