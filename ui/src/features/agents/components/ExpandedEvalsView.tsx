import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Play, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { MetricBar } from '@/features/agents/components/MetricBar';
import { Sparkline } from '@/features/agents/components/Sparkline';
import { cn } from '@/lib/utils';
import type { EvalMetric } from '@/features/agents/types';

const SPRING = { type: 'spring' as const, stiffness: 280, damping: 32 };
const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

type ExpandedEvalsViewProps = {
  metrics: EvalMetric[];
  evalScore: number;
  evalRunId: number;
  evaluating: boolean;
  onRunEval: () => void;
  onClose: () => void;
};

function scoreColor(value: number): string {
  if (value >= 80) return 'text-emerald-500';
  if (value >= 50) return 'text-amber-500';
  return 'text-rose-500';
}

const generateHistory = (currentValue: number | 'pass' | 'fail', count: number): number[] => {
  const base =
    typeof currentValue === 'number' ? currentValue : currentValue === 'pass' ? 94 : 28;
  return Array.from({ length: count }, (_, i) => {
    const t = i / Math.max(count - 1, 1);
    const drift = Math.sin(t * Math.PI * 2.4) * 6;
    const noise = (Math.random() - 0.5) * 4.5;
    return Math.max(0, Math.min(100, Math.round(base + drift + noise)));
  });
};

type Run = {
  id: string;
  timestamp: string;
  sampleSize: number;
  score: number;
};

const generateRuns = (currentScore: number, count: number): Run[] => {
  return Array.from({ length: count }, (_, i) => {
    const ago = (i + 1) * 6;
    const drift = Math.sin(i * 0.7) * 4;
    const score = Math.max(
      0,
      Math.min(100, Math.round(currentScore + drift + (Math.random() - 0.5) * 5))
    );
    return {
      id: `r_${i}`,
      timestamp: ago < 24 ? `${ago}h ago` : `${Math.round(ago / 24)}d ago`,
      sampleSize: 60 + Math.floor(Math.random() * 240),
      score,
    };
  });
};

export function ExpandedEvalsView({
  metrics,
  evalScore,
  evalRunId,
  evaluating,
  onRunEval,
  onClose,
}: ExpandedEvalsViewProps) {
  const histories = useMemo(
    () =>
      metrics.map((metric) => ({
        name: metric.name,
        series: generateHistory(metric.value, 12),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [evalRunId, metrics.length]
  );

  const runs = useMemo(
    () => generateRuns(evalScore, 8),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [evalRunId]
  );

  return (
    <>
      <motion.div
        key="evals-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25 }}
        className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      <motion.div
        key="evals-panel"
        layoutId="agent-evals-panel"
        transition={SPRING}
        className="fixed inset-4 z-50 flex flex-col overflow-hidden rounded-2xl border border-border/50 bg-card/95 shadow-2xl md:inset-8"
      >
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3, delay: 0.18, ease: EASE_OUT_EXPO }}
          className="flex items-center justify-between border-b border-border/40 px-6 py-4"
        >
          <div className="flex flex-col gap-0.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
              Eval breakdown
            </span>
            <h2
              className={cn(
                'text-2xl font-medium tracking-tight tabular-nums',
                scoreColor(evalScore)
              )}
            >
              {evalScore}%
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={onRunEval} disabled={evaluating} className="gap-1.5">
              {evaluating ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Play className="size-3.5 fill-current" />
              )}
              {evaluating ? 'Running…' : 'Run Eval'}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label="Close"
              className="size-8"
            >
              <X className="size-4" />
            </Button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, delay: 0.22, ease: EASE_OUT_EXPO }}
          className="flex-1 overflow-y-auto px-6 py-6"
        >
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-3">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
                Per-metric trend · last 12 runs
              </span>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {metrics.map((metric, i) => (
                  <ExpandedMetricRow
                    key={metric.name}
                    metric={metric}
                    history={histories[i]?.series ?? []}
                    index={i}
                    runId={evalRunId}
                  />
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
                Run history
              </span>
              <div className="flex flex-col gap-px rounded-xl border border-border/40 bg-card/30 p-1">
                {runs.map((run, i) => (
                  <motion.div
                    key={run.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3, delay: 0.34 + i * 0.04, ease: 'easeOut' }}
                    className="flex items-center justify-between gap-3 rounded-lg px-3 py-2 transition-colors hover:bg-card/60"
                  >
                    <code className="font-mono text-[11px] text-muted-foreground/70">
                      {run.timestamp}
                    </code>
                    <div className="flex shrink-0 items-center gap-4 font-mono text-[11px] tabular-nums text-muted-foreground">
                      <span>{run.sampleSize} traces</span>
                      <span className={cn('w-10 text-right', scoreColor(run.score))}>
                        {run.score}%
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/40 bg-card/20 px-5 py-6 text-center">
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground/60">
                Failing traces drill-down
              </span>
              <span className="font-mono text-[11px] text-muted-foreground/50">
                Surface traces that scored low per metric, with judge reasoning. Coming soon.
              </span>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </>
  );
}

type ExpandedMetricRowProps = {
  metric: EvalMetric;
  history: number[];
  index: number;
  runId: number;
};

function ExpandedMetricRow({ metric, history, index, runId }: ExpandedMetricRowProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.26 + index * 0.05, ease: EASE_OUT_EXPO }}
      className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-4 py-4"
    >
      <MetricBar metric={metric} index={index} runId={runId} />
      <div className="text-foreground/45">
        <Sparkline values={history} width={400} height={32} delay={0.32 + index * 0.05} />
      </div>
    </motion.div>
  );
}
