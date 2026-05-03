import { motion } from 'framer-motion';

import { MetricBar } from '@/features/agents/components/MetricBar';
import { MetricTile } from '@/features/agents/components/MetricTile';
import { RecentTracesList } from '@/features/agents/components/RecentTracesList';
import { Sparkline } from '@/features/agents/components/Sparkline';
import { FrameworkChip, StackChip } from '@/features/agents/components/StackChip';
import type { AgentSummary, EvalMetric } from '@/features/agents/types';

const formatTraces = (n: number): string => n.toLocaleString();
const formatLatency = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
const formatErrorRate = (rate: number): string => `${(rate * 100).toFixed(1)}%`;
const formatCost = (n: number): string => `$${n.toFixed(4)}`;

type OverviewTabProps = {
  agent: AgentSummary;
  metrics?: EvalMetric[];
  evalRunId?: number;
  evaluating?: boolean;
};

export function OverviewTab({
  agent,
  metrics,
  evalRunId = 0,
  evaluating = false,
}: OverviewTabProps) {
  const displayMetrics = metrics ?? agent.metrics;

  return (
    <div className="flex flex-col gap-6 px-6 py-6">
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col gap-2"
      >
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
          Stack
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <FrameworkChip framework={agent.framework} delay={0.04} />
          {agent.stack.map((chip, i) => (
            <StackChip
              key={`${chip.kind}-${chip.label}`}
              kind={chip.kind}
              label={chip.label}
              delay={0.08 + i * 0.04}
            />
          ))}
        </div>
      </motion.div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricTile label="Traces 24h" value={formatTraces(agent.traces24h)} delay={0.12} />
        <MetricTile
          label="p95 latency"
          value={formatLatency(agent.p95LatencyMs)}
          delay={0.17}
        />
        <MetricTile
          label="Error rate"
          value={formatErrorRate(agent.errorRate)}
          delay={0.22}
        />
        <MetricTile label="$ / trace" value={formatCost(agent.costPerTrace)} delay={0.27} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="rounded-xl border border-border/40 bg-card/30 px-5 py-4"
      >
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Trace volume · 24h
          </span>
          <span className="font-mono text-[11px] text-muted-foreground/60">30min buckets</span>
        </div>
        <div className="mt-3 text-foreground/55">
          <Sparkline values={agent.traceVolume} width={800} height={64} delay={0.36} />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="flex flex-col gap-3 md:col-span-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Recent traces
          </span>
          <RecentTracesList traces={agent.recentTraces} delay={0.4} />
        </div>
        <div className="flex flex-col gap-3">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Eval breakdown
          </span>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{
              opacity: evaluating ? 0.55 : 1,
              y: 0,
            }}
            transition={{ duration: 0.4, delay: 0.44, ease: [0.16, 1, 0.3, 1] }}
            className="relative flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-4 py-4"
          >
            {evaluating ? <ShimmerOverlay /> : null}
            {displayMetrics.map((metric, i) => (
              <MetricBar
                key={`${metric.name}-${evalRunId}`}
                metric={metric}
                index={i}
                runId={evalRunId}
              />
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

function ShimmerOverlay() {
  return (
    <motion.div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden rounded-xl"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
    >
      <motion.div
        className="absolute inset-y-0 -left-1/3 w-1/3 bg-gradient-to-r from-transparent via-foreground/10 to-transparent"
        animate={{ x: ['0%', '400%'] }}
        transition={{ duration: 1.4, ease: 'linear', repeat: Infinity }}
      />
    </motion.div>
  );
}
