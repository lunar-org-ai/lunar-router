import { motion } from 'framer-motion';

import { MetricBar } from '@/features/agents/components/MetricBar';
import { MetricTile } from '@/features/agents/components/MetricTile';
import { RecentTracesList } from '@/features/agents/components/RecentTracesList';
import { Sparkline } from '@/features/agents/components/Sparkline';
import type { AgentSummary } from '@/features/agents/types';

const formatTraces = (n: number): string => n.toLocaleString();
const formatLatency = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
const formatErrorRate = (rate: number): string => `${(rate * 100).toFixed(1)}%`;
const formatCost = (n: number): string => `$${n.toFixed(4)}`;

type OverviewTabProps = {
  agent: AgentSummary;
};

export function OverviewTab({ agent }: OverviewTabProps) {
  return (
    <div className="flex flex-col gap-6 px-6 py-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricTile label="Traces 24h" value={formatTraces(agent.traces24h)} delay={0.04} />
        <MetricTile
          label="p95 latency"
          value={formatLatency(agent.p95LatencyMs)}
          delay={0.09}
        />
        <MetricTile
          label="Error rate"
          value={formatErrorRate(agent.errorRate)}
          delay={0.14}
        />
        <MetricTile label="$ / trace" value={formatCost(agent.costPerTrace)} delay={0.19} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.22, ease: [0.16, 1, 0.3, 1] }}
        className="rounded-xl border border-border/40 bg-card/30 px-5 py-4"
      >
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Trace volume · 24h
          </span>
          <span className="font-mono text-[11px] text-muted-foreground/60">30min buckets</span>
        </div>
        <div className="mt-3 text-foreground/55">
          <Sparkline values={agent.traceVolume} width={800} height={64} delay={0.28} />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="flex flex-col gap-3 md:col-span-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Recent traces
          </span>
          <RecentTracesList traces={agent.recentTraces} delay={0.32} />
        </div>
        <div className="flex flex-col gap-3">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Eval breakdown
          </span>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.36, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-4 py-4"
          >
            {agent.metrics.map((metric, i) => (
              <MetricBar key={metric.name} metric={metric} index={i} />
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
