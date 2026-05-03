import { motion } from 'framer-motion';
import { X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Sparkline } from '@/features/agents/components/Sparkline';
import type { AgentSummary } from '@/features/agents/types';

const SPRING = { type: 'spring' as const, stiffness: 280, damping: 32 };
const EASE = [0.16, 1, 0.3, 1] as const;

const formatTraces = (n: number): string => n.toLocaleString();
const formatLatency = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
const formatErrorRate = (rate: number): string => `${(rate * 100).toFixed(1)}%`;

type Props = {
  agent: AgentSummary;
  onClose: () => void;
};

export function ExpandedTrafficView({ agent, onClose }: Props) {
  const peak = Math.max(...agent.traceVolume, 1);
  const avg = Math.round(
    agent.traceVolume.reduce((a, b) => a + b, 0) / agent.traceVolume.length
  );
  const total = agent.traces24h;

  const latencySeries = agent.traceVolume.map((v) => {
    const base = agent.p95LatencyMs;
    return Math.round(base * (0.85 + (v / peak) * 0.32));
  });

  return (
    <motion.div
      key="traffic-panel"
      layoutId="agent-traffic-panel"
      transition={SPRING}
      className="absolute inset-0 flex flex-col overflow-hidden rounded-xl border border-border/50 bg-card/95"
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
              Traffic · {agent.name}
            </span>
            <h2 className="text-lg font-medium tracking-tight tabular-nums">
              {formatTraces(total)}{' '}
              <span className="font-normal text-muted-foreground/60">traces · last 24h</span>
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
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Tile label="Total" value={formatTraces(total)} />
              <Tile label="Peak" value={formatTraces(peak)} hint="per 30min bucket" />
              <Tile label="Avg" value={formatTraces(avg)} hint="per 30min bucket" />
              <Tile label="Error rate" value={formatErrorRate(agent.errorRate)} />
            </div>

            <div className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-5 py-4">
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
                  Trace volume · 24h
                </span>
                <span className="font-mono text-[11px] text-muted-foreground/60">
                  30min buckets
                </span>
              </div>
              <div className="text-foreground/55">
                <Sparkline
                  values={agent.traceVolume}
                  width={1200}
                  height={140}
                  delay={0.3}
                />
              </div>
            </div>

            <div className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-5 py-4">
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
                  Latency · estimated p95
                </span>
                <span className="font-mono text-[11px] text-muted-foreground/60">
                  {formatLatency(agent.p95LatencyMs)} avg
                </span>
              </div>
              <div className="text-amber-500/55">
                <Sparkline values={latencySeries} width={1200} height={80} delay={0.36} />
              </div>
            </div>

            <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/40 bg-card/20 px-5 py-6 text-center">
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground/60">
                Per-status & per-model breakdown
              </span>
              <span className="font-mono text-[11px] text-muted-foreground/50">
                Stacked timeline of ok/error and traffic share by model. Coming soon.
              </span>
            </div>
          </div>
        </motion.div>
    </motion.div>
  );
}

type TileProps = {
  label: string;
  value: string;
  hint?: string;
};

function Tile({ label, value, hint }: TileProps) {
  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-border/40 bg-card/30 px-4 py-3">
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
        {label}
      </span>
      <span className="text-xl font-medium tracking-tight tabular-nums">{value}</span>
      {hint ? (
        <span className="font-mono text-[11px] text-muted-foreground/60">{hint}</span>
      ) : null}
    </div>
  );
}
