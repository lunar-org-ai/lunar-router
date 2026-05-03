import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Sparkline } from '@/features/agents/components/Sparkline';
import { StatusPill } from '@/features/agents/components/StatusPill';
import type { AgentFramework, AgentSummary } from '@/features/agents/types';

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

const formatTraces = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return n.toString();
};

const formatLatency = (ms: number): string => {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
};

const formatErrorRate = (rate: number): string => `${(rate * 100).toFixed(1)}%`;

const formatAgo = (iso: string): string => {
  const diffMin = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const hours = Math.floor(diffMin / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
};

type AgentCardProps = {
  agent: AgentSummary;
  index?: number;
};

export function AgentCard({ agent, index = 0 }: AgentCardProps) {
  const Icon = FRAMEWORK_ICONS[agent.framework];
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, delay: 0.04 + index * 0.05, ease: [0.16, 1, 0.3, 1] }}
    >
      <Link
        to={`/agents/${agent.id}`}
        className="group flex flex-col gap-4 rounded-xl border border-border/40 bg-card/30 px-5 py-4 transition-colors hover:border-border/70 hover:bg-card/50"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg">
              <Icon size={36} />
            </span>
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="truncate text-sm font-medium leading-tight">{agent.name}</span>
              <span className="font-mono text-[11px] text-muted-foreground/70">
                {agent.framework}
              </span>
            </div>
          </div>
          <StatusPill status={agent.status} />
        </div>

        <div className="text-foreground/45">
          <Sparkline values={agent.traceVolume} width={300} height={36} />
        </div>

        <div className="grid grid-cols-3 gap-3 border-t border-border/30 pt-3">
          <Stat label="Traces 24h" value={formatTraces(agent.traces24h)} />
          <Stat label="p95" value={formatLatency(agent.p95LatencyMs)} />
          <Stat label="Errors" value={formatErrorRate(agent.errorRate)} />
        </div>

        <div className="flex items-center justify-between font-mono text-[11px] text-muted-foreground/60">
          <span>eval {agent.evalScore}%</span>
          <span>last seen {formatAgo(agent.lastSeenAt)}</span>
        </div>
      </Link>
    </motion.div>
  );
}

type StatProps = {
  label: string;
  value: string;
};

function Stat({ label, value }: StatProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">
        {label}
      </span>
      <span className="font-mono text-sm tabular-nums">{value}</span>
    </div>
  );
}
