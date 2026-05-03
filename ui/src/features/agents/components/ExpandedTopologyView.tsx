import { motion } from 'framer-motion';
import {
  Brain,
  Database,
  type LucideIcon,
  ShieldCheck,
  Sparkles,
  Wrench,
  X,
} from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Button } from '@/components/ui/button';
import type {
  AgentFramework,
  StackComponent,
  StackKind,
} from '@/features/agents/types';

const SPRING = { type: 'spring' as const, stiffness: 280, damping: 32 };
const EASE = [0.16, 1, 0.3, 1] as const;

const KIND_ICON: Record<StackKind, LucideIcon> = {
  model: Sparkles,
  tool: Wrench,
  vectorstore: Database,
  memory: Brain,
  guardrail: ShieldCheck,
};

const KIND_LABEL: Record<StackKind, string> = {
  model: 'Model',
  tool: 'Tool',
  vectorstore: 'Vector store',
  memory: 'Memory',
  guardrail: 'Guardrail',
};

const FRAMEWORK_LABEL: Record<AgentFramework, string> = {
  langchain: 'LangChain',
  langgraph: 'LangGraph',
  crewai: 'CrewAI',
  'openai-agents': 'OpenAI Agents',
};

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

type Props = {
  framework: AgentFramework;
  agentName: string;
  stack: StackComponent[];
  onClose: () => void;
};

export function ExpandedTopologyView({ framework, agentName, stack, onClose }: Props) {
  return (
    <motion.div
      key="topology-panel"
      layoutId="agent-topology-panel"
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
              Topology · {agentName}
            </span>
            <h2 className="text-lg font-medium tracking-tight">
              {stack.length + 1} components observed
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
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
              <FrameworkBlock framework={framework} index={0} />
              {stack.map((component, i) => (
                <ComponentBlock
                  key={`${component.kind}-${component.label}`}
                  component={component}
                  index={i + 1}
                />
              ))}
            </div>

            <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/40 bg-card/20 px-5 py-6 text-center">
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground/60">
                Graph view
              </span>
              <span className="font-mono text-[11px] text-muted-foreground/50">
                DAG of observed spans with parent → child edges. Coming soon.
              </span>
            </div>
          </div>
        </motion.div>
    </motion.div>
  );
}

type FrameworkBlockProps = {
  framework: AgentFramework;
  index: number;
};

function FrameworkBlock({ framework, index }: FrameworkBlockProps) {
  const Icon = FRAMEWORK_ICONS[framework];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.26 + index * 0.04, ease: EASE }}
      className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-4 py-4"
    >
      <div className="flex items-center gap-3">
        <span className="flex size-9 items-center justify-center overflow-hidden rounded-lg">
          <Icon size={36} />
        </span>
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium leading-tight">
            {FRAMEWORK_LABEL[framework]}
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            Framework
          </span>
        </div>
      </div>
      <div className="border-t border-border/30 pt-3 font-mono text-[11px] text-muted-foreground/70">
        Auto-instrumented via opentracy-{framework}
      </div>
    </motion.div>
  );
}

const formatLatency = (ms?: number): string => {
  if (ms === undefined) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
};
const formatCount = (n?: number): string => {
  if (n === undefined) return '—';
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return n.toString();
};
const formatPct = (n?: number): string => {
  if (n === undefined) return '—';
  return `${(n * 100).toFixed(1)}%`;
};

type ComponentBlockProps = {
  component: StackComponent;
  index: number;
};

function ComponentBlock({ component, index }: ComponentBlockProps) {
  const Icon = KIND_ICON[component.kind];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.26 + index * 0.04, ease: EASE }}
      className="flex flex-col gap-3 rounded-xl border border-border/40 bg-card/30 px-4 py-4"
    >
      <div className="flex items-center gap-3">
        <span className="flex size-9 items-center justify-center rounded-lg bg-card/60">
          <Icon className="size-4 text-muted-foreground/80" />
        </span>
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="truncate text-sm font-medium leading-tight">{component.label}</span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
            {KIND_LABEL[component.kind]}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 border-t border-border/30 pt-3">
        <Stat label="Calls 24h" value={formatCount(component.callCount)} />
        <Stat label="Success" value={formatPct(component.successRate)} />
        <Stat label="Avg latency" value={formatLatency(component.avgLatencyMs)} />
        <Stat label="Cost share" value={formatPct(component.costShare)} />
      </div>
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
      <span className="font-mono text-xs tabular-nums">{value}</span>
    </div>
  );
}
