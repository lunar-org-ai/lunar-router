import { motion } from 'framer-motion';
import { Brain, Database, type LucideIcon, ShieldCheck, Sparkles, Wrench } from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { cn } from '@/lib/utils';
import type { AgentFramework, StackKind } from '@/features/agents/types';

const KIND_ICON: Record<StackKind, LucideIcon> = {
  model: Sparkles,
  tool: Wrench,
  vectorstore: Database,
  memory: Brain,
  guardrail: ShieldCheck,
};

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

const FRAMEWORK_LABEL: Record<AgentFramework, string> = {
  langchain: 'LangChain',
  langgraph: 'LangGraph',
  crewai: 'CrewAI',
  'openai-agents': 'OpenAI Agents',
};

const SHELL =
  'inline-flex items-center gap-1.5 rounded-full border border-border/40 bg-card/40 px-2.5 py-1 font-mono text-[11px] text-foreground/85';

type FrameworkChipProps = {
  framework: AgentFramework;
  delay?: number;
};

export function FrameworkChip({ framework, delay = 0 }: FrameworkChipProps) {
  const Icon = FRAMEWORK_ICONS[framework];
  return (
    <motion.span
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay, ease: [0.16, 1, 0.3, 1] }}
      className={cn(SHELL, 'pl-1.5')}
    >
      <span className="flex size-4 items-center justify-center overflow-hidden rounded-full">
        <Icon size={16} />
      </span>
      {FRAMEWORK_LABEL[framework]}
    </motion.span>
  );
}

type StackChipProps = {
  kind: StackKind;
  label: string;
  delay?: number;
};

export function StackChip({ kind, label, delay = 0 }: StackChipProps) {
  const Icon = KIND_ICON[kind];
  return (
    <motion.span
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay, ease: [0.16, 1, 0.3, 1] }}
      className={SHELL}
    >
      <Icon className="size-3 text-muted-foreground/70" />
      {label}
    </motion.span>
  );
}
