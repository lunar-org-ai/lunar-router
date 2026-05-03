import { Loader2, Play, RotateCcw } from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Button } from '@/components/ui/button';
import { StatusPill } from '@/features/agents/components/StatusPill';
import type { AgentFramework, AgentStatus } from '@/features/agents/types';

const FRAMEWORK_ICONS: Record<AgentFramework, React.ComponentType<{ size?: number }>> = {
  langchain: LangChain.Avatar,
  langgraph: LangGraph.Avatar,
  crewai: CrewAI.Avatar,
  'openai-agents': OpenAI.Avatar,
};

type AgentHeaderProps = {
  agentName: string;
  framework: AgentFramework;
  status: AgentStatus;
  onRunEval?: () => void;
  onReset?: () => void;
  evaluating?: boolean;
};

export function AgentHeader({
  agentName,
  framework,
  status,
  onRunEval,
  onReset,
  evaluating = false,
}: AgentHeaderProps) {
  const Icon = FRAMEWORK_ICONS[framework];

  return (
    <header className="flex items-center justify-between border-b border-border/40 px-6 py-5">
      <div className="flex items-center gap-3">
        <span className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-lg">
          <Icon size={36} />
        </span>
        <div className="flex flex-col gap-0.5">
          <h1 className="text-lg font-medium tracking-tight">{agentName}</h1>
          <StatusPill status={status} />
        </div>
      </div>

      <div className="flex items-center gap-2">
        {onReset ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="gap-1.5 text-muted-foreground"
          >
            <RotateCcw className="size-3.5" />
            Reset
          </Button>
        ) : null}
        {onRunEval ? (
          <Button size="sm" onClick={onRunEval} disabled={evaluating} className="gap-1.5">
            {evaluating ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Play className="size-3.5 fill-current" />
            )}
            {evaluating ? 'Running…' : 'Run Eval'}
          </Button>
        ) : null}
      </div>
    </header>
  );
}
