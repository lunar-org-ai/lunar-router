import { useEffect, useState } from 'react';
import { animate, motion, useMotionValue } from 'framer-motion';
import { Loader2, Play, RotateCcw } from 'lucide-react';
import { CrewAI, LangChain, LangGraph, OpenAI } from '@lobehub/icons';

import { Button } from '@/components/ui/button';
import { StatusPill } from '@/features/agents/components/StatusPill';
import { cn } from '@/lib/utils';
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
  evalScore?: number;
  evalRunId?: number;
  onRunEval?: () => void;
  onReset?: () => void;
  evaluating?: boolean;
};

export function AgentHeader({
  agentName,
  framework,
  status,
  evalScore,
  evalRunId = 0,
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

      <div className="flex items-center gap-4">
        {typeof evalScore === 'number' ? (
          <EvalScoreReadout
            score={evalScore}
            runId={evalRunId}
            evaluating={evaluating}
          />
        ) : null}
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

type EvalScoreReadoutProps = {
  score: number;
  runId: number;
  evaluating: boolean;
};

function scoreColor(value: number): string {
  if (value >= 80) return 'text-emerald-500';
  if (value >= 50) return 'text-amber-500';
  return 'text-rose-500';
}

function EvalScoreReadout({ score, runId, evaluating }: EvalScoreReadoutProps) {
  const motionVal = useMotionValue(score);
  const [display, setDisplay] = useState(score);

  useEffect(() => {
    const controls = animate(motionVal, score, {
      duration: 0.85,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (latest) => setDisplay(Math.round(latest)),
    });
    return () => controls.stop();
  }, [score, runId, motionVal]);

  return (
    <div className="flex flex-col items-end gap-0.5 leading-none">
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/70">
        Eval score
      </span>
      <motion.span
        animate={{ opacity: evaluating ? 0.55 : 1 }}
        transition={{ duration: 0.25 }}
        className={cn('font-mono text-base font-medium tabular-nums', scoreColor(display))}
      >
        {display}%
      </motion.span>
    </div>
  );
}
