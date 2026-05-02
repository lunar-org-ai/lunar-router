import { LayoutGrid, Loader2, Play, RotateCcw } from 'lucide-react';

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type AgentHeaderProps = {
  agentId: string;
  agentName: string;
  version: string;
  overall: number;
  showActions?: boolean;
  scoreState?: 'visible' | 'pending' | 'evaluating';
  onRunEval?: () => void;
  onReset?: () => void;
  evalDisabled?: boolean;
};

function scoreColor(value: number) {
  if (value >= 80) return 'text-emerald-500';
  if (value >= 50) return 'text-amber-500';
  return 'text-rose-500';
}

export function AgentHeader({
  agentId,
  agentName,
  version,
  overall,
  showActions = true,
  scoreState = 'visible',
  onRunEval,
  onReset,
  evalDisabled = false,
}: AgentHeaderProps) {
  return (
    <header className="flex flex-col gap-3 border-b border-border/40 px-6 pb-4 pt-5">
      <Breadcrumb>
        <BreadcrumbList className="text-xs">
          <BreadcrumbItem>
            <BreadcrumbLink href="/agents">agents</BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage>{agentId}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex size-8 items-center justify-center rounded-md border border-border/40 bg-card/40 text-muted-foreground">
            <LayoutGrid className="size-4" />
          </div>
          <h1 className="text-xl font-medium tracking-tight">{agentName}</h1>
          <Badge
            variant="outline"
            className="border-border/40 bg-background/60 text-xs text-muted-foreground"
          >
            {version}
          </Badge>
        </div>

        {showActions ? (
          <div className="flex items-center gap-4">
            <ScoreReadout state={scoreState} value={overall} />
            {onReset ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={onReset}
                className="gap-1.5 text-muted-foreground"
              >
                <RotateCcw className="size-3.5" />
                Reset demo
              </Button>
            ) : null}
            <Button size="sm" onClick={onRunEval} disabled={evalDisabled} className="gap-1.5">
              {scoreState === 'evaluating' ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Play className="size-3.5 fill-current" />
              )}
              {scoreState === 'evaluating' ? 'Running…' : 'Run Eval'}
            </Button>
          </div>
        ) : null}
      </div>
    </header>
  );
}

type ScoreReadoutProps = {
  state: 'visible' | 'pending' | 'evaluating';
  value: number;
};

function ScoreReadout({ state, value }: ScoreReadoutProps) {
  if (state === 'pending') {
    return <span className="font-mono text-base font-medium text-muted-foreground/60">—</span>;
  }
  if (state === 'evaluating') {
    return (
      <span className="font-mono text-base font-medium text-muted-foreground">…</span>
    );
  }
  return (
    <span className={cn('font-mono text-base font-medium', scoreColor(value))}>{value}%</span>
  );
}
