import { LayoutGrid, Play } from 'lucide-react';

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
};

function scoreColor(value: number) {
  if (value >= 80) return 'text-emerald-500';
  if (value >= 50) return 'text-amber-500';
  return 'text-rose-500';
}

export function AgentHeader({ agentId, agentName, version, overall }: AgentHeaderProps) {
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

        <div className="flex items-center gap-4">
          <span className={cn('font-mono text-base font-medium', scoreColor(overall))}>
            {overall}%
          </span>
          <Button size="sm" className="gap-1.5">
            <Play className="size-3.5 fill-current" />
            Run Eval
          </Button>
        </div>
      </div>
    </header>
  );
}
