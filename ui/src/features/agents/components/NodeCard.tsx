import { GitBranch, MessageSquare, Search, Send, Sparkles, type LucideIcon } from 'lucide-react';
import { motion } from 'framer-motion';

import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { AgentNode, NodeType } from '@/features/agents/types';

const ICONS: Record<NodeType, LucideIcon> = {
  agent: Sparkles,
  tool: Search,
  llm: MessageSquare,
  router: GitBranch,
  output: Send,
};

type NodeCardProps = {
  node: AgentNode;
  className?: string;
};

export function NodeCard({ node, className }: NodeCardProps) {
  const Icon = ICONS[node.type];
  const hasMetrics = Boolean(node.cost) || Boolean(node.latency);

  return (
    <Card
      className={cn(
        'flex w-[420px] flex-row items-center gap-4 rounded-xl border-border/40 bg-card/40 px-4 py-3 shadow-none backdrop-blur-sm',
        className
      )}
    >
      <div className="flex size-9 shrink-0 items-center justify-center rounded-md border border-border/40 bg-background/60 text-muted-foreground">
        <Icon className="size-4" />
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="text-sm font-medium leading-tight text-foreground">{node.title}</div>
        {node.subtitle ? (
          <div className="font-mono text-xs leading-tight text-muted-foreground">
            {node.subtitle}
          </div>
        ) : null}
        {hasMetrics ? (
          <motion.div
            className="mt-1 flex items-center gap-2 font-mono text-[10px] text-muted-foreground/60"
            initial={{ opacity: 0, y: 2 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
          >
            {node.cost ? (
              <span className="rounded-sm bg-emerald-500/10 px-1 py-px text-emerald-500/80">
                {node.cost}
              </span>
            ) : null}
            {node.latency ? <span>{node.latency}</span> : null}
          </motion.div>
        ) : null}
      </div>

      {node.meta ? (
        <div className="font-mono text-xs text-muted-foreground">{node.meta}</div>
      ) : null}

      {node.badge ? (
        <Badge
          variant="outline"
          className="border-border/40 bg-background/60 text-[10px] uppercase tracking-wider text-muted-foreground"
        >
          {node.badge}
        </Badge>
      ) : null}
    </Card>
  );
}
