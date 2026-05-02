import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import { cn } from '@/lib/utils';
import type { AgentNode } from '@/features/agents/types';

type AgentGraphProps = {
  nodes: AgentNode[];
  revealedCount?: number;
  highlightLast?: boolean;
};

export function AgentGraph({ nodes, revealedCount, highlightLast = false }: AgentGraphProps) {
  const visibleCount = revealedCount ?? nodes.length;
  const visibleNodes = nodes.slice(0, visibleCount);
  const lastIndex = visibleNodes.length - 1;

  return (
    <div className="flex w-full flex-1 items-center justify-center px-8 py-16">
      <ol className="flex flex-col items-center">
        {visibleNodes.map((node, index) => {
          const isLast = index === lastIndex;
          const pulse = highlightLast && isLast;
          return (
            <li key={node.id} className="flex flex-col items-center">
              {index > 0 ? (
                <motion.div
                  aria-hidden
                  initial={{ scaleY: 0, opacity: 0 }}
                  animate={{ scaleY: 1, opacity: 1 }}
                  transition={{ duration: 0.2, ease: 'easeOut' }}
                  style={{ originY: 0 }}
                  className="my-3 h-8 border-l border-dashed border-border/50"
                />
              ) : null}
              <motion.div
                initial={{ opacity: 0, y: 8, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className={cn(
                  'relative',
                  pulse && 'after:pointer-events-none after:absolute after:inset-0 after:rounded-xl after:ring-2 after:ring-emerald-400/30 after:animate-pulse'
                )}
              >
                <NodeCard node={node} />
              </motion.div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
