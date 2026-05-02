import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import { cn } from '@/lib/utils';
import type { AgentNode } from '@/features/agents/types';

export type GraphFlowMode = 'discovery' | 'build' | 'static';

type AgentGraphProps = {
  nodes: AgentNode[];
  revealedCount?: number;
  flowMode?: GraphFlowMode;
};

export function AgentGraph({ nodes, revealedCount, flowMode = 'static' }: AgentGraphProps) {
  const visibleCount = revealedCount ?? nodes.length;
  const visibleNodes = nodes.slice(0, visibleCount);
  const lastIndex = visibleNodes.length - 1;
  const isDiscovery = flowMode === 'discovery';
  const isBuild = flowMode === 'build';

  return (
    <div className="flex w-full flex-1 items-center justify-center px-8 py-16">
      <ol className="flex flex-col items-center">
        {visibleNodes.map((node, index) => {
          const isLast = index === lastIndex;
          const pulse = isDiscovery && isLast;
          return (
            <li key={node.id} className="flex flex-col items-center">
              {index > 0 ? (
                <Connector
                  flowing={isDiscovery}
                  connectorIndex={index - 1}
                  buildMode={isBuild}
                />
              ) : null}
              <motion.div
                initial={isBuild ? { opacity: 0, scale: 0.94 } : { opacity: 0, y: 8, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{
                  duration: isBuild ? 0.3 : 0.3,
                  delay: isBuild ? 0.05 : 0,
                  ease: 'easeOut',
                }}
                className={cn(
                  'relative',
                  pulse &&
                    'after:pointer-events-none after:absolute after:inset-0 after:rounded-xl after:ring-2 after:ring-emerald-400/30 after:animate-pulse'
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

type ConnectorProps = {
  flowing: boolean;
  connectorIndex: number;
  buildMode: boolean;
};

function Connector({ flowing, connectorIndex, buildMode }: ConnectorProps) {
  return (
    <div aria-hidden className="relative my-3 h-8 w-px">
      <motion.div
        initial={{ scaleY: 0, opacity: 0 }}
        animate={{ scaleY: 1, opacity: 1 }}
        transition={{
          duration: buildMode ? 0.2 : 0.2,
          delay: buildMode ? connectorIndex * 0.04 : 0,
          ease: 'easeOut',
        }}
        style={{ originY: 0 }}
        className="absolute inset-0 border-l border-dashed border-border/50"
      />
      {flowing ? (
        <motion.span
          className="absolute -left-[2px] size-1 rounded-full bg-emerald-400/80 shadow-[0_0_6px_rgba(52,211,153,0.6)]"
          initial={{ top: '-4px', opacity: 0 }}
          animate={{ top: ['-4px', 'calc(100% + 4px)'], opacity: [0, 1, 1, 0] }}
          transition={{
            duration: 1.4,
            repeat: Infinity,
            ease: 'easeIn',
            delay: connectorIndex * 0.2,
            times: [0, 0.1, 0.9, 1],
          }}
        />
      ) : null}
    </div>
  );
}
