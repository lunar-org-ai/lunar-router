import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import type { AgentNode } from '@/features/agents/types';

export type GraphFlowMode = 'discovery' | 'build' | 'static';

type AgentGraphProps = {
  nodes: AgentNode[];
  revealedCount?: number;
  flowMode?: GraphFlowMode;
};

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

export function AgentGraph({ nodes, revealedCount, flowMode = 'static' }: AgentGraphProps) {
  const visibleCount = revealedCount ?? nodes.length;
  const visibleNodes = nodes.slice(0, visibleCount);
  const isBuild = flowMode === 'build';

  return (
    <div className="flex w-full flex-1 items-center justify-center px-8 py-16">
      <ol className="flex flex-col items-center">
        {visibleNodes.map((node, index) => {
          const nodeDelay = isBuild ? 0.04 + index * 0.025 : 0.05;
          const connectorDelay = isBuild ? index * 0.025 : 0;

          return (
            <li key={node.id} className="flex flex-col items-center">
              {index > 0 ? <Connector delay={connectorDelay} /> : null}

              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.28,
                  delay: nodeDelay,
                  ease: EASE_OUT_EXPO,
                }}
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
  delay: number;
};

function Connector({ delay }: ConnectorProps) {
  return (
    <motion.div
      aria-hidden
      className="my-3 h-8 border-l border-dashed border-border/40"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.22, delay, ease: EASE_OUT_EXPO }}
    />
  );
}
