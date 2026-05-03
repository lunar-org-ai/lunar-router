import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import type { AgentNode } from '@/features/agents/types';

export type GraphFlowMode = 'discovery' | 'static';

type AgentGraphProps = {
  nodes: AgentNode[];
  revealedCount?: number;
  flowMode?: GraphFlowMode;
};

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

export function AgentGraph({ nodes, revealedCount, flowMode = 'static' }: AgentGraphProps) {
  const visibleCount = revealedCount ?? nodes.length;
  const visibleNodes = nodes.slice(0, visibleCount);
  const isStatic = flowMode === 'static';

  return (
    <div className="flex w-full flex-1 items-center justify-center px-8 py-16">
      <ol className="flex flex-col items-center">
        {visibleNodes.map((node, index) => {
          const cascadeDelay = isStatic ? index * 0.07 : 0;
          const nodeDelay = cascadeDelay + (index === 0 ? 0.04 : 0.18);
          return (
            <li key={node.id} className="flex flex-col items-center">
              {index > 0 ? <Connector delay={cascadeDelay} /> : null}
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: nodeDelay, ease: EASE_OUT_EXPO }}
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
      className="my-3 h-8 w-px origin-top bg-gradient-to-b from-transparent via-foreground/30 to-transparent"
      initial={{ scaleY: 0, opacity: 0 }}
      animate={{ scaleY: 1, opacity: 1 }}
      transition={{ duration: 0.42, delay, ease: EASE_OUT_EXPO }}
    />
  );
}
