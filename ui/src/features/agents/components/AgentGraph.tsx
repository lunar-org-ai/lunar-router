import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import type { AgentNode } from '@/features/agents/types';

type AgentGraphProps = {
  nodes: AgentNode[];
};

export function AgentGraph({ nodes }: AgentGraphProps) {
  return (
    <div className="flex w-full flex-1 items-center justify-center px-8 py-16">
      <ol className="flex flex-col items-center">
        {nodes.map((node, index) => (
          <li key={node.id} className="flex flex-col items-center">
            {index > 0 ? (
              <div
                aria-hidden
                className="my-3 h-8 border-l border-dashed border-border/50"
              />
            ) : null}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05, duration: 0.25, ease: 'easeOut' }}
            >
              <NodeCard node={node} />
            </motion.div>
          </li>
        ))}
      </ol>
    </div>
  );
}
