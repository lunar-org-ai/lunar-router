import { motion } from 'framer-motion';

import { NodeCard } from '@/features/agents/components/NodeCard';
import type { AgentNode } from '@/features/agents/types';

export type GraphFlowMode = 'discovery' | 'build' | 'static';

type AgentGraphProps = {
  nodes: AgentNode[];
  revealedCount?: number;
  flowMode?: GraphFlowMode;
};

const MODERN_EASE = [0.22, 1, 0.36, 1] as const;

const SPRING_NODE = {
  type: 'spring',
  stiffness: 380,
  damping: 30,
  mass: 0.55,
} as const;

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
          const showPulse = isDiscovery && isLast;
          const nodeDelay = isBuild ? 0.06 + index * 0.04 : 0.08;

          return (
            <li key={node.id} className="flex flex-col items-center">
              {index > 0 ? (
                <Connector
                  flowing={isDiscovery}
                  connectorIndex={index - 1}
                  drawDelay={isBuild ? index * 0.04 : 0}
                />
              ) : null}

              <motion.div
                className="relative"
                initial={{ opacity: 0, y: isBuild ? 0 : 4, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ ...SPRING_NODE, delay: nodeDelay }}
              >
                <NodeCard node={node} />
                {showPulse ? <PulseRing /> : null}
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
  drawDelay: number;
};

function Connector({ flowing, connectorIndex, drawDelay }: ConnectorProps) {
  return (
    <div aria-hidden className="relative my-3 h-8 w-px">
      <svg
        width="2"
        height="32"
        viewBox="0 0 2 32"
        className="absolute -left-[0.5px] inset-y-0 overflow-visible text-border/60"
      >
        <motion.line
          x1="1"
          y1="0"
          x2="1"
          y2="32"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="3 3"
          strokeLinecap="round"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{
            duration: 0.35,
            delay: drawDelay,
            ease: MODERN_EASE,
          }}
        />
      </svg>

      {flowing ? (
        <motion.span
          aria-hidden
          className="absolute -left-[1.5px] size-[3px] rounded-full bg-emerald-400"
          style={{ filter: 'blur(0.4px)' }}
          initial={{ top: '-2px', opacity: 0 }}
          animate={{
            top: ['-2px', 'calc(100% + 2px)'],
            opacity: [0, 1, 1, 0],
          }}
          transition={{
            duration: 1.2,
            repeat: Infinity,
            ease: [0.4, 0, 0.6, 1],
            delay: 0.4 + connectorIndex * 0.16,
            times: [0, 0.15, 0.85, 1],
          }}
        />
      ) : null}
    </div>
  );
}

function PulseRing() {
  return (
    <motion.span
      aria-hidden
      className="pointer-events-none absolute inset-0 rounded-xl ring-2 ring-emerald-400/40"
      initial={{ opacity: 0, scale: 1 }}
      animate={{ opacity: [0, 0.6, 0.2, 0.6], scale: [1, 1.025, 1.01, 1.025] }}
      transition={{
        duration: 1.8,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  );
}
