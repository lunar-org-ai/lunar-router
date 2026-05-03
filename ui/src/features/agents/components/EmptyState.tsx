import { Plug, Workflow } from 'lucide-react';
import { motion } from 'framer-motion';

import { Button } from '@/components/ui/button';

type EmptyStateProps = {
  onImport: () => void;
};

export function EmptyState({ onImport }: EmptyStateProps) {
  return (
    <div className="flex flex-1 items-center justify-center px-8">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="flex max-w-md flex-col items-center gap-6 text-center"
      >
        <div className="flex size-14 items-center justify-center rounded-2xl border border-border/40 bg-card/40 text-muted-foreground">
          <Workflow className="size-6" />
        </div>

        <div className="flex flex-col gap-2">
          <h2 className="text-lg font-medium tracking-tight">Import an agent</h2>
          <p className="font-mono text-xs leading-relaxed text-muted-foreground">
            Bring in an existing agent built with LangChain, LangGraph, CrewAI, or the OpenAI
            Agents SDK — OpenTracy auto-instruments it from a single import.
          </p>
        </div>

        <Button onClick={onImport} className="gap-2">
          <Plug className="size-3.5" />
          Import Agent
        </Button>
      </motion.div>
    </div>
  );
}
