import { Plug } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { Button } from '@/components/ui/button';
import { AgentCard } from '@/features/agents/components/AgentCard';
import { OnboardingFlow } from '@/features/agents/components/OnboardingFlow';
import { useAgentImport } from '@/features/agents/hooks/useAgentImport';
import { useAgentsList } from '@/features/agents/hooks/useAgentsList';
import { DEFAULT_PROJECT_NAME, slugify } from '@/features/agents/state';

export default function AgentsIndexPage() {
  const navigate = useNavigate();
  const { agents, addAgent } = useAgentsList();

  const importer = useAgentImport({
    onComplete: ({ name, framework }) => {
      const trimmed = name.trim();
      const finalName = trimmed.length > 0 ? trimmed : DEFAULT_PROJECT_NAME;
      const slug = slugify(finalName);
      addAgent({ slug, name: finalName, framework });
      navigate(`/agents/${slug}`);
    },
  });

  const isImporting = importer.phase !== 'closed';

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <header className="flex items-center justify-between border-b border-border/40 px-6 py-5">
        <div className="flex flex-col gap-0.5">
          <h1 className="text-xl font-medium tracking-tight">Agents</h1>
          <span className="font-mono text-[11px] text-muted-foreground/70">
            {agents.length} {agents.length === 1 ? 'agent' : 'agents'} instrumented
          </span>
        </div>
        {!isImporting ? (
          <Button size="sm" onClick={importer.open} className="gap-1.5">
            <Plug className="size-3.5" />
            Import Agent
          </Button>
        ) : null}
      </header>

      {isImporting ? (
        <OnboardingFlow
          phase={importer.phase as Exclude<typeof importer.phase, 'closed'>}
          step={importer.step}
          framework={importer.framework}
          name={importer.name}
          onSelectFramework={importer.selectFramework}
          onSetName={importer.setName}
          onAdvance={importer.advance}
          onBack={importer.back}
          onSubmit={importer.submit}
          onCancel={importer.close}
        />
      ) : (
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {agents.map((agent, index) => (
              <AgentCard key={agent.id} agent={agent} index={index} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
