import { useParams } from 'react-router-dom';

import { AgentGraph } from '@/features/agents/components/AgentGraph';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { AgentSidebar } from '@/features/agents/components/AgentSidebar';
import { EmptyState } from '@/features/agents/components/EmptyState';
import { EvalPanel } from '@/features/agents/components/EvalPanel';
import { ImportAgentModal } from '@/features/agents/components/ImportAgentModal';
import { mockSupportAgent } from '@/features/agents/data/mock-agent';
import { useAgentImport } from '@/features/agents/hooks/useAgentImport';

export default function AgentViewPage() {
  const { agentId } = useParams<{ agentId: string }>();
  const run = mockSupportAgent;

  const {
    phase,
    modalStep,
    framework,
    revealedCount,
    runId,
    openImport,
    closeModal,
    selectFramework,
    advanceStep,
    runSimulation,
    runEval,
    reset,
  } = useAgentImport(run.nodes.length);

  const showEmpty = phase === 'empty';
  const showGraphArea = phase === 'discovering' || phase === 'evaluating' || phase === 'ready';
  const evaluating = phase === 'evaluating';
  const headerScoreState =
    phase === 'ready' ? 'visible' : phase === 'evaluating' ? 'evaluating' : 'pending';

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <AgentHeader
        agentId={agentId ?? run.id}
        agentName={run.agentName}
        version={run.version}
        overall={run.overall}
        showActions={!showEmpty}
        scoreState={headerScoreState}
        onRunEval={phase === 'ready' ? runEval : undefined}
        onReset={phase === 'ready' ? reset : undefined}
        evalDisabled={phase !== 'ready'}
      />

      <div className="flex min-h-0 flex-1">
        <AgentSidebar />
        <main className="flex min-w-0 flex-1">
          {showEmpty ? (
            <EmptyState onImport={openImport} />
          ) : (
            <>
              <AgentGraph
                nodes={run.nodes}
                revealedCount={phase === 'discovering' ? revealedCount : run.nodes.length}
                highlightLast={phase === 'discovering'}
              />
              {showGraphArea ? (
                <EvalPanel run={run} evaluating={evaluating} runId={runId} />
              ) : null}
            </>
          )}
        </main>
      </div>

      <ImportAgentModal
        open={phase === 'modal'}
        step={modalStep}
        framework={framework}
        onOpenChange={(open) => {
          if (!open) closeModal();
        }}
        onSelectFramework={selectFramework}
        onAdvance={advanceStep}
        onRunSimulation={runSimulation}
      />
    </div>
  );
}
