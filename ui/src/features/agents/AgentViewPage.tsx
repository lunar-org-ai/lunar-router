import { useMemo } from 'react';

import { AgentGraph } from '@/features/agents/components/AgentGraph';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { EmptyState } from '@/features/agents/components/EmptyState';
import { EvalPanel } from '@/features/agents/components/EvalPanel';
import { OnboardingFlow } from '@/features/agents/components/OnboardingFlow';
import { useAgentImport } from '@/features/agents/hooks/useAgentImport';
import { DEFAULT_PROJECT_NAME } from '@/features/agents/state';
import { SUPPORT_TEMPLATE } from '@/features/agents/templates';
import type { AgentRun } from '@/features/agents/types';

export default function AgentViewPage() {
  const {
    phase,
    modalStep,
    framework,
    name,
    revealedCount,
    runId,
    openImport,
    closeModal,
    selectFramework,
    setName,
    advanceStep,
    runSimulation,
    runEval,
    reset,
  } = useAgentImport(SUPPORT_TEMPLATE.nodes.length);

  const trimmedName = name.trim();
  const displayName = trimmedName.length > 0 ? trimmedName : DEFAULT_PROJECT_NAME;

  const run: AgentRun = useMemo(
    () => ({
      id: SUPPORT_TEMPLATE.id,
      agentName: displayName,
      version: 'v1',
      runLabel: `Run ${runId || 1}`,
      nodes: SUPPORT_TEMPLATE.nodes,
      metrics: SUPPORT_TEMPLATE.metrics,
      overall: SUPPORT_TEMPLATE.overall,
      warning: SUPPORT_TEMPLATE.warning,
    }),
    [displayName, runId]
  );

  const showEmpty = phase === 'empty';
  const showOnboarding = phase === 'modal';
  const showGraphArea = phase === 'discovering' || phase === 'evaluating' || phase === 'ready';
  const evaluating = phase === 'evaluating';
  const headerScoreState =
    phase === 'ready' ? 'visible' : phase === 'evaluating' ? 'evaluating' : 'pending';

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <AgentHeader
        agentName={run.agentName}
        overall={run.overall}
        showActions={showGraphArea}
        scoreState={headerScoreState}
        onRunEval={phase === 'ready' ? runEval : undefined}
        onReset={phase === 'ready' ? reset : undefined}
        evalDisabled={phase !== 'ready'}
      />

      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1">
          {showEmpty ? <EmptyState onImport={openImport} /> : null}

          {showOnboarding ? (
            <OnboardingFlow
              step={modalStep}
              framework={framework}
              name={name}
              onSelectFramework={selectFramework}
              onSetName={setName}
              onAdvance={advanceStep}
              onRunSimulation={runSimulation}
              onCancel={closeModal}
            />
          ) : null}

          {showGraphArea ? (
            <>
              <AgentGraph
                nodes={run.nodes}
                revealedCount={phase === 'discovering' ? revealedCount : run.nodes.length}
                flowMode={phase === 'discovering' ? 'discovery' : 'static'}
              />
              <EvalPanel run={run} evaluating={evaluating} runId={runId} />
            </>
          ) : null}
        </main>
      </div>
    </div>
  );
}
