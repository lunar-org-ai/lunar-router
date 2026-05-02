import { useMemo } from 'react';

import { AgentGraph } from '@/features/agents/components/AgentGraph';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { EmptyState } from '@/features/agents/components/EmptyState';
import { EvalPanel } from '@/features/agents/components/EvalPanel';
import { OnboardingFlow } from '@/features/agents/components/OnboardingFlow';
import { useAgentImport } from '@/features/agents/hooks/useAgentImport';
import {
  AGENT_TEMPLATES,
  SUPPORT_TEMPLATE_ID,
  findTemplate,
} from '@/features/agents/templates';
import type { AgentRun } from '@/features/agents/types';

export default function AgentViewPage() {
  const fallbackTemplate = AGENT_TEMPLATES[0];

  const {
    phase,
    mode,
    modalStep,
    framework,
    templateId,
    name,
    revealedCount,
    runId,
    openImport,
    openCreate,
    closeModal,
    selectFramework,
    selectTemplate,
    setName,
    advanceStep,
    runSimulation,
    runEval,
    reset,
  } = useAgentImport(fallbackTemplate.nodes.length);

  const activeTemplate = useMemo(() => {
    if (mode === 'create') {
      return findTemplate(templateId) ?? fallbackTemplate;
    }
    return findTemplate(SUPPORT_TEMPLATE_ID) ?? fallbackTemplate;
  }, [mode, templateId, fallbackTemplate]);

  const displayName = mode === 'create' && name.trim() ? name.trim() : activeTemplate.name;

  const run: AgentRun = useMemo(
    () => ({
      id: activeTemplate.id,
      agentName: displayName,
      version: 'v1',
      runLabel: `Run ${runId || 1}`,
      nodes: activeTemplate.nodes,
      metrics: activeTemplate.metrics,
      overall: activeTemplate.overall,
      warning: activeTemplate.warning,
    }),
    [activeTemplate, displayName, runId]
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
          {showEmpty ? <EmptyState onImport={openImport} onCreate={openCreate} /> : null}

          {showOnboarding && mode ? (
            <OnboardingFlow
              mode={mode}
              step={modalStep}
              framework={framework}
              templateId={templateId}
              name={name}
              onSelectFramework={selectFramework}
              onSelectTemplate={selectTemplate}
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
                revealedCount={
                  phase === 'discovering' && mode === 'import'
                    ? revealedCount
                    : run.nodes.length
                }
                flowMode={
                  phase === 'discovering'
                    ? mode === 'create'
                      ? 'build'
                      : 'discovery'
                    : 'static'
                }
              />
              <EvalPanel run={run} evaluating={evaluating} runId={runId} />
            </>
          ) : null}
        </main>
      </div>
    </div>
  );
}
