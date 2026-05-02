import { useParams } from 'react-router-dom';

import { AgentGraph } from '@/features/agents/components/AgentGraph';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { AgentSidebar } from '@/features/agents/components/AgentSidebar';
import { EvalPanel } from '@/features/agents/components/EvalPanel';
import { mockSupportAgent } from '@/features/agents/data/mock-agent';

export default function AgentViewPage() {
  const { agentId } = useParams<{ agentId: string }>();
  const run = mockSupportAgent;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <AgentHeader
        agentId={agentId ?? run.id}
        agentName={run.agentName}
        version={run.version}
        overall={run.overall}
      />

      <div className="flex min-h-0 flex-1">
        <AgentSidebar />
        <main className="flex min-w-0 flex-1">
          <AgentGraph nodes={run.nodes} />
          <EvalPanel run={run} />
        </main>
      </div>
    </div>
  );
}
