import { useNavigate, useParams } from 'react-router-dom';

import { Button } from '@/components/ui/button';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { OverviewTab } from '@/features/agents/components/OverviewTab';
import { useAgentsList } from '@/features/agents/hooks/useAgentsList';
import { cn } from '@/lib/utils';

type TabKey = 'overview' | 'traces' | 'topology' | 'evals';

const TABS: { key: TabKey; label: string; enabled: boolean }[] = [
  { key: 'overview', label: 'Overview', enabled: true },
  { key: 'traces', label: 'Traces', enabled: false },
  { key: 'topology', label: 'Topology', enabled: false },
  { key: 'evals', label: 'Evals', enabled: false },
];

export default function AgentDetailPage() {
  const { agentId } = useParams<{ agentId: string }>();
  const navigate = useNavigate();
  const { findAgent, removeAgent } = useAgentsList();

  if (!agentId) {
    navigate('/agents', { replace: true });
    return null;
  }

  const agent = findAgent(agentId);

  if (!agent) {
    return (
      <div className="flex h-full min-h-0 flex-1 flex-col items-center justify-center gap-4 bg-background px-6 text-center">
        <h1 className="text-lg font-medium tracking-tight">Agent not found</h1>
        <p className="font-mono text-xs text-muted-foreground">
          We couldn't find an agent with id <code>{agentId}</code>.
        </p>
        <Button size="sm" variant="outline" onClick={() => navigate('/agents')}>
          Back to agents
        </Button>
      </div>
    );
  }

  const handleReset = agent.isMock
    ? undefined
    : () => {
        removeAgent(agent.id);
        navigate('/agents');
      };

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <AgentHeader
        agentName={agent.name}
        framework={agent.framework}
        status={agent.status}
        onReset={handleReset}
      />
      <DetailTabs activeTab="overview" />
      <div className="flex-1 overflow-y-auto">
        <OverviewTab agent={agent} />
      </div>
    </div>
  );
}

type DetailTabsProps = {
  activeTab: TabKey;
};

function DetailTabs({ activeTab }: DetailTabsProps) {
  return (
    <div className="flex items-center gap-1 border-b border-border/40 px-6">
      {TABS.map((tab) => {
        const isActive = tab.key === activeTab;
        return (
          <button
            key={tab.key}
            type="button"
            disabled={!tab.enabled}
            className={cn(
              'relative -mb-px flex items-center gap-2 border-b-2 px-3 py-3 text-sm transition-colors',
              isActive
                ? 'border-foreground text-foreground'
                : 'border-transparent text-muted-foreground/60',
              !tab.enabled
                ? 'cursor-not-allowed'
                : !isActive && 'hover:text-foreground'
            )}
          >
            {tab.label}
            {!tab.enabled ? (
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/40">
                Soon
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
