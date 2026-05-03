import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { Button } from '@/components/ui/button';
import { AgentHeader } from '@/features/agents/components/AgentHeader';
import { OverviewTab } from '@/features/agents/components/OverviewTab';
import { useAgentsList } from '@/features/agents/hooks/useAgentsList';
import { cn } from '@/lib/utils';
import type { EvalMetric } from '@/features/agents/types';

const EVAL_DURATION_MS = 2200;

type TabKey = 'overview' | 'traces' | 'topology' | 'evals';

const TABS: { key: TabKey; label: string; enabled: boolean }[] = [
  { key: 'overview', label: 'Overview', enabled: true },
  { key: 'traces', label: 'Traces', enabled: false },
  { key: 'topology', label: 'Topology', enabled: false },
  { key: 'evals', label: 'Evals', enabled: false },
];

const perturbMetric = (metric: EvalMetric): EvalMetric => {
  if (typeof metric.value !== 'number') return metric;
  const delta = Math.round((Math.random() - 0.5) * 12);
  const next = Math.max(0, Math.min(100, metric.value + delta));
  return { ...metric, value: next };
};

const computeOverall = (metrics: EvalMetric[]): number => {
  const numerics = metrics
    .map((metric) => metric.value)
    .filter((value): value is number => typeof value === 'number');
  if (numerics.length === 0) return 0;
  const sum = numerics.reduce((acc, val) => acc + val, 0);
  return Math.round(sum / numerics.length);
};

export default function AgentDetailPage() {
  const { agentId } = useParams<{ agentId: string }>();
  const navigate = useNavigate();
  const { findAgent, removeAgent } = useAgentsList();

  const agent = agentId ? findAgent(agentId) : null;

  const [evaluating, setEvaluating] = useState(false);
  const [metrics, setMetrics] = useState<EvalMetric[]>(() => agent?.metrics ?? []);
  const [evalScore, setEvalScore] = useState<number>(() => agent?.evalScore ?? 0);
  const [evalRunId, setEvalRunId] = useState(0);
  const evalTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!agent) return;
    setMetrics(agent.metrics);
    setEvalScore(agent.evalScore);
  }, [agent?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    return () => {
      if (evalTimeoutRef.current) clearTimeout(evalTimeoutRef.current);
    };
  }, []);

  const handleRunEval = useCallback(() => {
    if (!agent || evaluating) return;
    setEvaluating(true);
    evalTimeoutRef.current = setTimeout(() => {
      const nextMetrics = metrics.map(perturbMetric);
      setMetrics(nextMetrics);
      setEvalScore(computeOverall(nextMetrics));
      setEvalRunId((prev) => prev + 1);
      setEvaluating(false);
    }, EVAL_DURATION_MS);
  }, [agent, evaluating, metrics]);

  const handleReset = useMemo(() => {
    if (!agent || agent.isMock) return undefined;
    return () => {
      removeAgent(agent.id);
      navigate('/agents');
    };
  }, [agent, removeAgent, navigate]);

  if (!agentId) {
    navigate('/agents', { replace: true });
    return null;
  }

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

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col bg-background">
      <AgentHeader
        agentName={agent.name}
        framework={agent.framework}
        status={agent.status}
        evalScore={evalScore}
        evalRunId={evalRunId}
        onRunEval={handleRunEval}
        onReset={handleReset}
        evaluating={evaluating}
      />
      <DetailTabs activeTab="overview" />
      <div className="flex-1 overflow-y-auto">
        <OverviewTab
          agent={agent}
          metrics={metrics}
          evalRunId={evalRunId}
          evaluating={evaluating}
        />
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
