import { useCallback, useEffect, useMemo, useState } from 'react';

import { MOCK_AGENTS, buildFreshAgent } from '@/features/agents/mockAgents';
import {
  type StoredAgent,
  clearStoredAgents,
  loadStoredAgents,
  saveStoredAgents,
} from '@/features/agents/state';
import type { AgentFramework, AgentSummary } from '@/features/agents/types';

type AddAgentInput = {
  slug: string;
  name: string;
  framework: AgentFramework;
};

type UseAgentsListResult = {
  agents: AgentSummary[];
  findAgent: (id: string) => AgentSummary | null;
  addAgent: (input: AddAgentInput) => AgentSummary;
  removeAgent: (slug: string) => void;
  resetAll: () => void;
};

export function useAgentsList(): UseAgentsListResult {
  const [stored, setStored] = useState<StoredAgent[]>(() => loadStoredAgents());

  useEffect(() => {
    saveStoredAgents(stored);
  }, [stored]);

  const agents = useMemo<AgentSummary[]>(() => {
    const imported = stored.map((entry) =>
      buildFreshAgent(entry.slug, entry.name, entry.framework)
    );
    return [...imported, ...MOCK_AGENTS];
  }, [stored]);

  const findAgent = useCallback(
    (id: string) => agents.find((agent) => agent.id === id) ?? null,
    [agents]
  );

  const addAgent = useCallback(
    ({ slug, name, framework }: AddAgentInput) => {
      const importedAt = new Date().toISOString();
      setStored((prev) => {
        const without = prev.filter((entry) => entry.slug !== slug);
        return [{ slug, name, framework, importedAt }, ...without];
      });
      return buildFreshAgent(slug, name, framework);
    },
    []
  );

  const removeAgent = useCallback((slug: string) => {
    setStored((prev) => prev.filter((entry) => entry.slug !== slug));
  }, []);

  const resetAll = useCallback(() => {
    clearStoredAgents();
    setStored([]);
  }, []);

  return { agents, findAgent, addAgent, removeAgent, resetAll };
}
