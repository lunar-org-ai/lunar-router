/**
 * Harness API service — connects UI to the agent system + ledger.
 */

import { useCallback, useMemo } from 'react';
import { API_BASE_URL } from '../config/api';

const BASE = API_BASE_URL;

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

export interface AgentConfig {
  name: string;
  description: string;
  model: string;
  temperature: number;
  max_tokens: number;
  output_schema: {
    type: string;
    fields: Record<string, { type: string; description?: string }>;
  };
  system_prompt: string;
}

export interface AgentRunResult {
  agent: string;
  result: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

export type ObjectiveDirection = 'higher_is_better' | 'lower_is_better';

export interface GuardrailSpec {
  type: string;
  threshold?: number | null;
  min_n?: number | null;
}

export interface Objective {
  id: string;
  description: string;
  compute_fn: string;
  unit: string;
  direction: ObjectiveDirection;
  baseline: number | null;
  target: number | null;
  window_hours: number;
  update_cadence: string;
  dimensions: string[];
  owner_agents: string[];
  guardrails: GuardrailSpec[];
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

export type LedgerEntryType =
  | 'signal'
  | 'run'
  | 'observation'
  | 'proposal'
  | 'decision'
  | 'action'
  | 'lesson';

export type LedgerOutcome = 'ok' | 'failed' | 'rolled_back' | 'skipped';

export interface LedgerEntry {
  id: string;
  ts: string;
  type: LedgerEntryType;
  objective_id: string | null;
  subject: string | null;
  agent: string | null;
  parameters_in: Record<string, unknown>;
  data: Record<string, unknown>;
  parent_id: string | null;
  tags: string[];
  duration_ms: number | null;
  cost_usd: number | null;
  outcome: LedgerOutcome | null;
}

export interface LedgerFilters {
  type?: LedgerEntryType | '';
  objective_id?: string;
  agent?: string;
  limit?: number;
}

export interface ObjectiveMeasurementPoint {
  ts: string;
  value: number;
  sample_size: number | null;
  id: string;
}

export interface ObjectiveTimeSeries {
  objective_id: string;
  window_hours: number;
  measurements: ObjectiveMeasurementPoint[];
  markers: LedgerEntry[];
}

// ---------------------------------------------------------------------------
// Proposals (Phase 1.5 — operator surface for write paths)
// ---------------------------------------------------------------------------

export type ProposalStatus =
  | 'pending'
  | 'approved'
  | 'rejected'
  | 'rejected_by_critic'
  | 'executed'
  | 'failed';

export interface CriticVerdict {
  decision: 'approve' | 'reject';
  rationale: string;
  estimated_cost_usd: number;
  estimated_benefit: string;
  decision_entry_id: string;
}

/**
 * A proposal is a `type='proposal'` ledger row. Status is computed
 * server-side by walking children for a `proposal_resolved` decision
 * tag. The verdict that gated the proposal is stored under `data`.
 */
export interface Proposal extends LedgerEntry {
  status: ProposalStatus;
}

export interface ProposalFilters {
  status?: ProposalStatus | '';
  objective_id?: string;
  limit?: number;
}

export interface ApproveResult {
  proposal_id: string;
  decision: 'approved' | 'rejected';
  decision_entry_id: string;
  verdict?: CriticVerdict;
}

// ---------------------------------------------------------------------------
// Setup status (drives the in-product Setup Guide)
// ---------------------------------------------------------------------------

export interface HarnessSetupStatus {
  configured_providers: string[];
  required_providers: string[];
  missing_providers: string[];
  critic: {
    agent: string;
    model: string | null;
    provider: string | null;
    ready: boolean;
  };
  mcp: {
    transport: 'http';
    path: string;
    name: string;
  };
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export function useHarnessService() {
  const listAgents = useCallback(async (): Promise<AgentConfig[]> => {
    const res = await fetch(`${BASE}/v1/harness/agents`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.agents || [];
  }, []);

  const getAgent = useCallback(async (name: string): Promise<AgentConfig | null> => {
    const res = await fetch(`${BASE}/v1/harness/agents/${name}`);
    if (!res.ok) return null;
    return res.json();
  }, []);

  const runAgent = useCallback(
    async (name: string, input: string, useTools = false): Promise<AgentRunResult> => {
      const res = await fetch(`${BASE}/v1/harness/run/${name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input, use_tools: useTools }),
      });
      if (!res.ok) throw new Error(`Agent run failed: ${res.status}`);
      return res.json();
    },
    []
  );

  const listObjectives = useCallback(async (): Promise<Objective[]> => {
    const res = await fetch(`${BASE}/v1/harness/objectives`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.objectives || [];
  }, []);

  const listLedgerEntries = useCallback(
    async (filters: LedgerFilters = {}): Promise<LedgerEntry[]> => {
      const params = new URLSearchParams();
      if (filters.type) params.set('type', filters.type);
      if (filters.objective_id) params.set('objective_id', filters.objective_id);
      if (filters.agent) params.set('agent', filters.agent);
      params.set('limit', String(filters.limit ?? 100));
      const res = await fetch(`${BASE}/v1/harness/ledger?${params.toString()}`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.entries || [];
    },
    []
  );

  const getLedgerEntry = useCallback(
    async (id: string): Promise<LedgerEntry | null> => {
      const res = await fetch(`${BASE}/v1/harness/ledger/${id}`);
      if (!res.ok) return null;
      return res.json();
    },
    []
  );

  const getChain = useCallback(async (id: string): Promise<LedgerEntry[]> => {
    const res = await fetch(`${BASE}/v1/harness/ledger/${id}/chain`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.entries || [];
  }, []);

  const getObjectiveTimeSeries = useCallback(
    async (objectiveId: string, hours = 168): Promise<ObjectiveTimeSeries> => {
      const params = new URLSearchParams({ hours: String(hours) });
      const res = await fetch(
        `${BASE}/v1/harness/objectives/${objectiveId}/time-series?${params.toString()}`,
      );
      if (!res.ok) {
        return { objective_id: objectiveId, window_hours: hours, measurements: [], markers: [] };
      }
      return res.json();
    },
    [],
  );

  const listProposals = useCallback(
    async (filters: ProposalFilters = {}): Promise<Proposal[]> => {
      const params = new URLSearchParams();
      if (filters.status) params.set('status', filters.status);
      if (filters.objective_id) params.set('objective_id', filters.objective_id);
      params.set('limit', String(filters.limit ?? 100));
      const res = await fetch(`${BASE}/v1/harness/proposals?${params.toString()}`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.proposals || [];
    },
    [],
  );

  const getProposal = useCallback(async (id: string): Promise<Proposal | null> => {
    const res = await fetch(`${BASE}/v1/harness/proposals/${id}`);
    if (!res.ok) return null;
    return res.json();
  }, []);

  const approveProposal = useCallback(async (id: string): Promise<ApproveResult> => {
    const res = await fetch(`${BASE}/v1/harness/proposals/${id}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`approve failed: ${res.status}`);
    return res.json();
  }, []);

  const rejectProposal = useCallback(
    async (id: string, reason = ''): Promise<ApproveResult> => {
      const res = await fetch(`${BASE}/v1/harness/proposals/${id}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      });
      if (!res.ok) throw new Error(`reject failed: ${res.status}`);
      return res.json();
    },
    [],
  );

  const getSetupStatus = useCallback(async (): Promise<HarnessSetupStatus | null> => {
    const res = await fetch(`${BASE}/v1/harness/setup-status`);
    if (!res.ok) return null;
    return res.json();
  }, []);

  const saveProviderKey = useCallback(
    async (provider: string, apiKey: string): Promise<void> => {
      const res = await fetch(`${BASE}/v1/secrets/${provider}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`save key failed (${res.status}): ${detail}`);
      }
    },
    [],
  );

  // Memoize the returned object so `service` has stable identity across
  // renders. Without this, every parent re-render produced a new `service`
  // object; any `useEffect(..., [service])` or `useCallback(..., [service])`
  // would then re-fire on every render, creating an infinite fetch loop
  // that tripped the browser's replaceState rate limit.
  return useMemo(
    () => ({
      listAgents,
      getAgent,
      runAgent,
      listObjectives,
      listLedgerEntries,
      getLedgerEntry,
      getChain,
      getObjectiveTimeSeries,
      listProposals,
      getProposal,
      approveProposal,
      rejectProposal,
      getSetupStatus,
      saveProviderKey,
    }),
    [
      listAgents,
      getAgent,
      runAgent,
      listObjectives,
      listLedgerEntries,
      getLedgerEntry,
      getChain,
      getObjectiveTimeSeries,
      listProposals,
      getProposal,
      approveProposal,
      rejectProposal,
      getSetupStatus,
      saveProviderKey,
    ],
  );
}
