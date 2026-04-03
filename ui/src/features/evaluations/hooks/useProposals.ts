import { useState, useCallback, useEffect, useMemo } from 'react';
import { toast } from 'sonner';
import { useEvaluationsService } from '../api/evaluationsService';
import type { Proposal } from '../types/evaluationsTypes';

function normalizeProposal(p: Record<string, unknown>): Proposal {
  return { ...p, id: (p.proposal_id as string) || (p.id as string) } as Proposal;
}

export function useProposals() {
  const accessToken = '';
  const service = useEvaluationsService();

  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await service.listProposals(accessToken);
      setProposals((res.proposals || []).map(normalizeProposal));
    } catch (err) {
      console.error('[useProposals] refresh failed:', err);
    } finally {
      setLoading(false);
    }
  }, [service]);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    service
      .listProposals(accessToken)
      .then((res) => {
        if (!cancelled) setProposals((res.proposals || []).map(normalizeProposal));
      })
      .catch((err) => console.error('[useProposals] load failed:', err))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [service]);

  const approveProposal = useCallback(
    async (id: string) => {
      setProposals((prev) => prev.filter((p) => p.id !== id)); // optimistic
      try {
        const result = await service.approveProposal(accessToken, id);
        toast.success('Proposal approved');
        return result.execution_result;
      } catch (err) {
        console.error('[useProposals] approve failed:', err);
        toast.error('Failed to approve proposal');
        refresh();
        return null;
      }
    },
    [service, refresh]
  );

  const rejectProposal = useCallback(
    async (id: string, reason?: string) => {
      setProposals((prev) => prev.filter((p) => p.id !== id)); // optimistic
      try {
        await service.rejectProposal(accessToken, id, reason);
        toast.success('Proposal dismissed');
      } catch (err) {
        console.error('[useProposals] reject failed:', err);
        toast.error('Failed to dismiss proposal');
        refresh();
      }
    },
    [service, refresh]
  );

  const pendingCount = useMemo(
    () => proposals.filter((p) => p.status === 'pending').length,
    [proposals]
  );

  return { proposals, loading, pendingCount, approveProposal, rejectProposal, refresh };
}
