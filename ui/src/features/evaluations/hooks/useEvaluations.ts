import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useEvaluationsService } from '../api/evaluationsService';
import type {
  Evaluation,
  EvaluationResults,
  EvaluationStatus,
  EvaluationStatusResponse,
  CreateEvaluationRequest,
} from '../types';

const POLLING_INTERVALS: Record<EvaluationStatus, number> = {
  queued: 2000,
  starting: 2000,
  running: 3000,
  completed: 0,
  failed: 0,
  cancelled: 0,
};

const ACTIVE_STATUSES: EvaluationStatus[] = ['queued', 'starting', 'running'];
const TERMINAL_STATUSES: EvaluationStatus[] = ['completed', 'failed', 'cancelled'];

export function useEvaluations() {
  const accessToken = 'no-auth';
  const service = useEvaluationsService();
  const loaded = useRef(false);

  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const activeEvaluations = useMemo(
    () => evaluations.filter((e) => ACTIVE_STATUSES.includes(e.status)),
    [evaluations]
  );

  const activeSignature = useMemo(
    () =>
      activeEvaluations
        .map((e) => `${e.id}:${e.status}`)
        .sort()
        .join(','),
    [activeEvaluations]
  );

  const hasQueuedOrStarting = useMemo(
    () => activeEvaluations.some((e) => e.status === 'queued' || e.status === 'starting'),
    [activeEvaluations]
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const fresh = await service.listEvaluations(accessToken);
      setEvaluations((prev) =>
        fresh.map((f) => {
          const existing = prev.find((e) => e.id === f.id);
          return existing?.error && !f.error && f.status === 'failed'
            ? { ...f, error: existing.error }
            : f;
        })
      );
    } catch (err) {
      console.error('[useEvaluations] refresh failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch evaluations');
    } finally {
      setLoading(false);
    }
  }, [accessToken, service]);

  const create = useCallback(
    async (request: CreateEvaluationRequest): Promise<Evaluation | null> => {
      try {
        const evaluation = await service.createEvaluation(accessToken, request);
        setEvaluations((prev) => [evaluation, ...prev]);
        return evaluation;
      } catch (err) {
        console.error('[useEvaluations] create failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to create evaluation');
        return null;
      }
    },
    [accessToken, service]
  );

  const remove = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        await service.deleteEvaluation(accessToken, id);
        setEvaluations((prev) => prev.filter((e) => e.id !== id));
        return true;
      } catch (err) {
        console.error('[useEvaluations] delete failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to delete evaluation');
        return false;
      }
    },
    [accessToken, service]
  );

  const cancel = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        const ok = await service.cancelEvaluation(accessToken, id);
        if (ok) {
          setEvaluations((prev) =>
            prev.map((e) => (e.id === id ? { ...e, status: 'cancelled' as EvaluationStatus } : e))
          );
        }
        return ok;
      } catch (err) {
        console.error('[useEvaluations] cancel failed:', err);
        setError(err instanceof Error ? err.message : 'Failed to cancel evaluation');
        return false;
      }
    },
    [accessToken, service]
  );

  const getResults = useCallback(
    async (id: string): Promise<{ results: EvaluationResults; samples_total?: number } | null> => {
      try {
        return await service.getEvaluationResults(accessToken, id, { include_samples: true });
      } catch (err) {
        console.error('[useEvaluations] getResults failed:', err);
        return null;
      }
    },
    [accessToken, service]
  );

  const getStatus = useCallback(
    async (id: string): Promise<EvaluationStatusResponse | null> => {
      try {
        return await service.getEvaluationStatus(accessToken, id);
      } catch (err) {
        console.error('[useEvaluations] getStatus failed:', err);
        return null;
      }
    },
    [accessToken, service]
  );

  const getOne = useCallback(
    async (id: string): Promise<Evaluation | null> => {
      try {
        return await service.getEvaluation(accessToken, id);
      } catch (err) {
        console.error('[useEvaluations] getOne failed:', err);
        return null;
      }
    },
    [accessToken, service]
  );

  const clearError = useCallback(() => setError(null), []);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (activeEvaluations.length === 0) return;

    const interval = hasQueuedOrStarting ? POLLING_INTERVALS.queued : POLLING_INTERVALS.running;

    const pollAll = async () => {
      const results = await Promise.allSettled(
        activeEvaluations.map(async (ev) => {
          const status = await service.getEvaluationStatus(accessToken, ev.id);
          return { id: ev.id, status };
        })
      );

      const updates = results
        .filter(
          (r): r is PromiseFulfilledResult<{ id: string; status: EvaluationStatusResponse }> =>
            r.status === 'fulfilled'
        )
        .map((r) => r.value);

      if (updates.length === 0) return;

      setEvaluations((prev) =>
        prev.map((e) => {
          const u = updates.find((upd) => upd.id === e.id);
          if (!u) return e;
          return {
            ...e,
            status: u.status.status,
            progress: u.status.progress,
            started_at: u.status.started_at,
            updated_at: u.status.updated_at,
            error: u.status.error,
          };
        })
      );

      const hasTerminal = updates.some((u) => TERMINAL_STATUSES.includes(u.status.status));
      if (hasTerminal) setTimeout(refresh, 500);
    };

    pollAll();
    const id = setInterval(pollAll, interval);
    return () => clearInterval(id);
  }, [activeEvaluations, activeSignature, hasQueuedOrStarting, refresh, service]);

  return {
    evaluations,
    loading,
    error,
    refreshEvaluations: refresh,
    createEvaluation: create,
    deleteEvaluation: remove,
    cancelEvaluation: cancel,
    getEvaluation: getOne,
    getEvaluationResults: getResults,
    getEvaluationStatus: getStatus,
    clearError,
  };
}
